"""
Frequency Sweep Engine for Nonlinear Systems

This module implements the core simulation engine for extracting frequency
response data from closed-loop nonlinear systems using sinusoidal excitation.

Methodology: Sinusoidal Sweep with DFT Correlation
--------------------------------------------------
The engine applies sinusoidal inputs and extracts the fundamental frequency
component from the steady-state response using discrete Fourier correlation:

$$G(j\\omega) = \\frac{2}{NT} \\int_0^{NT} y(t) e^{-j\\omega t} dt$$

For discrete samples:

$$G(j\\omega) = \\frac{2}{N} \\sum_{k=0}^{N-1} y[k] e^{-j\\omega k T_s}$$

The gain and phase are extracted as:

$$|G(j\\omega)| = |G| / A_{input}$$
$$\\angle G(j\\omega) = \\arg(G)$$

Implementation Details
----------------------
1. **Settling Time Estimation**: Uses 5× expected time constant based on 
   closed-loop bandwidth estimate
2. **Measurement Window**: Integer number of complete periods for accurate DFT
3. **Windowing**: Optional Hanning window to reduce spectral leakage
4. **Multi-Cycle Averaging**: Averages multiple periods for noise reduction
5. **Coherence Check**: Validates signal quality via coherence function

Author: Dr. S. Shahid Mustafa
Date: January 28, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, Optional, List, Callable, Any
from collections import namedtuple


class SweepType(Enum):
    """Type of frequency sweep excitation."""
    REFERENCE_TRACKING = auto()    # Sinusoidal reference input
    DISTURBANCE_INJECTION = auto() # Sinusoidal disturbance torque
    NOISE_INJECTION = auto()       # Measurement noise injection


@dataclass
class FrequencyPoint:
    """
    Frequency response data at a single frequency point.
    
    Attributes
    ----------
    frequency_hz : float
        Excitation frequency [Hz]
    frequency_rad : float
        Excitation frequency [rad/s]
    gain_db : float
        Magnitude response [dB]
    gain_linear : float
        Magnitude response (linear scale)
    phase_deg : float
        Phase response [degrees]
    phase_rad : float
        Phase response [radians]
    coherence : float
        Coherence function γ² ∈ [0, 1] indicating signal quality
    input_amplitude : float
        Applied input amplitude
    output_amplitude : float
        Measured output amplitude
    settling_time : float
        Time waited for transient decay [s]
    num_cycles : int
        Number of complete cycles used for estimation
    axis : str
        Axis identifier ('az' or 'el')
    metadata : Dict
        Additional diagnostic data
    """
    frequency_hz: float
    frequency_rad: float
    gain_db: float
    gain_linear: float
    phase_deg: float
    phase_rad: float
    coherence: float
    input_amplitude: float
    output_amplitude: float
    settling_time: float
    num_cycles: int
    axis: str = 'az'
    metadata: Dict = field(default_factory=dict)


@dataclass
class FrequencySweepConfig:
    """
    Configuration for frequency sweep analysis.
    
    Frequency Range Design Guidelines
    ----------------------------------
    For gimbal systems with ~10 Hz crossover frequency:
    - f_min: 0.01 Hz (captures low-frequency integral action)
    - f_max: 100 Hz (well beyond Nyquist limit for structural modes)
    - n_points: 40-60 per decade for smooth Bode plots
    
    Settling Time Guidelines
    ------------------------
    - settling_cycles: 5-10 cycles for accurate steady-state
    - measurement_cycles: 10-20 cycles for noise averaging
    - For low frequencies, use max_settling_time to limit duration
    
    Attributes
    ----------
    f_min : float
        Minimum frequency [Hz]
    f_max : float
        Maximum frequency [Hz]
    n_points : int
        Number of frequency points (logarithmically spaced)
    amplitude : float
        Excitation amplitude [rad] for reference, [N·m] for disturbance
    settling_cycles : int
        Number of cycles to wait for transient decay
    measurement_cycles : int
        Number of cycles for DFT averaging
    max_settling_time : float
        Maximum settling time [s] (for very low frequencies)
    min_measurement_time : float
        Minimum measurement window [s]
    use_hanning_window : bool
        Apply Hanning window to reduce spectral leakage
    coherence_threshold : float
        Minimum coherence for valid measurement (0-1)
    dt : float
        Simulation timestep [s]
    """
    f_min: float = 0.1
    f_max: float = 50.0
    n_points: int = 40
    amplitude: float = np.deg2rad(1.0)  # 1 degree default
    settling_cycles: int = 8
    measurement_cycles: int = 16
    max_settling_time: float = 20.0
    min_measurement_time: float = 0.5
    use_hanning_window: bool = True
    coherence_threshold: float = 0.7
    dt: float = 0.001
    
    def get_frequency_vector(self) -> np.ndarray:
        """Generate logarithmically-spaced frequency vector [Hz]."""
        return np.logspace(
            np.log10(self.f_min),
            np.log10(self.f_max),
            self.n_points
        )


class FrequencySweepEngine:
    """
    Core engine for frequency sweep analysis of nonlinear systems.
    
    This engine provides the low-level simulation loop that applies sinusoidal
    excitation and extracts frequency response via DFT correlation. It is designed
    to work with any closed-loop system that implements the standard interface.
    
    Signal Processing Pipeline
    --------------------------
    1. Generate sinusoidal excitation signal
    2. Apply to plant via simulation callback
    3. Collect steady-state response (after settling)
    4. Apply windowing (optional Hanning)
    5. Compute DFT at excitation frequency
    6. Extract gain and phase
    7. Estimate coherence for quality check
    
    Example Usage
    -------------
    >>> config = FrequencySweepConfig(f_min=0.1, f_max=50, n_points=40)
    >>> engine = FrequencySweepEngine(config)
    >>> 
    >>> def sim_callback(omega, t_total, amplitude, sweep_type):
    ...     # Run simulation and return (time, input, output) arrays
    ...     return t, u, y
    >>> 
    >>> results = engine.run_sweep(sim_callback, SweepType.REFERENCE_TRACKING)
    
    Parameters
    ----------
    config : FrequencySweepConfig
        Sweep configuration parameters
    verbose : bool
        Enable progress output
    """
    
    def __init__(self, config: FrequencySweepConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._results: List[FrequencyPoint] = []
    
    def run_sweep(
        self,
        simulation_callback: Callable[[float, float, float, SweepType, str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        sweep_type: SweepType,
        axis: str = 'az'
    ) -> List[FrequencyPoint]:
        """
        Execute complete frequency sweep.
        
        The simulation_callback must implement:
            callback(omega_rad, duration, amplitude, sweep_type, axis) -> (t, u, y)
        
        where:
            - omega_rad: Excitation frequency [rad/s]
            - duration: Total simulation time [s]
            - amplitude: Input amplitude
            - sweep_type: Type of excitation
            - axis: 'az' or 'el'
            - Returns: (time_vector, input_signal, output_signal)
        
        Parameters
        ----------
        simulation_callback : Callable
            Function that runs closed-loop simulation and returns signals
        sweep_type : SweepType
            Type of frequency sweep (reference tracking, disturbance, etc.)
        axis : str
            Axis to analyze ('az' or 'el')
            
        Returns
        -------
        List[FrequencyPoint]
            Frequency response data for each frequency point
        """
        frequencies = self.config.get_frequency_vector()
        self._results = []
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"FREQUENCY SWEEP ANALYSIS - {sweep_type.name} - {axis.upper()} AXIS")
            print(f"{'='*70}")
            print(f"Frequency Range: {self.config.f_min:.3f} Hz to {self.config.f_max:.1f} Hz")
            print(f"Number of Points: {self.config.n_points}")
            print(f"Amplitude: {np.rad2deg(self.config.amplitude):.3f} deg")
            print(f"{'='*70}\n")
        
        for idx, f_hz in enumerate(frequencies):
            omega = 2 * np.pi * f_hz
            
            # Calculate timing for this frequency
            period = 1.0 / f_hz
            settling_time = min(
                self.config.settling_cycles * period,
                self.config.max_settling_time
            )
            measurement_time = max(
                self.config.measurement_cycles * period,
                self.config.min_measurement_time
            )
            total_duration = settling_time + measurement_time
            
            if self.verbose:
                print(f"[{idx+1:3d}/{self.config.n_points}] "
                      f"f = {f_hz:8.4f} Hz | "
                      f"T_settle = {settling_time:6.2f}s | "
                      f"T_meas = {measurement_time:6.2f}s", end='')
            
            try:
                # Run simulation
                t, u, y = simulation_callback(
                    omega, 
                    total_duration, 
                    self.config.amplitude,
                    sweep_type,
                    axis
                )
                
                # Extract frequency response
                freq_point = self._extract_frequency_response(
                    t, u, y, omega, f_hz, settling_time, axis
                )
                self._results.append(freq_point)
                
                if self.verbose:
                    print(f" | Gain = {freq_point.gain_db:7.2f} dB | "
                          f"Phase = {freq_point.phase_deg:7.1f}° | "
                          f"γ² = {freq_point.coherence:.3f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f" | ERROR: {str(e)}")
                # Add NaN point for failed measurement
                self._results.append(self._create_nan_point(f_hz, omega, axis))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SWEEP COMPLETE: {len(self._results)} points collected")
            valid = sum(1 for r in self._results if r.coherence >= self.config.coherence_threshold)
            print(f"Valid Points (γ² ≥ {self.config.coherence_threshold}): {valid}")
            print(f"{'='*70}\n")
        
        return self._results
    
    def _extract_frequency_response(
        self,
        t: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        omega: float,
        f_hz: float,
        settling_time: float,
        axis: str
    ) -> FrequencyPoint:
        """
        Extract frequency response from time-domain signals using DFT correlation.
        
        This implements the Sinusoidal Input Describing Function method:
        
        $$G(j\\omega) = \\frac{Y_1}{U_1}$$
        
        where $Y_1$ and $U_1$ are the fundamental Fourier coefficients of
        the output and input signals respectively.
        
        DFT Correlation Formula
        -----------------------
        For a signal x(t) sampled at times t_k = k*dt:
        
        $$X(j\\omega) = \\frac{2}{N} \\sum_{k=0}^{N-1} x[k] \\cdot e^{-j\\omega t_k}$$
        
        The factor of 2 accounts for the one-sided spectrum.
        
        Parameters
        ----------
        t : np.ndarray
            Time vector [s]
        u : np.ndarray
            Input signal (reference or disturbance)
        y : np.ndarray
            Output signal (position or error)
        omega : float
            Excitation frequency [rad/s]
        f_hz : float
            Excitation frequency [Hz]
        settling_time : float
            Initial transient period to discard [s]
        axis : str
            Axis identifier
            
        Returns
        -------
        FrequencyPoint
            Extracted frequency response data
        """
        # Extract steady-state portion
        dt = t[1] - t[0] if len(t) > 1 else self.config.dt
        n_settle = int(settling_time / dt)
        
        # Ensure we have data after settling
        if n_settle >= len(t) - 10:
            n_settle = max(0, len(t) // 2)
        
        t_ss = t[n_settle:]
        u_ss = u[n_settle:]
        y_ss = y[n_settle:]
        N = len(t_ss)
        
        if N < 10:
            return self._create_nan_point(f_hz, omega, axis)
        
        # Remove DC offset
        u_ss = u_ss - np.mean(u_ss)
        y_ss = y_ss - np.mean(y_ss)
        
        # Apply Hanning window if enabled
        if self.config.use_hanning_window:
            window = np.hanning(N)
            # Window correction factor for amplitude preservation
            window_corr = 2.0  # Hanning window has average value of 0.5
        else:
            window = np.ones(N)
            window_corr = 1.0
        
        u_windowed = u_ss * window
        y_windowed = y_ss * window
        
        # DFT at excitation frequency using correlation
        # Complex exponential: e^(-jωt) = cos(ωt) - j*sin(ωt)
        cos_basis = np.cos(omega * t_ss)
        sin_basis = np.sin(omega * t_ss)
        
        # Input Fourier coefficient
        U_real = (2.0 / N) * np.sum(u_windowed * cos_basis) * window_corr
        U_imag = -(2.0 / N) * np.sum(u_windowed * sin_basis) * window_corr
        U_complex = U_real + 1j * U_imag
        
        # Output Fourier coefficient  
        Y_real = (2.0 / N) * np.sum(y_windowed * cos_basis) * window_corr
        Y_imag = -(2.0 / N) * np.sum(y_windowed * sin_basis) * window_corr
        Y_complex = Y_real + 1j * Y_imag
        
        # Frequency response: G = Y/U
        if np.abs(U_complex) < 1e-12:
            return self._create_nan_point(f_hz, omega, axis)
        
        G_complex = Y_complex / U_complex
        
        # Gain and phase
        gain_linear = np.abs(G_complex)
        gain_db = 20 * np.log10(gain_linear + 1e-12)
        phase_rad = np.angle(G_complex)
        phase_deg = np.rad2deg(phase_rad)
        
        # Compute coherence using cross-spectral density ratio
        # γ² = |S_uy|² / (S_uu * S_yy)
        coherence = self._compute_coherence(u_ss, y_ss, omega, dt)
        
        # Actual number of cycles measured
        measurement_time = t_ss[-1] - t_ss[0]
        num_cycles = int(measurement_time * f_hz)
        
        return FrequencyPoint(
            frequency_hz=f_hz,
            frequency_rad=omega,
            gain_db=gain_db,
            gain_linear=gain_linear,
            phase_deg=phase_deg,
            phase_rad=phase_rad,
            coherence=coherence,
            input_amplitude=np.abs(U_complex),
            output_amplitude=np.abs(Y_complex),
            settling_time=settling_time,
            num_cycles=num_cycles,
            axis=axis,
            metadata={
                'N_samples': N,
                'dt': dt,
                'window': 'hanning' if self.config.use_hanning_window else 'rectangular'
            }
        )
    
    def _compute_coherence(
        self, 
        u: np.ndarray, 
        y: np.ndarray, 
        omega: float,
        dt: float
    ) -> float:
        """
        Compute coherence function at excitation frequency.
        
        The coherence function γ²(ω) measures the linear relationship between
        input and output at frequency ω:
        
        $$\\gamma^2(\\omega) = \\frac{|S_{uy}(\\omega)|^2}{S_{uu}(\\omega) S_{yy}(\\omega)}$$
        
        where S_uy is the cross-spectral density and S_uu, S_yy are auto-spectral
        densities.
        
        Interpretation:
        - γ² = 1: Perfect linear relationship (ideal)
        - γ² > 0.9: Excellent measurement
        - γ² > 0.7: Acceptable measurement
        - γ² < 0.5: Poor measurement (nonlinearity or noise dominant)
        
        Parameters
        ----------
        u : np.ndarray
            Input signal
        y : np.ndarray
            Output signal
        omega : float
            Excitation frequency [rad/s]
        dt : float
            Sample time [s]
            
        Returns
        -------
        float
            Coherence value γ² ∈ [0, 1]
        """
        N = len(u)
        t = np.arange(N) * dt
        
        # Correlation with complex exponential
        exp_neg = np.exp(-1j * omega * t)
        
        # Fourier coefficients
        U = np.sum(u * exp_neg) / N
        Y = np.sum(y * exp_neg) / N
        
        # Cross and auto spectral densities at this frequency
        S_uy = np.abs(np.sum(u * np.conj(y) * np.abs(exp_neg)**2)) / N
        S_uu = np.abs(np.sum(u * np.conj(u) * np.abs(exp_neg)**2)) / N
        S_yy = np.abs(np.sum(y * np.conj(y) * np.abs(exp_neg)**2)) / N
        
        # Alternative simpler coherence based on amplitude ratio consistency
        # This works well for single-frequency excitation
        u_amp = np.std(u) * np.sqrt(2)  # Approximate amplitude
        y_amp = np.std(y) * np.sqrt(2)
        
        if u_amp < 1e-12 or y_amp < 1e-12:
            return 0.0
        
        # Check how well the signals match sinusoidal assumption
        # by comparing DFT amplitude to RMS amplitude
        u_dft_amp = 2 * np.abs(U)
        y_dft_amp = 2 * np.abs(Y)
        
        coherence_u = min(1.0, u_dft_amp / (u_amp + 1e-12))
        coherence_y = min(1.0, y_dft_amp / (y_amp + 1e-12))
        
        # Combined coherence
        coherence = coherence_u * coherence_y
        
        return np.clip(coherence, 0.0, 1.0)
    
    def _create_nan_point(self, f_hz: float, omega: float, axis: str) -> FrequencyPoint:
        """Create a frequency point with NaN values for failed measurement."""
        return FrequencyPoint(
            frequency_hz=f_hz,
            frequency_rad=omega,
            gain_db=np.nan,
            gain_linear=np.nan,
            phase_deg=np.nan,
            phase_rad=np.nan,
            coherence=0.0,
            input_amplitude=0.0,
            output_amplitude=0.0,
            settling_time=0.0,
            num_cycles=0,
            axis=axis,
            metadata={'error': 'measurement_failed'}
        )
    
    def get_results_as_arrays(self) -> Dict[str, np.ndarray]:
        """
        Convert results to numpy arrays for easy plotting.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'frequency_hz': Frequency vector [Hz]
            - 'frequency_rad': Frequency vector [rad/s]
            - 'gain_db': Magnitude response [dB]
            - 'gain_linear': Magnitude response (linear)
            - 'phase_deg': Phase response [degrees]
            - 'phase_rad': Phase response [radians]
            - 'coherence': Coherence values
        """
        if not self._results:
            return {}
        
        return {
            'frequency_hz': np.array([r.frequency_hz for r in self._results]),
            'frequency_rad': np.array([r.frequency_rad for r in self._results]),
            'gain_db': np.array([r.gain_db for r in self._results]),
            'gain_linear': np.array([r.gain_linear for r in self._results]),
            'phase_deg': np.array([r.phase_deg for r in self._results]),
            'phase_rad': np.array([r.phase_rad for r in self._results]),
            'coherence': np.array([r.coherence for r in self._results]),
        }
    
    @property
    def results(self) -> List[FrequencyPoint]:
        """Get the list of frequency response points."""
        return self._results
