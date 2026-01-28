"""
Frequency Response Analyzer for Comparative Controller Analysis

This module provides the high-level orchestration layer for comparing frequency
response characteristics across multiple controller architectures. It manages:

1. Controller configuration and initialization
2. Simulation callbacks for each controller type
3. Open-loop and closed-loop frequency response extraction
4. Sensitivity and complementary sensitivity function computation

Key Frequency-Domain Performance Metrics
----------------------------------------
For disturbance rejection and noise attenuation analysis:

**Sensitivity Function S(jω)**:
- S(jω) = E(jω) / D(jω) - Transfer from disturbance to error
- |S(jω)| < 1: Disturbances attenuated
- |S(jω)| > 1: Disturbances amplified
- Peak |S|: Robustness indicator (Ms < 2 desired, Ms < 1.5 excellent)

**Complementary Sensitivity T(jω)**:
- T(jω) = Y(jω) / R(jω) - Transfer from reference to output  
- T(jω) = 1 - S(jω) for unity feedback
- Bandwidth: frequency where |T| = -3dB
- |T| roll-off: noise rejection at high frequencies

**Loop Transfer Function L(jω)**:
- L = G_c * G_p (controller × plant)
- Gain margin: 1/|L| where ∠L = -180°
- Phase margin: 180° + ∠L where |L| = 1

Author: Dr. S. Shahid Mustafa
Date: January 28, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, Optional, List, Any
import copy

# Local imports
from .frequency_sweep_engine import (
    FrequencySweepEngine,
    FrequencySweepConfig,
    FrequencyPoint,
    SweepType
)


class ControllerType(Enum):
    """Controller architecture types for comparison."""
    OPEN_LOOP = auto()      # No feedback control
    PID = auto()            # Standard PID controller
    FBL = auto()            # Feedback Linearization
    FBL_NDOB = auto()       # Feedback Linearization + NDOB


@dataclass
class FrequencyResponseData:
    """
    Complete frequency response dataset for one controller configuration.
    
    Contains all transfer functions needed for comprehensive analysis:
    - Closed-loop tracking response
    - Sensitivity function (disturbance rejection)
    - Complementary sensitivity (noise rejection)
    - Control effort transfer function
    
    Attributes
    ----------
    controller_type : ControllerType
        Controller architecture
    axis : str
        Axis analyzed ('az' or 'el')
    frequencies_hz : np.ndarray
        Frequency vector [Hz]
    frequencies_rad : np.ndarray
        Frequency vector [rad/s]
    closed_loop_gain_db : np.ndarray
        |T(jω)| in dB
    closed_loop_phase_deg : np.ndarray
        ∠T(jω) in degrees
    sensitivity_gain_db : np.ndarray
        |S(jω)| in dB
    sensitivity_phase_deg : np.ndarray
        ∠S(jω) in degrees
    control_effort_gain_db : np.ndarray
        |U(jω)/R(jω)| in dB
    coherence : np.ndarray
        Measurement quality metric
    bandwidth_hz : float
        -3dB bandwidth of closed-loop system
    peak_sensitivity : float
        max|S(jω)| (Ms) - robustness indicator
    gain_margin_db : float
        Gain margin [dB]
    phase_margin_deg : float
        Phase margin [degrees]
    metadata : Dict
        Additional analysis metadata
    """
    controller_type: ControllerType
    axis: str
    frequencies_hz: np.ndarray
    frequencies_rad: np.ndarray
    closed_loop_gain_db: np.ndarray
    closed_loop_phase_deg: np.ndarray
    sensitivity_gain_db: np.ndarray
    sensitivity_phase_deg: np.ndarray
    control_effort_gain_db: np.ndarray
    coherence: np.ndarray
    bandwidth_hz: float = 0.0
    peak_sensitivity: float = 0.0
    gain_margin_db: float = np.inf
    phase_margin_deg: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class AnalyzerConfig:
    """
    Configuration for the frequency response analyzer.
    
    Attributes
    ----------
    sweep_config : FrequencySweepConfig
        Configuration for the sweep engine
    analyze_both_axes : bool
        If True, analyze both Az and El axes
    compute_stability_margins : bool
        Compute gain/phase margins from open-loop response
    verbose : bool
        Enable detailed progress output
    """
    sweep_config: FrequencySweepConfig = field(default_factory=FrequencySweepConfig)
    analyze_both_axes: bool = False
    compute_stability_margins: bool = True
    verbose: bool = True


class FrequencyResponseAnalyzer:
    """
    High-Level Frequency Response Analyzer for Gimbal Control Systems.
    
    This class orchestrates the complete frequency response analysis workflow:
    
    1. Initialize simulation infrastructure
    2. Configure each controller type
    3. Execute frequency sweeps (tracking and disturbance)
    4. Compute derived metrics (bandwidth, margins, peak sensitivity)
    5. Package results for visualization
    
    The analyzer uses a callback-based design to decouple from specific
    simulation implementations, allowing reuse across different test setups.
    
    Analysis Types
    --------------
    - **Closed-Loop Tracking**: R(s) → Y(s), measures command following
    - **Sensitivity**: D(s) → E(s), measures disturbance rejection
    - **Control Sensitivity**: R(s) → U(s), measures control effort
    
    Example Usage
    -------------
    >>> from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
    >>> 
    >>> config = AnalyzerConfig(
    ...     sweep_config=FrequencySweepConfig(f_min=0.1, f_max=50, n_points=40)
    ... )
    >>> analyzer = FrequencyResponseAnalyzer(config)
    >>> 
    >>> # Register simulation callback for each controller
    >>> analyzer.register_controller(ControllerType.PID, pid_sim_callback)
    >>> analyzer.register_controller(ControllerType.FBL, fbl_sim_callback)
    >>> 
    >>> # Run analysis
    >>> results = analyzer.run_comparative_analysis()
    
    Parameters
    ----------
    config : AnalyzerConfig
        Analyzer configuration
    dynamics : GimbalDynamics, optional
        Plant dynamics model (used for open-loop analysis)
    """
    
    def __init__(
        self, 
        config: AnalyzerConfig,
        dynamics: Optional[Any] = None
    ):
        self.config = config
        self.dynamics = dynamics
        self._sweep_engine = FrequencySweepEngine(
            config.sweep_config, 
            verbose=config.verbose
        )
        self._controller_callbacks: Dict[ControllerType, callable] = {}
        self._results: Dict[ControllerType, FrequencyResponseData] = {}
    
    def register_controller(
        self,
        controller_type: ControllerType,
        simulation_callback: callable
    ) -> None:
        """
        Register a simulation callback for a controller type.
        
        The callback must implement the interface:
            callback(omega, duration, amplitude, sweep_type, axis) -> (t, u, y)
        
        Parameters
        ----------
        controller_type : ControllerType
            Type of controller
        simulation_callback : callable
            Function that runs simulation and returns time-domain signals
        """
        self._controller_callbacks[controller_type] = simulation_callback
    
    def run_comparative_analysis(
        self,
        axes: Optional[List[str]] = None
    ) -> Dict[ControllerType, FrequencyResponseData]:
        """
        Execute comparative frequency response analysis for all registered controllers.
        
        This runs both tracking and disturbance sweeps for each controller,
        computing all transfer functions and derived metrics.
        
        Parameters
        ----------
        axes : List[str], optional
            Axes to analyze. Default: ['az'] or ['az', 'el'] if analyze_both_axes
            
        Returns
        -------
        Dict[ControllerType, FrequencyResponseData]
            Frequency response data for each controller type
        """
        if axes is None:
            axes = ['az', 'el'] if self.config.analyze_both_axes else ['az']
        
        self._results = {}
        
        for controller_type, callback in self._controller_callbacks.items():
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"ANALYZING: {controller_type.name}")
                print(f"{'='*70}")
            
            for axis in axes:
                # Run closed-loop tracking sweep
                tracking_results = self._sweep_engine.run_sweep(
                    callback,
                    SweepType.REFERENCE_TRACKING,
                    axis
                )
                
                # Run disturbance rejection sweep
                sensitivity_results = self._sweep_engine.run_sweep(
                    callback,
                    SweepType.DISTURBANCE_INJECTION,
                    axis
                )
                
                # Package results
                freq_data = self._package_results(
                    controller_type,
                    axis,
                    tracking_results,
                    sensitivity_results
                )
                
                # Store with composite key
                key = controller_type
                if len(axes) > 1:
                    self._results[(controller_type, axis)] = freq_data
                else:
                    self._results[controller_type] = freq_data
        
        return self._results
    
    def run_single_controller_analysis(
        self,
        controller_type: ControllerType,
        axis: str = 'az'
    ) -> FrequencyResponseData:
        """
        Run frequency response analysis for a single controller.
        
        Parameters
        ----------
        controller_type : ControllerType
            Controller to analyze
        axis : str
            Axis to analyze ('az' or 'el')
            
        Returns
        -------
        FrequencyResponseData
            Complete frequency response data
        """
        if controller_type not in self._controller_callbacks:
            raise ValueError(f"Controller {controller_type} not registered")
        
        callback = self._controller_callbacks[controller_type]
        
        # Run closed-loop tracking sweep
        tracking_results = self._sweep_engine.run_sweep(
            callback,
            SweepType.REFERENCE_TRACKING,
            axis
        )
        
        # Run disturbance rejection sweep
        sensitivity_results = self._sweep_engine.run_sweep(
            callback,
            SweepType.DISTURBANCE_INJECTION,
            axis
        )
        
        return self._package_results(
            controller_type,
            axis,
            tracking_results,
            sensitivity_results
        )
    
    def _package_results(
        self,
        controller_type: ControllerType,
        axis: str,
        tracking_results: List[FrequencyPoint],
        sensitivity_results: List[FrequencyPoint]
    ) -> FrequencyResponseData:
        """
        Package sweep results into FrequencyResponseData structure.
        
        Also computes derived metrics:
        - Bandwidth (-3dB frequency)
        - Peak sensitivity (max|S|)
        - Stability margins (if open-loop data available)
        """
        # Extract arrays from tracking results
        n_points = len(tracking_results)
        frequencies_hz = np.array([r.frequency_hz for r in tracking_results])
        frequencies_rad = np.array([r.frequency_rad for r in tracking_results])
        
        cl_gain_db = np.array([r.gain_db for r in tracking_results])
        cl_phase_deg = np.array([r.phase_deg for r in tracking_results])
        cl_coherence = np.array([r.coherence for r in tracking_results])
        
        # Extract sensitivity results
        sens_gain_db = np.array([r.gain_db for r in sensitivity_results])
        sens_phase_deg = np.array([r.phase_deg for r in sensitivity_results])
        
        # Control effort (approximated from tracking for now)
        # In full implementation, would need separate sweep
        control_effort_db = cl_gain_db.copy()  # Placeholder
        
        # Compute bandwidth (frequency where |T| drops below -3dB)
        bandwidth_hz = self._compute_bandwidth(frequencies_hz, cl_gain_db)
        
        # Compute peak sensitivity
        peak_sensitivity = self._compute_peak_sensitivity(sens_gain_db)
        
        # Compute stability margins (requires open-loop data)
        gain_margin, phase_margin = np.inf, 0.0
        if self.config.compute_stability_margins:
            # Would need loop transfer function - approximation here
            gain_margin, phase_margin = self._estimate_stability_margins(
                frequencies_hz, cl_gain_db, cl_phase_deg
            )
        
        return FrequencyResponseData(
            controller_type=controller_type,
            axis=axis,
            frequencies_hz=frequencies_hz,
            frequencies_rad=frequencies_rad,
            closed_loop_gain_db=cl_gain_db,
            closed_loop_phase_deg=cl_phase_deg,
            sensitivity_gain_db=sens_gain_db,
            sensitivity_phase_deg=sens_phase_deg,
            control_effort_gain_db=control_effort_db,
            coherence=cl_coherence,
            bandwidth_hz=bandwidth_hz,
            peak_sensitivity=peak_sensitivity,
            gain_margin_db=gain_margin,
            phase_margin_deg=phase_margin,
            metadata={
                'n_points': n_points,
                'f_min': frequencies_hz[0] if len(frequencies_hz) > 0 else 0,
                'f_max': frequencies_hz[-1] if len(frequencies_hz) > 0 else 0,
            }
        )
    
    def _compute_bandwidth(
        self, 
        frequencies: np.ndarray, 
        gain_db: np.ndarray
    ) -> float:
        """
        Compute -3dB bandwidth from frequency response.
        
        The bandwidth is defined as the frequency where |T(jω)| = -3dB
        (or 70.7% of DC gain).
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency vector [Hz]
        gain_db : np.ndarray
            Magnitude response [dB]
            
        Returns
        -------
        float
            Bandwidth [Hz], or f_max if not found
        """
        # Find DC gain (lowest frequency)
        valid_idx = ~np.isnan(gain_db)
        if not np.any(valid_idx):
            return 0.0
        
        dc_gain = gain_db[valid_idx][0]
        threshold = dc_gain - 3.0
        
        # Find first crossing below -3dB
        below_threshold = gain_db < threshold
        crossings = np.where(below_threshold & valid_idx)[0]
        
        if len(crossings) > 0:
            # Interpolate for more accurate estimate
            idx = crossings[0]
            if idx > 0:
                # Linear interpolation between points
                f1, g1 = frequencies[idx-1], gain_db[idx-1]
                f2, g2 = frequencies[idx], gain_db[idx]
                if g1 != g2:
                    f_bw = f1 + (threshold - g1) * (f2 - f1) / (g2 - g1)
                    return f_bw
            return frequencies[idx]
        
        return frequencies[-1]  # Bandwidth exceeds max frequency
    
    def _compute_peak_sensitivity(self, sens_gain_db: np.ndarray) -> float:
        """
        Compute peak sensitivity Ms = max|S(jω)|.
        
        The peak sensitivity is a key robustness indicator:
        - Ms < 1.5 (3.5 dB): Excellent robustness
        - Ms < 2.0 (6.0 dB): Good robustness
        - Ms > 2.0 (6.0 dB): Poor robustness, potential for oscillation
        
        Parameters
        ----------
        sens_gain_db : np.ndarray
            Sensitivity magnitude [dB]
            
        Returns
        -------
        float
            Peak sensitivity Ms (linear scale)
        """
        valid = ~np.isnan(sens_gain_db)
        if not np.any(valid):
            return np.inf
        
        peak_db = np.max(sens_gain_db[valid])
        return 10 ** (peak_db / 20.0)
    
    def _estimate_stability_margins(
        self,
        frequencies: np.ndarray,
        gain_db: np.ndarray,
        phase_deg: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate stability margins from closed-loop response.
        
        For true margins, open-loop transfer function is needed.
        This provides an approximation based on closed-loop characteristics.
        
        The relationship T = L/(1+L) gives:
        - At |T| peak (resonance): phase margin ≈ 2*asin(1/(2*Mp))
        - Gain margin: related to how close |T| approaches infinity
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency vector [Hz]
        gain_db : np.ndarray
            Closed-loop magnitude [dB]
        phase_deg : np.ndarray
            Closed-loop phase [degrees]
            
        Returns
        -------
        Tuple[float, float]
            (gain_margin_db, phase_margin_deg)
        """
        valid = ~np.isnan(gain_db)
        if not np.any(valid):
            return np.inf, 0.0
        
        # Peak closed-loop gain indicates proximity to instability
        peak_gain_db = np.max(gain_db[valid])
        Mp = 10 ** (peak_gain_db / 20.0)
        
        # Approximate phase margin from peak magnitude
        # For second-order systems: PM ≈ 2*arcsin(1/(2*Mp))
        if Mp > 0.5:
            arg = 1.0 / (2.0 * Mp)
            if arg <= 1.0:
                phase_margin = 2.0 * np.rad2deg(np.arcsin(arg))
            else:
                phase_margin = 90.0  # High damping
        else:
            phase_margin = 90.0
        
        # Approximate gain margin from sensitivity peak
        # GM ≈ 1 + 1/Ms where Ms = max|S|
        Ms = Mp  # Rough approximation
        if Ms > 0:
            gm_linear = 1 + 1.0 / Ms
            gain_margin = 20 * np.log10(gm_linear)
        else:
            gain_margin = np.inf
        
        return gain_margin, phase_margin
    
    @property
    def results(self) -> Dict[ControllerType, FrequencyResponseData]:
        """Get analysis results."""
        return self._results
    
    def get_comparative_summary(self) -> Dict[str, Any]:
        """
        Generate summary comparison of all analyzed controllers.
        
        Returns
        -------
        Dict
            Summary metrics for each controller
        """
        summary = {}
        
        for key, data in self._results.items():
            if isinstance(key, tuple):
                name = f"{key[0].name}_{key[1]}"
            else:
                name = key.name
            
            summary[name] = {
                'bandwidth_hz': data.bandwidth_hz,
                'peak_sensitivity': data.peak_sensitivity,
                'peak_sensitivity_db': 20 * np.log10(data.peak_sensitivity + 1e-12),
                'gain_margin_db': data.gain_margin_db,
                'phase_margin_deg': data.phase_margin_deg,
                'dc_gain_db': data.closed_loop_gain_db[0] if len(data.closed_loop_gain_db) > 0 else np.nan,
            }
        
        return summary
