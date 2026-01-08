"""
Environmental Disturbance Models for Laser Communication Terminal

This module implements realistic external disturbances that affect pointing
accuracy in operational conditions:

1. Wind Loading: Time-varying aerodynamic torques on gimbal structure
2. Ground Vibration: Micro-seismic and structural vibrations at base mount
3. Structural Noise: Internal mechanical noise from gears, motors, bearings

All disturbances use deterministic seeded random number generation for
reproducible simulation results.

Physical Justification:
----------------------
- Wind: Largest low-frequency disturbance, dominates below 5 Hz
- Ground vibration: Affects 10-100 Hz range, couples through mount
- Structural noise: High-frequency (>100 Hz), low amplitude

Integration with MuJoCo:
-----------------------
Disturbances are applied as external torques via mjData.qfrc_applied[] for
joints, or as base position/orientation perturbations for ground vibration.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import scipy.signal as signal


@dataclass
class DisturbanceState:
    """Container for disturbance contributions."""
    
    # Wind torques [N·m]
    wind_torque_az: float = 0.0
    wind_torque_el: float = 0.0
    
    # Vibration-induced accelerations [m/s²] or [rad/s²]
    vibration_accel_x: float = 0.0
    vibration_accel_y: float = 0.0
    vibration_accel_z: float = 0.0
    vibration_accel_roll: float = 0.0
    vibration_accel_pitch: float = 0.0
    vibration_accel_yaw: float = 0.0
    
    # Structural noise torques [N·m]
    structural_torque_az: float = 0.0
    structural_torque_el: float = 0.0
    
    # Total combined torques [N·m]
    total_torque_az: float = 0.0
    total_torque_el: float = 0.0


class EnvironmentalDisturbances:
    """
    Comprehensive environmental disturbance generator.
    
    This class models all external perturbations that affect gimbal pointing
    accuracy in real-world operation. Disturbances are generated using
    physics-based models with configurable severity levels.
    
    Wind Model:
    ----------
    Uses Dryden wind turbulence model (simplified) with colored noise.
    Torque proportional to velocity squared and exposed area.
    
    Ground Vibration:
    ----------------
    Band-limited white noise (BLWN) filtered to seismic frequency range
    (10-100 Hz). Models building vibrations and ground motion.
    
    Structural Noise:
    ----------------
    High-frequency mechanical noise from motor commutation, gear mesh,
    bearing roughness. Modeled as filtered white noise > 100 Hz.
    
    Usage:
    ------
    >>> config = {
    ...     'wind_rms': 0.5,  # N·m
    ...     'vibration_psd': 1e-6,  # (m/s²)²/Hz
    ...     'structural_noise_std': 0.01,  # N·m
    ...     'seed': 42
    ... }
    >>> disturbances = EnvironmentalDisturbances(config)
    >>> state = disturbances.step(dt=0.001, gimbal_az=0.1, gimbal_el=0.5)
    >>> torque_az = state.total_torque_az
    """
    
    def __init__(self, config: Dict):
        """
        Initialize environmental disturbance generator.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary:
            Wind Loading:
            - 'wind_rms': RMS wind torque magnitude [N·m]
            - 'wind_correlation_time': Time constant for wind gusts [s]
            - 'wind_enabled': Enable wind disturbances [bool]
            
            Ground Vibration:
            - 'vibration_psd': Power spectral density [(m/s²)²/Hz]
            - 'vibration_freq_low': Lower frequency bound [Hz]
            - 'vibration_freq_high': Upper frequency bound [Hz]
            - 'vibration_enabled': Enable vibration [bool]
            
            Structural Noise:
            - 'structural_noise_std': Standard deviation [N·m]
            - 'structural_freq_low': Lower frequency bound [Hz]
            - 'structural_freq_high': Upper frequency bound [Hz]
            - 'structural_enabled': Enable structural noise [bool]
            
            General:
            - 'seed': Random seed for reproducibility
        """
        self.config = config
        
        # Deterministic RNG
        self.seed = config.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)
        
        # Enable flags
        self.wind_enabled = config.get('wind_enabled', True)
        self.vibration_enabled = config.get('vibration_enabled', True)
        self.structural_enabled = config.get('structural_enabled', True)
        
        # Wind parameters
        self.wind_rms = config.get('wind_rms', 0.5)  # N·m
        self.wind_correlation_time = config.get('wind_correlation_time', 2.0)  # s
        
        # Wind state variables (first-order Gauss-Markov process)
        self.wind_state_az = 0.0
        self.wind_state_el = 0.0
        
        # Ground vibration parameters
        self.vibration_psd = config.get('vibration_psd', 1e-6)  # (m/s²)²/Hz
        self.vibration_freq_low = config.get('vibration_freq_low', 10.0)  # Hz
        self.vibration_freq_high = config.get('vibration_freq_high', 100.0)  # Hz
        
        # Vibration filters (2nd-order Butterworth bandpass)
        self._init_vibration_filters()
        
        # Structural noise parameters
        self.structural_noise_std = config.get('structural_noise_std', 0.01)  # N·m
        self.structural_freq_low = config.get('structural_freq_low', 100.0)  # Hz
        self.structural_freq_high = config.get('structural_freq_high', 500.0)  # Hz
        
        # Structural noise filters
        self._init_structural_filters()
        
        # Iteration counter
        self.iteration = 0
        
    def _init_vibration_filters(self) -> None:
        """Initialize bandpass filters for ground vibration."""
        # Design 2nd-order Butterworth bandpass filter
        # Note: Filter design needs sampling frequency, will be done on first step
        self.vibration_filter_states = {
            'x': np.zeros(4),  # Linear X acceleration filter state
            'y': np.zeros(4),  # Linear Y acceleration filter state
            'z': np.zeros(4),  # Linear Z (vertical) acceleration filter state
            'roll': np.zeros(4),
            'pitch': np.zeros(4),
            'yaw': np.zeros(4)
        }
        self.vibration_filter_coeffs = None  # Will be initialized on first step
        
    def _init_structural_filters(self) -> None:
        """Initialize bandpass filters for structural noise."""
        self.structural_filter_states = {
            'az': np.zeros(4),
            'el': np.zeros(4)
        }
        self.structural_filter_coeffs = None
    
    def _design_bandpass_filter(
        self, 
        freq_low: float, 
        freq_high: float, 
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design 2nd-order Butterworth bandpass filter.
        
        Parameters
        ----------
        freq_low : float
            Lower cutoff frequency [Hz]
        freq_high : float
            Upper cutoff frequency [Hz]
        fs : float
            Sampling frequency [Hz]
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (b, a) - Filter coefficients
        """
        nyquist = 0.5 * fs
        low = freq_low / nyquist
        high = freq_high / nyquist
        
        # Ensure valid frequency range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = signal.butter(2, [low, high], btype='band')
        return b, a
    
    def _apply_filter(
        self, 
        x: float, 
        state: np.ndarray, 
        b: np.ndarray, 
        a: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Apply IIR filter to single sample.
        
        Uses Direct Form II transposed structure.
        
        Parameters
        ----------
        x : float
            Input sample
        state : np.ndarray
            Filter state (will be modified)
        b : np.ndarray
            Numerator coefficients
        a : np.ndarray
            Denominator coefficients
            
        Returns
        -------
        Tuple[float, np.ndarray]
            (output, new_state)
        """
        # Direct Form II transposed
        y = b[0] * x + state[0]
        
        # Update state
        new_state = state.copy()
        for i in range(len(state) - 1):
            new_state[i] = b[i+1] * x - a[i+1] * y + state[i+1]
        new_state[-1] = b[-1] * x - a[-1] * y if len(b) > len(state) else 0.0
        
        return y, new_state
    
    def _compute_wind_torque(self, dt: float) -> Tuple[float, float]:
        """
        Compute wind-induced torques using Gauss-Markov process.
        
        Wind is modeled as first-order colored noise:
        dw/dt = -w/T_c + sqrt(2*σ²/T_c) * η(t)
        
        where T_c is correlation time, σ is RMS, η is white noise.
        
        Parameters
        ----------
        dt : float
            Time step [s]
            
        Returns
        -------
        Tuple[float, float]
            (wind_torque_az, wind_torque_el) [N·m]
        """
        if not self.wind_enabled:
            return 0.0, 0.0
        
        # Gauss-Markov parameters
        beta = 1.0 / self.wind_correlation_time
        sigma = self.wind_rms
        
        # Discrete-time update
        # x[k+1] = exp(-beta*dt) * x[k] + w[k]
        # where w ~ N(0, sigma² * (1 - exp(-2*beta*dt)))
        phi = np.exp(-beta * dt)
        noise_variance = sigma**2 * (1 - phi**2)
        
        # Update azimuth wind state
        noise_az = self.rng.normal(0.0, np.sqrt(noise_variance))
        self.wind_state_az = phi * self.wind_state_az + noise_az
        
        # Update elevation wind state (independent)
        noise_el = self.rng.normal(0.0, np.sqrt(noise_variance))
        self.wind_state_el = phi * self.wind_state_el + noise_el
        
        return self.wind_state_az, self.wind_state_el
    
    def _compute_ground_vibration(
        self, 
        dt: float
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Compute ground vibration accelerations.
        
        Generates band-limited white noise in seismic frequency range.
        Models building vibrations, ground motion, and structural resonances.
        
        Parameters
        ----------
        dt : float
            Time step [s]
            
        Returns
        -------
        Tuple[float, float, float, float, float, float]
            (accel_x, accel_y, accel_z, accel_roll, accel_pitch, accel_yaw)
            Linear accelerations [m/s²], angular accelerations [rad/s²]
        """
        if not self.vibration_enabled:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Initialize filters on first call
        fs = 1.0 / dt
        if self.vibration_filter_coeffs is None:
            b, a = self._design_bandpass_filter(
                self.vibration_freq_low,
                self.vibration_freq_high,
                fs
            )
            self.vibration_filter_coeffs = (b, a)
        
        b, a = self.vibration_filter_coeffs
        
        # Generate white noise and filter for each DOF
        # PSD scaling: noise_std = sqrt(PSD * BW * fs/2)
        bandwidth = self.vibration_freq_high - self.vibration_freq_low
        noise_std = np.sqrt(self.vibration_psd * bandwidth * fs / 2.0)
        
        # Linear vibrations
        noise_x = self.rng.normal(0.0, noise_std)
        accel_x, self.vibration_filter_states['x'] = self._apply_filter(
            noise_x, self.vibration_filter_states['x'], b, a
        )
        
        noise_y = self.rng.normal(0.0, noise_std)
        accel_y, self.vibration_filter_states['y'] = self._apply_filter(
            noise_y, self.vibration_filter_states['y'], b, a
        )
        
        noise_z = self.rng.normal(0.0, noise_std)
        accel_z, self.vibration_filter_states['z'] = self._apply_filter(
            noise_z, self.vibration_filter_states['z'], b, a
        )
        
        # Angular vibrations (typically smaller than linear)
        angular_scale = 1e-3  # rad/s² per m/s²
        
        noise_roll = self.rng.normal(0.0, noise_std * angular_scale)
        accel_roll, self.vibration_filter_states['roll'] = self._apply_filter(
            noise_roll, self.vibration_filter_states['roll'], b, a
        )
        
        noise_pitch = self.rng.normal(0.0, noise_std * angular_scale)
        accel_pitch, self.vibration_filter_states['pitch'] = self._apply_filter(
            noise_pitch, self.vibration_filter_states['pitch'], b, a
        )
        
        noise_yaw = self.rng.normal(0.0, noise_std * angular_scale)
        accel_yaw, self.vibration_filter_states['yaw'] = self._apply_filter(
            noise_yaw, self.vibration_filter_states['yaw'], b, a
        )
        
        return accel_x, accel_y, accel_z, accel_roll, accel_pitch, accel_yaw
    
    def _compute_structural_noise(self, dt: float) -> Tuple[float, float]:
        """
        Compute structural noise torques.
        
        Models high-frequency mechanical noise from:
        - Motor commutation ripple
        - Gear mesh vibrations
        - Bearing roughness
        - Encoder quantization effects
        
        Parameters
        ----------
        dt : float
            Time step [s]
            
        Returns
        -------
        Tuple[float, float]
            (noise_torque_az, noise_torque_el) [N·m]
        """
        if not self.structural_enabled:
            return 0.0, 0.0
        
        # Initialize filters on first call
        fs = 1.0 / dt
        if self.structural_filter_coeffs is None:
            b, a = self._design_bandpass_filter(
                self.structural_freq_low,
                self.structural_freq_high,
                fs
            )
            self.structural_filter_coeffs = (b, a)
        
        b, a = self.structural_filter_coeffs
        
        # Generate white noise and filter
        noise_az_raw = self.rng.normal(0.0, self.structural_noise_std)
        noise_az, self.structural_filter_states['az'] = self._apply_filter(
            noise_az_raw, self.structural_filter_states['az'], b, a
        )
        
        noise_el_raw = self.rng.normal(0.0, self.structural_noise_std)
        noise_el, self.structural_filter_states['el'] = self._apply_filter(
            noise_el_raw, self.structural_filter_states['el'], b, a
        )
        
        return noise_az, noise_el
    
    def step(
        self, 
        dt: float,
        gimbal_az: float = 0.0,
        gimbal_el: float = 0.0,
        gimbal_vel_az: float = 0.0,
        gimbal_vel_el: float = 0.0
    ) -> DisturbanceState:
        """
        Compute all disturbances for current timestep.
        
        This is the main interface called by the simulation runner at each
        time step to generate all environmental disturbances.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        gimbal_az : float, optional
            Current azimuth angle [rad] (for angle-dependent disturbances)
        gimbal_el : float, optional
            Current elevation angle [rad]
        gimbal_vel_az : float, optional
            Azimuth angular velocity [rad/s]
        gimbal_vel_el : float, optional
            Elevation angular velocity [rad/s]
            
        Returns
        -------
        DisturbanceState
            Complete disturbance state with all contributions
        """
        state = DisturbanceState()
        
        # 1. Wind loading
        wind_az, wind_el = self._compute_wind_torque(dt)
        
        # Scale wind by angle (more exposure at higher elevation)
        # Simple model: wind increases with cos(elevation)
        wind_scale = max(0.5, np.abs(np.cos(gimbal_el)))
        state.wind_torque_az = wind_az * wind_scale
        state.wind_torque_el = wind_el * wind_scale
        
        # 2. Ground vibration
        (state.vibration_accel_x,
         state.vibration_accel_y,
         state.vibration_accel_z,
         state.vibration_accel_roll,
         state.vibration_accel_pitch,
         state.vibration_accel_yaw) = self._compute_ground_vibration(dt)
        
        # 3. Structural noise
        state.structural_torque_az, state.structural_torque_el = \
            self._compute_structural_noise(dt)
        
        # 4. Compute total combined torques
        state.total_torque_az = (state.wind_torque_az + 
                                  state.structural_torque_az)
        state.total_torque_el = (state.wind_torque_el + 
                                  state.structural_torque_el)
        
        # Note: Vibration accelerations are not directly added to torques
        # They should be applied to the base body in MuJoCo as position/orientation
        # perturbations, which then couple through the structure.
        
        self.iteration += 1
        
        return state
    
    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about disturbance generation.
        
        Returns
        -------
        Dict
            Diagnostic data including RMS values, spectral properties
        """
        return {
            'iteration': self.iteration,
            'wind_enabled': self.wind_enabled,
            'wind_rms': self.wind_rms,
            'wind_state_az': self.wind_state_az,
            'wind_state_el': self.wind_state_el,
            'vibration_enabled': self.vibration_enabled,
            'vibration_psd': self.vibration_psd,
            'vibration_freq_range': (self.vibration_freq_low, 
                                      self.vibration_freq_high),
            'structural_enabled': self.structural_enabled,
            'structural_noise_std': self.structural_noise_std,
            'structural_freq_range': (self.structural_freq_low,
                                       self.structural_freq_high)
        }
    
    def reset(self) -> None:
        """Reset disturbance generator to initial conditions."""
        # Reset wind states
        self.wind_state_az = 0.0
        self.wind_state_el = 0.0
        
        # Reset filter states
        for key in self.vibration_filter_states:
            self.vibration_filter_states[key] = np.zeros(4)
        
        for key in self.structural_filter_states:
            self.structural_filter_states[key] = np.zeros(4)
        
        # Reset RNG to initial seed for repeatability
        self.rng = np.random.default_rng(self.seed)
        
        self.iteration = 0


class SimpleDisturbanceModel:
    """
    Simplified disturbance model for quick testing.
    
    Provides basic white noise disturbances without filtering or
    sophisticated physics models. Useful for preliminary testing
    and debugging.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize simple disturbance model.
        
        Parameters
        ----------
        config : Dict
            Configuration:
            - 'torque_std': Torque noise standard deviation [N·m]
            - 'seed': Random seed
        """
        self.torque_std = config.get('torque_std', 0.1)
        self.seed = config.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)
        
    def step(self, dt: float, **kwargs) -> DisturbanceState:
        """
        Generate simple white noise disturbances.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        **kwargs
            Ignored (for compatibility)
            
        Returns
        -------
        DisturbanceState
            Disturbance state with white noise torques
        """
        state = DisturbanceState()
        
        # Pure white noise
        state.total_torque_az = self.rng.normal(0.0, self.torque_std)
        state.total_torque_el = self.rng.normal(0.0, self.torque_std)
        
        state.wind_torque_az = state.total_torque_az
        state.wind_torque_el = state.total_torque_el
        
        return state
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.rng = np.random.default_rng(self.seed)
