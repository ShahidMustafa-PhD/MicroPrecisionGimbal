"""
Environmental Disturbance Models for Laser Communication Terminal

This module implements high-fidelity environmental disturbance models for
industrial-grade gimbal performance verification. All disturbances are
designed to be injected into the plant dynamics as additive torques:

    M(q)q̈ + C(q,q̇)q̇ + G(q) = τ_control + τ_disturbance

The disturbances exist ONLY in the plant (simulation environment), creating
the plant-model dissonance that the NDOB must estimate and compensate.

==============================================================================
DESIGN NOTES: Physical Modeling of Environmental Disturbances
==============================================================================

1. WIND/GUST LOADING (Dryden Turbulence Model)
   ------------------------------------------
   Wind torque on a gimbal is computed from aerodynamic pressure:
   
   τ_wind = 0.5 × ρ × V² × Cd × A × L
   
   where:
   - ρ = air density [kg/m³] (≈1.225 at sea level)
   - V = wind velocity [m/s]
   - Cd = drag coefficient (≈1.2 for flat plate)
   - A = exposed cross-sectional area [m²]
   - L = moment arm from rotation axis [m]
   
   The Dryden turbulence model generates V(t) = V_mean + V_gust(t) where:
   - V_mean: Steady-state wind (constant bias)
   - V_gust: Stochastic gusts via shaping filter on white noise
   
   Transfer function (longitudinal Dryden):
   
   H_u(s) = σ_u × sqrt(2 × V_mean / (π × L_u)) × (1 + sqrt(3) × L_u/V_mean × s) / (1 + L_u/V_mean × s)²
   
   For low-altitude (<1000 ft) conditions:
   - L_u = 200 m (turbulence scale length)
   - σ_u = 0.1 × V_mean (turbulence intensity for light conditions)

2. STRUCTURAL VIBRATION (PSD-Based Model)
   --------------------------------------
   Platform/mast vibrations are modeled using Power Spectral Density (PSD)
   approach with multiple resonance peaks representing structural modes.
   
   The acceleration PSD is modeled as sum of modal contributions:
   
   S_a(f) = Σ [A_i / ((1 - (f/f_i)²)² + (2ζ_i × f/f_i)²)]
   
   where:
   - A_i = modal amplitude [m²/s⁴/Hz]
   - f_i = i-th mode natural frequency [Hz]
   - ζ_i = modal damping ratio
   
   Torque from base acceleration: τ = I × α + m × L × a
   where I = moment of inertia, α = angular accel, a = linear accel

3. CORRELATION TIME (τ) FOR STOCHASTIC PROCESSES
   ----------------------------------------------
   - Wind gusts: τ ≈ L_u / V_mean (typically 2-10 seconds)
   - Structural vibration: τ ≈ 1 / (2π × f_mode) (10-100 ms for 10-100 Hz modes)
   - NDOB can effectively reject disturbances with τ >> 1/λ (observer bandwidth)

==============================================================================

Author: Dr. S. Shahid Mustafa (Principal Control Systems Engineer)
Date: January 26, 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
import scipy.signal as signal


@dataclass
class DisturbanceState:
    """Container for disturbance contributions at a single timestep."""
    
    # Wind/gust torques [N·m]
    wind_torque_az: float = 0.0
    wind_torque_el: float = 0.0
    
    # Vibration-induced torques [N·m]
    vibration_torque_az: float = 0.0
    vibration_torque_el: float = 0.0
    
    # Vibration-induced accelerations (for reference/logging)
    vibration_accel_x: float = 0.0   # [m/s²]
    vibration_accel_y: float = 0.0   # [m/s²]
    vibration_accel_z: float = 0.0   # [m/s²]
    vibration_accel_roll: float = 0.0   # [rad/s²]
    vibration_accel_pitch: float = 0.0  # [rad/s²]
    vibration_accel_yaw: float = 0.0    # [rad/s²]
    
    # Structural noise torques [N·m]
    structural_torque_az: float = 0.0
    structural_torque_el: float = 0.0
    
    # Total combined torques [N·m] - this is τ_disturbance in the dynamics
    total_torque_az: float = 0.0
    total_torque_el: float = 0.0
    
    # Diagnostic: individual wind components
    wind_steady_az: float = 0.0
    wind_steady_el: float = 0.0
    wind_gust_az: float = 0.0
    wind_gust_el: float = 0.0


@dataclass
class DrydenWindConfig:
    """
    Configuration for Dryden turbulence wind model.
    
    Attributes
    ----------
    enabled : bool
        Enable wind disturbance generation
    start_time : float
        Time to begin wind injection [s]
    mean_velocity : float
        Mean wind speed [m/s] (V_mean)
    turbulence_intensity : float
        Relative intensity σ_u/V_mean (0.1 = light, 0.2 = moderate, 0.3 = severe)
    scale_length : float
        Turbulence scale length L_u [m] (200m typical for low altitude)
    direction_deg : float
        Wind direction relative to gimbal frame [deg]
    gimbal_area : float
        Effective exposed area of gimbal [m²]
    gimbal_arm : float
        Moment arm from wind force center to rotation axis [m]
    drag_coefficient : float
        Aerodynamic drag coefficient Cd
    air_density : float
        Air density ρ [kg/m³]
    correlation_time : float
        Override correlation time τ [s]. If 0, computed from L_u/V_mean
    """
    enabled: bool = True
    start_time: float = 0.0
    mean_velocity: float = 5.0           # m/s (light breeze)
    turbulence_intensity: float = 0.15   # Light-moderate turbulence
    scale_length: float = 200.0          # m (low altitude)
    direction_deg: float = 45.0          # Hits both axes
    gimbal_area: float = 0.02            # m² (typical small gimbal)
    gimbal_arm: float = 0.15             # m
    drag_coefficient: float = 1.2        # Flat plate
    air_density: float = 1.225           # kg/m³ (sea level)
    correlation_time: float = 0.0        # Auto-compute if 0


@dataclass
class StructuralVibrationConfig:
    """
    Configuration for PSD-based structural vibration model.
    
    Attributes
    ----------
    enabled : bool
        Enable vibration disturbance generation
    start_time : float
        Time to begin vibration injection [s]
    modal_frequencies : List[float]
        Natural frequencies of structural modes [Hz]
    modal_dampings : List[float]
        Damping ratios for each mode
    modal_amplitudes : List[float]
        Amplitude scaling for each mode [(m/s²)²/Hz]
    inertia_coupling : float
        Coupling factor from base acceleration to gimbal torque [N·m/(m/s²)]
    noise_floor_psd : float
        Broadband noise floor PSD [(m/s²)²/Hz]
    """
    enabled: bool = True
    start_time: float = 0.0
    modal_frequencies: List[float] = field(default_factory=lambda: [15.0, 45.0, 80.0])
    modal_dampings: List[float] = field(default_factory=lambda: [0.02, 0.015, 0.01])
    modal_amplitudes: List[float] = field(default_factory=lambda: [1e-3, 5e-4, 2e-4])
    inertia_coupling: float = 0.1        # N·m per m/s² base acceleration
    noise_floor_psd: float = 1e-6        # (m/s²)²/Hz


@dataclass 
class EnvironmentalDisturbanceConfig:
    """
    Master configuration for all environmental disturbances.
    
    This dataclass aggregates all disturbance parameters for clean integration
    with the simulation runner.
    """
    enabled: bool = True
    seed: int = 42
    wind: DrydenWindConfig = field(default_factory=DrydenWindConfig)
    vibration: StructuralVibrationConfig = field(default_factory=StructuralVibrationConfig)
    # Structural noise (high-frequency mechanical)
    structural_noise_enabled: bool = True
    structural_noise_std: float = 0.005  # N·m
    structural_noise_freq_low: float = 100.0   # Hz
    structural_noise_freq_high: float = 500.0  # Hz


class DrydenTurbulenceFilter:
    """
    Discrete-time implementation of Dryden turbulence shaping filter.
    
    Implements the simplified Dryden transfer function for longitudinal
    turbulence as a discrete-time state-space system.
    
    Reference: MIL-F-8785C, MIL-STD-1797
    """
    
    def __init__(self, config: DrydenWindConfig, dt: float):
        """
        Initialize Dryden turbulence filter.
        
        Parameters
        ----------
        config : DrydenWindConfig
            Wind configuration parameters
        dt : float
            Sampling period [s]
        """
        self.config = config
        self.dt = dt
        
        # Compute derived parameters
        V = max(config.mean_velocity, 0.1)  # Avoid division by zero
        L = config.scale_length
        sigma = config.turbulence_intensity * V
        
        # Time constant
        if config.correlation_time > 0:
            tau = config.correlation_time
        else:
            tau = L / V  # Physical correlation time
        
        # Store for diagnostics
        self.tau = tau
        self.sigma = sigma
        
        # Simplified first-order Dryden (Gauss-Markov process)
        # Continuous: dx/dt = -x/τ + sqrt(2σ²/τ) * w(t)
        # Discrete: x[k+1] = φ*x[k] + noise
        self.phi = np.exp(-dt / tau)
        self.noise_std = sigma * np.sqrt(1 - self.phi**2)
        
        # State variables
        self.state_az = 0.0
        self.state_el = 0.0
        
    def step(self, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Generate one sample of turbulent velocity fluctuation.
        
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        Tuple[float, float]
            (velocity_fluctuation_az, velocity_fluctuation_el) [m/s]
        """
        # Gauss-Markov update
        self.state_az = self.phi * self.state_az + rng.normal(0, self.noise_std)
        self.state_el = self.phi * self.state_el + rng.normal(0, self.noise_std)
        
        return self.state_az, self.state_el
    
    def reset(self) -> None:
        """Reset filter state."""
        self.state_az = 0.0
        self.state_el = 0.0


class ModalVibrationFilter:
    """
    PSD-based structural vibration generator using modal superposition.
    
    Models structural resonances as second-order bandpass filters excited
    by white noise. Each mode contributes to the overall acceleration PSD.
    """
    
    def __init__(self, config: StructuralVibrationConfig, dt: float):
        """
        Initialize modal vibration filters.
        
        Parameters
        ----------
        config : StructuralVibrationConfig
            Vibration configuration
        dt : float
            Sampling period [s]
        """
        self.config = config
        self.dt = dt
        self.fs = 1.0 / dt
        
        # Initialize filters for each mode
        self.mode_filters_az: List[Dict] = []
        self.mode_filters_el: List[Dict] = []
        
        for i, (f_n, zeta, amp) in enumerate(zip(
            config.modal_frequencies,
            config.modal_dampings, 
            config.modal_amplitudes
        )):
            # Design bandpass filter centered on mode frequency
            # Use narrow bandwidth: BW = 2*zeta*f_n
            bw = max(2 * zeta * f_n, 1.0)  # Minimum 1 Hz bandwidth
            f_low = max(f_n - bw, 0.5)
            f_high = min(f_n + bw, 0.95 * self.fs / 2)
            
            if f_high <= f_low:
                continue  # Skip invalid frequency range
                
            try:
                b, a = self._design_bandpass(f_low, f_high, order=2)
            except Exception:
                continue  # Skip if filter design fails
                
            # Amplitude scaling
            noise_gain = np.sqrt(amp * 2 * bw)
            
            self.mode_filters_az.append({
                'b': b, 'a': a,
                'state': np.zeros(max(len(a), len(b)) - 1),
                'gain': noise_gain
            })
            self.mode_filters_el.append({
                'b': b, 'a': a,
                'state': np.zeros(max(len(a), len(b)) - 1),
                'gain': noise_gain
            })
        
        # Broadband noise filter (optional noise floor)
        if config.noise_floor_psd > 0:
            self.noise_gain = np.sqrt(config.noise_floor_psd * self.fs / 2)
        else:
            self.noise_gain = 0.0
            
    def _design_bandpass(self, f_low: float, f_high: float, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth bandpass filter."""
        nyquist = 0.5 * self.fs
        low = max(f_low / nyquist, 0.001)
        high = min(f_high / nyquist, 0.999)
        
        if high <= low:
            raise ValueError("Invalid frequency range")
            
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a
    
    def step(self, rng: np.random.Generator) -> Tuple[float, float]:
        """
        Generate one sample of vibration-induced acceleration.
        
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        Tuple[float, float]
            (accel_az, accel_el) [m/s²]
        """
        accel_az = 0.0
        accel_el = 0.0
        
        # Modal contributions
        for filt in self.mode_filters_az:
            noise = rng.normal(0, 1) * filt['gain']
            y, filt['state'] = signal.lfilter(
                filt['b'], filt['a'], [noise], zi=filt['state']
            )
            accel_az += y[0]
            
        for filt in self.mode_filters_el:
            noise = rng.normal(0, 1) * filt['gain']
            y, filt['state'] = signal.lfilter(
                filt['b'], filt['a'], [noise], zi=filt['state']
            )
            accel_el += y[0]
        
        # Add noise floor
        if self.noise_gain > 0:
            accel_az += rng.normal(0, self.noise_gain)
            accel_el += rng.normal(0, self.noise_gain)
            
        return accel_az, accel_el
    
    def reset(self) -> None:
        """Reset all filter states."""
        for filt in self.mode_filters_az:
            filt['state'] = np.zeros_like(filt['state'])
        for filt in self.mode_filters_el:
            filt['state'] = np.zeros_like(filt['state'])


class EnvironmentalDisturbances:
    """
    High-Fidelity Environmental Disturbance Generator.
    
    Implements industrial-grade disturbance models for gimbal performance
    verification, including:
    
    1. **Dryden Wind Turbulence**: Colored noise gust model with steady bias
    2. **PSD-Based Structural Vibration**: Modal superposition approach
    3. **High-Frequency Structural Noise**: Mechanical noise floor
    
    All disturbances are output as additive torques τ_d for injection into
    the plant dynamics equation:
    
        M(q)q̈ + C(q,q̇)q̇ + G(q) = τ_control + τ_disturbance
    
    The NDOB observer estimates τ_disturbance = d_hat and compensates by
    subtracting it from the control command.
    
    Example
    -------
    >>> config = EnvironmentalDisturbanceConfig(
    ...     wind=DrydenWindConfig(mean_velocity=8.0, start_time=2.0),
    ...     vibration=StructuralVibrationConfig(modal_frequencies=[20.0, 50.0])
    ... )
    >>> disturbances = EnvironmentalDisturbances(config, dt=0.001)
    >>> state = disturbances.step(t=5.0)
    >>> tau_d = np.array([state.total_torque_az, state.total_torque_el])
    """
    
    def __init__(
        self, 
        config: EnvironmentalDisturbanceConfig,
        dt: float = 0.001
    ):
        """
        Initialize environmental disturbance generator.
        
        Parameters
        ----------
        config : EnvironmentalDisturbanceConfig
            Master configuration dataclass
        dt : float
            Simulation timestep [s] (needed for filter design)
        """
        self.config = config
        self.dt = dt
        
        # Deterministic RNG
        self.rng = np.random.default_rng(config.seed)
        
        # Initialize Dryden wind turbulence filter
        self.dryden_filter: Optional[DrydenTurbulenceFilter] = None
        if config.wind.enabled:
            self.dryden_filter = DrydenTurbulenceFilter(config.wind, dt)
            
        # Initialize modal vibration filter
        self.vibration_filter: Optional[ModalVibrationFilter] = None
        if config.vibration.enabled:
            self.vibration_filter = ModalVibrationFilter(config.vibration, dt)
            
        # Structural noise bandpass filter
        self.structural_filter_b: Optional[np.ndarray] = None
        self.structural_filter_a: Optional[np.ndarray] = None
        self.structural_state_az = np.zeros(4)
        self.structural_state_el = np.zeros(4)
        
        if config.structural_noise_enabled:
            self._init_structural_filter()
        
        # Iteration counter and time
        self.iteration = 0
        
    def _init_structural_filter(self) -> None:
        """Initialize high-frequency structural noise filter."""
        fs = 1.0 / self.dt
        nyquist = 0.5 * fs
        
        f_low = self.config.structural_noise_freq_low / nyquist
        f_high = self.config.structural_noise_freq_high / nyquist
        
        # Clamp to valid range
        f_low = max(0.001, min(f_low, 0.95))
        f_high = max(f_low + 0.01, min(f_high, 0.999))
        
        if f_high > f_low:
            self.structural_filter_b, self.structural_filter_a = signal.butter(
                2, [f_low, f_high], btype='band'
            )
            
    def _compute_wind_torque(self, t: float) -> Tuple[float, float, float, float, float, float]:
        """
        Compute wind-induced torques using Dryden turbulence model.
        
        Parameters
        ----------
        t : float
            Current simulation time [s]
            
        Returns
        -------
        Tuple[float, float, float, float, float, float]
            (torque_az, torque_el, steady_az, steady_el, gust_az, gust_el)
        """
        if not self.config.wind.enabled or self.dryden_filter is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        cfg = self.config.wind
        
        # Check start time
        if t < cfg.start_time:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        # Mean wind velocity components (steady-state)
        direction_rad = np.deg2rad(cfg.direction_deg)
        V_mean_az = cfg.mean_velocity * np.cos(direction_rad)
        V_mean_el = cfg.mean_velocity * np.sin(direction_rad)
        
        # Gust components (stochastic from Dryden filter)
        V_gust_az, V_gust_el = self.dryden_filter.step(self.rng)
        
        # Total velocity
        V_az = V_mean_az + V_gust_az
        V_el = V_mean_el + V_gust_el
        
        # Aerodynamic pressure: P = 0.5 * ρ * V²
        # Torque: τ = P * A * Cd * L
        # τ = 0.5 * ρ * V² * Cd * A * L
        rho = cfg.air_density
        Cd = cfg.drag_coefficient
        A = cfg.gimbal_area
        L = cfg.gimbal_arm
        
        # Signed torque (considers wind direction)
        torque_coeff = 0.5 * rho * Cd * A * L
        
        # Total torque (proportional to V|V| to preserve sign)
        torque_az = torque_coeff * V_az * np.abs(V_az)
        torque_el = torque_coeff * V_el * np.abs(V_el)
        
        # Steady components only
        steady_az = torque_coeff * V_mean_az * np.abs(V_mean_az)
        steady_el = torque_coeff * V_mean_el * np.abs(V_mean_el)
        
        # Gust components (difference from steady)
        gust_az = torque_az - steady_az
        gust_el = torque_el - steady_el
        
        return torque_az, torque_el, steady_az, steady_el, gust_az, gust_el
    
    def _compute_vibration_torque(self, t: float) -> Tuple[float, float, float, float]:
        """
        Compute vibration-induced torques from PSD-based model.
        
        Parameters
        ----------
        t : float
            Current simulation time [s]
            
        Returns
        -------
        Tuple[float, float, float, float]
            (torque_az, torque_el, accel_az, accel_el)
        """
        if not self.config.vibration.enabled or self.vibration_filter is None:
            return 0.0, 0.0, 0.0, 0.0
            
        cfg = self.config.vibration
        
        # Check start time
        if t < cfg.start_time:
            return 0.0, 0.0, 0.0, 0.0
            
        # Generate acceleration from modal filters
        accel_az, accel_el = self.vibration_filter.step(self.rng)
        
        # Convert acceleration to torque via inertia coupling
        # τ = I_coupling × a
        torque_az = cfg.inertia_coupling * accel_az
        torque_el = cfg.inertia_coupling * accel_el
        
        return torque_az, torque_el, accel_az, accel_el
    
    def _compute_structural_noise(self, t: float) -> Tuple[float, float]:
        """
        Compute high-frequency structural noise.
        
        Parameters
        ----------
        t : float
            Current simulation time [s]
            
        Returns
        -------
        Tuple[float, float]
            (noise_torque_az, noise_torque_el) [N·m]
        """
        if not self.config.structural_noise_enabled:
            return 0.0, 0.0
            
        if self.structural_filter_b is None:
            # No filter, just white noise
            return (
                self.rng.normal(0, self.config.structural_noise_std),
                self.rng.normal(0, self.config.structural_noise_std)
            )
        
        # Filter white noise for band-limited structural noise
        noise_az_raw = self.rng.normal(0, self.config.structural_noise_std)
        noise_el_raw = self.rng.normal(0, self.config.structural_noise_std)
        
        y_az, self.structural_state_az = signal.lfilter(
            self.structural_filter_b, 
            self.structural_filter_a,
            [noise_az_raw],
            zi=self.structural_state_az
        )
        
        y_el, self.structural_state_el = signal.lfilter(
            self.structural_filter_b,
            self.structural_filter_a, 
            [noise_el_raw],
            zi=self.structural_state_el
        )
        
        return y_az[0], y_el[0]
    
    def step(
        self,
        t: float,
        gimbal_az: float = 0.0,
        gimbal_el: float = 0.0,
        gimbal_vel_az: float = 0.0,
        gimbal_vel_el: float = 0.0
    ) -> DisturbanceState:
        """
        Compute all disturbances for current timestep.
        
        This is the main interface called by the simulation runner to generate
        τ_disturbance for injection into plant dynamics.
        
        Parameters
        ----------
        t : float
            Current simulation time [s]
        gimbal_az : float, optional
            Current azimuth angle [rad]
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
        if not self.config.enabled:
            return DisturbanceState()
            
        state = DisturbanceState()
        
        # 1. Wind/Gust Loading (Dryden model)
        (state.wind_torque_az, 
         state.wind_torque_el,
         state.wind_steady_az,
         state.wind_steady_el,
         state.wind_gust_az,
         state.wind_gust_el) = self._compute_wind_torque(t)
        
        # Optional: Scale wind by gimbal elevation (more exposed at horizon)
        if self.config.wind.enabled:
            exposure_factor = max(0.3, np.cos(gimbal_el))
            state.wind_torque_az *= exposure_factor
            state.wind_torque_el *= exposure_factor
        
        # 2. Structural Vibration (PSD-based)
        (state.vibration_torque_az,
         state.vibration_torque_el,
         state.vibration_accel_x,  # Store as x for logging
         state.vibration_accel_y) = self._compute_vibration_torque(t)
        
        # 3. Structural Noise (high-frequency)
        (state.structural_torque_az,
         state.structural_torque_el) = self._compute_structural_noise(t)
        
        # 4. Compute total combined torques (τ_disturbance)
        state.total_torque_az = (
            state.wind_torque_az + 
            state.vibration_torque_az + 
            state.structural_torque_az
        )
        state.total_torque_el = (
            state.wind_torque_el + 
            state.vibration_torque_el + 
            state.structural_torque_el
        )
        
        self.iteration += 1
        return state
    
    def get_statistics(self) -> Dict:
        """
        Get statistical properties of disturbance sources.
        
        Returns
        -------
        Dict
            Statistical parameters (mean, variance, correlation time)
        """
        stats = {
            'iteration': self.iteration,
            'enabled': self.config.enabled,
        }
        
        if self.config.wind.enabled and self.dryden_filter:
            stats['wind'] = {
                'mean_velocity': self.config.wind.mean_velocity,
                'turbulence_intensity': self.config.wind.turbulence_intensity,
                'correlation_time_s': self.dryden_filter.tau,
                'gust_std_m_s': self.dryden_filter.sigma,
                'start_time_s': self.config.wind.start_time
            }
            
        if self.config.vibration.enabled:
            stats['vibration'] = {
                'modal_frequencies_hz': self.config.vibration.modal_frequencies,
                'modal_dampings': self.config.vibration.modal_dampings,
                'inertia_coupling_Nm_per_m_s2': self.config.vibration.inertia_coupling,
                'start_time_s': self.config.vibration.start_time
            }
            
        if self.config.structural_noise_enabled:
            stats['structural_noise'] = {
                'std_Nm': self.config.structural_noise_std,
                'freq_range_hz': (
                    self.config.structural_noise_freq_low,
                    self.config.structural_noise_freq_high
                )
            }
            
        return stats
    
    def get_diagnostics(self) -> Dict:
        """Alias for get_statistics (backward compatibility)."""
        return self.get_statistics()
    
    def reset(self) -> None:
        """Reset disturbance generator to initial conditions."""
        self.rng = np.random.default_rng(self.config.seed)
        
        if self.dryden_filter:
            self.dryden_filter.reset()
            
        if self.vibration_filter:
            self.vibration_filter.reset()
            
        self.structural_state_az = np.zeros(4)
        self.structural_state_el = np.zeros(4)
        
        self.iteration = 0


class SimpleDisturbanceModel:
    """
    Simplified disturbance model for rapid prototyping and testing.
    
    Provides configurable disturbances with both:
    1. Constant bias (steady-state offset)
    2. Stochastic component (white or colored noise)
    
    This model is useful for:
    - Quick NDOB tuning verification
    - Controller robustness testing
    - Understanding disturbance rejection basics
    
    Example
    -------
    >>> config = {
    ...     'torque_bias': [0.1, 0.05],      # Constant bias [Az, El]
    ...     'torque_std': 0.02,               # Noise RMS
    ...     'correlation_time': 0.5,          # Colored noise τ
    ...     'start_time': 2.0,                # Injection time
    ...     'ramp_duration': 1.0,             # Gradual onset
    ... }
    >>> disturbances = SimpleDisturbanceModel(config)
    >>> state = disturbances.step(dt=0.001, t=3.0)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize simple disturbance model.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary:
            - 'enabled': bool (default True)
            - 'torque_bias': [az, el] constant offset [N·m]
            - 'torque_std': noise RMS [N·m]
            - 'correlation_time': noise τ [s] (0 = white noise)
            - 'start_time': injection start [s]
            - 'ramp_duration': gradual onset [s]
            - 'seed': random seed
        """
        self.enabled = config.get('enabled', True)
        self.torque_bias = np.array(config.get('torque_bias', [0.0, 0.0]))
        self.torque_std = config.get('torque_std', 0.1)
        self.correlation_time = config.get('correlation_time', 0.0)
        self.start_time = config.get('start_time', 0.0)
        self.ramp_duration = config.get('ramp_duration', 0.0)
        self.seed = config.get('seed', 42)
        
        self.rng = np.random.default_rng(self.seed)
        
        # State for colored noise
        self.noise_state = np.array([0.0, 0.0])
        
    def step(self, dt: float, t: float = 0.0, **kwargs) -> DisturbanceState:
        """
        Generate disturbance for current timestep.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        t : float
            Current time [s]
        **kwargs
            Ignored (compatibility)
            
        Returns
        -------
        DisturbanceState
            Disturbance state
        """
        state = DisturbanceState()
        
        if not self.enabled or t < self.start_time:
            return state
            
        # Compute ramp factor for gradual onset
        if self.ramp_duration > 0:
            t_since_start = t - self.start_time
            ramp_factor = min(1.0, t_since_start / self.ramp_duration)
        else:
            ramp_factor = 1.0
            
        # Bias component
        bias_az = self.torque_bias[0] * ramp_factor
        bias_el = self.torque_bias[1] * ramp_factor
        
        # Noise component
        if self.correlation_time > 0:
            # Gauss-Markov (colored noise)
            tau = self.correlation_time
            phi = np.exp(-dt / tau)
            noise_var = self.torque_std**2 * (1 - phi**2)
            
            self.noise_state[0] = phi * self.noise_state[0] + self.rng.normal(0, np.sqrt(noise_var))
            self.noise_state[1] = phi * self.noise_state[1] + self.rng.normal(0, np.sqrt(noise_var))
            
            noise_az = self.noise_state[0] * ramp_factor
            noise_el = self.noise_state[1] * ramp_factor
        else:
            # White noise
            noise_az = self.rng.normal(0, self.torque_std) * ramp_factor
            noise_el = self.rng.normal(0, self.torque_std) * ramp_factor
        
        # Total disturbance
        state.total_torque_az = bias_az + noise_az
        state.total_torque_el = bias_el + noise_el
        
        # Store components for analysis
        state.wind_torque_az = bias_az  # Use wind fields for bias
        state.wind_torque_el = bias_el
        state.structural_torque_az = noise_az  # Use structural for noise
        state.structural_torque_el = noise_el
        
        return state
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.rng = np.random.default_rng(self.seed)
        self.noise_state = np.array([0.0, 0.0])


# =============================================================================
# Factory functions for easy configuration
# =============================================================================

def create_default_disturbances(dt: float = 0.001) -> EnvironmentalDisturbances:
    """Create disturbance generator with default settings."""
    config = EnvironmentalDisturbanceConfig()
    return EnvironmentalDisturbances(config, dt)


def create_wind_disturbance(
    mean_velocity: float = 5.0,
    start_time: float = 2.0,
    turbulence_intensity: float = 0.15,
    dt: float = 0.001
) -> EnvironmentalDisturbances:
    """Create disturbance generator with wind only."""
    config = EnvironmentalDisturbanceConfig(
        wind=DrydenWindConfig(
            enabled=True,
            mean_velocity=mean_velocity,
            start_time=start_time,
            turbulence_intensity=turbulence_intensity
        ),
        vibration=StructuralVibrationConfig(enabled=False),
        structural_noise_enabled=False
    )
    return EnvironmentalDisturbances(config, dt)


def create_vibration_disturbance(
    modal_frequencies: List[float] = [15.0, 45.0],
    start_time: float = 0.0,
    inertia_coupling: float = 0.1,
    dt: float = 0.001
) -> EnvironmentalDisturbances:
    """Create disturbance generator with vibration only."""
    config = EnvironmentalDisturbanceConfig(
        wind=DrydenWindConfig(enabled=False),
        vibration=StructuralVibrationConfig(
            enabled=True,
            modal_frequencies=modal_frequencies,
            start_time=start_time,
            inertia_coupling=inertia_coupling
        ),
        structural_noise_enabled=False
    )
    return EnvironmentalDisturbances(config, dt)


def create_environmental_disturbances(
    config_dict: Dict[str, Any],
    dt: float = 0.001
) -> EnvironmentalDisturbances:
    """
    Factory function to create EnvironmentalDisturbances from a dictionary.
    
    This is the primary interface for runtime configuration, especially
    when config is loaded from JSON or passed through SimulationConfig.
    
    Parameters
    ----------
    config_dict : dict
        Dictionary with keys:
        - 'wind': dict with DrydenWindConfig fields
        - 'vibration': dict with StructuralVibrationConfig fields
        - 'structural_noise': dict with 'enabled', 'std', 'freq_low', 'freq_high'
        - 'seed': int, RNG seed for reproducibility
    dt : float
        Simulation timestep [s]
        
    Returns
    -------
    EnvironmentalDisturbances
        Configured disturbance generator
        
    Example
    -------
    >>> cfg = {
    ...     'wind': {'enabled': True, 'turbulence_intensity': 0.2, 'mean_velocity': 8.0},
    ...     'vibration': {'enabled': True, 'modal_frequencies': [15, 45, 80]},
    ...     'structural_noise': {'enabled': True, 'std': 0.005}
    ... }
    >>> disturbances = create_environmental_disturbances(cfg, dt=0.001)
    """
    # Extract sub-configs with defaults
    wind_cfg = config_dict.get('wind', {})
    vib_cfg = config_dict.get('vibration', {})
    noise_cfg = config_dict.get('structural_noise', {})
    
    # Build DrydenWindConfig
    # Map user-friendly names to dataclass field names
    wind_config = DrydenWindConfig(
        enabled=wind_cfg.get('enabled', True),
        start_time=wind_cfg.get('start_time', 0.0),
        mean_velocity=wind_cfg.get('mean_velocity', 5.0),
        turbulence_intensity=wind_cfg.get('turbulence_intensity', 0.15),
        scale_length=wind_cfg.get('scale_length', 200.0),
        direction_deg=wind_cfg.get('direction_deg', 45.0),
        gimbal_area=wind_cfg.get('gimbal_area', 0.02),
        gimbal_arm=wind_cfg.get('gimbal_arm', 0.15),
        drag_coefficient=wind_cfg.get('drag_coefficient', 1.2),
        air_density=wind_cfg.get('air_density', 1.225),
        correlation_time=wind_cfg.get('correlation_time', 0.0)
    )
    
    # Build StructuralVibrationConfig
    vibration_config = StructuralVibrationConfig(
        enabled=vib_cfg.get('enabled', True),
        start_time=vib_cfg.get('start_time', 0.0),
        modal_frequencies=vib_cfg.get('modal_frequencies', [15.0, 45.0, 80.0]),
        modal_dampings=vib_cfg.get('modal_dampings', [0.02, 0.015, 0.01]),
        modal_amplitudes=vib_cfg.get('modal_amplitudes', [1e-3, 5e-4, 2e-4]),
        inertia_coupling=vib_cfg.get('inertia_coupling', 0.1),
        noise_floor_psd=vib_cfg.get('noise_floor_psd', 1e-6)
    )
    
    # Build master config
    env_config = EnvironmentalDisturbanceConfig(
        enabled=config_dict.get('enabled', True),
        seed=config_dict.get('seed', 42),
        wind=wind_config,
        vibration=vibration_config,
        structural_noise_enabled=noise_cfg.get('enabled', True),
        structural_noise_std=noise_cfg.get('std', 0.005),
        structural_noise_freq_low=noise_cfg.get('freq_low', 100.0),
        structural_noise_freq_high=noise_cfg.get('freq_high', 500.0)
    )
    
    return EnvironmentalDisturbances(env_config, dt)
