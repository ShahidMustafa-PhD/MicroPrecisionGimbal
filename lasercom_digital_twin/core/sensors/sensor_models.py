"""
Sensor Models for Laser Communication Terminal

This module implements high-fidelity models of industrial-grade sensors used in
precision pointing systems. All models capture critical non-ideal effects that
limit system performance, including noise, bias, drift, quantization, and latency.

All noise generation uses seeded random number generators to ensure deterministic
execution for debugging and continuous integration.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from collections import deque


class SensorModel(ABC):
    """
    Abstract base class for all sensor models.
    
    Defines the standard interface for sensor measurements including:
    - Configuration-driven initialization
    - Deterministic random number generation via seeding
    - Time-dependent state updates
    - True-to-measured value conversion
    """
    
    def __init__(self, config: dict, seed: int = 42):
        """
        Initialize the sensor model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with sensor-specific parameters
        seed : int, optional
            Random number generator seed for deterministic execution
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        
    @abstractmethod
    def measure(self, true_value: float) -> float:
        """
        Convert true physical value to measured sensor output.
        
        Parameters
        ----------
        true_value : float
            True physical quantity being measured
            
        Returns
        -------
        float
            Measured value including all non-ideal effects
        """
        pass
    
    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update time-dependent sensor states (e.g., bias drift).
        
        Parameters
        ----------
        dt : float
            Time step [s]
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset sensor to initial conditions.
        """
        pass


class AbsoluteEncoder(SensorModel):
    """
    Absolute angular encoder model for gimbal position measurement.
    
    Models a high-resolution optical encoder with:
    - White noise from electronic and optical sources
    - Static bias offset
    - Quantization effects from ADC discretization
    
    Typical specifications:
    - Resolution: 18-24 bits per revolution
    - Accuracy: 5-50 arcsec RMS
    """
    
    def __init__(self, config: dict, seed: int = 42):
        """
        Initialize absolute encoder model.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'resolution_bits': ADC resolution [bits] (e.g., 20)
            - 'noise_rms': White noise RMS [rad] (e.g., 2.4e-5)
            - 'bias': Static bias offset [rad] (e.g., 1e-5)
            - 'range_min': Minimum angle [rad] (default: 0)
            - 'range_max': Maximum angle [rad] (default: 2π)
        seed : int, optional
            RNG seed for deterministic noise generation
        """
        super().__init__(config, seed)
        
        # Noise parameters
        self.noise_rms: float = config.get('noise_rms', 2.4e-5)  # rad (~5 arcsec)
        self.bias: float = config.get('bias', 1.0e-5)  # rad
        
        # Quantization parameters
        self.resolution_bits: int = config.get('resolution_bits', 20)  # 20-bit encoder
        self.range_min: float = config.get('range_min', 0.0)  # rad
        self.range_max: float = config.get('range_max', 2.0 * np.pi)  # rad
        
        # Compute quantization step size
        self.q_step: float = (self.range_max - self.range_min) / (2**self.resolution_bits)
        
    def measure(self, true_value: float) -> float:
        """
        Measure angular position with encoder non-idealities.
        
        Parameters
        ----------
        true_value : float
            True angular position [rad]
            
        Returns
        -------
        float
            Measured position [rad] including noise, bias, and quantization
        """
        # Add white noise
        noise = self.rng.normal(0.0, self.noise_rms)
        
        # Add bias---
        measured = true_value + self.bias + noise
        
        # Apply quantization
        quantized = self._quantize(measured)
        
        return quantized
    
    def _quantize(self, value: float) -> float:
        """
        Apply ADC quantization to continuous value.
        
        Parameters
        ----------
        value : float
            Continuous input value [rad]
            
        Returns
        -------
        float
            Quantized value [rad]
        """
        # Normalize to range
        normalized = (value - self.range_min) / (self.range_max - self.range_min)
        
        # Quantize to integer levels
        levels = int(normalized * (2**self.resolution_bits))
        levels = np.clip(levels, 0, 2**self.resolution_bits - 1)
        
        # Convert back to physical units
        quantized = self.range_min + levels * self.q_step
        
        return quantized
    
    def update(self, dt: float) -> None:
        """
        Update encoder state (no time-dependent effects for absolute encoder).
        
        Parameters
        ----------
        dt : float
            Time step [s]
        """
        pass  # Absolute encoders have no drift in this model
    
    def reset(self) -> None:
        """
        Reset encoder to initial state.
        """
        pass  # No internal state to reset


class RateGyro(SensorModel):
    """
    Rate gyroscope (angular velocity sensor) model.
    
    Models a MEMS or fiber-optic gyroscope with:
    - White noise (angle random walk, ARW)
    - Bias instability (modeled as random walk)
    - Fixed latency (delay buffer)
    
    Typical specifications:
    - Noise density: 0.001-0.1 deg/√hr
    - Bias stability: 0.1-10 deg/hr
    - Latency: 1-10 ms
    """
    
    def __init__(self, config: dict, seed: int = 42):
        """
        Initialize rate gyroscope model.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'noise_density': Angle random walk [rad/√s] (e.g., 1e-6)
            - 'bias_initial': Initial bias offset [rad/s] (e.g., 1e-4)
            - 'bias_drift_rate': Bias random walk rate [rad/s/√s] (e.g., 1e-7)
            - 'latency': Fixed time delay [s] (e.g., 0.001)
            - 'sample_rate': Nominal sample rate [Hz] (e.g., 1000)
        seed : int, optional
            RNG seed for deterministic noise generation
        """
        super().__init__(config, seed)
        
        # Noise parameters
        self.noise_density: float = config.get('noise_density', 1.0e-6)  # rad/√s
        self.bias: float = config.get('bias_initial', 1.0e-4)  # rad/s
        self.bias_drift_rate: float = config.get('bias_drift_rate', 1.0e-7)  # rad/s/√s
        
        # Latency modeling (fixed delay buffer)
        self.latency: float = config.get('latency', 0.001)  # s (1 ms)
        self.sample_rate: float = config.get('sample_rate', 1000.0)  # Hz
        
        # Initialize delay buffer
        buffer_length = max(1, int(self.latency * self.sample_rate))
        self.delay_buffer: deque = deque([0.0] * buffer_length, maxlen=buffer_length)
        
    def measure(self, true_value: float) -> float:
        """
        Measure angular velocity with gyro non-idealities.
        
        Parameters
        ----------
        true_value : float
            True angular velocity [rad/s]
            
        Returns
        -------
        float
            Measured angular velocity [rad/s] including noise, bias, and latency
        """
        # Add white noise (scale by sqrt of sample time for discrete-time)
        dt_sample = 1.0 / self.sample_rate
        noise = self.rng.normal(0.0, self.noise_density / np.sqrt(dt_sample))
        
        # Add current bias
        measured = true_value + self.bias + noise
        
        # Add to delay buffer
        self.delay_buffer.append(measured)
        
        # Return delayed measurement
        return self.delay_buffer[0]
    
    def update(self, dt: float) -> None:
        """
        Update gyro bias drift using random walk model.
        
        Bias evolves as: bias(t+dt) = bias(t) + N(0, σ_drift * √dt)
        
        Parameters
        ----------
        dt : float
            Time step [s]
        """
        # Random walk bias drift
        bias_change = self.rng.normal(0.0, self.bias_drift_rate * np.sqrt(dt))
        self.bias += bias_change
    
    def reset(self) -> None:
        """
        Reset gyro to initial conditions.
        """
        self.bias = self.config.get('bias_initial', 1.0e-4)
        buffer_length = max(1, int(self.latency * self.sample_rate))
        self.delay_buffer = deque([0.0] * buffer_length, maxlen=buffer_length)


class IncrementalEncoder(SensorModel):
    """
    Incremental (relative) encoder model for velocity measurement.
    
    Models a quadrature encoder that measures velocity by counting edge transitions.
    Includes:
    - White noise from edge detection uncertainty
    - Sampling jitter (variable measurement interval)
    - Integration error when converting to position
    
    Typical specifications:
    - Pulses per revolution: 1000-10000
    - Velocity accuracy: 0.1-1% of reading
    """
    
    def __init__(self, config: dict, seed: int = 42):
        """
        Initialize incremental encoder model.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'pulses_per_rev': Encoder line count [pulses/rev] (e.g., 5000)
            - 'noise_rms': Velocity noise RMS [rad/s] (e.g., 1e-4)
            - 'jitter_std': Sampling time jitter std dev [s] (e.g., 1e-6)
            - 'nominal_dt': Nominal sampling interval [s] (e.g., 0.001)
        seed : int, optional
            RNG seed for deterministic noise generation
        """
        super().__init__(config, seed)
        
        # Encoder parameters
        self.pulses_per_rev: int = config.get('pulses_per_rev', 5000)
        self.noise_rms: float = config.get('noise_rms', 1.0e-4)  # rad/s
        
        # Sampling jitter parameters
        self.jitter_std: float = config.get('jitter_std', 1.0e-6)  # s
        self.nominal_dt: float = config.get('nominal_dt', 0.001)  # s
        
        # State for integration
        self.integrated_position: float = 0.0
        self.last_measurement_time: float = 0.0
        
    def measure(self, true_value: float) -> float:
        """
        Measure angular velocity with incremental encoder non-idealities.
        
        Parameters
        ----------
        true_value : float
            True angular velocity [rad/s]
            
        Returns
        -------
        float
            Measured velocity [rad/s] including noise and jitter effects
        """
        # Add white noise
        noise = self.rng.normal(0.0, self.noise_rms)
        
        # Sampling jitter affects the effective measurement by introducing
        # uncertainty in the time base used for velocity calculation
        jitter = self.rng.normal(0.0, self.jitter_std)
        dt_actual = self.nominal_dt + jitter
        
        # Velocity measurement with jitter-induced error
        # If true position changed by Δθ in time dt_actual, but we think it was nominal_dt,
        # we get velocity error
        velocity_error_from_jitter = true_value * (dt_actual - self.nominal_dt) / self.nominal_dt
        
        measured = true_value + noise + velocity_error_from_jitter
        
        return measured
    
    def update(self, dt: float) -> None:
        """
        Update integrated position (for tracking cumulative error).
        
        Parameters
        ----------
        dt : float
            Time step [s]
        """
        # This is a placeholder for tracking integration error
        # In a full implementation, this would accumulate velocity measurements
        pass
    
    def reset(self) -> None:
        """
        Reset encoder state.
        """
        self.integrated_position = 0.0
        self.last_measurement_time = 0.0
