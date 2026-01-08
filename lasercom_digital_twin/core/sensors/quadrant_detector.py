"""
Quadrant Detector (QPD) Model for Optical Pointing Error Sensing

This module implements a high-fidelity model of a quadrant photodetector used
to measure line-of-sight (LOS) pointing errors in laser communication terminals.
The QPD converts the spatial position of a laser spot into electrical signals
from four quadrant photodiodes.
"""

import numpy as np
from typing import Tuple


class QuadrantDetector:
    """
    Quadrant photodetector model for fine pointing error measurement.
    
    The QPD consists of four photodiodes arranged in quadrants. The position
    of the laser spot is determined from the normalized error signal (NES):
    
        NES_x = (V1 + V4 - V2 - V3) / (V1 + V2 + V3 + V4)
        NES_y = (V1 + V2 - V3 - V4) / (V1 + V2 + V3 + V4)
    
    Non-ideal effects modeled:
    - Non-linear sensitivity near detector edges
    - High white noise from photon shot noise and detector electronics
    - Static voltage bias from amplifier offsets
    
    Typical specifications:
    - Linear range: ±100-500 µrad
    - Sensitivity: 0.01-0.1 V/µrad
    - Noise: 10-100 nV/√Hz
    """
    
    def __init__(self, config: dict, seed: int = 42):
        """
        Initialize quadrant detector model.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'sensitivity': Linear sensitivity [V/rad] (e.g., 2000.0)
            - 'linear_range': Linear region half-width [rad] (e.g., 100e-6)
            - 'noise_voltage_rms': Voltage noise RMS [V] (e.g., 1e-4)
            - 'bias_x': X-axis voltage bias [V] (e.g., 0.001)
            - 'bias_y': Y-axis voltage bias [V] (e.g., 0.001)
            - 'saturation_voltage': Maximum output voltage [V] (e.g., 10.0)
            - 'nonlinearity_factor': Non-linearity coefficient (e.g., 0.15)
            - 'detector_size_um': Physical detector size [µm] (e.g., 5000.0)
            - 'spot_size_um': Laser spot 1/e² radius [µm] (e.g., 50.0)
        seed : int, optional
            RNG seed for deterministic noise generation
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Sensitivity and range
        self.sensitivity: float = config.get('sensitivity', 2000.0)  # V/rad
        self.linear_range: float = config.get('linear_range', 100.0e-6)  # rad (±100 µrad)
        
        # Noise parameters
        self.noise_voltage_rms: float = config.get('noise_voltage_rms', 1.0e-4)  # V
        
        # Bias offsets
        self.bias_x: float = config.get('bias_x', 0.001)  # V
        self.bias_y: float = config.get('bias_y', 0.001)  # V
        
        # Saturation
        self.saturation_voltage: float = config.get('saturation_voltage', 10.0)  # V
        
        # Non-linearity parameter
        self.nonlinearity_factor: float = config.get('nonlinearity_factor', 0.15)
        
        # Physical detector parameters for micron-level accuracy analysis
        self.detector_size_um: float = config.get('detector_size_um', 5000.0)  # µm (5 mm)
        self.spot_size_um: float = config.get('spot_size_um', 50.0)  # µm
        
        # Compute derived parameters for NES analysis
        self._compute_nes_parameters()
    
    def _compute_nes_parameters(self) -> None:
        """
        Compute derived NES sensitivity parameters.
        
        For a Gaussian beam profile on a quadrant detector:
        NES_sensitivity = 4 / (π * spot_diameter)
        
        This gives the slope dNES/dx in units of [1/µm].
        """
        # NES sensitivity (slope of NES vs. position curve)
        # Units: [dimensionless / µm]
        self.nes_sensitivity_per_um = 4.0 / (np.pi * self.spot_size_um)
        
        # Voltage sensitivity (combines NES sensitivity with V/rad conversion)
        # Units: [V/µm]
        self.voltage_sensitivity_per_um = self.sensitivity * self.linear_range * self.nes_sensitivity_per_um
    
    def _apply_nonlinearity(self, position: float) -> float:
        """
        Apply non-linear sensitivity characteristic.
        
        The detector sensitivity decreases as the spot approaches the edge
        of the detector. Modeled using a cubic correction term:
        
            V = S * x * (1 - k * (x/x_max)²)
        
        where k is the non-linearity factor.
        
        Parameters
        ----------
        position : float
            Normalized position error (in linear range units)
            
        Returns
        -------
        float
            Position with non-linear correction applied
        """
        # Normalize to linear range
        normalized_pos = position / self.linear_range
        
        # Apply cubic non-linearity (reduces sensitivity at edges)
        nonlinear_correction = 1.0 - self.nonlinearity_factor * normalized_pos**2
        
        # Ensure positive sensitivity
        nonlinear_correction = max(0.1, nonlinear_correction)
        
        return position * nonlinear_correction
    
    def measure(
        self, 
        tip_error: float, 
        tilt_error: float
    ) -> Tuple[float, float]:
        """
        Measure LOS pointing error and convert to QPD voltage signals.
        
        This is the main measurement method that converts true optical
        pointing errors (in radians) to electrical signals (in volts).
        
        Parameters
        ----------
        tip_error : float
            True tip (X-axis) pointing error [rad]
        tilt_error : float
            True tilt (Y-axis) pointing error [rad]
            
        Returns
        -------
        Tuple[float, float]
            (voltage_x, voltage_y) - QPD output voltages [V]
        """
        # Apply non-linear sensitivity characteristic
        tip_corrected = self._apply_nonlinearity(tip_error)
        tilt_corrected = self._apply_nonlinearity(tilt_error)
        
        # Convert to voltage (linear within corrected range)
        voltage_x_ideal = self.sensitivity * tip_corrected
        voltage_y_ideal = self.sensitivity * tilt_corrected
        
        # Add white noise (independent for each axis)
        noise_x = self.rng.normal(0.0, self.noise_voltage_rms)
        noise_y = self.rng.normal(0.0, self.noise_voltage_rms)
        
        voltage_x = voltage_x_ideal + self.bias_x + noise_x
        voltage_y = voltage_y_ideal + self.bias_y + noise_y
        
        # Apply saturation
        voltage_x = np.clip(voltage_x, -self.saturation_voltage, self.saturation_voltage)
        voltage_y = np.clip(voltage_y, -self.saturation_voltage, self.saturation_voltage)
        
        return voltage_x, voltage_y
    
    def voltage_to_angle(
        self, 
        voltage_x: float, 
        voltage_y: float
    ) -> Tuple[float, float]:
        """
        Convert QPD voltages back to angular error estimate.
        
        This is the inverse operation, used by the control system to
        interpret the QPD measurements. Note that this assumes linear
        sensitivity and does not perfectly invert the non-linearity.
        
        Parameters
        ----------
        voltage_x : float
            X-axis voltage measurement [V]
        voltage_y : float
            Y-axis voltage measurement [V]
            
        Returns
        -------
        Tuple[float, float]
            (tip_estimate, tilt_estimate) - Estimated pointing errors [rad]
        """
        # Remove known biases
        voltage_x_corrected = voltage_x - self.bias_x
        voltage_y_corrected = voltage_y - self.bias_y
        
        # Convert to angle using linear sensitivity (inverse)
        tip_estimate = voltage_x_corrected / self.sensitivity
        tilt_estimate = voltage_y_corrected / self.sensitivity
        
        return tip_estimate, tilt_estimate
    
    def get_quadrant_voltages(
        self, 
        tip_error: float, 
        tilt_error: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute individual quadrant photodiode voltages.
        
        This method provides the raw four-quadrant signals before
        computing the normalized error signal (NES). Useful for
        detailed signal processing analysis.
        
        Parameters
        ----------
        tip_error : float
            True tip (X-axis) pointing error [rad]
        tilt_error : float
            True tilt (Y-axis) pointing error [rad]
            
        Returns
        -------
        Tuple[float, float, float, float]
            (V1, V2, V3, V4) - Individual quadrant voltages [V]
            Quadrant layout:
                Q1 | Q2    (+x,+y) | (-x,+y)
                ---|---  = --------|--------
                Q4 | Q3    (+x,-y) | (-x,-y)
        """
        # Get normalized error signal (NES) components
        voltage_x, voltage_y = self.measure(tip_error, tilt_error)
        
        # Assume uniform illumination with total power = 1.0 (normalized)
        # Individual quadrant signals derived from NES definition
        total_power = 1.0  # Normalized total signal
        
        # Simplified model: linear approximation of quadrant distribution
        # V1 ~ total * (1 + NES_x + NES_y) / 4
        # V2 ~ total * (1 - NES_x + NES_y) / 4
        # V3 ~ total * (1 - NES_x - NES_y) / 4
        # V4 ~ total * (1 + NES_x - NES_y) / 4
        
        # Normalize to get NES components
        nes_x = voltage_x / (self.sensitivity * self.linear_range)
        nes_y = voltage_y / (self.sensitivity * self.linear_range)
        
        # Compute quadrant voltages (with small signal approximation)
        v1 = total_power * 0.25 * (1.0 + nes_x + nes_y)
        v2 = total_power * 0.25 * (1.0 - nes_x + nes_y)
        v3 = total_power * 0.25 * (1.0 - nes_x - nes_y)
        v4 = total_power * 0.25 * (1.0 + nes_x - nes_y)
        
        # Add noise to each quadrant independently
        noise = self.rng.normal(0.0, self.noise_voltage_rms * 0.5, 4)
        v1 += noise[0]
        v2 += noise[1]
        v3 += noise[2]
        v4 += noise[3]
        
        # Ensure non-negative (photodiodes produce positive current)
        v1 = max(0.0, v1)
        v2 = max(0.0, v2)
        v3 = max(0.0, v3)
        v4 = max(0.0, v4)
        
        return v1, v2, v3, v4
    
    def compute_nes_from_position(
        self,
        x_position_um: float,
        y_position_um: float
    ) -> Tuple[float, float]:
        """
        Compute normalized error signal (NES) from spot position.
        
        The NES is a dimensionless quantity ranging from -1 to +1 that
        represents the normalized position of the laser spot on the detector.
        
        NES Definition:
        --------------
        NES_x = (V1 + V4 - V2 - V3) / (V1 + V2 + V3 + V4)
        NES_y = (V1 + V2 - V3 - V4) / (V1 + V2 + V3 + V4)
        
        For small displacements with Gaussian beam:
        NES_x ≈ k_x * x  where k_x = 4/(π*w)
        
        Parameters
        ----------
        x_position_um : float
            X position of spot on focal plane [µm]
        y_position_um : float
            Y position of spot on focal plane [µm]
            
        Returns
        -------
        Tuple[float, float]
            (NES_x, NES_y) - Normalized error signals [dimensionless]
        """
        # Apply NES sensitivity
        nes_x = self.nes_sensitivity_per_um * x_position_um
        nes_y = self.nes_sensitivity_per_um * y_position_um
        
        # Clip to physical range [-1, +1]
        nes_x = np.clip(nes_x, -1.0, 1.0)
        nes_y = np.clip(nes_y, -1.0, 1.0)
        
        return nes_x, nes_y
    
    def measure_from_focal_plane(
        self,
        x_position_um: float,
        y_position_um: float
    ) -> Tuple[float, float]:
        """
        Measure QPD output from focal plane position in micrometers.
        
        This method bridges the gap between the optical chain (which computes
        focal plane positions in µm) and the sensor output (voltages).
        
        Parameters
        ----------
        x_position_um : float
            X position on focal plane [µm]
        y_position_um : float
            Y position on focal plane [µm]
            
        Returns
        -------
        Tuple[float, float]
            (voltage_x, voltage_y) - QPD output voltages [V]
        """
        # Compute NES from position
        nes_x, nes_y = self.compute_nes_from_position(x_position_um, y_position_um)
        
        # Convert NES to voltage (linear approximation)
        voltage_x_ideal = nes_x * self.sensitivity * self.linear_range
        voltage_y_ideal = nes_y * self.sensitivity * self.linear_range
        
        # Add noise
        noise_x = self.rng.normal(0.0, self.noise_voltage_rms)
        noise_y = self.rng.normal(0.0, self.noise_voltage_rms)
        
        voltage_x = voltage_x_ideal + self.bias_x + noise_x
        voltage_y = voltage_y_ideal + self.bias_y + noise_y
        
        # Apply saturation
        voltage_x = np.clip(voltage_x, -self.saturation_voltage, self.saturation_voltage)
        voltage_y = np.clip(voltage_y, -self.saturation_voltage, self.saturation_voltage)
        
        return voltage_x, voltage_y
    
    def get_micron_accuracy_metrics(self) -> dict:
        """
        Get key metrics for micron-level pointing accuracy analysis.
        
        This method provides the critical parameters that define the
        relationship between physical spot position and sensor output,
        which determines the achievable pointing accuracy.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'spot_size_um': Laser spot 1/e² radius [µm]
            - 'detector_size_um': Physical detector size [µm]
            - 'nes_sensitivity_per_um': NES change per micron [1/µm]
            - 'voltage_sensitivity_per_um': Voltage change per micron [V/µm]
            - 'noise_voltage_rms': RMS voltage noise [V]
            - 'noise_limited_accuracy_um': Position accuracy [µm]
            - 'noise_limited_accuracy_urad': Angular accuracy [µrad] (if applicable)
        """
        metrics = {
            'spot_size_um': self.spot_size_um,
            'detector_size_um': self.detector_size_um,
            'nes_sensitivity_per_um': self.nes_sensitivity_per_um,
            'voltage_sensitivity_per_um': self.voltage_sensitivity_per_um,
            'noise_voltage_rms': self.noise_voltage_rms,
            'noise_limited_accuracy_um': self.noise_limited_accuracy_um,
            'sensitivity_V_per_rad': self.sensitivity,
            'linear_range_rad': self.linear_range,
            'linear_range_urad': self.linear_range * 1e6
        }
        
        return metrics
    
    def reset(self) -> None:
        """
        Reset detector state (no internal state in current model).
        """
        pass
    
    def update(self, dt: float) -> None:
        """
        Update time-dependent effects (no drift in current model).
        
        Parameters
        ----------
        dt : float
            Time step [s]
        """
        pass
