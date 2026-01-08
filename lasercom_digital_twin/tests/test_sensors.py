"""
Unit tests for sensor models.

This module contains pytest-based unit tests for all sensor models including
encoders, gyroscopes, and the quadrant detector. Tests verify non-ideal effects
such as quantization, bias drift, noise characteristics, and non-linear sensitivity.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensors.sensor_models import AbsoluteEncoder, RateGyro, IncrementalEncoder
from core.sensors.quadrant_detector import QuadrantDetector


class TestAbsoluteEncoder:
    """Test suite for AbsoluteEncoder."""
    
    @pytest.fixture
    def encoder_config(self):
        """Standard encoder configuration for testing."""
        return {
            'resolution_bits': 20,
            'noise_rms': 2.4e-5,  # ~5 arcsec
            'bias': 1.0e-5,  # rad
            'range_min': 0.0,
            'range_max': 2.0 * np.pi
        }
    
    def test_initialization(self, encoder_config):
        """Test encoder initializes with correct parameters."""
        encoder = AbsoluteEncoder(encoder_config, seed=42)
        
        assert encoder.resolution_bits == 20
        assert encoder.noise_rms == 2.4e-5
        assert encoder.bias == 1.0e-5
        assert encoder.q_step > 0
    
    def test_quantization_observable(self, encoder_config):
        """
        Test that quantization effects are observable.
        
        Two very close input values should map to the same quantized output
        if they fall within one quantization step.
        """
        encoder = AbsoluteEncoder(encoder_config, seed=42)
        
        # Use a fixed angle and measure multiple times (noise should average out over many samples)
        # But quantization should remain
        true_angle = 1.0  # rad
        
        # Take multiple measurements with same input
        measurements = [encoder.measure(true_angle) for _ in range(100)]
        
        # All measurements should be quantized to discrete levels
        # Check that values cluster around quantization steps
        unique_values = np.unique(measurements)
        
        # Should have discrete levels, not continuous
        # With noise, we might see a few adjacent quantization levels
        assert len(unique_values) < 50, \
            f"Too many unique values ({len(unique_values)}), quantization not evident"
        
        # Check quantization step size is approximately correct
        if len(unique_values) > 1:
            min_diff = np.min(np.diff(sorted(unique_values)))
            expected_q_step = encoder.q_step
            
            # Minimum difference should be at least one quantization step
            assert min_diff >= expected_q_step * 0.9, \
                f"Quantization step too small: {min_diff} < {expected_q_step}"
    
    def test_bias_present(self, encoder_config):
        """
        Test that static bias is present in measurements.
        """
        encoder = AbsoluteEncoder(encoder_config, seed=42)
        
        true_angle = 1.0  # rad
        
        # Take many measurements and average to remove noise
        measurements = [encoder.measure(true_angle) for _ in range(1000)]
        mean_measurement = np.mean(measurements)
        
        # Mean should be offset from true value by approximately the bias
        measured_bias = mean_measurement - true_angle
        
        # Should be within reasonable tolerance of configured bias
        # (quantization adds some uncertainty)
        assert np.abs(measured_bias - encoder_config['bias']) < 5 * encoder.q_step, \
            f"Bias not correct: measured={measured_bias}, expected={encoder_config['bias']}"
    
    def test_deterministic_with_seed(self):
        """
        Test that same seed produces identical measurement sequences.
        """
        config = {
            'resolution_bits': 20,
            'noise_rms': 2.4e-5,
            'bias': 1.0e-5
        }
        
        # Create two encoders with same seed
        encoder1 = AbsoluteEncoder(config, seed=123)
        encoder2 = AbsoluteEncoder(config, seed=123)
        
        # Generate measurement sequences
        true_angle = 1.5
        measurements1 = [encoder1.measure(true_angle) for _ in range(10)]
        measurements2 = [encoder2.measure(true_angle) for _ in range(10)]
        
        # Should be identical
        assert np.allclose(measurements1, measurements2), \
            "Same seed should produce identical measurements"


class TestRateGyro:
    """Test suite for RateGyro."""
    
    @pytest.fixture
    def gyro_config(self):
        """Standard gyro configuration for testing."""
        return {
            'noise_density': 1.0e-6,  # rad/√s
            'bias_initial': 1.0e-4,  # rad/s
            'bias_drift_rate': 1.0e-7,  # rad/s/√s
            'latency': 0.001,  # s (1 ms)
            'sample_rate': 1000.0  # Hz
        }
    
    def test_initialization(self, gyro_config):
        """Test gyro initializes with correct parameters."""
        gyro = RateGyro(gyro_config, seed=42)
        
        assert gyro.noise_density == 1.0e-6
        assert gyro.bias == 1.0e-4
        assert gyro.latency == 0.001
        assert len(gyro.delay_buffer) > 0
    
    def test_bias_drift_observable(self, gyro_config):
        """
        Test that bias drift is observable over time.
        
        The bias should change as a random walk when update() is called.
        """
        gyro = RateGyro(gyro_config, seed=42)
        
        initial_bias = gyro.bias
        
        # Update bias for many steps
        dt = 0.001  # s
        n_steps = 10000
        
        for _ in range(n_steps):
            gyro.update(dt)
        
        final_bias = gyro.bias
        
        # Bias should have drifted
        bias_change = np.abs(final_bias - initial_bias)
        
        # Expected RMS change: σ_drift * sqrt(N*dt)
        expected_rms_change = gyro_config['bias_drift_rate'] * np.sqrt(n_steps * dt)
        
        # Should be of similar order of magnitude (within factor of 5 for statistical variation)
        assert bias_change > expected_rms_change * 0.2, \
            f"Bias drift too small: {bias_change} << {expected_rms_change}"
        
        assert bias_change < expected_rms_change * 5.0, \
            f"Bias drift too large: {bias_change} >> {expected_rms_change}"
    
    def test_latency_present(self, gyro_config):
        """
        Test that latency delay is present in measurements.
        
        Apply step input and verify output is delayed.
        """
        gyro = RateGyro(gyro_config, seed=42)
        
        # Measure zero rate to fill buffer
        for _ in range(10):
            gyro.measure(0.0)
        
        # Apply step input
        true_rate_initial = 0.0
        true_rate_final = 1.0  # rad/s
        
        # First measurement after step should still be near zero (delayed)
        first_measurement = gyro.measure(true_rate_final)
        
        # Should be close to initial rate, not final rate
        assert np.abs(first_measurement - true_rate_initial) < 0.1, \
            f"Latency not working: immediate response {first_measurement} to step input"
        
        # After filling the buffer, should converge to final rate
        for _ in range(20):
            measurement = gyro.measure(true_rate_final)
        
        # Now should be close to final rate (within noise)
        assert np.abs(measurement - true_rate_final) < 0.2, \
            f"Failed to converge to final rate: {measurement} vs {true_rate_final}"
    
    def test_reset(self, gyro_config):
        """Test that reset properly clears gyro state."""
        gyro = RateGyro(gyro_config, seed=42)
        
        # Modify state
        for _ in range(100):
            gyro.update(0.01)
            gyro.measure(1.0)
        
        initial_bias = gyro_config['bias_initial']
        assert gyro.bias != initial_bias, "Bias should have drifted"
        
        # Reset
        gyro.reset()
        
        assert gyro.bias == initial_bias, "Bias should be reset to initial value"
        assert all(v == 0.0 for v in gyro.delay_buffer), "Buffer should be cleared"


class TestIncrementalEncoder:
    """Test suite for IncrementalEncoder."""
    
    @pytest.fixture
    def inc_encoder_config(self):
        """Standard incremental encoder configuration."""
        return {
            'pulses_per_rev': 5000,
            'noise_rms': 1.0e-4,  # rad/s
            'jitter_std': 1.0e-6,  # s
            'nominal_dt': 0.001  # s (1 kHz)
        }
    
    def test_initialization(self, inc_encoder_config):
        """Test incremental encoder initializes correctly."""
        encoder = IncrementalEncoder(inc_encoder_config, seed=42)
        
        assert encoder.pulses_per_rev == 5000
        assert encoder.noise_rms == 1.0e-4
        assert encoder.jitter_std == 1.0e-6
    
    def test_sampling_jitter_effect(self, inc_encoder_config):
        """
        Test that sampling jitter introduces measurement uncertainty.
        
        Measuring the same true velocity multiple times should show variation
        due to jitter.
        """
        encoder = IncrementalEncoder(inc_encoder_config, seed=42)
        
        true_velocity = 1.0  # rad/s
        
        # Take many measurements of same velocity
        measurements = [encoder.measure(true_velocity) for _ in range(1000)]
        
        # Standard deviation should include both noise and jitter contributions
        std_measured = np.std(measurements)
        
        # Expected: noise_rms + jitter-induced error
        # Jitter error ~ velocity * jitter_std / nominal_dt
        expected_jitter_error = true_velocity * inc_encoder_config['jitter_std'] / inc_encoder_config['nominal_dt']
        total_expected_std = np.sqrt(inc_encoder_config['noise_rms']**2 + expected_jitter_error**2)
        
        # Measured std should be of similar magnitude
        assert std_measured > total_expected_std * 0.5, \
            f"Measured std too low: {std_measured} < {total_expected_std}"
        
        assert std_measured < total_expected_std * 2.0, \
            f"Measured std too high: {std_measured} > {total_expected_std}"
    
    def test_deterministic_with_seed(self, inc_encoder_config):
        """Test deterministic behavior with same seed."""
        encoder1 = IncrementalEncoder(inc_encoder_config, seed=99)
        encoder2 = IncrementalEncoder(inc_encoder_config, seed=99)
        
        true_velocity = 0.5
        measurements1 = [encoder1.measure(true_velocity) for _ in range(10)]
        measurements2 = [encoder2.measure(true_velocity) for _ in range(10)]
        
        assert np.allclose(measurements1, measurements2), \
            "Same seed should produce identical measurements"


class TestQuadrantDetector:
    """Test suite for QuadrantDetector."""
    
    @pytest.fixture
    def qpd_config(self):
        """Standard QPD configuration."""
        return {
            'sensitivity': 2000.0,  # V/rad
            'linear_range': 100.0e-6,  # rad (±100 µrad)
            'noise_voltage_rms': 1.0e-4,  # V
            'bias_x': 0.001,  # V
            'bias_y': 0.001,  # V
            'saturation_voltage': 10.0,  # V
            'nonlinearity_factor': 0.15
        }
    
    def test_initialization(self, qpd_config):
        """Test QPD initializes with correct parameters."""
        qpd = QuadrantDetector(qpd_config, seed=42)
        
        assert qpd.sensitivity == 2000.0
        assert qpd.linear_range == 100.0e-6
        assert qpd.bias_x == 0.001
    
    def test_nonlinearity_observable(self, qpd_config):
        """
        Test that non-linear sensitivity is observable.
        
        The sensitivity (dV/dθ) should decrease at larger angles.
        """
        qpd = QuadrantDetector(qpd_config, seed=42)
        
        # Measure at small angle (linear region)
        angle_small = qpd_config['linear_range'] * 0.1  # 10% of range
        v_x_small, _ = qpd.measure(angle_small, 0.0)
        
        # Measure at large angle (near edge)
        angle_large = qpd_config['linear_range'] * 0.9  # 90% of range
        v_x_large, _ = qpd.measure(angle_large, 0.0)
        
        # Compute effective sensitivities
        sensitivity_small = v_x_small / angle_small
        sensitivity_large = v_x_large / angle_large
        
        # Sensitivity should decrease at larger angles due to non-linearity
        assert sensitivity_large < sensitivity_small * 0.95, \
            f"Non-linearity not evident: S_large={sensitivity_large}, S_small={sensitivity_small}"
    
    def test_noise_present(self, qpd_config):
        """
        Test that white noise is present in measurements.
        """
        qpd = QuadrantDetector(qpd_config, seed=42)
        
        # Measure same angle multiple times
        tip_error = 50.0e-6  # 50 µrad
        tilt_error = 0.0
        
        measurements_x = [qpd.measure(tip_error, tilt_error)[0] for _ in range(1000)]
        
        # Standard deviation should be approximately noise_voltage_rms
        std_measured = np.std(measurements_x)
        
        # Should be within factor of 2 (statistical variation)
        assert std_measured > qpd_config['noise_voltage_rms'] * 0.5, \
            f"Noise too low: {std_measured} < {qpd_config['noise_voltage_rms']}"
        
        assert std_measured < qpd_config['noise_voltage_rms'] * 2.0, \
            f"Noise too high: {std_measured} > {qpd_config['noise_voltage_rms']}"
    
    def test_bias_present(self, qpd_config):
        """
        Test that voltage bias is present in measurements.
        """
        qpd = QuadrantDetector(qpd_config, seed=42)
        
        # Measure at zero error
        tip_error = 0.0
        tilt_error = 0.0
        
        # Take many measurements and average
        measurements_x = [qpd.measure(tip_error, tilt_error)[0] for _ in range(1000)]
        measurements_y = [qpd.measure(tip_error, tilt_error)[1] for _ in range(1000)]
        
        mean_x = np.mean(measurements_x)
        mean_y = np.mean(measurements_y)
        
        # Means should be approximately the bias values
        assert np.abs(mean_x - qpd_config['bias_x']) < 0.001, \
            f"X-axis bias not correct: {mean_x} vs {qpd_config['bias_x']}"
        
        assert np.abs(mean_y - qpd_config['bias_y']) < 0.001, \
            f"Y-axis bias not correct: {mean_y} vs {qpd_config['bias_y']}"
    
    def test_voltage_to_angle_inversion(self, qpd_config):
        """
        Test that voltage_to_angle approximately inverts the measurement.
        """
        qpd = QuadrantDetector(qpd_config, seed=42)
        
        # True errors
        tip_true = 50.0e-6  # 50 µrad
        tilt_true = 30.0e-6  # 30 µrad
        
        # Measure
        v_x, v_y = qpd.measure(tip_true, tilt_true)
        
        # Invert
        tip_est, tilt_est = qpd.voltage_to_angle(v_x, v_y)
        
        # Should be close to true values (within noise and non-linearity)
        assert np.abs(tip_est - tip_true) < 10.0e-6, \
            f"Tip inversion error too large: {tip_est} vs {tip_true}"
        
        assert np.abs(tilt_est - tilt_true) < 10.0e-6, \
            f"Tilt inversion error too large: {tilt_est} vs {tilt_true}"
    
    def test_quadrant_voltages_sum(self, qpd_config):
        """
        Test that individual quadrant voltages sum to total power.
        """
        qpd = QuadrantDetector(qpd_config, seed=42)
        
        # Measure with small error
        tip_error = 10.0e-6
        tilt_error = 10.0e-6
        
        v1, v2, v3, v4 = qpd.get_quadrant_voltages(tip_error, tilt_error)
        
        # Sum should be approximately 1.0 (normalized power)
        total = v1 + v2 + v3 + v4
        
        assert np.abs(total - 1.0) < 0.2, \
            f"Quadrant voltages don't sum correctly: {total} vs 1.0"
    
    def test_deterministic_with_seed(self, qpd_config):
        """Test deterministic behavior with same seed."""
        qpd1 = QuadrantDetector(qpd_config, seed=777)
        qpd2 = QuadrantDetector(qpd_config, seed=777)
        
        tip_error = 50.0e-6
        tilt_error = 30.0e-6
        
        measurements1 = [qpd1.measure(tip_error, tilt_error) for _ in range(10)]
        measurements2 = [qpd2.measure(tip_error, tilt_error) for _ in range(10)]
        
        assert np.allclose(measurements1, measurements2), \
            "Same seed should produce identical measurements"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
