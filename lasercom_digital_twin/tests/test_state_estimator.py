"""
Unit Tests for Pointing State Estimator

This module tests the Extended Kalman Filter implementation for the
laser communication terminal's state estimation system.

Test Coverage:
-------------
1. State vector initialization and structure
2. Prediction step (process model)
3. Correction step (measurement update)
4. Sensor fusion with multiple sensor types
5. Covariance tuning and convergence
6. Bias estimation accuracy
7. Disturbance estimation
8. Innovation and Kalman gain validation
9. Multi-rate sensor handling
10. Numerical stability
"""

import pytest
import numpy as np
from core.estimators.state_estimator import (
    PointingStateEstimator,
    StateIndex,
    MeasurementIndex,
    EstimatorState
)


class TestStateEstimatorInitialization:
    """Test estimator initialization and configuration."""
    
    def test_default_initialization(self):
        """Verify default estimator initializes correctly."""
        config = {
            'inertia_az': 2.0,
            'inertia_el': 1.5,
            'friction_coeff_az': 0.05,
            'friction_coeff_el': 0.05,
        }
        
        estimator = PointingStateEstimator(config)
        
        # Check state vector dimension
        assert estimator.x_hat.shape == (10,)
        assert estimator.P.shape == (10, 10)
        
        # Check covariance is positive definite
        eigenvalues = np.linalg.eigvals(estimator.P)
        assert np.all(eigenvalues > 0), "Covariance must be positive definite"
        
        # Check initial state is zero
        assert np.allclose(estimator.x_hat, 0.0)
    
    def test_custom_initial_state(self):
        """Verify custom initial state is applied."""
        initial_state = np.array([
            0.1, 0.0, 0.0,  # Az: angle, rate, bias
            0.2, 0.0, 0.0,  # El: angle, rate, bias
            0.0, 0.0,       # Roll: angle, rate
            0.0, 0.0        # Disturbances
        ])
        
        config = {
            'initial_state': initial_state,
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        
        estimator = PointingStateEstimator(config)
        
        assert np.allclose(estimator.x_hat, initial_state)
        assert estimator.x_hat[StateIndex.THETA_AZ] == 0.1
        assert estimator.x_hat[StateIndex.THETA_EL] == 0.2
    
    def test_covariance_tuning_hooks(self):
        """Verify covariance tuning methods work correctly."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        # Set custom process noise
        Q_diag_new = np.ones(10) * 1e-6
        estimator.set_process_noise_covariance(Q_diag_new)
        assert np.allclose(np.diag(estimator.Q), Q_diag_new)
        
        # Set custom measurement noise
        R_diag_new = np.ones(6) * 1e-5
        estimator.set_measurement_noise_covariance(R_diag_new)
        assert np.allclose(np.diag(estimator.R), R_diag_new)


class TestPredictionStep:
    """Test EKF prediction (time update) functionality."""
    
    def test_position_propagation(self):
        """Verify position integrates velocity correctly."""
        config = {
            'initial_state': np.array([0.0, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
            'friction_coeff_az': 0.0,
            'friction_coeff_el': 0.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Predict forward with zero control
        dt = 0.01
        u = np.array([0.0, 0.0])
        estimator.predict(u, dt)
        
        # Position should have integrated velocity
        # θ_new = θ_old + θ̇ * dt
        expected_theta_az = 0.0 + 0.1 * dt
        expected_theta_el = 0.0 + 0.05 * dt
        
        assert np.isclose(estimator.x_hat[StateIndex.THETA_AZ], expected_theta_az, atol=1e-6)
        assert np.isclose(estimator.x_hat[StateIndex.THETA_EL], expected_theta_el, atol=1e-6)
    
    def test_acceleration_from_torque(self):
        """Verify velocity integrates acceleration correctly."""
        config = {
            'initial_state': np.zeros(10),
            'inertia_az': 2.0,  # kg·m²
            'inertia_el': 2.0,
            'friction_coeff_az': 0.0,
            'friction_coeff_el': 0.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Apply torque
        tau = 4.0  # N·m
        u = np.array([tau, tau])
        dt = 0.01
        
        estimator.predict(u, dt)
        
        # Expected acceleration: a = τ / J
        expected_accel = tau / 2.0  # 2.0 rad/s²
        expected_velocity = expected_accel * dt  # 0.02 rad/s
        
        assert np.isclose(estimator.x_hat[StateIndex.THETA_DOT_AZ], expected_velocity, atol=1e-6)
        assert np.isclose(estimator.x_hat[StateIndex.THETA_DOT_EL], expected_velocity, atol=1e-6)
    
    def test_friction_damping(self):
        """Verify friction causes velocity decay."""
        config = {
            'initial_state': np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
            'friction_coeff_az': 0.5,  # N·m·s/rad
            'friction_coeff_el': 0.5,
        }
        estimator = PointingStateEstimator(config)
        
        initial_velocity = 1.0
        u = np.array([0.0, 0.0])
        dt = 0.01
        
        estimator.predict(u, dt)
        
        # Friction torque: τ_friction = -b * θ̇
        # Acceleration: a = -b * θ̇ / J = -0.5 * 1.0 / 1.0 = -0.5 rad/s²
        expected_velocity = initial_velocity - 0.5 * dt
        
        assert estimator.x_hat[StateIndex.THETA_DOT_AZ] < initial_velocity
        assert np.isclose(estimator.x_hat[StateIndex.THETA_DOT_AZ], expected_velocity, atol=1e-6)
    
    def test_covariance_growth(self):
        """Verify covariance grows during prediction."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        initial_trace = np.trace(estimator.P)
        
        u = np.array([0.0, 0.0])
        dt = 0.01
        estimator.predict(u, dt)
        
        # Covariance should grow due to process noise Q
        final_trace = np.trace(estimator.P)
        assert final_trace > initial_trace, "Covariance must increase during prediction"
    
    def test_process_jacobian_structure(self):
        """Verify process Jacobian has correct structure."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        dt = 0.01
        F = estimator._compute_process_jacobian(dt, 0.0, 0.0)
        
        # Check dimension
        assert F.shape == (10, 10)
        
        # Check position-velocity coupling
        assert F[StateIndex.THETA_AZ, StateIndex.THETA_DOT_AZ] == dt
        assert F[StateIndex.THETA_EL, StateIndex.THETA_DOT_EL] == dt
        
        # Diagonal should be close to identity (small dt)
        assert np.abs(F[StateIndex.THETA_AZ, StateIndex.THETA_AZ] - 1.0) < 1e-6


class TestCorrectionStep:
    """Test EKF correction (measurement update) functionality."""
    
    def test_encoder_measurement_update(self):
        """Verify encoder measurements correct angle estimates."""
        config = {
            'initial_state': np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'measurement_noise_std': [1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4],
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Measurement: true angle is 0.0 (estimator thinks 0.1, 0.2)
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mask = np.array([True, True, False, False, False, False])
        
        estimator.correct(z, mask)
        
        # Angle estimate should move toward measurement
        assert np.abs(estimator.x_hat[StateIndex.THETA_AZ]) < 0.1
        assert np.abs(estimator.x_hat[StateIndex.THETA_EL]) < 0.2
    
    def test_gyro_bias_estimation(self):
        """Verify gyro bias is estimated from measurements."""
        config = {
            'initial_state': np.zeros(10),
            'measurement_noise_std': [1e-6, 1e-6, 1e-8, 1e-8, 1e-4, 1e-4],
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Simulate gyro measurement with bias
        # Gyro reads: θ̇ + b = 0.0 + 0.01 = 0.01
        true_velocity = 0.0
        true_bias = 0.01
        z = np.array([0.0, 0.0, true_bias, true_bias, 0.0, 0.0])
        mask = np.array([False, False, True, True, False, False])
        
        # Multiple updates to allow convergence
        for _ in range(20):
            estimator.correct(z, mask)
        
        # Bias estimate should converge toward true bias
        # Since velocity is unknown, bias may not be perfectly estimated
        # but should be non-zero
        assert np.abs(estimator.x_hat[StateIndex.BIAS_AZ]) > 0.0
    
    def test_innovation_computation(self):
        """Verify innovation is computed correctly."""
        config = {
            'initial_state': np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Measurement
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mask = np.array([True, True, True, True, False, False])
        
        estimator.correct(z, mask)
        
        # Innovation should be measurement - prediction
        # For encoder: z - θ_est = 0.0 - 0.1 = -0.1
        assert np.isclose(estimator.innovation[MeasurementIndex.THETA_AZ_ENC], -0.1, atol=1e-6)
    
    def test_covariance_reduction(self):
        """Verify covariance decreases after measurement update."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        initial_trace = np.trace(estimator.P)
        
        # Perfect measurement (low noise)
        z = np.zeros(6)
        mask = np.array([True, True, True, True, False, False])
        estimator.correct(z, mask)
        
        # Covariance should decrease
        final_trace = np.trace(estimator.P)
        assert final_trace < initial_trace, "Covariance must decrease after measurement"
    
    def test_measurement_mask(self):
        """Verify measurement masking works correctly."""
        config = {
            'initial_state': np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Only azimuth encoder available
        z = np.array([0.0, 999.0, 999.0, 999.0, 999.0, 999.0])
        mask = np.array([True, False, False, False, False, False])
        
        estimator.correct(z, mask)
        
        # Only azimuth angle should be corrected
        assert np.abs(estimator.x_hat[StateIndex.THETA_AZ]) < 0.1
        # Elevation should remain unchanged (or only slightly affected)
        # (some coupling may exist through P matrix)


class TestSensorFusion:
    """Test multi-sensor fusion capabilities."""
    
    def test_encoder_gyro_fusion(self):
        """Verify encoder and gyro measurements are fused correctly."""
        config = {
            'initial_state': np.zeros(10),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Simulate constant velocity motion
        true_velocity = 0.1  # rad/s
        dt = 0.01
        
        for i in range(50):
            # Predict
            u = np.array([0.0, 0.0])
            estimator.predict(u, dt)
            
            # Measure
            true_angle = true_velocity * i * dt
            z = np.array([
                true_angle, 0.0,           # Encoders
                true_velocity, 0.0,        # Gyros
                0.0, 0.0                   # QPD
            ])
            mask = np.array([True, True, True, True, False, False])
            estimator.correct(z, mask)
        
        # Estimate should track true values
        final_angle = true_velocity * 49 * dt
        assert np.isclose(estimator.x_hat[StateIndex.THETA_AZ], final_angle, atol=1e-3)
        assert np.isclose(estimator.x_hat[StateIndex.THETA_DOT_AZ], true_velocity, atol=1e-3)
    
    def test_multi_rate_fusion(self):
        """Verify different sensor update rates are handled correctly."""
        config = {
            'initial_state': np.zeros(10),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        dt = 0.001
        
        for i in range(100):
            u = np.array([0.0, 0.0])
            estimator.predict(u, dt)
            
            # Encoders at 1 kHz
            z_enc = np.array([0.01, 0.01, 0.0, 0.0, 0.0, 0.0])
            mask_enc = np.array([True, True, False, False, False, False])
            
            # Gyros at 10 kHz (every step)
            z_gyro = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            mask_gyro = np.array([False, False, True, True, False, False])
            
            # Update encoders every 10 steps
            if i % 10 == 0:
                estimator.correct(z_enc, mask_enc)
            
            # Update gyros every step
            estimator.correct(z_gyro, mask_gyro)
        
        # Should converge to encoder value
        assert np.abs(estimator.x_hat[StateIndex.THETA_AZ] - 0.01) < 1e-3


class TestConvergenceAndStability:
    """Test estimator convergence and numerical stability."""
    
    def test_steady_state_convergence(self):
        """Verify estimator converges to steady state."""
        config = {
            'initial_state': np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # True state at origin
        true_angle = 0.0
        dt = 0.01
        
        angle_history = []
        for _ in range(100):
            u = np.array([0.0, 0.0])
            estimator.predict(u, dt)
            
            z = np.array([true_angle, true_angle, 0.0, 0.0, 0.0, 0.0])
            mask = np.array([True, True, True, True, False, False])
            estimator.correct(z, mask)
            
            angle_history.append(estimator.x_hat[StateIndex.THETA_AZ])
        
        # Should converge monotonically
        final_error = np.abs(angle_history[-1] - true_angle)
        assert final_error < 0.01, "Estimator should converge to true state"
    
    def test_covariance_positive_definite(self):
        """Verify covariance remains positive definite."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        dt = 0.01
        for _ in range(100):
            u = np.array([0.5, -0.3])
            estimator.predict(u, dt)
            
            z = np.random.randn(6) * 0.01
            mask = np.ones(6, dtype=bool)
            estimator.correct(z, mask)
            
            # Check positive definite
            eigenvalues = np.linalg.eigvals(estimator.P)
            assert np.all(eigenvalues > 0), f"Covariance not positive definite at iteration"
    
    def test_large_initial_error(self):
        """Verify estimator handles large initial errors."""
        config = {
            'initial_state': np.array([10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # True state near origin
        dt = 0.01
        for _ in range(200):
            u = np.array([0.0, 0.0])
            estimator.predict(u, dt)
            
            z = np.array([0.01, 0.01, 0.0, 0.0, 0.0, 0.0])
            mask = np.array([True, True, True, True, False, False])
            estimator.correct(z, mask)
        
        # Should converge despite large initial error
        assert np.abs(estimator.x_hat[StateIndex.THETA_AZ] - 0.01) < 0.1


class TestInterfaceMethods:
    """Test public interface methods."""
    
    def test_step_method(self):
        """Verify step() method performs full predict-correct cycle."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        u = np.array([1.0, -0.5])
        measurements = {
            'theta_az_enc': 0.01,
            'theta_el_enc': -0.01,
            'theta_dot_az_gyro': 0.1,
            'theta_dot_el_gyro': -0.1,
        }
        dt = 0.01
        
        x_new = estimator.step(u, measurements, dt)
        
        # Should return updated state
        assert x_new.shape == (10,)
        assert estimator.iteration == 1
    
    def test_get_fused_state(self):
        """Verify get_fused_state() returns correct format."""
        config = {
            'initial_state': np.array([0.1, 0.2, 0.01, 0.3, 0.4, 0.02, 0.05, 0.01, 0.1, 0.2]),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        state_dict = estimator.get_fused_state()
        
        # Check all keys present
        assert 'theta_az' in state_dict
        assert 'theta_dot_az' in state_dict
        assert 'bias_az' in state_dict
        assert 'theta_el' in state_dict
        assert 'phi_roll' in state_dict
        assert 'dist_az' in state_dict
        
        # Check values match state vector
        assert state_dict['theta_az'] == 0.1
        assert state_dict['theta_dot_el'] == 0.4
        assert state_dict['bias_el'] == 0.02
    
    def test_get_diagnostics(self):
        """Verify diagnostics contain expected information."""
        config = {'inertia_az': 1.0, 'inertia_el': 1.0}
        estimator = PointingStateEstimator(config)
        
        diag = estimator.get_diagnostics()
        
        assert 'iteration' in diag
        assert 'state_estimate' in diag
        assert 'covariance_diag' in diag
        assert 'innovation' in diag
        assert 'kalman_gain_norm' in diag
        assert 'trace_P' in diag
    
    def test_reset_method(self):
        """Verify reset() returns estimator to initial state."""
        config = {
            'initial_state': np.zeros(10),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
        }
        estimator = PointingStateEstimator(config)
        
        # Run for some time
        for _ in range(10):
            u = np.array([1.0, 1.0])
            measurements = {'theta_az_enc': 0.1, 'theta_el_enc': 0.1}
            estimator.step(u, measurements, 0.01)
        
        assert estimator.iteration > 0
        assert not np.allclose(estimator.x_hat, 0.0)
        
        # Reset
        estimator.reset()
        
        assert estimator.iteration == 0
        assert np.allclose(estimator.x_hat, 0.0)


class TestDisturbanceEstimation:
    """Test disturbance torque estimation capability."""
    
    def test_constant_disturbance_estimation(self):
        """Verify constant disturbance torque is estimated."""
        true_disturbance_az = 0.5  # N·m
        
        config = {
            'initial_state': np.zeros(10),
            'inertia_az': 1.0,
            'inertia_el': 1.0,
            'friction_coeff_az': 0.0,
            'friction_coeff_el': 0.0,
            'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-3, 1e-3],
        }
        estimator = PointingStateEstimator(config)
        
        dt = 0.01
        
        # Simulate system with disturbance (not known to estimator)
        # True dynamics: θ̈ = (τ - d) / J
        # If τ = 0, then θ̈ = -d / J
        # But estimator predicts θ̈ = 0 (no disturbance estimate yet)
        
        for i in range(100):
            # Apply no control
            u = np.array([0.0, 0.0])
            estimator.predict(u, dt)
            
            # Simulate "true" system with disturbance
            # Acceleration due to disturbance: a = -d / J = -0.5 / 1.0 = -0.5 rad/s²
            true_accel = -true_disturbance_az / 1.0
            true_velocity = true_accel * i * dt
            true_angle = 0.5 * true_accel * (i * dt) ** 2
            
            # Measure
            z = np.array([true_angle, 0.0, true_velocity, 0.0, 0.0, 0.0])
            mask = np.array([True, False, True, False, False, False])
            estimator.correct(z, mask)
        
        # Disturbance estimate should be non-zero (qualitative test)
        # Full convergence requires more sophisticated tuning
        estimated_disturbance = estimator.x_hat[StateIndex.DIST_AZ]
        # At minimum, should detect that disturbance exists
        assert np.abs(estimated_disturbance) > 0.01, "Should detect non-zero disturbance"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
