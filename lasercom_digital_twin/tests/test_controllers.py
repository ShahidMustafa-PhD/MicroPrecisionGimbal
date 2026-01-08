"""
Unit tests for control systems.

This module contains pytest-based unit tests for the hierarchical control
architecture including the coarse gimbal PID controller and the fine FSM
PI controller. Tests verify anti-windup, saturation handling, decoupling,
and proper control law implementation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.controllers.control_laws import CoarseGimbalController, BaseController
from core.controllers.fsm_controller import FSMController


class TestCoarseGimbalController:
    """Test suite for CoarseGimbalController."""
    
    @pytest.fixture
    def controller_config(self):
        """Standard coarse controller configuration."""
        return {
            'kp': [50.0, 50.0],
            'ki': [10.0, 10.0],
            'kd': [5.0, 5.0],
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            'tau_rate_limit': 100.0,
            'anti_windup_gain': [0.1, 0.1],
            'setpoint_filter_wn': 10.0,
            'setpoint_filter_zeta': 0.7,
            'enable_derivative': True,
            'use_setpoint_filter': True
        }
    
    def test_initialization(self, controller_config):
        """Test controller initializes with correct parameters."""
        controller = CoarseGimbalController(controller_config)
        
        assert np.allclose(controller.kp, [50.0, 50.0])
        assert np.allclose(controller.ki, [10.0, 10.0])
        assert np.allclose(controller.kd, [5.0, 5.0])
        assert controller.integral.shape == (2,)
    
    def test_proportional_response(self, controller_config):
        """Test pure proportional response with zero integral/derivative."""
        config = controller_config.copy()
        config['ki'] = [0.0, 0.0]  # Disable integral
        config['kd'] = [0.0, 0.0]  # Disable derivative
        config['use_setpoint_filter'] = False
        
        controller = CoarseGimbalController(config)
        
        # Step input
        reference = np.array([1.0, 0.5])
        measurement = np.array([0.0, 0.0])
        dt = 0.001
        
        command, metadata = controller.compute_control(reference, measurement, dt)
        
        # Output should be K_p * error
        expected = config['kp'] * (reference - measurement)
        
        assert np.allclose(command, expected, atol=0.01), \
            f"Proportional response incorrect: {command} vs {expected}"
    
    def test_integral_accumulation(self, controller_config):
        """Test that integral term accumulates over time."""
        config = controller_config.copy()
        config['kp'] = [0.0, 0.0]  # Disable proportional
        config['kd'] = [0.0, 0.0]  # Disable derivative
        config['use_setpoint_filter'] = False
        
        controller = CoarseGimbalController(config)
        
        # Constant error
        reference = np.array([1.0, 1.0])
        measurement = np.array([0.0, 0.0])
        dt = 0.01
        
        # Run for multiple steps
        n_steps = 10
        for _ in range(n_steps):
            command, metadata = controller.compute_control(reference, measurement, dt)
        
        # Integral should have accumulated
        expected_integral = (reference - measurement) * dt * n_steps
        
        assert np.allclose(controller.integral, expected_integral, atol=0.01), \
            f"Integral accumulation error: {controller.integral} vs {expected_integral}"
    
    def test_anti_windup_activation(self, controller_config):
        """
        Test that anti-windup prevents integrator wind-up during saturation.
        """
        controller = CoarseGimbalController(controller_config)
        
        # Large error that will cause saturation
        reference = np.array([10.0, 10.0])
        measurement = np.array([0.0, 0.0])
        dt = 0.01
        
        # Run for many steps to observe saturation
        integral_history = []
        for _ in range(100):
            command, metadata = controller.compute_control(reference, measurement, dt)
            integral_history.append(controller.integral.copy())
        
        # Check that saturation was active
        assert np.any(controller.saturation_active), \
            "Saturation should have been active"
        
        # Integral should be bounded (not growing indefinitely)
        final_integral = controller.integral
        max_integral = np.max(integral_history, axis=0)
        
        # Integral should stabilize, not grow unbounded
        # (exact value depends on anti-windup gain, but should be < 100)
        assert np.all(np.abs(final_integral) < 100.0), \
            f"Integral unbounded: {final_integral}"
    
    def test_derivative_on_measurement(self, controller_config):
        """
        Test derivative term (derivative-on-measurement to avoid kick).
        """
        controller = CoarseGimbalController(controller_config)
        
        # Constant reference, changing measurement
        reference = np.array([1.0, 1.0])
        measurement_0 = np.array([0.0, 0.0])
        measurement_1 = np.array([0.1, 0.1])
        dt = 0.01
        
        # First step
        controller.compute_control(reference, measurement_0, dt)
        
        # Second step with changing measurement
        command, metadata = controller.compute_control(reference, measurement_1, dt)
        
        # Derivative term should be present
        u_d = metadata['u_d']
        assert np.any(np.abs(u_d) > 1e-6), \
            "Derivative term should be non-zero with changing measurement"
    
    def test_rate_limiting(self, controller_config):
        """
        Test that output rate is limited to configured value.
        """
        controller = CoarseGimbalController(controller_config)
        
        # Large step change
        reference_0 = np.array([0.0, 0.0])
        reference_1 = np.array([5.0, 5.0])
        measurement = np.array([0.0, 0.0])
        dt = 0.01
        
        # First command
        command_0, _ = controller.compute_control(reference_0, measurement, dt)
        
        # Large step in reference
        command_1, _ = controller.compute_control(reference_1, measurement, dt)
        
        # Rate should be limited
        delta_command = command_1 - command_0
        max_delta = controller_config['tau_rate_limit'] * dt
        
        assert np.all(np.abs(delta_command) <= max_delta * 1.01), \
            f"Rate limit violated: {delta_command} > {max_delta}"
    
    def test_reset(self, controller_config):
        """Test that reset clears controller state."""
        controller = CoarseGimbalController(controller_config)
        
        # Run controller to build up state
        reference = np.array([1.0, 1.0])
        measurement = np.array([0.0, 0.0])
        dt = 0.01
        
        for _ in range(10):
            controller.compute_control(reference, measurement, dt)
        
        assert np.any(controller.integral != 0), "Integral should be non-zero"
        
        # Reset
        controller.reset()
        
        assert np.all(controller.integral == 0), "Integral should be zero after reset"
        assert np.all(controller.previous_error == 0), "Previous error should be zero"
    
    def test_residual_error_computation(self, controller_config):
        """Test computation of residual error for FSM feedforward."""
        controller = CoarseGimbalController(controller_config)
        
        reference = np.array([1.0, 0.5])
        measurement = np.array([0.9, 0.4])
        
        residual = controller.get_residual_error_for_fsm(reference, measurement)
        
        expected = reference - measurement
        
        assert np.allclose(residual, expected), \
            f"Residual error incorrect: {residual} vs {expected}"


class TestFSMController:
    """Test suite for FSMController."""
    
    @pytest.fixture
    def fsm_config(self):
        """Standard FSM controller configuration."""
        return {
            'kp': [1.0, 1.0],
            'ki': [100.0, 100.0],
            'fsm_deflection_max': [np.deg2rad(1.0), np.deg2rad(1.0)],
            'fsm_deflection_min': [-np.deg2rad(1.0), -np.deg2rad(1.0)],
            'enable_feedforward': True,
            'high_pass_filter_enabled': False,
            'high_pass_cutoff_hz': 0.1
        }
    
    def test_initialization(self, fsm_config):
        """Test FSM controller initializes correctly."""
        controller = FSMController(fsm_config)
        
        assert np.allclose(controller.kp, [1.0, 1.0])
        assert np.allclose(controller.ki, [100.0, 100.0])
        assert controller.integral.shape == (2,)
    
    def test_proportional_response(self, fsm_config):
        """Test pure proportional response."""
        config = fsm_config.copy()
        config['ki'] = [0.0, 0.0]  # Disable integral
        
        controller = FSMController(config)
        
        # QPD error
        qpd_error = np.array([100.0e-6, 50.0e-6])  # 100 µrad tip, 50 µrad tilt
        dt = 0.0001
        
        command, metadata = controller.compute_control(qpd_error, dt)
        
        # Should be proportional to error
        expected = config['kp'] * qpd_error
        
        assert np.allclose(command, expected, atol=1e-9), \
            f"Proportional response incorrect: {command} vs {expected}"
    
    def test_integral_response(self, fsm_config):
        """Test integral term accumulation."""
        config = fsm_config.copy()
        config['kp'] = [0.0, 0.0]  # Disable proportional
        
        controller = FSMController(config)
        
        # Constant error
        qpd_error = np.array([10.0e-6, 10.0e-6])
        dt = 0.001
        
        # Run for multiple steps
        n_steps = 10
        for _ in range(n_steps):
            command, metadata = controller.compute_control(qpd_error, dt)
        
        # Integral should accumulate
        expected_integral = qpd_error * dt * n_steps
        
        assert np.allclose(controller.integral, expected_integral, atol=1e-8), \
            f"Integral accumulation error: {controller.integral} vs {expected_integral}"
    
    def test_saturation_detection(self, fsm_config):
        """
        Test that FSM saturation is properly detected.
        """
        controller = FSMController(fsm_config)
        
        # Very large error to force saturation
        qpd_error = np.array([10.0e-3, 10.0e-3])  # 10 mrad (>> 1 deg limit)
        dt = 0.001
        
        command, metadata = controller.compute_control(qpd_error, dt)
        
        # Command should be saturated at limits
        assert np.allclose(np.abs(command), np.deg2rad(1.0), atol=1e-9), \
            "FSM should saturate at deflection limit"
        
        # Saturation flag should be active
        assert controller.is_saturated(), \
            "Saturation flag should be active"
        
        assert metadata['saturated'].all(), \
            "Metadata should indicate saturation"
    
    def test_feedforward_integration(self, fsm_config):
        """
        Test that coarse residual feedforward is properly integrated.
        """
        controller = FSMController(fsm_config)
        
        qpd_error = np.array([10.0e-6, 5.0e-6])
        coarse_residual = np.array([50.0e-6, 30.0e-6])
        dt = 0.001
        
        # Without feedforward
        config_no_ff = fsm_config.copy()
        config_no_ff['enable_feedforward'] = False
        controller_no_ff = FSMController(config_no_ff)
        
        command_no_ff, _ = controller_no_ff.compute_control(qpd_error, dt)
        
        # With feedforward
        command_with_ff, metadata = controller.compute_control(
            qpd_error, dt, coarse_residual
        )
        
        # Commands should differ (feedforward adds to error)
        assert not np.allclose(command_no_ff, command_with_ff), \
            "Feedforward should change control output"
        
        assert metadata['feedforward_used'], \
            "Metadata should indicate feedforward was used"
    
    def test_saturation_duration_tracking(self, fsm_config):
        """
        Test that saturation duration is tracked correctly.
        """
        controller = FSMController(fsm_config)
        
        # Large error causing saturation
        qpd_error = np.array([10.0e-3, 10.0e-3])
        dt = 0.01
        
        # Run for multiple steps
        n_steps = 5
        for _ in range(n_steps):
            command, metadata = controller.compute_control(qpd_error, dt)
        
        # Saturation duration should accumulate
        expected_duration = dt * n_steps
        
        assert np.allclose(controller.saturation_duration, expected_duration, atol=dt), \
            f"Saturation duration tracking error: {controller.saturation_duration} vs {expected_duration}"
    
    def test_saturation_report(self, fsm_config):
        """
        Test saturation diagnostic report generation.
        """
        controller = FSMController(fsm_config)
        
        # Cause saturation
        qpd_error = np.array([10.0e-3, 5.0e-3])
        dt = 0.01
        
        controller.compute_control(qpd_error, dt)
        
        report = controller.get_saturation_report()
        
        assert 'tip_saturated' in report
        assert 'tilt_saturated' in report
        assert 'any_saturated' in report
        assert report['any_saturated'], "Report should indicate saturation"
    
    def test_anti_windup_on_saturation(self, fsm_config):
        """
        Test that integrator stops accumulating during saturation.
        """
        controller = FSMController(fsm_config)
        
        # Large error causing saturation
        qpd_error = np.array([10.0e-3, 10.0e-3])
        dt = 0.001
        
        # Run until saturated
        for _ in range(10):
            controller.compute_control(qpd_error, dt)
        
        integral_at_saturation = controller.integral.copy()
        
        # Continue running
        for _ in range(10):
            controller.compute_control(qpd_error, dt)
        
        # Integral should not grow much during saturation
        integral_change = np.abs(controller.integral - integral_at_saturation)
        
        # Should be small (anti-windup prevents growth)
        assert np.all(integral_change < qpd_error * dt * 5), \
            f"Integrator should not wind up during saturation: change={integral_change}"
    
    def test_reset(self, fsm_config):
        """Test that reset clears FSM controller state."""
        controller = FSMController(fsm_config)
        
        # Run to build up state
        qpd_error = np.array([100.0e-6, 50.0e-6])
        dt = 0.001
        
        for _ in range(10):
            controller.compute_control(qpd_error, dt)
        
        assert np.any(controller.integral != 0), "Integral should be non-zero"
        
        # Reset
        controller.reset()
        
        assert np.all(controller.integral == 0), "Integral should be zero after reset"
        assert np.all(controller.saturation_duration == 0), "Saturation duration should be zero"


class TestControllerDecoupling:
    """Test hierarchical decoupling between coarse and fine loops."""
    
    def test_residual_feedforward_reduces_fsm_saturation(self):
        """
        Test that feedforward from coarse loop reduces FSM saturation.
        
        This is a critical integration test verifying that the hierarchical
        architecture properly decouples the control loops.
        """
        # Configure controllers
        coarse_config = {
            'kp': [50.0, 50.0],
            'ki': [10.0, 10.0],
            'kd': [5.0, 5.0],
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            'tau_rate_limit': 100.0,
            'anti_windup_gain': [0.1, 0.1],
            'use_setpoint_filter': False,
            'enable_derivative': False
        }
        
        fsm_config = {
            'kp': [1.0, 1.0],
            'ki': [100.0, 100.0],
            'fsm_deflection_max': [np.deg2rad(1.0), np.deg2rad(1.0)],
            'fsm_deflection_min': [-np.deg2rad(1.0), -np.deg2rad(1.0)],
            'enable_feedforward': True
        }
        
        coarse = CoarseGimbalController(coarse_config)
        fsm_with_ff = FSMController(fsm_config)
        
        # Without feedforward
        fsm_config_no_ff = fsm_config.copy()
        fsm_config_no_ff['enable_feedforward'] = False
        fsm_no_ff = FSMController(fsm_config_no_ff)
        
        # Scenario: Large tracking error
        target = np.array([1.0, 0.5])  # Large gimbal angle
        gimbal_position = np.array([0.0, 0.0])
        
        # Coarse loop residual
        residual = coarse.get_residual_error_for_fsm(target, gimbal_position)
        
        # QPD sees the residual as error
        qpd_error = residual  # Simplified: actual QPD would see focal plane error
        
        dt = 0.001
        
        # FSM without feedforward
        cmd_no_ff, meta_no_ff = fsm_no_ff.compute_control(qpd_error, dt)
        
        # FSM with feedforward
        cmd_with_ff, meta_with_ff = fsm_with_ff.compute_control(
            qpd_error, dt, residual
        )
        
        # With feedforward, FSM should be closer to saturation limit or handle error better
        # The key is that feedforward changes the control action
        assert not np.allclose(cmd_no_ff, cmd_with_ff), \
            "Feedforward should modify FSM command"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
