"""
Unit tests for actuator models.

This module contains pytest-based unit tests for the gimbal motor and FSM actuator
models, verifying critical non-ideal behaviors including torque saturation and hysteresis.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.actuators.motor_models import GimbalMotorModel
from core.actuators.fsm_actuator import FSMActuatorModel


class TestGimbalMotorModel:
    """Test suite for GimbalMotorModel."""
    
    @pytest.fixture
    def motor_config(self):
        """Standard motor configuration for testing."""
        return {
            'R': 2.0,  # Ohm
            'L': 0.002,  # H
            'K_t': 0.15,  # N·m/A
            'K_e': 0.15,  # V·s/rad
            'tau_max': 5.0,  # N·m
            'tau_min': -5.0,  # N·m
            'cogging_amplitude': 0.1,  # N·m
            'cogging_frequency': 12
        }
    
    def test_initialization(self, motor_config):
        """Test that motor initializes with correct parameters."""
        motor = GimbalMotorModel(motor_config)
        
        assert motor.R == 2.0
        assert motor.L == 0.002
        assert motor.K_t == 0.15
        assert motor.tau_max == 5.0
        assert motor.current == 0.0
    
    def test_torque_saturation_positive(self, motor_config):
        """
        Test that motor correctly saturates torque at positive limit.
        
        This test applies a very high voltage command that would theoretically
        produce torque exceeding tau_max, and verifies the output is clamped.
        """
        motor = GimbalMotorModel(motor_config)
        
        # Apply very high voltage to drive current high
        v_cmd = 100.0  # V - intentionally excessive
        omega_m = 0.0  # rad/s - stationary
        theta_m = 0.0  # rad
        dt = 0.001  # s
        
        # Run for multiple steps to let current build up
        for _ in range(100):
            tau_output = motor.step(v_cmd, omega_m, theta_m, dt)
        
        # Verify saturation: output should not exceed tau_max
        # Even with cogging torque, saturation should enforce limit
        assert tau_output <= motor_config['tau_max'], \
            f"Positive torque saturation failed: {tau_output} > {motor_config['tau_max']}"
    
    def test_torque_saturation_negative(self, motor_config):
        """
        Test that motor correctly saturates torque at negative limit.
        """
        motor = GimbalMotorModel(motor_config)
        
        # Apply very high negative voltage
        v_cmd = -100.0  # V
        omega_m = 0.0  # rad/s
        theta_m = 0.0  # rad
        dt = 0.001  # s
        
        # Run for multiple steps to let current build up
        for _ in range(100):
            tau_output = motor.step(v_cmd, omega_m, theta_m, dt)
        
        # Verify saturation at negative limit
        assert tau_output >= motor_config['tau_min'], \
            f"Negative torque saturation failed: {tau_output} < {motor_config['tau_min']}"
    
    def test_cogging_torque_periodicity(self, motor_config):
        """
        Test that cogging torque exhibits expected periodic behavior.
        """
        motor = GimbalMotorModel(motor_config)
        
        # Sample cogging torque over one mechanical revolution
        theta_samples = np.linspace(0, 2*np.pi, 100)
        cogging_values = [motor._compute_cogging_torque(theta) for theta in theta_samples]
        
        # Cogging should repeat with specified frequency
        # Check that pattern repeats (value at 0 ≈ value at 2π)
        assert np.abs(cogging_values[0] - cogging_values[-1]) < 0.01, \
            "Cogging torque is not periodic"
        
        # Verify amplitude is within expected range
        max_cogging = np.max(np.abs(cogging_values))
        expected_max = motor_config['cogging_amplitude'] * 1.3  # Including harmonics
        assert max_cogging <= expected_max, \
            f"Cogging amplitude exceeds expected: {max_cogging} > {expected_max}"
    
    def test_electrical_dynamics_response(self, motor_config):
        """
        Test that electrical dynamics respond correctly to step input.
        """
        motor = GimbalMotorModel(motor_config)
        
        # Apply step voltage input
        v_cmd = 10.0  # V
        omega_m = 0.0  # rad/s
        theta_m = 0.0  # rad
        dt = 0.0001  # s - small timestep for accuracy
        
        # Current should increase exponentially with time constant L/R
        tau_elec = motor_config['L'] / motor_config['R']  # Expected time constant
        
        time = []
        current = []
        for i in range(500):
            motor.step(v_cmd, omega_m, theta_m, dt)
            time.append(i * dt)
            current.append(motor.current)
        
        # At t ≈ 5*tau, current should be near steady-state
        steady_state_index = int(5 * tau_elec / dt)
        if steady_state_index < len(current):
            expected_steady_state = v_cmd / motor_config['R']
            actual_steady_state = current[steady_state_index]
            
            # Should be within 1% of expected
            assert np.abs(actual_steady_state - expected_steady_state) / expected_steady_state < 0.01, \
                f"Steady-state current incorrect: {actual_steady_state} vs {expected_steady_state}"
    
    def test_reset(self, motor_config):
        """Test that reset properly clears motor state."""
        motor = GimbalMotorModel(motor_config)
        
        # Run motor to build up state
        for _ in range(10):
            motor.step(10.0, 0.0, 0.0, 0.001)
        
        assert motor.current != 0.0, "Motor current should be non-zero before reset"
        
        # Reset
        motor.reset()
        
        assert motor.current == 0.0, "Motor current should be zero after reset"


class TestFSMActuatorModel:
    """Test suite for FSMActuatorModel."""
    
    @pytest.fixture
    def fsm_config(self):
        """Standard FSM configuration for testing."""
        return {
            'omega_n': 2000.0,  # rad/s
            'zeta': 0.7,  # -
            'alpha_max': np.deg2rad(1.0),  # rad
            'alpha_min': np.deg2rad(-1.0),  # rad
            'beta_max': np.deg2rad(1.0),  # rad
            'beta_min': np.deg2rad(-1.0),  # rad
            'rate_limit': np.deg2rad(500.0),  # rad/s
            'hysteresis_width': np.deg2rad(0.002),  # rad
            'hysteresis_gain': 0.2
        }
    
    def test_initialization(self, fsm_config):
        """Test that FSM initializes with correct parameters."""
        fsm = FSMActuatorModel(fsm_config)
        
        assert fsm.omega_n == 2000.0
        assert fsm.zeta == 0.7
        assert fsm.alpha == 0.0
        assert fsm.beta == 0.0
    
    def test_hysteresis_observable(self, fsm_config):
        """
        Test that hysteresis is observable in the FSM response.
        
        This test applies a triangular wave command and verifies that the
        output position exhibits hysteresis (i.e., different paths for
        increasing vs. decreasing commands).
        """
        fsm = FSMActuatorModel(fsm_config)
        
        # Generate triangular wave command
        dt = 0.0001  # s - fine timestep
        t_period = 0.01  # s - period of triangular wave
        n_steps = int(t_period / dt)
        
        # Create triangular wave (0 -> max -> 0 -> min -> 0)
        amplitude = np.deg2rad(0.5)
        cmd_sequence = []
        
        # Ramp up
        cmd_sequence.extend(np.linspace(0, amplitude, n_steps // 4))
        # Ramp down to negative
        cmd_sequence.extend(np.linspace(amplitude, -amplitude, n_steps // 2))
        # Ramp back to zero
        cmd_sequence.extend(np.linspace(-amplitude, 0, n_steps // 4))
        
        # Track positions during upward and downward sweeps
        positions_up = []
        positions_down = []
        commands = []
        
        for i, cmd in enumerate(cmd_sequence):
            alpha, _ = fsm.step(cmd, 0.0, dt)
            commands.append(cmd)
            
            # Record positions during specific phases
            if i < n_steps // 4:  # Upward sweep
                positions_up.append((cmd, alpha))
            elif i > n_steps // 4 and i < 3 * n_steps // 4:  # Downward sweep
                positions_down.append((cmd, alpha))
        
        # Check for hysteresis: at same command level, position should differ
        # Find a common command value in both sweeps
        if len(positions_up) > 10 and len(positions_down) > 10:
            # Sample in middle of range
            cmd_test = amplitude / 2
            
            # Find closest positions in each sweep
            pos_up = [p[1] for p in positions_up if np.abs(p[0] - cmd_test) < amplitude * 0.1]
            pos_down = [p[1] for p in positions_down if np.abs(p[0] - cmd_test) < amplitude * 0.1]
            
            if len(pos_up) > 0 and len(pos_down) > 0:
                avg_pos_up = np.mean(pos_up)
                avg_pos_down = np.mean(pos_down)
                
                # Positions should differ due to hysteresis
                hysteresis_error = np.abs(avg_pos_up - avg_pos_down)
                
                # Should be measurable (at least 10% of hysteresis width)
                min_observable = 0.1 * fsm_config['hysteresis_width']
                assert hysteresis_error > min_observable, \
                    f"Hysteresis not observable: error={hysteresis_error:.2e}, min={min_observable:.2e}"
    
    def test_rate_limiting(self, fsm_config):
        """
        Test that FSM correctly limits angular rate.
        """
        fsm = FSMActuatorModel(fsm_config)
        
        # Apply large step command
        cmd_initial = 0.0
        cmd_final = np.deg2rad(0.5)
        dt = 0.00001  # s - very fine timestep
        
        # Run for a few steps and measure maximum velocity
        max_velocity = 0.0
        for _ in range(1000):
            alpha, _ = fsm.step(cmd_final, 0.0, dt)
            if np.abs(fsm.alpha_dot) > max_velocity:
                max_velocity = np.abs(fsm.alpha_dot)
        
        # Maximum velocity should not exceed rate limit
        assert max_velocity <= fsm_config['rate_limit'] * 1.01, \
            f"Rate limit violated: {max_velocity} > {fsm_config['rate_limit']}"
    
    def test_position_saturation(self, fsm_config):
        """
        Test that FSM position saturates at configured limits.
        """
        fsm = FSMActuatorModel(fsm_config)
        
        # Command beyond positive limit
        cmd_excessive = np.deg2rad(5.0)  # Way beyond limit
        dt = 0.0001
        
        # Run until settled
        for _ in range(10000):
            alpha, _ = fsm.step(cmd_excessive, 0.0, dt)
        
        # Should saturate at max
        assert alpha <= fsm_config['alpha_max'], \
            f"Position limit violated: {alpha} > {fsm_config['alpha_max']}"
    
    def test_second_order_dynamics(self, fsm_config):
        """
        Test that FSM exhibits proper second-order step response characteristics.
        """
        fsm = FSMActuatorModel(fsm_config)
        
        # Apply step command
        cmd = np.deg2rad(0.1)
        dt = 0.00001  # s
        
        positions = []
        times = []
        
        # Simulate for sufficient time
        n_steps = 5000
        for i in range(n_steps):
            alpha, _ = fsm.step(cmd, 0.0, dt)
            positions.append(alpha)
            times.append(i * dt)
        
        # For underdamped system (zeta < 1), should see overshoot
        max_position = np.max(positions)
        final_position = positions[-1]
        
        # Check that we reach steady-state
        assert np.abs(final_position - cmd) < cmd * 0.05, \
            "FSM did not reach commanded position"
        
        # For zeta = 0.7, expect some overshoot
        if fsm_config['zeta'] < 1.0:
            overshoot = (max_position - cmd) / cmd
            assert overshoot > 0, "Expected overshoot for underdamped system"
    
    def test_reset(self, fsm_config):
        """Test that reset properly clears FSM state."""
        fsm = FSMActuatorModel(fsm_config)
        
        # Run FSM to build up state
        for _ in range(100):
            fsm.step(np.deg2rad(0.5), np.deg2rad(0.3), 0.001)
        
        assert fsm.alpha != 0.0, "Alpha should be non-zero before reset"
        assert fsm.beta != 0.0, "Beta should be non-zero before reset"
        
        # Reset
        fsm.reset()
        
        assert fsm.alpha == 0.0, "Alpha should be zero after reset"
        assert fsm.beta == 0.0, "Beta should be zero after reset"
        assert fsm.alpha_dot == 0.0, "Alpha velocity should be zero after reset"
        assert fsm.beta_dot == 0.0, "Beta velocity should be zero after reset"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
