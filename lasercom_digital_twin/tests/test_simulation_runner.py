"""
Unit Tests for Integrated Simulation Runner

This module tests the complete digital twin simulation framework including:
- Multi-rate execution timing
- Component integration
- Data logging infrastructure
- Closed-loop stability
- Performance metrics computation
"""

import pytest
import numpy as np
from core.simulation.simulation_runner import (
    DigitalTwinRunner,
    SimulationConfig,
    SimulationState
)


class TestSimulationInitialization:
    """Test simulation runner initialization."""
    
    def test_default_initialization(self):
        """Verify default configuration initializes correctly."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        # Check timing initialized
        assert runner.time == 0.0
        assert runner.iteration == 0
        
        # Check all components exist
        assert runner.motor_az is not None
        assert runner.motor_el is not None
        assert runner.fsm is not None
        assert runner.encoder_az is not None
        assert runner.encoder_el is not None
        assert runner.gyro_az is not None
        assert runner.gyro_el is not None
        assert runner.qpd is not None
        assert runner.estimator is not None
        assert runner.coarse_controller is not None
        assert runner.fsm_controller is not None
        
    def test_custom_timing_configuration(self):
        """Verify custom timing rates are applied."""
        config = SimulationConfig(
            dt_sim=0.0005,      # 0.5 ms
            dt_coarse=0.020,    # 20 ms
            dt_fine=0.002,      # 2 ms
            seed=42
        )
        runner = DigitalTwinRunner(config)
        
        assert runner.config.dt_sim == 0.0005
        assert runner.config.dt_coarse == 0.020
        assert runner.config.dt_fine == 0.002
    
    def test_deterministic_seeding(self):
        """Verify same seed produces identical initialization."""
        config1 = SimulationConfig(seed=12345)
        config2 = SimulationConfig(seed=12345)
        
        runner1 = DigitalTwinRunner(config1)
        runner2 = DigitalTwinRunner(config2)
        
        # Both should start at same state
        assert runner1.q_az == runner2.q_az
        assert runner1.q_el == runner2.q_el


class TestMultiRateExecution:
    """Test multi-rate loop timing and execution."""
    
    def test_coarse_update_rate(self):
        """Verify coarse controller updates at correct rate."""
        config = SimulationConfig(
            dt_sim=0.001,       # 1 ms
            dt_coarse=0.010,    # 10 ms
            seed=42
        )
        runner = DigitalTwinRunner(config)
        
        # Run for 100 ms
        runner.run_simulation(duration=0.1)
        
        # Check coarse updates occurred
        # Should have ~10 updates (100 ms / 10 ms)
        # Estimator iteration counts updates
        assert runner.estimator.iteration >= 8
        assert runner.estimator.iteration <= 12
    
    def test_fine_update_rate(self):
        """Verify fine controller updates at correct rate."""
        config = SimulationConfig(
            dt_sim=0.001,
            dt_fine=0.005,      # 5 ms
            seed=42
        )
        runner = DigitalTwinRunner(config)
        
        # Run for 50 ms
        runner.run_simulation(duration=0.05)
        
        # Fine controller should update ~10 times (50 ms / 5 ms)
        # Check FSM has been updated
        assert runner.state.fsm_tip != 0.0 or runner.state.fsm_cmd_tip != 0.0
    
    def test_sensor_sampling_rate(self):
        """Verify sensors sample at configured rates."""
        config = SimulationConfig(
            dt_sim=0.001,
            dt_encoder=0.002,   # 2 ms
            seed=42
        )
        runner = DigitalTwinRunner(config)
        
        # Manually step and check sensor updates
        for i in range(10):
            runner.time = i * config.dt_sim
            runner._step_dynamics(config.dt_sim)
            runner._sample_sensors()
        
        # Encoder should have updated at least once
        assert runner.state.z_enc_az != 0.0 or runner.state.z_enc_el != 0.0
    
    def test_multi_rate_synchronization(self):
        """Verify different rates execute in proper order."""
        config = SimulationConfig(
            dt_sim=0.001,
            dt_coarse=0.010,
            dt_fine=0.002,
            seed=42
        )
        runner = DigitalTwinRunner(config)
        
        # Run short simulation
        results = runner.run_simulation(duration=0.05)
        
        # All rates should have executed
        assert len(results['log_data']['time']) > 0
        assert results['n_samples'] > 0


class TestClosedLoopBehavior:
    """Test closed-loop tracking and stability."""
    
    def test_target_tracking_convergence(self):
        """Verify system tracks target setpoint."""
        config = SimulationConfig(
            seed=42,
            target_az=np.deg2rad(10.0),
            target_el=np.deg2rad(45.0),
            target_enabled=True
        )
        runner = DigitalTwinRunner(config)
        
        # Run for 2 seconds
        results = runner.run_simulation(duration=2.0)
        
        # Should converge toward target
        final_az = results['final_az']
        final_el = results['final_el']
        target_az = config.target_az
        target_el = config.target_el
        
        # Check convergence (within 1 degree)
        assert np.abs(final_az - target_az) < np.deg2rad(1.0)
        assert np.abs(final_el - target_el) < np.deg2rad(1.0)
    
    def test_stability_no_divergence(self):
        """Verify system remains stable (no runaway)."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=1.0)
        
        # Check states remain bounded
        log_arrays = results['log_arrays']
        assert np.all(np.abs(log_arrays['q_az']) < 10.0)  # Within ±10 rad
        assert np.all(np.abs(log_arrays['q_el']) < 10.0)
        assert np.all(np.abs(log_arrays['torque_az']) < 50.0)  # Reasonable torque
        assert np.all(np.abs(log_arrays['torque_el']) < 50.0)
    
    def test_fsm_operates_within_limits(self):
        """Verify FSM stays within mechanical limits."""
        config = SimulationConfig(
            seed=42,
            target_az=np.deg2rad(5.0),
            target_el=np.deg2rad(30.0)
        )
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=1.0)
        
        # FSM should stay within ±0.01 rad
        log_arrays = results['log_arrays']
        fsm_max = runner.fsm.config['max_angle']
        assert np.all(np.abs(log_arrays['fsm_tip']) <= fsm_max * 1.1)  # Allow 10% margin
        assert np.all(np.abs(log_arrays['fsm_tilt']) <= fsm_max * 1.1)


class TestDataLogging:
    """Test data logging infrastructure."""
    
    def test_all_signals_logged(self):
        """Verify all required signals are logged."""
        config = SimulationConfig(seed=42, log_period=0.001)
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=0.1)
        log_data = results['log_data']
        
        # Check all mandatory signals present
        required_signals = [
            'time', 'q_az', 'q_el', 'torque_az', 'torque_el',
            'fsm_tip', 'fsm_tilt', 'fsm_saturated',
            'z_qpd_nes_x', 'z_qpd_nes_y', 'los_error_x', 'los_error_y'
        ]
        
        for signal in required_signals:
            assert signal in log_data, f"Signal '{signal}' not logged"
            assert len(log_data[signal]) > 0, f"Signal '{signal}' has no data"
    
    def test_log_period_respected(self):
        """Verify data logs at specified period."""
        log_period = 0.005  # 5 ms
        config = SimulationConfig(seed=42, log_period=log_period)
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=0.1)
        
        # Check approximate number of samples
        expected_samples = int(0.1 / log_period)
        actual_samples = results['n_samples']
        
        # Allow ±10% tolerance
        assert actual_samples >= expected_samples * 0.9
        assert actual_samples <= expected_samples * 1.1
    
    def test_time_vector_monotonic(self):
        """Verify logged time is strictly increasing."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=0.1)
        time_vector = results['log_arrays']['time']
        
        # Check monotonically increasing
        assert np.all(np.diff(time_vector) > 0)


class TestPerformanceMetrics:
    """Test performance metric computation."""
    
    def test_rms_error_computation(self):
        """Verify RMS error is computed correctly."""
        config = SimulationConfig(
            seed=42,
            target_az=0.1,
            target_el=0.2
        )
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=0.5)
        
        # RMS error should be non-negative
        assert results['los_error_rms'] >= 0.0
        assert results['los_error_rms_x'] >= 0.0
        assert results['los_error_rms_y'] >= 0.0
        
        # Total RMS should be >= individual components
        assert results['los_error_rms'] >= results['los_error_rms_x']
        assert results['los_error_rms'] >= results['los_error_rms_y']
    
    def test_saturation_percentage(self):
        """Verify FSM saturation percentage is computed."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=0.5)
        
        # Saturation percentage should be in [0, 100]
        assert 0.0 <= results['fsm_saturation_percent'] <= 100.0
    
    def test_estimation_error_metrics(self):
        """Verify estimation error is computed."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=0.5)
        
        # Estimation error should be small (good EKF)
        assert results['est_error_rms'] >= 0.0
        # With proper tuning, should be < 1 mrad
        assert results['est_error_rms'] < 1e-3


class TestComponentIntegration:
    """Test integration between subsystems."""
    
    def test_sensor_to_estimator_flow(self):
        """Verify sensor measurements reach estimator."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        # Run briefly
        results = runner.run_simulation(duration=0.1)
        
        # Estimator should have non-zero state
        fused_state = runner.estimator.get_fused_state()
        # At least velocity or angle should be non-zero
        state_norm = np.sqrt(
            fused_state['theta_az']**2 + fused_state['theta_el']**2 +
            fused_state['theta_dot_az']**2 + fused_state['theta_dot_el']**2
        )
        assert state_norm > 0.0
    
    def test_controller_to_actuator_flow(self):
        """Verify controller commands reach actuators."""
        config = SimulationConfig(
            seed=42,
            target_az=0.1,
            target_el=0.1
        )
        runner = DigitalTwinRunner(config)
        
        # Run for short time
        results = runner.run_simulation(duration=0.1)
        
        # Motor should have received torque commands
        log_arrays = results['log_arrays']
        torque_applied = np.any(log_arrays['torque_az'] != 0.0)
        assert torque_applied, "No torque commands applied"
    
    def test_optical_chain_integration(self):
        """Verify optical chain processes LOS correctly."""
        config = SimulationConfig(
            seed=42,
            target_az=0.05,
            target_el=0.05,
            target_enabled=True
        )
        runner = DigitalTwinRunner(config)
        
        # Run simulation
        results = runner.run_simulation(duration=0.5)
        
        # QPD should receive non-zero signals
        log_arrays = results['log_arrays']
        qpd_active = np.any(log_arrays['z_qpd_nes_x'] != 0.0) or \
                     np.any(log_arrays['z_qpd_nes_y'] != 0.0)
        assert qpd_active, "QPD not receiving signals"
    
    def test_hierarchical_control_decoupling(self):
        """Verify coarse and fine controllers operate together."""
        config = SimulationConfig(
            seed=42,
            target_az=0.1,
            target_el=0.1
        )
        runner = DigitalTwinRunner(config)
        
        results = runner.run_simulation(duration=1.0)
        
        # Both controllers should be active
        log_arrays = results['log_arrays']
        coarse_active = np.any(log_arrays['torque_az'] != 0.0)
        fine_active = np.any(log_arrays['fsm_cmd_tip'] != 0.0)
        
        assert coarse_active, "Coarse controller not active"
        assert fine_active, "Fine controller not active"


class TestResetFunctionality:
    """Test simulation reset capability."""
    
    def test_reset_returns_to_initial_state(self):
        """Verify reset clears all state."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        # Run simulation
        runner.run_simulation(duration=0.5)
        
        # State should be non-zero
        assert runner.time > 0.0
        assert runner.iteration > 0
        
        # Reset
        runner.reset()
        
        # Should return to zero
        assert runner.time == 0.0
        assert runner.iteration == 0
        assert runner.q_az == 0.0
        assert runner.q_el == 0.0
        assert len(runner.log_data) == 0
    
    def test_multiple_runs_after_reset(self):
        """Verify multiple runs produce consistent results."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        # First run
        results1 = runner.run_simulation(duration=0.1)
        rms1 = results1['los_error_rms']
        
        # Reset and second run
        runner.reset()
        results2 = runner.run_simulation(duration=0.1)
        rms2 = results2['los_error_rms']
        
        # Should be identical (deterministic)
        assert np.isclose(rms1, rms2, rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_duration_simulation(self):
        """Verify zero-duration simulation doesn't crash."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        # Should handle gracefully
        results = runner.run_simulation(duration=0.0)
        assert results['n_samples'] == 0
    
    def test_very_short_simulation(self):
        """Verify very short simulation works."""
        config = SimulationConfig(seed=42)
        runner = DigitalTwinRunner(config)
        
        # Single timestep
        results = runner.run_simulation(duration=0.001)
        assert results['n_samples'] >= 0
    
    def test_large_target_angles(self):
        """Verify system handles large target angles."""
        config = SimulationConfig(
            seed=42,
            target_az=np.deg2rad(45.0),
            target_el=np.deg2rad(60.0)
        )
        runner = DigitalTwinRunner(config)
        
        # Should not crash
        results = runner.run_simulation(duration=0.5)
        assert results['n_samples'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
