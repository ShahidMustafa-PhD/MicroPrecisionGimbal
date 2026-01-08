"""
Unit tests for fault injection framework.

Tests verify:
1. Deterministic fault scheduling
2. Correct fault activation/deactivation timing
3. Parameter application to components
4. Composite fault scenarios
"""

import pytest
import numpy as np
from lasercom_digital_twin.core.simulation.fault_injector import (
    FaultInjector,
    FaultType,
    FaultEvent,
    CompositeFaultScenario,
    create_fault_injector
)


class TestFaultEvent:
    """Test FaultEvent dataclass."""
    
    def test_initialization(self):
        """Test fault event creation."""
        event = FaultEvent(
            fault_type=FaultType.SENSOR_DROPOUT,
            start_time=2.0,
            duration=1.0,
            target='gyro_az'
        )
        
        assert event.fault_type == FaultType.SENSOR_DROPOUT
        assert event.start_time == 2.0
        assert event.duration == 1.0
        assert event.target == 'gyro_az'
        assert not event.active
        
    def test_is_active_before_start(self):
        """Test fault not active before start time."""
        event = FaultEvent(
            fault_type=FaultType.SENSOR_DROPOUT,
            start_time=5.0,
            duration=2.0
        )
        
        assert not event.is_active(3.0)
        assert not event.is_active(4.99)
        
    def test_is_active_during_duration(self):
        """Test fault active during specified duration."""
        event = FaultEvent(
            fault_type=FaultType.SENSOR_DROPOUT,
            start_time=5.0,
            duration=2.0
        )
        
        assert event.is_active(5.0)
        assert event.is_active(6.0)
        assert event.is_active(6.99)
        
    def test_is_active_after_end(self):
        """Test fault not active after duration expires."""
        event = FaultEvent(
            fault_type=FaultType.SENSOR_DROPOUT,
            start_time=5.0,
            duration=2.0
        )
        
        assert not event.is_active(7.01)
        assert not event.is_active(10.0)
        
    def test_permanent_fault(self):
        """Test fault with duration=None is permanent."""
        event = FaultEvent(
            fault_type=FaultType.BACKLASH_GROWTH,
            start_time=5.0,
            duration=None
        )
        
        assert not event.is_active(4.0)
        assert event.is_active(5.0)
        assert event.is_active(100.0)
        assert event.is_active(1000.0)
        
    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        event = FaultEvent(
            fault_type=FaultType.SENSOR_DROPOUT,
            start_time=5.0,
            duration=2.0
        )
        
        assert event.get_elapsed_time(3.0) == 0.0
        assert event.get_elapsed_time(5.0) == 0.0
        assert event.get_elapsed_time(6.5) == 1.5
        assert event.get_elapsed_time(10.0) == 0.0


class TestFaultInjector:
    """Test FaultInjector class."""
    
    def test_initialization_empty(self):
        """Test initialization with no faults."""
        config = {'seed': 42, 'faults': []}
        
        injector = FaultInjector(config)
        
        assert injector.seed == 42
        assert len(injector.fault_events) == 0
        
    def test_initialization_with_faults(self):
        """Test initialization with fault schedule."""
        config = {
            'seed': 42,
            'faults': [
                {
                    'type': 'sensor_dropout',
                    'target': 'gyro_az',
                    'start_time': 2.0,
                    'duration': 1.0
                },
                {
                    'type': 'backlash_growth',
                    'target': 'az',
                    'start_time': 5.0,
                    'parameters': {'scale_factor': 2.0}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert len(injector.fault_events) == 2
        assert injector.fault_events[0].fault_type == FaultType.SENSOR_DROPOUT
        assert injector.fault_events[1].fault_type == FaultType.BACKLASH_GROWTH
        
    def test_invalid_fault_type(self):
        """Test that invalid fault type raises error."""
        config = {
            'faults': [
                {'type': 'invalid_fault_type', 'start_time': 0.0}
            ]
        }
        
        with pytest.raises(ValueError, match="Unknown fault type"):
            FaultInjector(config)
            
    def test_get_active_faults_none_active(self):
        """Test no active faults before start times."""
        config = {
            'faults': [
                {'type': 'sensor_dropout', 'start_time': 5.0}
            ]
        }
        
        injector = FaultInjector(config)
        active = injector.get_active_faults(current_time=3.0)
        
        assert len(active) == 0
        
    def test_get_active_faults_one_active(self):
        """Test single active fault."""
        config = {
            'faults': [
                {
                    'type': 'sensor_dropout',
                    'target': 'gyro_az',
                    'start_time': 2.0,
                    'duration': 3.0
                }
            ]
        }
        
        injector = FaultInjector(config)
        active = injector.get_active_faults(current_time=3.5)
        
        assert 'sensor_dropout' in active
        assert len(active['sensor_dropout']) == 1
        assert active['sensor_dropout'][0].target == 'gyro_az'
        
    def test_get_active_faults_multiple_active(self):
        """Test multiple simultaneous faults."""
        config = {
            'faults': [
                {'type': 'sensor_dropout', 'start_time': 2.0, 'duration': 5.0},
                {'type': 'backlash_growth', 'start_time': 3.0, 'duration': 5.0},
                {'type': 'sensor_bias', 'start_time': 3.5, 'duration': 5.0}
            ]
        }
        
        injector = FaultInjector(config)
        active = injector.get_active_faults(current_time=4.0)
        
        assert len(active) == 3
        assert 'sensor_dropout' in active
        assert 'backlash_growth' in active
        assert 'sensor_bias' in active
        
    def test_is_sensor_failed(self):
        """Test sensor failure detection."""
        config = {
            'faults': [
                {
                    'type': 'sensor_dropout',
                    'target': 'gyro_az',
                    'start_time': 2.0,
                    'duration': 1.0
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert not injector.is_sensor_failed('gyro_az', 1.0)
        assert injector.is_sensor_failed('gyro_az', 2.5)
        assert not injector.is_sensor_failed('gyro_az', 3.5)
        assert not injector.is_sensor_failed('gyro_el', 2.5)
        
    def test_get_sensor_bias(self):
        """Test sensor bias retrieval."""
        config = {
            'faults': [
                {
                    'type': 'sensor_bias',
                    'target': 'encoder_az',
                    'start_time': 2.0,
                    'duration': 3.0,
                    'parameters': {'bias_value': 1e-4}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_sensor_bias('encoder_az', 1.0) == 0.0
        assert injector.get_sensor_bias('encoder_az', 2.5) == 1e-4
        assert injector.get_sensor_bias('encoder_az', 6.0) == 0.0
        
    def test_get_sensor_bias_multiple(self):
        """Test multiple biases sum correctly."""
        config = {
            'faults': [
                {
                    'type': 'sensor_bias',
                    'target': 'gyro_az',
                    'start_time': 2.0,
                    'duration': None,
                    'parameters': {'bias_value': 0.1}
                },
                {
                    'type': 'sensor_bias',
                    'target': 'gyro_az',
                    'start_time': 5.0,
                    'duration': None,
                    'parameters': {'bias_value': 0.2}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert abs(injector.get_sensor_bias('gyro_az', 3.0) - 0.1) < 1e-10
        assert abs(injector.get_sensor_bias('gyro_az', 6.0) - 0.3) < 1e-10
        
    def test_get_sensor_noise_scale(self):
        """Test sensor noise scaling."""
        config = {
            'faults': [
                {
                    'type': 'sensor_noise',
                    'target': 'gyro_az',
                    'start_time': 3.0,
                    'duration': 2.0,
                    'parameters': {'noise_scale': 2.5}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_sensor_noise_scale('gyro_az', 2.0) == 1.0
        assert injector.get_sensor_noise_scale('gyro_az', 4.0) == 2.5
        assert injector.get_sensor_noise_scale('gyro_az', 6.0) == 1.0
        
    def test_get_backlash_scale_constant(self):
        """Test constant backlash scaling."""
        config = {
            'faults': [
                {
                    'type': 'backlash_growth',
                    'target': 'az',
                    'start_time': 2.0,
                    'duration': None,
                    'parameters': {'scale_factor': 2.0}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_backlash_scale('az', 1.0) == 1.0
        assert injector.get_backlash_scale('az', 3.0) == 2.0
        assert injector.get_backlash_scale('az', 10.0) == 2.0
        
    def test_get_backlash_scale_growth(self):
        """Test gradual backlash growth."""
        config = {
            'faults': [
                {
                    'type': 'backlash_growth',
                    'target': 'az',
                    'start_time': 5.0,
                    'duration': 10.0,
                    'parameters': {'growth_rate': 0.1}  # 10% per second
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_backlash_scale('az', 4.0) == 1.0
        assert injector.get_backlash_scale('az', 5.0) == 1.0
        assert abs(injector.get_backlash_scale('az', 7.0) - 1.2) < 1e-10
        assert abs(injector.get_backlash_scale('az', 10.0) - 1.5) < 1e-10
        
    def test_get_friction_scale(self):
        """Test friction scaling."""
        config = {
            'faults': [
                {
                    'type': 'friction_increase',
                    'target': 'el',
                    'start_time': 3.0,
                    'duration': 5.0,
                    'parameters': {'scale_factor': 1.5}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_friction_scale('el', 2.0) == 1.0
        assert injector.get_friction_scale('el', 5.0) == 1.5
        assert injector.get_friction_scale('el', 9.0) == 1.0
        
    def test_get_fsm_saturation_disturbance_step(self):
        """Test FSM saturation step disturbance."""
        config = {
            'faults': [
                {
                    'type': 'fsm_saturation',
                    'start_time': 5.0,
                    'duration': 2.0,
                    'parameters': {
                        'magnitude': 600e-6,
                        'type': 'step'
                    }
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        # Before activation
        result = injector.get_fsm_saturation_disturbance(4.0)
        assert result is None
        
        # During activation
        result = injector.get_fsm_saturation_disturbance(6.0)
        assert result is not None
        assert result['los_error_x'] == 600e-6
        assert result['type'] == 'step'
        
        # After deactivation
        result = injector.get_fsm_saturation_disturbance(8.0)
        assert result is None
        
    def test_get_fsm_saturation_disturbance_ramp(self):
        """Test FSM saturation ramp disturbance."""
        config = {
            'faults': [
                {
                    'type': 'fsm_saturation',
                    'start_time': 5.0,
                    'duration': 3.0,
                    'parameters': {
                        'magnitude': 800e-6,
                        'type': 'ramp',
                        'ramp_duration': 2.0
                    }
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        # At start
        result = injector.get_fsm_saturation_disturbance(5.0)
        assert result['los_error_x'] == 0.0
        
        # Halfway through ramp
        result = injector.get_fsm_saturation_disturbance(6.0)
        assert abs(result['los_error_x'] - 400e-6) < 1e-10
        
        # After ramp complete
        result = injector.get_fsm_saturation_disturbance(7.5)
        assert result['los_error_x'] == 800e-6
        
    def test_is_command_dropped(self):
        """Test command dropout detection."""
        config = {
            'faults': [
                {
                    'type': 'command_dropout',
                    'start_time': 3.0,
                    'duration': 0.5
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert not injector.is_command_dropped(2.0)
        assert injector.is_command_dropped(3.2)
        assert not injector.is_command_dropped(4.0)
        
    def test_get_power_scale(self):
        """Test power scaling."""
        config = {
            'faults': [
                {
                    'type': 'power_sag',
                    'start_time': 2.0,
                    'duration': 1.5,
                    'parameters': {'scale_factor': 0.6}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_power_scale(1.0) == 1.0
        assert injector.get_power_scale(2.5) == 0.6
        assert injector.get_power_scale(4.0) == 1.0
        
    def test_get_power_scale_multiple(self):
        """Test multiple power sags take minimum."""
        config = {
            'faults': [
                {
                    'type': 'power_sag',
                    'start_time': 2.0,
                    'duration': None,
                    'parameters': {'scale_factor': 0.8}
                },
                {
                    'type': 'power_sag',
                    'start_time': 5.0,
                    'duration': None,
                    'parameters': {'scale_factor': 0.6}
                }
            ]
        }
        
        injector = FaultInjector(config)
        
        assert injector.get_power_scale(3.0) == 0.8
        assert injector.get_power_scale(6.0) == 0.6
        
    def test_get_diagnostics(self):
        """Test diagnostic information."""
        config = {
            'faults': [
                {'type': 'sensor_dropout', 'start_time': 2.0},
                {'type': 'backlash_growth', 'start_time': 5.0}
            ]
        }
        
        injector = FaultInjector(config)
        injector.get_active_faults(3.0)
        
        diag = injector.get_diagnostics()
        
        assert diag['total_faults'] == 2
        assert 'fault_schedule' in diag
        assert len(diag['fault_schedule']) == 2
        
    def test_reset(self):
        """Test reset functionality."""
        config = {
            'faults': [
                {'type': 'sensor_dropout', 'start_time': 2.0}
            ]
        }
        
        injector = FaultInjector(config)
        
        # Activate fault
        injector.get_active_faults(3.0)
        assert len(injector.active_faults) > 0
        
        # Reset
        injector.reset()
        assert len(injector.active_faults) == 0
        assert injector.iteration == 0


class TestCompositeFaultScenario:
    """Test pre-configured fault scenarios."""
    
    def test_sensor_degradation_profile(self):
        """Test sensor degradation scenario."""
        config = CompositeFaultScenario.sensor_degradation_profile(
            start_time=3.0
        )
        
        assert 'faults' in config
        assert len(config['faults']) == 3
        
        # Check fault types
        types = [f['type'] for f in config['faults']]
        assert 'sensor_noise' in types
        assert 'sensor_bias' in types
        
    def test_mechanical_wear_profile(self):
        """Test mechanical wear scenario."""
        config = CompositeFaultScenario.mechanical_wear_profile(
            start_time=5.0
        )
        
        assert 'faults' in config
        assert len(config['faults']) == 3
        
        types = [f['type'] for f in config['faults']]
        assert 'friction_increase' in types
        assert 'backlash_growth' in types
        
    def test_mission_stress_test(self):
        """Test comprehensive stress test scenario."""
        config = CompositeFaultScenario.mission_stress_test()
        
        assert 'faults' in config
        assert len(config['faults']) >= 4
        
        types = [f['type'] for f in config['faults']]
        assert 'sensor_dropout' in types
        assert 'fsm_saturation' in types
        assert 'backlash_growth' in types


class TestFaultInjectorFactory:
    """Test factory function."""
    
    def test_create_none_scenario(self):
        """Test no-fault scenario."""
        injector = create_fault_injector('none')
        
        assert len(injector.fault_events) == 0
        
    def test_create_sensor_degradation(self):
        """Test sensor degradation scenario creation."""
        injector = create_fault_injector('sensor_degradation', start_time=2.0)
        
        assert len(injector.fault_events) > 0
        
    def test_create_mechanical_wear(self):
        """Test mechanical wear scenario creation."""
        injector = create_fault_injector('mechanical_wear', start_time=5.0)
        
        assert len(injector.fault_events) > 0
        
    def test_create_mission_stress(self):
        """Test mission stress scenario creation."""
        injector = create_fault_injector('mission_stress')
        
        assert len(injector.fault_events) > 0
        
    def test_create_custom(self):
        """Test custom scenario."""
        custom_faults = [
            {'type': 'sensor_dropout', 'start_time': 1.0}
        ]
        
        injector = create_fault_injector('custom', faults=custom_faults)
        
        assert len(injector.fault_events) == 1
        
    def test_invalid_scenario(self):
        """Test invalid scenario name raises error."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            create_fault_injector('invalid_scenario')


class TestFaultInjectorIntegration:
    """Test fault injector integration scenarios."""
    
    def test_time_sequence(self):
        """Test faults activate/deactivate correctly over time."""
        config = {
            'faults': [
                {'type': 'sensor_dropout', 'start_time': 2.0, 'duration': 1.0},
                {'type': 'backlash_growth', 'start_time': 4.0, 'duration': 2.0},
                {'type': 'power_sag', 'start_time': 7.0, 'duration': 1.0}
            ]
        }
        
        injector = FaultInjector(config)
        
        # t=0: No faults
        active = injector.get_active_faults(0.0)
        assert len(active) == 0
        
        # t=2.5: First fault active
        active = injector.get_active_faults(2.5)
        assert 'sensor_dropout' in active
        assert len(active) == 1
        
        # t=5.0: Second fault active
        active = injector.get_active_faults(5.0)
        assert 'backlash_growth' in active
        assert 'sensor_dropout' not in active
        
        # t=7.5: Third fault active
        active = injector.get_active_faults(7.5)
        assert 'power_sag' in active
        assert 'backlash_growth' not in active
        
        # t=10.0: No faults
        active = injector.get_active_faults(10.0)
        assert len(active) == 0
        
    def test_overlapping_faults(self):
        """Test multiple overlapping faults."""
        config = {
            'faults': [
                {'type': 'sensor_dropout', 'start_time': 2.0, 'duration': 5.0},
                {'type': 'backlash_growth', 'start_time': 4.0, 'duration': 5.0},
                {'type': 'power_sag', 'start_time': 6.0, 'duration': 5.0}
            ]
        }
        
        injector = FaultInjector(config)
        
        # t=6.5: All three should be active
        active = injector.get_active_faults(6.5)
        # sensor_dropout: 2.0 to 7.0 (active at 6.5)
        # backlash_growth: 4.0 to 9.0 (active at 6.5)
        # power_sag: 6.0 to 11.0 (active at 6.5)
        assert len(active) == 3, f"Expected 3 active faults, got {len(active)}: {list(active.keys())}"
        assert 'sensor_dropout' in active
        assert 'backlash_growth' in active
        assert 'power_sag' in active


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
