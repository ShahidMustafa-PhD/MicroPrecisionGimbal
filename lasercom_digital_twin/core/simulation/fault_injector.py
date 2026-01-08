"""
Fault Injection Framework for Digital Twin Robustness Testing

This module provides a flexible, deterministic fault injection system for
validating the digital twin's behavior under failure conditions. Faults can
be precisely scheduled and configured to test:

1. Sensor failures (dropout, bias, noise increase)
2. Mechanical degradation (backlash growth, friction increase)
3. Saturation events (external disturbances forcing control limits)
4. Communication failures (command dropouts, latency spikes)

Design Philosophy:
-----------------
All faults are time-triggered with deterministic behavior, enabling:
- Reproducible robustness testing
- Automated regression testing
- Monte Carlo failure analysis
- Fault recovery validation

Usage Pattern:
-------------
The FaultInjector is queried at each simulation timestep to determine
active faults. It returns fault parameters that the simulation runner
applies to the affected components.

Example:
--------
>>> config = {
...     'faults': [
...         {'type': 'sensor_dropout', 'target': 'gyro_az', 
...          'start_time': 2.0, 'duration': 0.5},
...         {'type': 'fsm_saturation', 'start_time': 3.0, 
...          'magnitude': 500e-6}
...     ]
... }
>>> injector = FaultInjector(config)
>>> active_faults = injector.get_active_faults(time=2.3)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class FaultType(Enum):
    """Enumeration of supported fault types."""
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_NOISE = "sensor_noise"
    BACKLASH_GROWTH = "backlash_growth"
    FRICTION_INCREASE = "friction_increase"
    FSM_SATURATION = "fsm_saturation"
    COMMAND_DROPOUT = "command_dropout"
    POWER_SAG = "power_sag"


@dataclass
class FaultEvent:
    """
    Definition of a single fault event.
    
    Attributes
    ----------
    fault_type : FaultType
        Type of fault to inject
    start_time : float
        Simulation time when fault activates [s]
    duration : float
        How long fault persists [s] (None = permanent)
    target : str
        Component/sensor identifier (e.g., 'gyro_az', 'encoder_el')
    parameters : Dict
        Fault-specific parameters (magnitude, bias value, etc.)
    active : bool
        Current activation state
    """
    fault_type: FaultType
    start_time: float
    duration: Optional[float] = None
    target: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    active: bool = False
    
    def is_active(self, current_time: float) -> bool:
        """
        Check if fault is active at given time.
        
        Parameters
        ----------
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        bool
            True if fault is currently active
        """
        if current_time < self.start_time:
            return False
        
        if self.duration is None:
            return True  # Permanent fault
        
        return current_time < (self.start_time + self.duration)
    
    def get_elapsed_time(self, current_time: float) -> float:
        """
        Get time since fault activation.
        
        Parameters
        ----------
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        float
            Elapsed time since activation [s], or 0 if not active
        """
        if not self.is_active(current_time):
            return 0.0
        return current_time - self.start_time


class FaultInjector:
    """
    Deterministic fault injection framework.
    
    This class manages a schedule of fault events and provides query
    interface for the simulation runner to determine active faults and
    their parameters at each timestep.
    
    Fault Categories:
    ----------------
    
    1. SENSOR FAULTS:
       - Dropout: Output set to zero or NaN
       - Bias: Constant offset added to measurement
       - Noise: Increased measurement noise variance
       
    2. MECHANICAL FAULTS:
       - Backlash Growth: Increased deadband in gearing
       - Friction Increase: Higher viscous/Coulomb friction
       
    3. CONTROL FAULTS:
       - FSM Saturation: External disturbance forcing limit
       - Command Dropout: Missing control commands
       - Power Sag: Reduced actuator authority
    
    Implementation:
    --------------
    Faults are checked every timestep via get_active_faults().
    The returned dictionary contains all currently active faults
    organized by type and target.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fault injection framework.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary:
            - 'faults': List of fault event definitions
            - 'seed': Random seed for stochastic faults
            
            Each fault definition dict must contain:
            - 'type': String matching FaultType enum
            - 'start_time': Activation time [s]
            - 'duration': Duration [s] (optional, None=permanent)
            - 'target': Component identifier (optional)
            - Additional parameters specific to fault type
        """
        self.config = config
        self.seed = config.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)
        
        # Parse fault schedule
        self.fault_events: List[FaultEvent] = []
        self._parse_fault_schedule(config.get('faults', []))
        
        # Track active faults
        self.active_faults: Dict[str, List[FaultEvent]] = {}
        
        # Iteration counter
        self.iteration = 0
        
    def _parse_fault_schedule(self, fault_list: List[Dict]) -> None:
        """
        Parse fault configuration into FaultEvent objects.
        
        Parameters
        ----------
        fault_list : List[Dict]
            List of fault definition dictionaries
        """
        for fault_dict in fault_list:
            # Parse fault type
            fault_type_str = fault_dict.get('type', '')
            try:
                fault_type = FaultType(fault_type_str)
            except ValueError:
                raise ValueError(
                    f"Unknown fault type: {fault_type_str}. "
                    f"Valid types: {[ft.value for ft in FaultType]}"
                )
            
            # Create fault event
            event = FaultEvent(
                fault_type=fault_type,
                start_time=fault_dict.get('start_time', 0.0),
                duration=fault_dict.get('duration'),
                target=fault_dict.get('target', ''),
                parameters=fault_dict.get('parameters', {})
            )
            
            self.fault_events.append(event)
        
        # Sort by start time for efficient processing
        self.fault_events.sort(key=lambda e: e.start_time)
    
    def get_active_faults(self, current_time: float) -> Dict[str, List[FaultEvent]]:
        """
        Get all currently active faults.
        
        This is the main interface called by simulation runner at each
        timestep to query fault status.
        
        Parameters
        ----------
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        Dict[str, List[FaultEvent]]
            Dictionary mapping fault types to active events:
            {
                'sensor_dropout': [FaultEvent(...), ...],
                'backlash_growth': [FaultEvent(...)],
                ...
            }
        """
        active = {}
        
        for event in self.fault_events:
            if event.is_active(current_time):
                fault_type_str = event.fault_type.value
                if fault_type_str not in active:
                    active[fault_type_str] = []
                active[fault_type_str].append(event)
        
        self.active_faults = active
        self.iteration += 1
        
        return active
    
    def is_sensor_failed(self, sensor_name: str, current_time: float) -> bool:
        """
        Check if a specific sensor is currently failed.
        
        Parameters
        ----------
        sensor_name : str
            Sensor identifier (e.g., 'gyro_az', 'encoder_el')
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        bool
            True if sensor has an active dropout fault
        """
        active = self.get_active_faults(current_time)
        
        dropout_faults = active.get(FaultType.SENSOR_DROPOUT.value, [])
        for fault in dropout_faults:
            if fault.target == sensor_name:
                return True
        
        return False
    
    def get_sensor_bias(self, sensor_name: str, current_time: float) -> float:
        """
        Get active bias for a specific sensor.
        
        Parameters
        ----------
        sensor_name : str
            Sensor identifier
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        float
            Bias value to add to sensor measurement
        """
        active = self.get_active_faults(current_time)
        
        bias_faults = active.get(FaultType.SENSOR_BIAS.value, [])
        total_bias = 0.0
        
        for fault in bias_faults:
            if fault.target == sensor_name:
                total_bias += fault.parameters.get('bias_value', 0.0)
        
        return total_bias
    
    def get_sensor_noise_scale(
        self, 
        sensor_name: str, 
        current_time: float
    ) -> float:
        """
        Get noise scaling factor for a sensor.
        
        Returns multiplicative factor for sensor noise standard deviation.
        Default is 1.0 (no change). Values > 1.0 increase noise.
        
        Parameters
        ----------
        sensor_name : str
            Sensor identifier
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        float
            Noise scale factor (multiply sensor noise std by this)
        """
        active = self.get_active_faults(current_time)
        
        noise_faults = active.get(FaultType.SENSOR_NOISE.value, [])
        max_scale = 1.0
        
        for fault in noise_faults:
            if fault.target == sensor_name:
                scale = fault.parameters.get('noise_scale', 1.0)
                max_scale = max(max_scale, scale)
        
        return max_scale
    
    def get_backlash_scale(self, joint_name: str, current_time: float) -> float:
        """
        Get backlash scaling factor for a joint.
        
        Parameters
        ----------
        joint_name : str
            Joint identifier ('az' or 'el')
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        float
            Backlash scale factor (multiply nominal backlash by this)
        """
        active = self.get_active_faults(current_time)
        
        backlash_faults = active.get(FaultType.BACKLASH_GROWTH.value, [])
        max_scale = 1.0
        
        for fault in backlash_faults:
            if fault.target == joint_name:
                # Support both instant jump and gradual growth
                scale = fault.parameters.get('scale_factor', 1.0)
                growth_rate = fault.parameters.get('growth_rate', None)
                
                if growth_rate is not None:
                    # Gradual growth: scale = 1 + growth_rate * elapsed_time
                    elapsed = fault.get_elapsed_time(current_time)
                    scale = 1.0 + growth_rate * elapsed
                
                max_scale = max(max_scale, scale)
        
        return max_scale
    
    def get_friction_scale(self, joint_name: str, current_time: float) -> float:
        """
        Get friction scaling factor for a joint.
        
        Parameters
        ----------
        joint_name : str
            Joint identifier ('az' or 'el')
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        float
            Friction scale factor (multiply nominal friction by this)
        """
        active = self.get_active_faults(current_time)
        
        friction_faults = active.get(FaultType.FRICTION_INCREASE.value, [])
        max_scale = 1.0
        
        for fault in friction_faults:
            if fault.target == joint_name:
                scale = fault.parameters.get('scale_factor', 1.0)
                max_scale = max(max_scale, scale)
        
        return max_scale
    
    def get_fsm_saturation_disturbance(
        self, 
        current_time: float
    ) -> Optional[Dict[str, float]]:
        """
        Get FSM saturation test disturbance.
        
        Returns external disturbance designed to force FSM to saturation,
        testing recovery behavior.
        
        Parameters
        ----------
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        Optional[Dict[str, float]]
            Disturbance parameters:
            {
                'los_error_x': LOS error X [rad],
                'los_error_y': LOS error Y [rad],
                'type': 'step' or 'ramp'
            }
            Returns None if no saturation test active
        """
        active = self.get_active_faults(current_time)
        
        saturation_faults = active.get(FaultType.FSM_SATURATION.value, [])
        
        if not saturation_faults:
            return None
        
        # Use first saturation fault (typically only one scheduled)
        fault = saturation_faults[0]
        
        magnitude = fault.parameters.get('magnitude', 500e-6)  # rad
        disturbance_type = fault.parameters.get('type', 'step')
        
        if disturbance_type == 'step':
            # Instant step
            return {
                'los_error_x': magnitude,
                'los_error_y': magnitude * 0.7,  # Slightly different Y
                'type': 'step'
            }
        elif disturbance_type == 'ramp':
            # Linear ramp
            elapsed = fault.get_elapsed_time(current_time)
            ramp_duration = fault.parameters.get('ramp_duration', 1.0)
            scale = min(1.0, elapsed / ramp_duration)
            
            return {
                'los_error_x': magnitude * scale,
                'los_error_y': magnitude * 0.7 * scale,
                'type': 'ramp'
            }
        
        return None
    
    def is_command_dropped(self, current_time: float) -> bool:
        """
        Check if control commands should be dropped.
        
        Parameters
        ----------
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        bool
            True if commands should not be sent to actuators
        """
        active = self.get_active_faults(current_time)
        
        dropout_faults = active.get(FaultType.COMMAND_DROPOUT.value, [])
        return len(dropout_faults) > 0
    
    def get_power_scale(self, current_time: float) -> float:
        """
        Get power/torque authority scaling factor.
        
        Models reduced actuator authority due to power sag, voltage drop,
        or thermal derating.
        
        Parameters
        ----------
        current_time : float
            Current simulation time [s]
            
        Returns
        -------
        float
            Power scale factor (multiply motor torque by this)
        """
        active = self.get_active_faults(current_time)
        
        power_faults = active.get(FaultType.POWER_SAG.value, [])
        min_scale = 1.0
        
        for fault in power_faults:
            scale = fault.parameters.get('scale_factor', 1.0)
            min_scale = min(min_scale, scale)
        
        return min_scale
    
    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about fault injection.
        
        Returns
        -------
        Dict
            Diagnostic data:
            {
                'total_faults': int,
                'active_faults': Dict,
                'fault_schedule': List[Dict],
                'iteration': int
            }
        """
        fault_schedule = []
        for event in self.fault_events:
            fault_schedule.append({
                'type': event.fault_type.value,
                'target': event.target,
                'start_time': event.start_time,
                'duration': event.duration,
                'parameters': event.parameters
            })
        
        return {
            'total_faults': len(self.fault_events),
            'active_fault_types': list(self.active_faults.keys()),
            'fault_schedule': fault_schedule,
            'iteration': self.iteration
        }
    
    def reset(self) -> None:
        """Reset fault injector to initial state."""
        for event in self.fault_events:
            event.active = False
        
        self.active_faults = {}
        self.rng = np.random.default_rng(self.seed)
        self.iteration = 0


class CompositeFaultScenario:
    """
    Pre-configured composite fault scenarios for standardized testing.
    
    Provides common fault combinations for regression testing:
    - Sensor degradation profile
    - Mechanical wear profile  
    - Mission duration stress test
    """
    
    @staticmethod
    def sensor_degradation_profile(
        start_time: float = 5.0,
        seed: int = 42
    ) -> Dict:
        """
        Gradual sensor degradation scenario.
        
        Tests system response to slowly degrading sensors over time.
        
        Parameters
        ----------
        start_time : float
            When degradation begins [s]
        seed : int
            Random seed
            
        Returns
        -------
        Dict
            FaultInjector configuration
        """
        return {
            'seed': seed,
            'faults': [
                # Gyro noise increases over time
                {
                    'type': 'sensor_noise',
                    'target': 'gyro_az',
                    'start_time': start_time,
                    'duration': None,  # Permanent
                    'parameters': {'noise_scale': 2.0}
                },
                {
                    'type': 'sensor_noise',
                    'target': 'gyro_el',
                    'start_time': start_time + 1.0,
                    'duration': None,
                    'parameters': {'noise_scale': 2.0}
                },
                # Encoder develops bias
                {
                    'type': 'sensor_bias',
                    'target': 'encoder_az',
                    'start_time': start_time + 2.0,
                    'duration': None,
                    'parameters': {'bias_value': 1e-4}  # 100 µrad bias
                }
            ]
        }
    
    @staticmethod
    def mechanical_wear_profile(
        start_time: float = 10.0,
        seed: int = 42
    ) -> Dict:
        """
        Progressive mechanical wear scenario.
        
        Models aging effects: increasing friction and backlash.
        
        Parameters
        ----------
        start_time : float
            When wear begins [s]
        seed : int
            Random seed
            
        Returns
        -------
        Dict
            FaultInjector configuration
        """
        return {
            'seed': seed,
            'faults': [
                # Friction increases gradually
                {
                    'type': 'friction_increase',
                    'target': 'az',
                    'start_time': start_time,
                    'duration': None,
                    'parameters': {'scale_factor': 1.5}
                },
                # Backlash grows over time
                {
                    'type': 'backlash_growth',
                    'target': 'az',
                    'start_time': start_time + 2.0,
                    'duration': 5.0,
                    'parameters': {'growth_rate': 0.1}  # 10% per second
                },
                {
                    'type': 'backlash_growth',
                    'target': 'el',
                    'start_time': start_time + 3.0,
                    'duration': 5.0,
                    'parameters': {'growth_rate': 0.1}
                }
            ]
        }
    
    @staticmethod
    def mission_stress_test(seed: int = 42) -> Dict:
        """
        Comprehensive stress test with multiple simultaneous faults.
        
        Tests worst-case scenario: sensor dropout + mechanical wear +
        saturation event all occurring in sequence.
        
        Parameters
        ----------
        seed : int
            Random seed
            
        Returns
        -------
        Dict
            FaultInjector configuration
        """
        return {
            'seed': seed,
            'faults': [
                # Initial sensor dropout
                {
                    'type': 'sensor_dropout',
                    'target': 'gyro_az',
                    'start_time': 2.0,
                    'duration': 1.0
                },
                # FSM saturation test
                {
                    'type': 'fsm_saturation',
                    'start_time': 5.0,
                    'duration': 2.0,
                    'parameters': {
                        'magnitude': 800e-6,  # Large disturbance
                        'type': 'step'
                    }
                },
                # Mechanical degradation
                {
                    'type': 'backlash_growth',
                    'target': 'az',
                    'start_time': 8.0,
                    'duration': None,
                    'parameters': {'scale_factor': 2.0}
                },
                # Power sag
                {
                    'type': 'power_sag',
                    'start_time': 10.0,
                    'duration': 1.5,
                    'parameters': {'scale_factor': 0.7}  # 30% power loss
                },
                # Command dropout
                {
                    'type': 'command_dropout',
                    'start_time': 12.0,
                    'duration': 0.5
                }
            ]
        }


def create_fault_injector(scenario: str = 'none', **kwargs) -> FaultInjector:
    """
    Factory function to create fault injector with pre-configured scenario.
    
    Parameters
    ----------
    scenario : str
        Scenario name:
        - 'none': No faults
        - 'sensor_degradation': Gradual sensor degradation
        - 'mechanical_wear': Progressive mechanical wear
        - 'mission_stress': Comprehensive stress test
        - 'custom': User-provided fault list
    **kwargs
        Scenario-specific parameters and custom fault list
        
    Returns
    -------
    FaultInjector
        Configured fault injector
        
    Example
    -------
    >>> injector = create_fault_injector('sensor_degradation', start_time=3.0)
    >>> injector = create_fault_injector('custom', faults=[...])
    """
    if scenario == 'none':
        config = {'seed': kwargs.get('seed', 42), 'faults': []}
    elif scenario == 'sensor_degradation':
        config = CompositeFaultScenario.sensor_degradation_profile(**kwargs)
    elif scenario == 'mechanical_wear':
        config = CompositeFaultScenario.mechanical_wear_profile(**kwargs)
    elif scenario == 'mission_stress':
        config = CompositeFaultScenario.mission_stress_test(**kwargs)
    elif scenario == 'custom':
        config = {
            'seed': kwargs.get('seed', 42),
            'faults': kwargs.get('faults', [])
        }
    else:
        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Valid: 'none', 'sensor_degradation', 'mechanical_wear', "
            f"'mission_stress', 'custom'"
        )
    
    return FaultInjector(config)
