"""
Integration Guide: Disturbances and Fault Injection

This document explains how to integrate the disturbance and fault injection
modules into the digital twin simulation runner.

═══════════════════════════════════════════════════════════════════════════
SECTION 1: DISTURBANCE INTEGRATION
═══════════════════════════════════════════════════════════════════════════

1.1 INITIALIZATION
------------------
Add disturbance module to simulation_runner.py initialization:

```python
from lasercom_digital_twin.core.disturbances.disturbance_models import (
    EnvironmentalDisturbances
)

class DigitalTwinRunner:
    def __init__(self, config: SimulationConfig):
        # ... existing initialization ...
        
        # Initialize disturbances
        disturbance_config = {
            'wind_rms': 0.5,  # N·m
            'wind_correlation_time': 2.0,  # s
            'wind_enabled': True,
            
            'vibration_psd': 1e-6,  # (m/s²)²/Hz
            'vibration_freq_low': 10.0,  # Hz
            'vibration_freq_high': 100.0,  # Hz
            'vibration_enabled': True,
            
            'structural_noise_std': 0.01,  # N·m
            'structural_freq_low': 100.0,  # Hz
            'structural_freq_high': 500.0,  # Hz
            'structural_enabled': True,
            
            'seed': config.seed
        }
        self.disturbances = EnvironmentalDisturbances(disturbance_config)
```

1.2 DISTURBANCE APPLICATION IN STEP LOOP
-----------------------------------------
In the main simulation loop, call disturbances.step() and apply torques:

```python
def run_simulation(self, duration: float):
    for i in range(n_steps):
        self.time = i * self.config.dt_sim
        
        # Generate disturbances
        disturbance_state = self.disturbances.step(
            dt=self.config.dt_sim,
            gimbal_az=self.gimbal_state.az,
            gimbal_el=self.gimbal_state.el,
            gimbal_vel_az=self.gimbal_state.vel_az,
            gimbal_vel_el=self.gimbal_state.vel_el
        )
        
        # Apply disturbances to dynamics
        self._step_dynamics_with_disturbances(
            dt=self.config.dt_sim,
            disturbance_state=disturbance_state
        )
        
        # ... rest of simulation loop ...
```

1.3 MODIFIED DYNAMICS FUNCTION
-------------------------------
Update _step_dynamics() to include disturbance torques:

```python
def _step_dynamics_with_disturbances(
    self, 
    dt: float,
    disturbance_state: DisturbanceState
):
    # Get control torques from motors
    tau_az_motor = self.motor_az.get_torque()
    tau_el_motor = self.motor_el.get_torque()
    
    # Add disturbance torques
    tau_az_total = tau_az_motor + disturbance_state.total_torque_az
    tau_el_total = tau_el_motor + disturbance_state.total_torque_el
    
    # Euler integration with total torque
    accel_az = (tau_az_total - self.friction_az * vel_az) / self.inertia_az
    vel_az_new = vel_az + accel_az * dt
    pos_az_new = pos_az + vel_az_new * dt
    
    # Similar for elevation...
```

1.4 GROUND VIBRATION APPLICATION (For MuJoCo)
----------------------------------------------
If using full MuJoCo integration, apply vibration to base body:

```python
# In MuJoCo integration:
if self.config.use_mujoco:
    # Apply base vibration as position perturbation
    base_body_id = mujoco.mj_name2id(
        self.mj_model, 
        mujoco.mjtObj.mjOBJ_BODY, 
        'base_mount'
    )
    
    # Integrate vibration accelerations to get displacements
    # (need to track velocity and position for vibration)
    self.vib_vel[0] += disturbance_state.vibration_accel_x * dt
    self.vib_pos[0] += self.vib_vel[0] * dt
    
    # Apply small displacement to base
    self.mj_data.xpos[base_body_id][0] += self.vib_pos[0]
    # ... similarly for Y, Z, roll, pitch, yaw
```

═══════════════════════════════════════════════════════════════════════════
SECTION 2: FAULT INJECTION INTEGRATION
═══════════════════════════════════════════════════════════════════════════

2.1 INITIALIZATION
------------------
Add fault injector to simulation_runner.py:

```python
from lasercom_digital_twin.core.simulation.fault_injector import (
    create_fault_injector
)

class DigitalTwinRunner:
    def __init__(self, config: SimulationConfig):
        # ... existing initialization ...
        
        # Initialize fault injector
        # Option 1: No faults
        self.fault_injector = create_fault_injector('none')
        
        # Option 2: Pre-configured scenario
        self.fault_injector = create_fault_injector(
            'mission_stress',
            seed=config.seed
        )
        
        # Option 3: Custom fault schedule
        custom_faults = [
            {
                'type': 'sensor_dropout',
                'target': 'gyro_az',
                'start_time': 2.0,
                'duration': 0.5
            },
            {
                'type': 'fsm_saturation',
                'start_time': 5.0,
                'duration': 2.0,
                'parameters': {'magnitude': 800e-6}
            }
        ]
        self.fault_injector = create_fault_injector(
            'custom',
            faults=custom_faults,
            seed=config.seed
        )
```

2.2 SENSOR FAULT APPLICATION
-----------------------------
Modify sensor measurement methods to check for faults:

```python
def _sample_sensors(self):
    # Check for gyro faults
    if self.fault_injector.is_sensor_failed('gyro_az', self.time):
        # Sensor dropout - return NaN or zero
        gyro_az = np.nan
    else:
        # Normal measurement
        gyro_az = self.gyro_az.measure(
            true_value=self.gimbal_state.vel_az
        )
        
        # Add fault-induced bias
        gyro_az += self.fault_injector.get_sensor_bias(
            'gyro_az', 
            self.time
        )
        
        # Scale noise if fault active
        noise_scale = self.fault_injector.get_sensor_noise_scale(
            'gyro_az',
            self.time
        )
        if noise_scale > 1.0:
            # Re-sample with increased noise
            original_std = self.gyro_az.noise_std
            self.gyro_az.noise_std = original_std * noise_scale
            gyro_az = self.gyro_az.measure(
                true_value=self.gimbal_state.vel_az
            )
            self.gyro_az.noise_std = original_std
    
    # Store measurement
    self.measurements['gyro_az'] = gyro_az
    
    # Similar for other sensors...
```

2.3 MECHANICAL FAULT APPLICATION
---------------------------------
Apply backlash and friction scaling in dynamics:

```python
def _step_dynamics_with_faults(self, dt: float):
    # Get fault scales
    backlash_scale_az = self.fault_injector.get_backlash_scale(
        'az', 
        self.time
    )
    friction_scale_az = self.fault_injector.get_friction_scale(
        'az',
        self.time
    )
    
    # Apply scaled backlash (modify effective position)
    backlash_nominal = 1e-5  # rad
    backlash_effective = backlash_nominal * backlash_scale_az
    
    # Apply in deadband model or gear model
    # (depends on your backlash implementation)
    
    # Apply scaled friction
    friction_torque = self.friction_coeff_az * friction_scale_az * vel_az
    
    # ... rest of dynamics ...
```

2.4 FSM SATURATION TEST
------------------------
Check for saturation test disturbances:

```python
def _update_qpd_measurement(self):
    # Normal LOS error calculation
    los_error_x, los_error_y = self._compute_los_error()
    
    # Check for FSM saturation test
    saturation_dist = self.fault_injector.get_fsm_saturation_disturbance(
        self.time
    )
    
    if saturation_dist is not None:
        # Add large error to force FSM to limits
        los_error_x += saturation_dist['los_error_x']
        los_error_y += saturation_dist['los_error_y']
    
    # Compute QPD measurement with disturbance
    qpd_x, qpd_y = self.qpd.measure_from_los_error(
        los_error_x,
        los_error_y
    )
```

2.5 COMMAND DROPOUT
-------------------
Check for command dropouts before sending to actuators:

```python
def _update_coarse_controller(self):
    # Compute control command
    torque_cmd = self.coarse_controller.compute_control(...)
    
    # Check for command dropout
    if self.fault_injector.is_command_dropped(self.time):
        # Don't send command - hold previous value
        torque_cmd = self.last_torque_cmd
    
    # Send to motor
    self.motor_az.set_torque(torque_cmd)
    self.last_torque_cmd = torque_cmd
```

2.6 POWER SAG
-------------
Scale motor torque authority:

```python
def _apply_motor_torques(self):
    # Get power scale
    power_scale = self.fault_injector.get_power_scale(self.time)
    
    # Scale motor output
    tau_az_commanded = self.motor_az.get_commanded_torque()
    tau_az_actual = tau_az_commanded * power_scale
    
    self.motor_az.set_actual_torque(tau_az_actual)
```

═══════════════════════════════════════════════════════════════════════════
SECTION 3: DATA LOGGING
═══════════════════════════════════════════════════════════════════════════

3.1 LOG DISTURBANCE SIGNALS
----------------------------
Add disturbance logging to _log_data():

```python
def _log_data(self):
    # ... existing logging ...
    
    # Log disturbances
    if hasattr(self, 'last_disturbance_state'):
        dist = self.last_disturbance_state
        self.telemetry['dist_wind_az'].append(dist.wind_torque_az)
        self.telemetry['dist_wind_el'].append(dist.wind_torque_el)
        self.telemetry['dist_structural_az'].append(dist.structural_torque_az)
        self.telemetry['dist_structural_el'].append(dist.structural_torque_el)
        self.telemetry['dist_total_az'].append(dist.total_torque_az)
        self.telemetry['dist_total_el'].append(dist.total_torque_el)
        self.telemetry['dist_vib_z'].append(dist.vibration_accel_z)
```

3.2 LOG FAULT STATUS
---------------------
Log active faults:

```python
def _log_data(self):
    # ... existing logging ...
    
    # Log fault status
    active_faults = self.fault_injector.get_active_faults(self.time)
    
    # Binary flags for each fault type
    self.telemetry['fault_sensor_dropout'].append(
        1 if 'sensor_dropout' in active_faults else 0
    )
    self.telemetry['fault_backlash'].append(
        1 if 'backlash_growth' in active_faults else 0
    )
    self.telemetry['fault_fsm_saturation'].append(
        1 if 'fsm_saturation' in active_faults else 0
    )
    
    # Log scaling factors
    self.telemetry['backlash_scale_az'].append(
        self.fault_injector.get_backlash_scale('az', self.time)
    )
    self.telemetry['power_scale'].append(
        self.fault_injector.get_power_scale(self.time)
    )
```

═══════════════════════════════════════════════════════════════════════════
SECTION 4: TESTING RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════

4.1 BASELINE TEST (NO FAULTS)
------------------------------
```python
# Establish baseline performance without disturbances/faults
config = SimulationConfig(dt_sim=0.001, seed=42)
runner = DigitalTwinRunner(config)
runner.disturbances = create_fault_injector('none')  # Disable
results = runner.run_simulation(duration=10.0)
```

4.2 DISTURBANCE-ONLY TEST
--------------------------
```python
# Test with disturbances but no faults
disturbance_config = {
    'wind_rms': 0.3,
    'vibration_psd': 5e-7,
    'structural_noise_std': 0.02,
    'seed': 42
}
runner.disturbances = EnvironmentalDisturbances(disturbance_config)
runner.fault_injector = create_fault_injector('none')
results = runner.run_simulation(duration=10.0)
```

4.3 SINGLE FAULT TEST
----------------------
```python
# Test individual fault response
custom_fault = {
    'faults': [
        {
            'type': 'sensor_dropout',
            'target': 'gyro_az',
            'start_time': 5.0,
            'duration': 1.0
        }
    ]
}
runner.fault_injector = FaultInjector(custom_fault)
results = runner.run_simulation(duration=15.0)

# Analyze recovery time after fault clears
```

4.4 STRESS TEST
---------------
```python
# Full stress test with all disturbances and faults
runner.fault_injector = create_fault_injector('mission_stress')
disturbance_config = {
    'wind_rms': 0.8,  # High wind
    'vibration_psd': 2e-6,  # High vibration
    'structural_noise_std': 0.05,  # High noise
    'seed': 42
}
runner.disturbances = EnvironmentalDisturbances(disturbance_config)
results = runner.run_simulation(duration=20.0)

# Verify system maintains stability
```

═══════════════════════════════════════════════════════════════════════════
SECTION 5: CONFIGURATION EXAMPLES
═══════════════════════════════════════════════════════════════════════════

5.1 REALISTIC OPERATIONAL CONDITIONS
-------------------------------------
```python
operational_disturbances = {
    'wind_rms': 0.2,  # Light wind
    'wind_correlation_time': 5.0,
    'vibration_psd': 5e-7,  # Building vibration
    'vibration_freq_low': 15.0,
    'vibration_freq_high': 80.0,
    'structural_noise_std': 0.005,  # Motor noise
    'structural_freq_low': 150.0,
    'structural_freq_high': 400.0,
    'seed': 42
}
```

5.2 HARSH ENVIRONMENT
---------------------
```python
harsh_disturbances = {
    'wind_rms': 1.0,  # Strong gusts
    'wind_correlation_time': 1.0,
    'vibration_psd': 5e-6,  # Heavy machinery nearby
    'structural_noise_std': 0.05,  # Worn bearings
    'seed': 42
}
```

5.3 SPACE ENVIRONMENT (LOW DISTURBANCE)
----------------------------------------
```python
space_disturbances = {
    'wind_rms': 0.0,  # No atmosphere
    'wind_enabled': False,
    'vibration_psd': 1e-8,  # Only spacecraft vibration
    'vibration_freq_low': 50.0,
    'vibration_freq_high': 200.0,
    'structural_noise_std': 0.001,  # Minimal
    'seed': 42
}
```

═══════════════════════════════════════════════════════════════════════════
SECTION 6: VALIDATION METRICS
═══════════════════════════════════════════════════════════════════════════

After integration, validate using:

1. **Disturbance Verification:**
   - Verify wind PSD matches theoretical Dryden model
   - Check vibration energy is in specified frequency band
   - Confirm structural noise has correct spectral content

2. **Fault Timing:**
   - Verify faults activate/deactivate at exact specified times
   - Confirm sensor dropouts produce NaN or zero correctly
   - Check fault parameter scaling is applied correctly

3. **System Response:**
   - LOS error should increase during disturbances
   - Estimator should track through sensor faults
   - Controller should recover after transient faults
   - No system crashes or instabilities

4. **Determinism:**
   - Same seed produces identical results
   - Fault sequences are perfectly repeatable

═══════════════════════════════════════════════════════════════════════════
END OF INTEGRATION GUIDE
═══════════════════════════════════════════════════════════════════════════
"""
