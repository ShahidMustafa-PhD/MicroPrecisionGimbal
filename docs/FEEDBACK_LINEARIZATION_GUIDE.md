# Feedback Linearization Controller Implementation

## Overview

This document describes the implementation of the Feedback Linearization Controller for the Laser Communication Digital Twin, including the complete signal flow architecture from sensors through estimators to actuators.

## Signal Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                              │
│  (sensors/sensor_models.py, sensors/quadrant_detector.py)  │
│                                                               │
│  - AbsoluteEncoder: θ_az, θ_el (noisy position)            │
│  - RateGyro: ω_az, ω_el (noisy angular velocity)           │
│  - QuadrantDetector: Δθ_fine (optical pointing error)      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Raw Measurements with noise, bias, drift
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   ESTIMATOR LAYER                            │
│           (estimators/state_estimator.py)                   │
│                                                               │
│  Extended Kalman Filter (EKF) performs:                     │
│  1. Prediction: x̂⁻ = f(x̂, u), P⁻ = FPF^T + Q              │
│  2. Measurement Update: x̂ = x̂⁻ + K(z - h(x̂⁻))            │
│  3. Disturbance Estimation: Estimates external torques      │
│                                                               │
│  State Vector (10 elements):                                │
│  [θ_az, θ̇_az, b_az, θ_el, θ̇_el, b_el,                     │
│   φ_roll, φ̇_roll, d_az, d_el]                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Filtered State Dictionary:
                 │ {
                 │   'theta_az': float,      # Filtered azimuth [rad]
                 │   'theta_el': float,      # Filtered elevation [rad]
                 │   'theta_dot_az': float,  # Filtered Az velocity [rad/s]
                 │   'theta_dot_el': float,  # Filtered El velocity [rad/s]
                 │   'dist_az': float,       # Estimated disturbance Az [N·m]
                 │   'dist_el': float        # Estimated disturbance El [N·m]
                 │ }
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  CONTROLLER LAYER                            │
│        (controllers/control_laws.py)                        │
│                                                               │
│  FeedbackLinearizationController:                           │
│  1. Extracts filtered state from EKF                        │
│  2. Computes linearizing terms:                             │
│     - M(q): Inertia matrix from dynamics model              │
│     - C(q,dq): Coriolis/centrifugal terms                  │
│     - G(q): Gravity vector                                  │
│  3. Defines virtual control:                                │
│     v = ddq_ref + Kd*ė + Kp*e                              │
│  4. Computes actual torque:                                 │
│     τ = M*v + C*dq + G - d̂                                 │
│                                                               │
│  Key advantage: Linear closed-loop behavior                 │
│  despite nonlinear plant dynamics                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Torque Commands: [τ_az, τ_el] [N·m]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   ACTUATOR LAYER                             │
│              (actuators/motor_models.py)                     │
│                                                               │
│  GimbalMotorModel:                                          │
│  - Converts torque to motor current                         │
│  - Models motor dynamics (R, L, Kt, Ke)                    │
│  - Includes non-idealities (cogging, saturation)           │
└─────────────────────────────────────────────────────────────┘
                 │
                 │ Applied Torques
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   DYNAMICS LAYER                             │
│        (dynamics/gimbal_dynamics.py or MuJoCo)              │
│                                                               │
│  Integrates equations of motion:                            │
│  M(q)q̈ + C(q,q̇)q̇ + G(q) = τ                               │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Sensor Layer (`lasercom_digital_twin/core/sensors/`)

**Files:**
- `sensor_models.py`: AbsoluteEncoder, RateGyro
- `quadrant_detector.py`: QuadrantDetector (QPD)

**Key Methods:**
```python
# Encoder measurement
theta_measured = encoder.measure(theta_true)  # Returns [rad] with noise

# Gyro measurement
omega_measured = gyro.measure(omega_true)  # Returns [rad/s] with bias

# QPD measurement
nes_x, nes_y = qpd.measure(theta_error_x, theta_error_y)  # Normalized error signal
```

**Sensor Models Include:**
- White noise
- Bias drift
- Quantization
- Latency
- Nonlinearities

### 2. Estimator Layer (`lasercom_digital_twin/core/estimators/`)

**File:** `state_estimator.py`

**Class:** `PointingStateEstimator`

**Key Methods:**
```python
# Initialize estimator
estimator = PointingStateEstimator(config)

# Update with measurements
estimator.step(
    u=np.array([tau_az, tau_el]),  # Control input
    measurements={
        'theta_az_enc': float,
        'theta_el_enc': float,
        'theta_dot_az_gyro': float,
        'theta_dot_el_gyro': float,
        'nes_x_qpd': float,
        'nes_y_qpd': float
    },
    dt=float
)

# Get filtered state
state_estimate = estimator.get_fused_state()
# Returns dictionary with:
# 'theta_az', 'theta_el', 'theta_dot_az', 'theta_dot_el',
# 'dist_az', 'dist_el', 'bias_az', 'bias_el', etc.
```

**EKF State Vector (10 elements):**
1. θ_az: Azimuth angle [rad]
2. θ̇_az: Azimuth velocity [rad/s]
3. b_az: Azimuth gyro bias [rad/s]
4. θ_el: Elevation angle [rad]
5. θ̇_el: Elevation velocity [rad/s]
6. b_el: Elevation gyro bias [rad/s]
7. φ_roll: Optical roll error [rad]
8. φ̇_roll: Roll rate [rad/s]
9. d_az: Azimuth disturbance torque [N·m]
10. d_el: Elevation disturbance torque [N·m]

### 3. Controller Layer (`lasercom_digital_twin/core/controllers/`)

**File:** `control_laws.py`

**Class:** `FeedbackLinearizationController`

**Initialization:**
```python
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics

# Create dynamics model
dynamics = GimbalDynamics(
    pan_mass=0.5,
    tilt_mass=0.25,
    cm_r=0.02,
    cm_h=0.005,
    gravity=9.81
)

# Create controller
config = {
    'kp': [100.0, 100.0],  # Position gains [1/s²]
    'kd': [20.0, 20.0],    # Velocity gains [1/s]
    'tau_max': [10.0, 10.0],  # Max torque [N·m]
    'tau_min': [-10.0, -10.0]  # Min torque [N·m]
}

controller = FeedbackLinearizationController(config, dynamics)
```

**Control Computation:**
```python
# At each control timestep:
tau_command, metadata = controller.compute_control(
    q_ref=np.array([target_az, target_el]),     # Desired position [rad]
    dq_ref=np.array([0.0, 0.0]),                # Desired velocity [rad/s]
    state_estimate=estimator.get_fused_state(), # EKF filtered state
    dt=0.01,                                     # Timestep [s]
    ddq_ref=None                                 # Desired accel (optional)
)
```

**Control Law:**
```
τ = M(q)·v + C(q,q̇)·q̇ + G(q) - d̂

where:
  v = q̈_ref + Kd·ė + Kp·e  (virtual control)
  e = q_ref - q             (position error)
  ė = q̇_ref - q̇            (velocity error)
  d̂ = [d_az, d_el]         (disturbance estimate from EKF)
```

**Linearizing Terms:**
- **M(q)**: Inertia matrix (2×2) from `dynamics.get_mass_matrix(q)`
- **C(q,q̇)**: Coriolis matrix (2×2) from `dynamics.get_coriolis_matrix(q, dq)`
- **G(q)**: Gravity vector (2×1) from `dynamics.get_gravity_vector(q)`

### 4. Dynamics Layer (`lasercom_digital_twin/core/dynamics/`)

**File:** `gimbal_dynamics.py`

**Class:** `GimbalDynamics`

**Key Methods:**
```python
dynamics = GimbalDynamics(pan_mass, tilt_mass, cm_r, cm_h, gravity)

# Get mass/inertia matrix
M = dynamics.get_mass_matrix(q)  # Returns 2×2 matrix

# Get Coriolis/centrifugal matrix
C = dynamics.get_coriolis_matrix(q, dq)  # Returns 2×2 matrix

# Get gravity vector
G = dynamics.get_gravity_vector(q)  # Returns 2-element vector

# Forward dynamics (for simulation)
ddq = dynamics.compute_forward_dynamics(q, dq, tau)  # Returns acceleration
```

## Integration in Simulation Runner

**File:** `lasercom_digital_twin/core/simulation/simulation_runner.py`

**Configuration:**
```python
config = SimulationConfig(
    use_feedback_linearization=True,  # Enable FL controller
    
    feedback_linearization_config={
        'kp': [150.0, 150.0],
        'kd': [30.0, 30.0],
        'tau_max': [10.0, 10.0],
        'tau_min': [-10.0, -10.0]
    },
    
    dynamics_config={
        'pan_mass': 0.5,
        'tilt_mass': 0.25,
        'cm_r': 0.02,
        'cm_h': 0.005,
        'gravity': 9.81
    }
)
```

**Simulation Loop:**
```python
# At each timestep:

# 1. SENSOR LAYER: Read sensors
z_enc_az = encoder_az.measure(true_q_az)
z_enc_el = encoder_el.measure(true_q_el)
z_gyro_az = gyro_az.measure(true_qd_az)
z_gyro_el = gyro_el.measure(true_qd_el)
z_qpd_x, z_qpd_y = qpd.measure(los_error_x, los_error_y)

# 2. ESTIMATOR LAYER: Fuse measurements
measurements = {
    'theta_az_enc': z_enc_az,
    'theta_el_enc': z_enc_el,
    'theta_dot_az_gyro': z_gyro_az,
    'theta_dot_el_gyro': z_gyro_el,
    'nes_x_qpd': z_qpd_x,
    'nes_y_qpd': z_qpd_y
}
estimator.step(u=[tau_az, tau_el], measurements=measurements, dt=dt)
state_estimate = estimator.get_fused_state()

# 3. CONTROLLER LAYER: Compute control
tau_command, metadata = controller.compute_control(
    q_ref=target,
    dq_ref=np.zeros(2),
    state_estimate=state_estimate,
    dt=dt
)

# 4. ACTUATOR LAYER: Apply torque
actual_tau_az = motor_az.step(tau_command[0], qd_az, q_az, dt)
actual_tau_el = motor_el.step(tau_command[1], qd_el, q_el, dt)

# 5. DYNAMICS LAYER: Integrate motion
ddq = dynamics.compute_forward_dynamics(q, dq, [actual_tau_az, actual_tau_el])
dq += ddq * dt
q += dq * dt
```

## Running the Demonstration

### Method 1: Run Demo Script
```bash
cd /path/to/MicroPrecisionGimbal
python demo_feedback_linearization.py
```

### Method 2: Use Simulation Runner Directly
```python
from lasercom_digital_twin.core.simulation.simulation_runner import main_feedback_linearization

results = main_feedback_linearization()
```

### Method 3: Custom Simulation
```python
from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig, 
    DigitalTwinRunner
)
import numpy as np

config = SimulationConfig(
    use_feedback_linearization=True,
    target_az=np.deg2rad(10.0),
    target_el=np.deg2rad(45.0),
    # ... other config ...
)

runner = DigitalTwinRunner(config)
results = runner.run_simulation(duration=10.0)

# Access results
print(f"LOS Error RMS: {results['los_error_rms']*1e6:.2f} µrad")
print(f"Torque RMS: {results['torque_rms_az']:.3f} N·m")
```

## Key Features

### 1. **Nonlinear Dynamics Cancellation**
The controller explicitly cancels:
- Mass/inertia variations with configuration
- Coriolis/centrifugal coupling between axes
- Gravity torques that vary with elevation angle

### 2. **Disturbance Compensation**
The EKF estimates external disturbances (friction, solar pressure, etc.) and the controller compensates via feedforward.

### 3. **High-Gain Stability**
Because the plant is linearized, much higher control gains can be used safely, leading to:
- Faster tracking response
- Better disturbance rejection
- Reduced steady-state error

### 4. **Modular Architecture**
Each layer is independent:
- Sensors can be swapped/upgraded
- Estimator can be replaced (EKF → UKF → Particle Filter)
- Controller is decoupled from sensor noise
- Dynamics model is reusable

## Comparison: PID vs Feedback Linearization

| Aspect | PID Controller | Feedback Linearization |
|--------|---------------|------------------------|
| **Model Requirement** | None | Accurate dynamics model |
| **Gain Tuning** | Empirical, conservative | Analytical, aggressive |
| **Nonlinear Compensation** | None | Explicit cancellation |
| **Disturbance Rejection** | Integral action only | Feedforward + feedback |
| **Tracking Performance** | Good | Excellent |
| **Robustness** | High (to model error) | Moderate (needs good model) |
| **Implementation** | Simple | Moderate complexity |

## Troubleshooting

### Issue: "Unindent amount does not match"
**Fixed.** The `FeedbackLinearizationController` class is now correctly unindented to module level.

### Issue: AttributeError: 'GimbalDynamics' object has no attribute 'compute_inertia_matrix'
**Fixed.** Updated method calls to match actual API:
- `get_mass_matrix(q)` instead of `compute_inertia_matrix(q)`
- `get_coriolis_matrix(q, dq)` instead of `compute_coriolis_matrix(q, dq)`
- `get_gravity_vector(q)` instead of `compute_gravity_vector(q)`

### Issue: Mutable default argument `ddq_ref: np.ndarray = np.zeros(2)`
**Fixed.** Changed to `ddq_ref: Optional[np.ndarray] = None` with initialization inside method.

## References

- `control_laws.py`: Controller implementations
- `gimbal_dynamics.py`: Lagrangian dynamics model
- `state_estimator.py`: EKF implementation
- `simulation_runner.py`: Integration and closed-loop simulation
- `demo_feedback_linearization.py`: Demonstration script

## Contact

For questions or issues, refer to the project documentation or repository.
