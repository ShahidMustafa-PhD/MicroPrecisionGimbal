# EKF Implementation with True Manipulator Dynamics

## Summary of Changes

Successfully implemented the true manipulator equation dynamics in the Extended Kalman Filter (EKF) for the `PointingStateEstimator` class.

## Implementation Details

### 1. **Import GimbalDynamics Class**
Added import statement to access the true physical dynamics model:
```python
from ..dynamics.gimbal_dynamics import GimbalDynamics
```

### 2. **Initialize GimbalDynamics Instance**
In the `__init__` method, created a `GimbalDynamics` instance with physical parameters from config:
- `pan_mass`: Mass of azimuth assembly (default: 0.5 kg)
- `tilt_mass`: Mass of elevation/optical payload (default: 0.25 kg)
- `cm_r`: Longitudinal CM offset (default: 0.002 m)
- `cm_h`: Lateral CM offset (default: 0.005 m)
- `gravity`: Gravitational acceleration (default: 9.81 m/s²)

### 3. **Updated `predict()` Method**
Replaced simplified rigid-body dynamics with the true manipulator equation:

**Mathematical Form:**
$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau + d$$

**Implementation Steps:**
1. Extract current state: `q = [θ_az, θ_el]`, `dq = [θ̇_az, θ̇_el]`
2. Get dynamics matrices from `GimbalDynamics`:
   - `M(q)`: Configuration-dependent inertia matrix (2×2)
   - `C(q, dq)`: Coriolis/centrifugal matrix (2×2)
   - `G(q)`: Gravity vector (2×1)
3. Compute effective torque: `τ_eff = τ_control - friction + disturbances`
4. Solve for accelerations: `M(q)q̈ = τ_eff - C(q,q̇)q̇ - G(q)`
5. Propagate state using Euler integration

**Key Features:**
- **Nonlinear Inertia Coupling**: M(q) captures Az-El coupling through elevation-dependent terms
- **Coriolis Effects**: C(q, dq) includes velocity-dependent coupling
- **Gravity Torques**: G(q) models gravitational effects on unbalanced mass
- **Friction Model**: Viscous damping included in effective torque

### 4. **Updated `_compute_process_jacobian()` Method**
Implemented proper linearization of the true dynamics for covariance propagation:

**Mathematical Form:**
$$F_k = I + \frac{\partial f}{\partial x} \cdot dt$$

**Jacobian Computation:**
- **Numerical Differentiation**: Used finite differences (ε = 1e-6) to compute:
  - `∂M/∂q`, `∂C/∂q`, `∂G/∂q` (w.r.t. positions)
  - `∂C/∂q̇` (w.r.t. velocities)
  
- **Acceleration Jacobians**:
  - `∂q̈/∂q = -M⁻¹[∂C/∂q·q̇ + ∂G/∂q]` (configuration sensitivity)
  - `∂q̈/∂q̇ = -M⁻¹[C + B_friction]` (velocity damping)
  
- **Disturbance Coupling**:
  - `∂q̈/∂d = M⁻¹` (direct disturbance influence)
  - Off-diagonal terms capture cross-axis coupling

**Jacobian Structure:**
```
F[θ_az, θ̇_az] = dt                    # Position kinematics
F[θ̇_az, θ_az] = 1 + (∂q̈/∂q_az) · dt  # Nonlinear position dependence
F[θ̇_az, θ_el] = (∂q̈/∂q_el) · dt      # Az-El coupling
F[θ̇_az, θ̇_az] = 1 + (∂q̈/∂q̇_az) · dt # Velocity damping
F[θ̇_az, d_az] = M_inv[0,0] · dt      # Disturbance influence
```

### 5. **Physical Effects Captured**

The implementation now correctly models:

1. **Configuration-Dependent Inertia**:
   - Azimuth inertia varies with elevation angle
   - Captured through `M(q)` dependence on `cos(θ_el)` and `sin(θ_el)`

2. **Coriolis/Centrifugal Forces**:
   - Velocity-dependent coupling between axes
   - Automatic generation via Christoffel symbols

3. **Gravity Compensation**:
   - Gravitational torque from CM offset: `τ_g = m·g·r·cos(θ_el)`
   - Affects elevation axis dynamics

4. **Axis Coupling**:
   - Off-diagonal terms in `M(q)` create dynamic coupling
   - Az motion affects El dynamics and vice versa

## Test Results

All tests passed successfully:

✓ Mass matrix computation matches `GimbalDynamics`  
✓ Coriolis matrix computation matches `GimbalDynamics`  
✓ Gravity vector computation matches `GimbalDynamics`  
✓ State propagation works correctly  
✓ Covariance increases during prediction (as expected)  
✓ Az-El coupling detected in Jacobian  
✓ 100-step prediction remains numerically stable  
✓ Covariance maintains positive definiteness  

## Configuration Parameters

To use the true dynamics, add these parameters to your EKF config:

```python
config = {
    # ... existing EKF parameters ...
    
    # GimbalDynamics parameters
    'pan_mass': 0.5,        # kg
    'tilt_mass': 0.25,      # kg
    'cm_r': 0.002,          # m (longitudinal offset)
    'cm_h': 0.005,          # m (lateral offset)
    'gravity': 9.81,        # m/s²
}
```

## Benefits

1. **Physics Fidelity**: EKF now uses the same dynamics as the true system
2. **Accurate Linearization**: Jacobian correctly captures nonlinear coupling
3. **Better State Estimation**: Improved prediction accuracy for large motions
4. **Consistent Framework**: Single source of truth for dynamics (GimbalDynamics)

## Notes

- The Jacobian uses numerical differentiation for generality and maintainability
- For real-time performance, analytical Jacobians could be derived if needed
- The implementation handles singular configurations robustly through `np.linalg.solve()`
- Friction is modeled separately as viscous damping (not part of GimbalDynamics)

## Files Modified

1. `lasercom_digital_twin/core/estimators/state_estimator.py`
   - Added `GimbalDynamics` import
   - Updated `__init__()` to create dynamics instance
   - Rewrote `predict()` to use M, C, G matrices
   - Rewrote `_compute_process_jacobian()` with proper linearization

## Validation

Run the validation test:
```bash
python test_ekf_true_dynamics.py
```

This validates:
- Dynamics consistency
- Jacobian structure
- Numerical stability
- Covariance propagation
