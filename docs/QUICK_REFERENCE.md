# Feedback Linearization Controller - Quick Reference

## ✅ What Was Fixed

| Issue | Status | Solution |
|-------|--------|----------|
| Indentation error in `FeedbackLinearizationController` | ✅ Fixed | Moved class to module level |
| Wrong method names (`compute_inertia_matrix`) | ✅ Fixed | Updated to `get_mass_matrix()` |
| Mutable default argument | ✅ Fixed | Changed to `Optional[np.ndarray] = None` |
| Missing signal flow integration | ✅ Implemented | Added complete architecture |

## 📊 Signal Flow Architecture

```
┌──────────────────────┐
│   SENSOR LAYER       │  AbsoluteEncoder, RateGyro, QPD
│   Raw Measurements   │  → Noisy θ, ω, pointing error
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ESTIMATOR LAYER     │  Extended Kalman Filter (EKF)
│  Sensor Fusion       │  → Filtered state + disturbance estimate
└──────────┬───────────┘
           │  state_estimate = {
           │    'theta_az', 'theta_el',
           │    'theta_dot_az', 'theta_dot_el',
           │    'dist_az', 'dist_el'
           │  }
           ▼
┌──────────────────────┐
│  CONTROLLER LAYER    │  FeedbackLinearizationController
│  Nonlinear Control   │  → τ = M(q)·v + C·dq + G - d̂
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ACTUATOR LAYER      │  GimbalMotorModel
│  Motor Dynamics      │  → Actual torque with non-idealities
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  DYNAMICS LAYER      │  GimbalDynamics or MuJoCo
│  Physics Simulation  │  → Updated position/velocity
└──────────────────────┘
```

## 🔑 Key Code Snippets

### 1. Sensor Measurements
```python
# sensors/sensor_models.py
z_enc_az = encoder_az.measure(true_q_az)
z_gyro_az = gyro_az.measure(true_qd_az)
z_qpd_x, z_qpd_y = qpd.measure(los_error_x, los_error_y)
```

### 2. EKF Fusion
```python
# estimators/state_estimator.py
measurements = {
    'theta_az_enc': z_enc_az,
    'theta_el_enc': z_enc_el,
    'theta_dot_az_gyro': z_gyro_az,
    'theta_dot_el_gyro': z_gyro_el,
    'nes_x_qpd': z_qpd_x,
    'nes_y_qpd': z_qpd_y
}
estimator.step(u=tau, measurements=measurements, dt=dt)
state_estimate = estimator.get_fused_state()
```

### 3. Feedback Linearization Control
```python
# controllers/control_laws.py
tau, metadata = controller.compute_control(
    q_ref=target,
    dq_ref=np.zeros(2),
    state_estimate=state_estimate,  # ← From EKF
    dt=0.01
)

# Inside: τ = M(q)·v + C(q,dq)·dq + G(q) - d̂
# where v = ddq_ref + Kd·ė + Kp·e
```

### 4. Complete Integration
```python
# simulation/simulation_runner.py
config = SimulationConfig(
    use_feedback_linearization=True,
    feedback_linearization_config={
        'kp': [150.0, 150.0],
        'kd': [30.0, 30.0]
    }
)
runner = DigitalTwinRunner(config)
results = runner.run_simulation(duration=10.0)
```

## 🎯 Control Law Explained

### Standard Form
```
τ = M(q)·v + C(q,q̇)·q̇ + G(q) - d̂
```

### Where:
- **M(q)**: Inertia matrix (2×2) - varies with configuration
- **C(q,q̇)**: Coriolis/centrifugal terms - coupling between axes
- **G(q)**: Gravity vector - varies with elevation angle
- **d̂**: Disturbance estimate from EKF
- **v**: Virtual control = ddq_ref + Kd·ė + Kp·e

### Result:
The closed-loop system becomes **linear**:
```
q̈ = v = ddq_ref + Kd·ė + Kp·e
```

This is a **double integrator** with PD feedback - easy to analyze and tune!

## 📁 Modified Files

| File | Purpose | Changes |
|------|---------|---------|
| `control_laws.py` | Controller implementation | Fixed FL class, updated API calls |
| `simulation_runner.py` | Integration & simulation | Added FL support, signal flow |
| `demo_feedback_linearization.py` | Demonstration | New comparison script |
| `FEEDBACK_LINEARIZATION_GUIDE.md` | Documentation | Complete guide |
| `IMPLEMENTATION_SUMMARY.md` | Summary | Detailed implementation notes |

## 🚀 How to Run

### Method 1: Demo Script
```bash
python demo_feedback_linearization.py
```

### Method 2: Direct Import
```python
from lasercom_digital_twin.core.simulation.simulation_runner import main_feedback_linearization
results = main_feedback_linearization()
```

### Method 3: Custom Configuration
```python
from lasercom_digital_twin.core.simulation.simulation_runner import SimulationConfig, DigitalTwinRunner

config = SimulationConfig(use_feedback_linearization=True, ...)
runner = DigitalTwinRunner(config)
results = runner.run_simulation(duration=10.0)
```

## 📈 Expected Performance

| Metric | PID Controller | Feedback Linearization |
|--------|----------------|------------------------|
| LOS Error RMS | ~50-100 µrad | ~20-50 µrad |
| Settling Time | ~2-3 s | ~0.5-1 s |
| Overshoot | 10-20% | <5% |
| Control Gains | Conservative | Aggressive (3-5× higher) |
| Robustness | High | Moderate (model-dependent) |

## ⚠️ Important Notes

1. **Controller NEVER sees raw sensor data** - only filtered state from EKF
2. **Dynamics model must be accurate** - FL performance depends on model quality
3. **EKF provides disturbance estimates** - critical for compensation
4. **Higher gains are stable** - linearization allows aggressive tuning
5. **Modular architecture** - each layer is independent

## 🔧 Tuning Guidelines

### Control Gains
- Start with `kp = [100, 100]`, `kd = [20, 20]`
- Increase until oscillations appear, then back off 20%
- FL allows 2-3× higher gains than PID

### EKF Tuning
- Process noise `Q`: How much model uncertainty
- Measurement noise `R`: Match sensor specifications
- Balance: Low `Q` → trust model, Low `R` → trust sensors

### Dynamics Model
- Accurate mass properties critical
- Verify center of mass offsets (cm_r, cm_h)
- Validate inertia matrix at multiple configurations

## 📚 Further Reading

- [FEEDBACK_LINEARIZATION_GUIDE.md](FEEDBACK_LINEARIZATION_GUIDE.md) - Complete implementation guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Detailed technical notes
- `control_laws.py` - Source code with inline documentation
- `simulation_runner.py` - Integration example

---

**Status**: ✅ All implementations complete and tested  
**Errors**: None  
**Ready**: Yes - run demo to see results!
