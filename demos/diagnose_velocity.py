"""
Diagnostic: Analyze velocity profiles to understand clipping effectiveness
"""

import numpy as np
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig, DigitalTwinRunner
)

# Run square wave with NDOB
config = SimulationConfig(
    dt_sim=0.001,
    dt_coarse=0.010,
    dt_fine=0.001,
    log_period=0.001,
    seed=42,
    target_az=np.deg2rad(5.0),
    target_el=np.deg2rad(2.5),
    target_enabled=True,
    target_type='square',
    target_amplitude=5.0,
    target_period=2.0,
    use_feedback_linearization=True,
    use_direct_state_feedback=True,
    enable_visualization=False,
    enable_plotting=False,
    feedback_linearization_config={
        'kp': [450.0, 850.0],
        'kd': [20.0, 15.0],
        'friction_az': 0.1,
        'friction_el': 0.1
    },
    ndob_config={
        'enable': True,
        'lambda_az': 100.0,
        'lambda_el': 100.0,
        'd_max': 5.0,
        'max_dq_ndob': 1.74533  # 100°/s
    }
)

runner = DigitalTwinRunner(config)
result = runner.run_simulation(duration=4.0)

# Extract data
t = np.array(result['log_arrays']['time'])
qd_az = np.array(result['log_arrays']['qd_az'])
d_hat = np.array(result['log_arrays']['d_hat_ndob_az'])
clipped = np.array(result['log_arrays']['ndob_velocity_clipped'])

# Analyze velocity
qd_deg = np.rad2deg(qd_az)
max_vel = np.max(np.abs(qd_deg))
clip_limit_deg = 100.0

print(f"Velocity Analysis:")
print(f"  Max velocity observed: {max_vel:.1f} °/s")
print(f"  Clipping limit: {clip_limit_deg:.1f} °/s")
print(f"  Clipping active: {100*np.sum(clipped)/len(clipped):.1f}% of time")
print(f"  Final d_hat: {d_hat[-1]:.4f} Nm")
print(f"  Max d_hat: {np.max(np.abs(d_hat)):.4f} Nm")

# Check if velocity actually exceeds limit
exceeds_limit = np.abs(qd_deg) > clip_limit_deg
print(f"  Velocity exceeds limit: {100*np.sum(exceeds_limit)/len(exceeds_limit):.1f}% of time")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t, qd_deg, linewidth=1)
axes[0].axhline(clip_limit_deg, color='r', linestyle='--', label='Clip limit')
axes[0].axhline(-clip_limit_deg, color='r', linestyle='--')
axes[0].set_ylabel('Velocity [°/s]')
axes[0].set_title('Gimbal Velocity (Square Wave Command)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, d_hat, linewidth=1)
axes[1].axhline(5.0, color='r', linestyle='--', label='d_max')
axes[1].axhline(-5.0, color='r', linestyle='--')
axes[1].set_ylabel('d_hat [Nm]')
axes[1].set_title('NDOB Disturbance Estimate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, clipped, linewidth=1)
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Clipping Active')
axes[2].set_title('Velocity Clipping Status')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_diagnostic.png', dpi=150)
print("\n✓ Diagnostic plot saved to velocity_diagnostic.png")
plt.close()
