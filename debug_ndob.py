"""Debug script to understand NDOB behavior with square wave."""

import numpy as np
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
from lasercom_digital_twin.core.n_dist_observer import NonlinearDisturbanceObserver, NDOBConfig
from lasercom_digital_twin.core.controllers.control_laws import FeedbackLinearizationController

# Create dynamics and NDOB
dyn = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25)

print("=" * 70)
print("TEST: Square wave tracking with friction")
print("=" * 70)

# Controller 1: Pure FBL with friction compensation
fbl_config = {
    'kp': [450.0, 850.0],
    'kd': [20.0, 15.0],
    'friction_az': 0.1,  # Compensation enabled
    'friction_el': 0.1,
    'conditional_friction': True
}
ctrl_fbl = FeedbackLinearizationController(fbl_config, dyn, ndob=None)

# Controller 2: FBL+NDOB with friction compensation ENABLED
# The gated NDOB provides ~0 during transients, so we still need friction comp.
# NDOB estimates ADDITIONAL unknown disturbances at steady state.
ndob = NonlinearDisturbanceObserver(dyn, NDOBConfig(lambda_az=20.0, lambda_el=20.0))
ndob_config = {
    'kp': [450.0, 850.0],
    'kd': [20.0, 15.0],
    'friction_az': 0.1,  # ENABLED - same as pure FBL
    'friction_el': 0.1,
    'conditional_friction': True
}
ctrl_ndob = FeedbackLinearizationController(ndob_config, dyn, ndob=ndob)

# Simulation parameters
dt = 0.001
duration = 5.0
steps = int(duration / dt)
friction_coeff = np.array([0.1, 0.1])  # PLANT friction

# Initial state
q_fbl = np.array([0.0, 0.0])
dq_fbl = np.array([0.0, 0.0])
q_ndob = np.array([0.0, 0.0])
dq_ndob = np.array([0.0, 0.0])

# Square wave parameters (like demo)
base_target = np.deg2rad(np.array([45.0, 45.0]))  # Base position
amplitude = np.deg2rad(20.0)  # ±20 degrees
period = 2.0  # 2 second period

def get_target(t):
    """Square wave target."""
    if (int(t / period) % 2) == 0:
        return base_target + np.array([amplitude, amplitude])
    else:
        return base_target - np.array([amplitude, amplitude])

# Storage
t_hist = []
q_fbl_hist = []
q_ndob_hist = []
error_fbl_hist = []
error_ndob_hist = []
d_hat_hist = []

print(f"Simulating {duration}s with dt={dt*1000}ms...")
print(f"Square wave: base={np.rad2deg(base_target)}°, amplitude=±{np.rad2deg(amplitude):.1f}°, period={period}s")
print(f"Plant friction: {friction_coeff}")
print()

for i in range(steps):
    t = i * dt
    q_ref = get_target(t)
    
    # FBL controller
    state_fbl = {
        'theta_az': q_fbl[0], 'theta_el': q_fbl[1],
        'theta_dot_az': dq_fbl[0], 'theta_dot_el': dq_fbl[1],
        'dist_az': 0.0, 'dist_el': 0.0
    }
    tau_fbl, m_fbl = ctrl_fbl.compute_control(q_ref, np.zeros(2), state_fbl, dt)
    
    # FBL+NDOB controller
    state_ndob = {
        'theta_az': q_ndob[0], 'theta_el': q_ndob[1],
        'theta_dot_az': dq_ndob[0], 'theta_dot_el': dq_ndob[1],
        'dist_az': 0.0, 'dist_el': 0.0
    }
    tau_ndob, m_ndob = ctrl_ndob.compute_control(q_ref, np.zeros(2), state_ndob, dt)
    
    # Simulate plant dynamics WITH friction
    M_fbl = dyn.get_mass_matrix(q_fbl)
    C_fbl = dyn.get_coriolis_matrix(q_fbl, dq_fbl)
    G_fbl = dyn.get_gravity_vector(q_fbl)
    d_fbl = -friction_coeff * dq_fbl
    rhs_fbl = tau_fbl - C_fbl @ dq_fbl - G_fbl + d_fbl
    ddq_fbl = np.linalg.solve(M_fbl, rhs_fbl)
    
    M_ndob = dyn.get_mass_matrix(q_ndob)
    C_ndob = dyn.get_coriolis_matrix(q_ndob, dq_ndob)
    G_ndob = dyn.get_gravity_vector(q_ndob)
    d_ndob = -friction_coeff * dq_ndob
    rhs_ndob = tau_ndob - C_ndob @ dq_ndob - G_ndob + d_ndob
    ddq_ndob = np.linalg.solve(M_ndob, rhs_ndob)
    
    # Integrate
    dq_fbl = dq_fbl + ddq_fbl * dt
    q_fbl = q_fbl + dq_fbl * dt
    dq_ndob = dq_ndob + ddq_ndob * dt
    q_ndob = q_ndob + dq_ndob * dt
    
    # Log
    t_hist.append(t)
    q_fbl_hist.append(q_fbl.copy())
    q_ndob_hist.append(q_ndob.copy())
    error_fbl_hist.append(q_ref - q_fbl)
    error_ndob_hist.append(q_ref - q_ndob)
    d_hat_hist.append(m_ndob['d_hat_ndob'].copy())

# Convert to arrays
q_fbl_hist = np.array(q_fbl_hist)
q_ndob_hist = np.array(q_ndob_hist)
error_fbl_hist = np.array(error_fbl_hist)
error_ndob_hist = np.array(error_ndob_hist)
d_hat_hist = np.array(d_hat_hist)
t_hist = np.array(t_hist)

# Compute RMS errors
rms_fbl = np.sqrt(np.mean(error_fbl_hist[:, 0]**2))
rms_ndob = np.sqrt(np.mean(error_ndob_hist[:, 0]**2))

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"RMS Error (Az axis):")
print(f"  FBL:      {np.rad2deg(rms_fbl):.4f}° = {rms_fbl*1e6:.2f} µrad")
print(f"  FBL+NDOB: {np.rad2deg(rms_ndob):.4f}° = {rms_ndob*1e6:.2f} µrad")
print(f"  Ratio (NDOB/FBL): {rms_ndob/rms_fbl:.2f}x")

# Sample at key points (just after target changes)
print("\nError at key points:")
print("  t(s)  | target(°) | FBL error(°) | NDOB error(°)")
print("-" * 55)
for t_sample in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
    idx = int(t_sample / dt)
    target = np.rad2deg(get_target(t_sample)[0])
    err_fbl = np.rad2deg(error_fbl_hist[idx, 0])
    err_ndob = np.rad2deg(error_ndob_hist[idx, 0])
    print(f"  {t_sample:.1f}   |   {target:.1f}  | {err_fbl:+10.4f}  | {err_ndob:+10.4f}")

