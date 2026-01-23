"""Debug script to investigate FBL+NDOB steady-state error."""

import numpy as np
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
from lasercom_digital_twin.core.n_dist_observer import NonlinearDisturbanceObserver, NDOBConfig
from lasercom_digital_twin.core.controllers.control_laws import FeedbackLinearizationController

# Create dynamics
dyn = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25)

print("=" * 70)
print("INVESTIGATING FBL+NDOB STEADY-STATE ERROR")
print("=" * 70)

# Controller configs matching demo_feedback_linearization.py
fbl_config = {
    'kp': [450.0, 850.0],
    'kd': [20.0, 15.0],
    'friction_az': 0.1,
    'friction_el': 0.1,
    'conditional_friction': True
}

ndob_config_dict = {
    'kp': [450.0, 850.0],
    'kd': [20.0, 15.0],
    'friction_az': 0.0,  # DISABLE friction comp to avoid double-counting
    'friction_el': 0.0,
    'conditional_friction': True
}

# Create controllers
ctrl_fbl = FeedbackLinearizationController(fbl_config, dyn, ndob=None)
ndob = NonlinearDisturbanceObserver(dyn, NDOBConfig(lambda_az=20.0, lambda_el=20.0))
ctrl_ndob = FeedbackLinearizationController(ndob_config_dict, dyn, ndob=ndob)

# Simulation parameters
dt = 0.001
friction_coeff = np.array([0.1, 0.1])  # Plant friction

print("\n--- TRACKING TEST WITH SQUARE WAVE ---")
print("Simulating square wave tracking to observe steady-state behavior...")

# Reset
ndob.reset()
ctrl_fbl.reset()
ctrl_ndob.reset()

# Initial state
q_fbl = np.array([0.0, 0.0])
dq_fbl = np.array([0.0, 0.0])
q_ndob = np.array([0.0, 0.0])
dq_ndob = np.array([0.0, 0.0])

# Square wave parameters (matching demo)
target_base = np.deg2rad(45.0)
amplitude = np.deg2rad(20.0)
period = 2.0

duration = 5.0
steps = int(duration / dt)

print(f"Duration: {duration}s, Square wave period: {period}s")
print(f"Target base: {np.rad2deg(target_base):.1f}°, Amplitude: ±{np.rad2deg(amplitude):.1f}°")
print()

# Storage for analysis
t_hist = []
q_ref_hist = []
q_fbl_hist = []
q_ndob_hist = []
d_hat_raw_hist = []
d_hat_hist = []
weight_hist = []
friction_comp_hist = []

for i in range(steps):
    t = i * dt
    
    # Square wave target
    phase = (t % period) / period
    if phase < 0.5:
        q_ref = np.array([target_base + amplitude, target_base + amplitude])
    else:
        q_ref = np.array([target_base - amplitude, target_base - amplitude])
    
    state_fbl = {
        'theta_az': q_fbl[0], 'theta_el': q_fbl[1],
        'theta_dot_az': dq_fbl[0], 'theta_dot_el': dq_fbl[1],
        'dist_az': 0.0, 'dist_el': 0.0
    }
    state_ndob = {
        'theta_az': q_ndob[0], 'theta_el': q_ndob[1],
        'theta_dot_az': dq_ndob[0], 'theta_dot_el': dq_ndob[1],
        'dist_az': 0.0, 'dist_el': 0.0
    }
    
    tau_fbl, m_fbl = ctrl_fbl.compute_control(q_ref, np.zeros(2), state_fbl, dt)
    tau_ndob, m_ndob = ctrl_ndob.compute_control(q_ref, np.zeros(2), state_ndob, dt)
    
    # Store
    t_hist.append(t)
    q_ref_hist.append(q_ref[0])
    q_fbl_hist.append(q_fbl[0])
    q_ndob_hist.append(q_ndob[0])
    d_hat_raw_hist.append(m_ndob.get('d_hat_ndob_raw', np.zeros(2))[0])
    d_hat_hist.append(m_ndob.get('d_hat_ndob', np.zeros(2))[0])
    weight = m_ndob.get('ndob_weight', np.ones(2))
    weight_hist.append(weight if isinstance(weight, float) else weight[0])
    friction_comp_hist.append(m_ndob.get('friction_comp', np.zeros(2))[0])
    
    # Simulate plant with friction
    M_fbl = dyn.get_mass_matrix(q_fbl)
    C_fbl = dyn.get_coriolis_matrix(q_fbl, dq_fbl)
    G_fbl = dyn.get_gravity_vector(q_fbl)
    d_plant_fbl = -friction_coeff * dq_fbl
    rhs_fbl = tau_fbl - C_fbl @ dq_fbl - G_fbl + d_plant_fbl
    ddq_fbl = np.linalg.solve(M_fbl, rhs_fbl)
    
    M_ndob = dyn.get_mass_matrix(q_ndob)
    C_ndob = dyn.get_coriolis_matrix(q_ndob, dq_ndob)
    G_ndob = dyn.get_gravity_vector(q_ndob)
    d_plant_ndob = -friction_coeff * dq_ndob
    rhs_ndob = tau_ndob - C_ndob @ dq_ndob - G_ndob + d_plant_ndob
    ddq_ndob = np.linalg.solve(M_ndob, rhs_ndob)
    
    # Integrate
    dq_fbl = dq_fbl + ddq_fbl * dt
    q_fbl = q_fbl + dq_fbl * dt
    dq_ndob = dq_ndob + ddq_ndob * dt
    q_ndob = q_ndob + dq_ndob * dt

# Convert to arrays
t_hist = np.array(t_hist)
q_ref_hist = np.array(q_ref_hist)
q_fbl_hist = np.array(q_fbl_hist)
q_ndob_hist = np.array(q_ndob_hist)
d_hat_raw_hist = np.array(d_hat_raw_hist)
d_hat_hist = np.array(d_hat_hist)
weight_hist = np.array(weight_hist)
friction_comp_hist = np.array(friction_comp_hist)

# Analysis
err_fbl = q_ref_hist - q_fbl_hist
err_ndob = q_ref_hist - q_ndob_hist

print("Time-sampled analysis:")
print("  t(s)  | q_ref(°) | err_fbl(°) | err_ndob(°) | d_hat_raw | d_hat | weight | fric_comp")
print("-" * 100)

sample_times = [0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 2.0, 2.5, 3.0, 4.0, 4.9]
for t_sample in sample_times:
    idx = int(t_sample / dt)
    if idx < len(t_hist):
        print(f"  {t_hist[idx]:.2f}  | {np.rad2deg(q_ref_hist[idx]):+7.2f} | {np.rad2deg(err_fbl[idx]):+9.4f} | "
              f"{np.rad2deg(err_ndob[idx]):+10.4f} | {d_hat_raw_hist[idx]:+8.4f} | {d_hat_hist[idx]:+6.4f} | "
              f"{weight_hist[idx]:.4f} | {friction_comp_hist[idx]:+7.4f}")

# RMS error comparison
rms_fbl = np.sqrt(np.mean(err_fbl**2))
rms_ndob = np.sqrt(np.mean(err_ndob**2))
print(f"\nRMS Error:")
print(f"  FBL:      {np.rad2deg(rms_fbl)*1000:.2f} m° ({rms_fbl*1e6:.1f} µrad)")
print(f"  FBL+NDOB: {np.rad2deg(rms_ndob)*1000:.2f} m° ({rms_ndob*1e6:.1f} µrad)")
print(f"  Ratio:    {rms_ndob/rms_fbl:.2f}x")

# Steady-state analysis (look at last 100ms before each target change)
print("\n--- STEADY-STATE ERROR ANALYSIS ---")
ss_indices = [
    (int(0.9/dt), int(1.0/dt)),   # Before first transition
    (int(1.9/dt), int(2.0/dt)),   # Before second transition
    (int(2.9/dt), int(3.0/dt)),   # Before third transition
    (int(3.9/dt), int(4.0/dt)),   # Before fourth transition
    (int(4.9/dt), int(5.0/dt)-1), # End
]

print("Steady-state (last 100ms before target change):")
for start, end in ss_indices:
    ss_err_fbl = np.mean(np.abs(err_fbl[start:end]))
    ss_err_ndob = np.mean(np.abs(err_ndob[start:end]))
    ss_d_hat_raw = np.mean(d_hat_raw_hist[start:end])
    ss_d_hat = np.mean(d_hat_hist[start:end])
    ss_weight = np.mean(weight_hist[start:end])
    ss_fric = np.mean(friction_comp_hist[start:end])
    t_range = f"{t_hist[start]:.1f}-{t_hist[end]:.1f}s"
    print(f"  {t_range}: FBL={np.rad2deg(ss_err_fbl)*1000:.3f}m°, NDOB={np.rad2deg(ss_err_ndob)*1000:.3f}m°, "
          f"d_hat_raw={ss_d_hat_raw:.5f}, d_hat={ss_d_hat:.5f}, w={ss_weight:.3f}, fric={ss_fric:.5f}")

# Check if NDOB is estimating friction when it shouldn't
print("\n--- ROOT CAUSE CHECK ---")
print("During steady-state (dq≈0), friction_comp≈0 and d_hat should also be ≈0.")
print("If d_hat_raw is non-zero at steady-state, NDOB is incorrectly estimating something.")
print()

# Look at a specific steady-state window
ss_window = slice(int(4.5/dt), int(4.9/dt))
print(f"Analysis window: t=4.5-4.9s (should be at steady-state)")
print(f"  Mean velocity magnitude: {np.abs(np.gradient(q_ndob_hist[ss_window], dt)).mean()*1000:.4f} mrad/s")
print(f"  Mean d_hat_raw: {d_hat_raw_hist[ss_window].mean():.6f}")
print(f"  Mean d_hat (weighted): {d_hat_hist[ss_window].mean():.6f}")
print(f"  Mean weight: {weight_hist[ss_window].mean():.4f}")
print(f"  Mean friction_comp: {friction_comp_hist[ss_window].mean():.6f}")
print(f"  Mean error FBL: {np.rad2deg(np.abs(err_fbl[ss_window]).mean())*1000:.4f} m°")
print(f"  Mean error NDOB: {np.rad2deg(np.abs(err_ndob[ss_window]).mean())*1000:.4f} m°")
