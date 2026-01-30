"""Debug script to check demo telemetry for FBL+NDOB."""

import numpy as np
import copy
from lasercom_digital_twin.core.simulation.simulation_runner import DigitalTwinRunner, SimulationConfig

# Create config matching demo_feedback_linearization.py
target_az_deg = 45.0
target_el_deg = 45.0
amplitude_deg = 20.0
period_s = 2.0
duration = 5.0

# Base FBL config - matching demo
config_fl = SimulationConfig(
    dt_sim=0.001,
    dt_coarse=0.010,
    dt_fine=0.001,
    log_period=0.001,
    seed=42,
    target_az=np.deg2rad(target_az_deg),
    target_el=np.deg2rad(target_el_deg),
    target_enabled=True,
    target_type='square',
    target_amplitude=amplitude_deg,
    target_period=period_s,
    use_feedback_linearization=True,
    use_direct_state_feedback=True,
    enable_visualization=False,
    enable_plotting=False,
    real_time_factor=0.0,
    vibration_enabled=False,  # Disabled for cleaner comparison
    feedback_linearization_config={
        'kp': [450.0, 850.0],
        'kd': [20.0, 15.0],
        'ki': [0.0, 0.0],
        'enable_integral': False,
        'tau_max': [0.5, 0.5],
        'tau_min': [-0.5, -0.5],
        'friction_az': 0.1,
        'friction_el': 0.1,
        'conditional_friction': True,
        'enable_disturbance_compensation': False,
    },
    ndob_config={
        'enable': False,
        'lambda_az': 0.0,
        'lambda_el': 0.0,
        'd_max': 5.0
    },
    dynamics_config={
        'pan_mass': 0.5,
        'tilt_mass': 0.25,
        'cm_r': 0.0,
        'cm_h': 0.0,
        'gravity': 9.81,
        'friction_az': 0.1,
        'friction_el': 0.1
    }
)

# Clone for NDOB
config_ndob = copy.deepcopy(config_fl)
config_ndob.ndob_config = {
    'enable': True,
    'lambda_az': 20.0,
    'lambda_el': 20.0,
    'd_max': 5.0
}
# Keep friction comp enabled
config_ndob.feedback_linearization_config['friction_az'] = 0.1
config_ndob.feedback_linearization_config['friction_el'] = 0.1

print("Running FBL...")
runner_fbl = DigitalTwinRunner(config_fl)
results_fbl = runner_fbl.run_simulation(duration=duration)
print(f"FBL LOS RMS: {results_fbl['los_error_rms']*1e6:.2f} µrad")

print("\nRunning FBL+NDOB...")
runner_ndob = DigitalTwinRunner(config_ndob)
results_ndob = runner_ndob.run_simulation(duration=duration)
print(f"FBL+NDOB LOS RMS: {results_ndob['los_error_rms']*1e6:.2f} µrad")

# Analyze telemetry
telemetry_fbl = results_fbl['telemetry']
telemetry_ndob = results_ndob['telemetry']

# Check the final few timesteps for steady-state analysis
print("\n--- TELEMETRY ANALYSIS ---")
print(f"Telemetry keys: {list(telemetry_ndob.keys())[:20]}...")

# Check NDOB estimates
if 'd_hat_ndob_az' in telemetry_ndob:
    print("\nNDOB estimates at various times:")
    for t_sample in [0.5, 0.9, 1.5, 1.9, 2.5, 4.5, 4.9]:
        idx = int(t_sample / 0.001)
        if idx < len(telemetry_ndob['time']):
            t = telemetry_ndob['time'][idx]
            d_hat_az = telemetry_ndob['d_hat_ndob_az'][idx]
            d_hat_el = telemetry_ndob['d_hat_ndob_el'][idx]
            print(f"  t={t:.2f}s: d_hat=[{d_hat_az:.6f}, {d_hat_el:.6f}]")

# Compare positions at steady-state windows
print("\n--- STEADY-STATE ERROR COMPARISON ---")
ss_windows = [
    (0.8, 1.0, "Before 1st switch"),
    (1.8, 2.0, "Before 2nd switch"),
    (4.8, 5.0, "End of sim")
]

for t_start, t_end, label in ss_windows:
    idx_start = int(t_start / 0.001)
    idx_end = int(t_end / 0.001)
    
    if idx_end > len(telemetry_fbl['time']):
        continue
    
    # FBL errors
    cmd_fbl = np.array(telemetry_fbl['theta_az_cmd'][idx_start:idx_end])
    meas_fbl = np.array(telemetry_fbl['theta_az_meas'][idx_start:idx_end])
    err_fbl = np.mean(np.abs(cmd_fbl - meas_fbl))
    
    # NDOB errors
    cmd_ndob = np.array(telemetry_ndob['theta_az_cmd'][idx_start:idx_end])
    meas_ndob = np.array(telemetry_ndob['theta_az_meas'][idx_start:idx_end])
    err_ndob = np.mean(np.abs(cmd_ndob - meas_ndob))
    
    print(f"\n{label} ({t_start:.1f}-{t_end:.1f}s):")
    print(f"  FBL:      SS error = {np.rad2deg(err_fbl)*1000:.4f} m° ({err_fbl*1e6:.1f} µrad)")
    print(f"  FBL+NDOB: SS error = {np.rad2deg(err_ndob)*1000:.4f} m° ({err_ndob*1e6:.1f} µrad)")
    
    # Also check NDOB d_hat in this window
    if 'd_hat_ndob_az' in telemetry_ndob:
        d_hat_mean = np.mean(telemetry_ndob['d_hat_ndob_az'][idx_start:idx_end])
        print(f"  NDOB d_hat_az mean: {d_hat_mean:.6f} N·m")
