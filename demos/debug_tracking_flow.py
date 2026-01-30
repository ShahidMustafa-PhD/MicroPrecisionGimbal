#!/usr/bin/env python3
"""
Debug script to trace signal flow for tracking failure diagnosis.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner
)


def debug_tracking():
    """Run a short simulation with debug output."""
    
    # Simple sine wave tracking
    config = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.001,  # 1 kHz coarse controller
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=0.0,
        target_el=0.0,
        target_enabled=True,
        target_type='sine',
        target_amplitude=20.0,  # 20 deg amplitude
        target_period=10.0,     # 10 second period (0.1 Hz)
        use_feedback_linearization=True,
        use_direct_state_feedback=False,  # Use EKF estimates
        enable_visualization=False,
        enable_plotting=False,
        dynamics_config={
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.0,
            'cm_h': 0.0,
            'gravity': 9.81
        },
        feedback_linearization_config={
            'kp': [1025.0, 1025.0],
            'kd': [64.0, 64.0],
            'ki': [1.0, 1.0],
            'enable_integral': False,
            'tau_max': [10.0, 10.0],   # Increased from 0.5 Nm
            'tau_min': [-10.0, -10.0],
            'friction_az': 0.0,
            'friction_el': 0.0,
            'conditional_friction': False,
        }
    )
    
    runner = DigitalTwinRunner(config)
    
    print("="*60)
    print("TRACKING DEBUG - Signal Flow Trace")
    print("="*60)
    print(f"Gains: Kp={config.feedback_linearization_config['kp']}, Kd={config.feedback_linearization_config['kd']}")
    print(f"Direct State Feedback: {config.use_direct_state_feedback}")
    print()
    
    # Run just a few steps manually
    duration = 10.0  # 10 seconds (full wave period)
    results = runner.run_simulation(duration=duration)
    
    # Extract log data
    log = results['log_arrays']
    t = log['time']
    q_az = log['q_az']
    q_el = log['q_el']
    target_az = log['target_az']
    target_el = log['target_el']
    tau_az = log['torque_az']
    tau_el = log['torque_el']
    los_error_x = log['los_error_x']
    los_error_y = log['los_error_y']
    est_az = log['est_az']
    z_enc_az = log['z_enc_az']
    
    # Print first 10 samples
    print()
    print("-"*120)
    print("First 10 timesteps:")
    print("-"*120)
    print(f"{'t[s]':>8} {'target_az[deg]':>14} {'q_az[deg]':>12} {'est_az[deg]':>12} {'est_err[deg]':>12} {'tau_az[Nm]':>12}")
    print("-"*120)
    for i in range(min(10, len(t))):
        est_err = np.rad2deg(q_az[i] - est_az[i])
        print(f"{t[i]:8.4f} {np.rad2deg(target_az[i]):14.6f} {np.rad2deg(q_az[i]):12.6f} {np.rad2deg(est_az[i]):12.6f} {est_err:12.6f} {tau_az[i]:12.6f}")
    
    print()
    print("-"*140)
    print("Samples at 1-second intervals:")
    print("-"*140)
    print(f"{'t[s]':>8} {'target[deg]':>12} {'q_az[deg]':>12} {'z_enc[deg]':>12} {'est_az[deg]':>12} {'est_err[deg]':>12} {'tau[Nm]':>10}")
    print("-"*140)
    # Sample at 1-second intervals
    dt = t[1] - t[0] if len(t) > 1 else 0.001
    for sec in range(0, 11):
        idx = min(int(sec / dt), len(t) - 1)
        est_err = np.rad2deg(q_az[idx] - est_az[idx])
        print(f"{t[idx]:8.4f} {np.rad2deg(target_az[idx]):12.4f} {np.rad2deg(q_az[idx]):12.4f} {np.rad2deg(z_enc_az[idx]):12.4f} {np.rad2deg(est_az[idx]):12.4f} {est_err:12.4f} {tau_az[idx]:10.4f}")
    
    # Check for issues
    print()
    print("="*60)
    print("DIAGNOSTICS")
    print("="*60)
    max_err = np.max(np.abs(q_az - target_az))
    rms_err = np.sqrt(np.mean((q_az - target_az)**2))
    los_rms = np.sqrt(np.mean(los_error_x**2) + np.mean(los_error_y**2))
    print(f"Position RMS Error: {np.rad2deg(rms_err):.4f} deg ({rms_err*1e6:.1f} µrad)")
    print(f"LOS RMS Error: {np.rad2deg(los_rms):.4f} deg ({los_rms*1e6:.1f} µrad)")
    print(f"Max Torque Commanded: {np.max(np.abs(tau_az)):.4f} Nm")
    print(f"Final Position: {np.rad2deg(q_az[-1]):.4f} deg")
    print(f"Final Target: {np.rad2deg(target_az[-1]):.4f} deg")
    
    # Velocity check
    vel_az = log.get('qd_az', np.zeros_like(t))
    print(f"Max Velocity: {np.rad2deg(np.max(np.abs(vel_az))):.4f} deg/s")
    
    # State estimation check
    est_az = log['est_az']
    print()
    print("-"*120)
    print("State Estimation Check (last 10 timesteps):")
    print("-"*120)
    print(f"{'t[s]':>8} {'q_az[deg]':>14} {'est_az[deg]':>14} {'est_err[deg]':>12}")
    print("-"*120)
    for i in range(-10, 0):
        est_err = np.rad2deg(q_az[i] - est_az[i])
        print(f"{t[i]:8.4f} {np.rad2deg(q_az[i]):14.6f} {np.rad2deg(est_az[i]):14.6f} {est_err:12.6f}")


if __name__ == "__main__":
    debug_tracking()
