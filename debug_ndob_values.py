#!/usr/bin/env python3
"""
Debug script to trace NDOB estimated values during simulation.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner
)


def debug_ndob():
    """Run simulation with NDOB and trace disturbance estimates."""
    
    config = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.010,  # 100 Hz coarse controller
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=0.0,
        target_el=0.0,
        target_enabled=True,
        target_type='sine',
        target_amplitude=20.0,  # 20 deg amplitude
        target_period=20.0,     # 20 second period (0.05 Hz)
        use_feedback_linearization=True,
        use_direct_state_feedback=False,
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
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            # CRITICAL: friction_az/el must be 0 when NDOB is enabled!
            # Otherwise friction gets double-compensated
            'friction_az': 0.0,
            'friction_el': 0.0,
            'conditional_friction': False,
            'enable_disturbance_compensation': False,
        },
        ndob_config={
            'enable': True,
            'lambda_az': 10.0,   # Very conservative
            'lambda_el': 5.0,    # Even more conservative for lower inertia
            'd_max': 0.1         # Limit to realistic friction level
        }
    )
    
    runner = DigitalTwinRunner(config)
    
    print("="*70)
    print("NDOB DEBUG - Disturbance Estimate Trace")
    print("="*70)
    print(f"NDOB Gains: lambda_az={config.ndob_config['lambda_az']}, lambda_el={config.ndob_config['lambda_el']}")
    print(f"d_max: {config.ndob_config['d_max']} Nm")
    print()
    
    # Run simulation
    duration = 5.0
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
    d_hat_az = log['d_hat_ndob_az']
    d_hat_el = log['d_hat_ndob_el']
    
    # Calculate tracking errors
    error_az = target_az - q_az
    error_el = target_el - q_el
    
    print()
    print("-"*120)
    print("NDOB Estimates at 0.5-second intervals:")
    print("-"*120)
    print(f"{'t[s]':>8} {'d_hat_az[Nm]':>14} {'d_hat_el[Nm]':>14} {'tau_az[Nm]':>12} {'tau_el[Nm]':>12} {'err_az[deg]':>12} {'err_el[deg]':>12}")
    print("-"*120)
    
    # Sample at 0.5-second intervals
    dt = t[1] - t[0] if len(t) > 1 else 0.001
    for sec_10 in range(0, int(duration*2)+1):
        sec = sec_10 * 0.5
        idx = min(int(sec / dt), len(t) - 1)
        print(f"{t[idx]:8.4f} {d_hat_az[idx]:14.6f} {d_hat_el[idx]:14.6f} {tau_az[idx]:12.6f} {tau_el[idx]:12.6f} {np.rad2deg(error_az[idx]):12.4f} {np.rad2deg(error_el[idx]):12.4f}")
    
    # Statistics
    print()
    print("="*70)
    print("NDOB STATISTICS")
    print("="*70)
    print(f"d_hat_az:  mean={np.mean(d_hat_az):+.6f}, std={np.std(d_hat_az):.6f}, max={np.max(np.abs(d_hat_az)):.6f} Nm")
    print(f"d_hat_el:  mean={np.mean(d_hat_el):+.6f}, std={np.std(d_hat_el):.6f}, max={np.max(np.abs(d_hat_el)):.6f} Nm")
    print()
    print(f"tau_az:    mean={np.mean(tau_az):+.6f}, std={np.std(tau_az):.6f}, max={np.max(np.abs(tau_az)):.6f} Nm")
    print(f"tau_el:    mean={np.mean(tau_el):+.6f}, std={np.std(tau_el):.6f}, max={np.max(np.abs(tau_el)):.6f} Nm")
    print()
    print(f"Tracking RMS Error:")
    print(f"  Az: {np.rad2deg(np.sqrt(np.mean(error_az**2))):.4f} deg ({np.sqrt(np.mean(error_az**2))*1e6:.1f} urad)")
    print(f"  El: {np.rad2deg(np.sqrt(np.mean(error_el**2))):.4f} deg ({np.sqrt(np.mean(error_el**2))*1e6:.1f} urad)")
    
    # Check for saturation
    d_max = config.ndob_config['d_max']
    saturated_az = np.sum(np.abs(d_hat_az) >= d_max * 0.99) / len(d_hat_az) * 100
    saturated_el = np.sum(np.abs(d_hat_el) >= d_max * 0.99) / len(d_hat_el) * 100
    print()
    print(f"NDOB Saturation (d >= {d_max}*0.99):")
    print(f"  Az: {saturated_az:.1f}% of time")
    print(f"  El: {saturated_el:.1f}% of time")
    
    # Check the ratio of d_hat to tau
    ratio_az = np.mean(np.abs(d_hat_az)) / (np.mean(np.abs(tau_az)) + 1e-9)
    ratio_el = np.mean(np.abs(d_hat_el)) / (np.mean(np.abs(tau_el)) + 1e-9)
    print()
    print(f"Disturbance-to-Control Ratio:")
    print(f"  Az: |d_hat|/|tau| = {ratio_az:.2f}")
    print(f"  El: |d_hat|/|tau| = {ratio_el:.2f}")
    
    if ratio_el > 1.0:
        print(f"  WARNING: El disturbance estimate exceeds control torque!")
        print(f"           This indicates NDOB is over-compensating!")


if __name__ == "__main__":
    debug_ndob()
