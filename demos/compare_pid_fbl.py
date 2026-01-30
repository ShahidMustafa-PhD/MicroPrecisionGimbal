#!/usr/bin/env python3
"""
Side-by-side comparison of PID vs FBL at the same operating point.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner
)


def compare_controllers():
    """Compare PID and FBL on sine tracking."""
    
    # Common settings
    base_config = {
        'dt_sim': 0.001,
        'dt_coarse': 0.010,
        'dt_fine': 0.001,
        'log_period': 0.001,
        'seed': 42,
        'target_az': 0.0,
        'target_el': 0.0,
        'target_enabled': True,
        'target_type': 'sine',
        'target_amplitude': 20.0,
        'target_period': 20.0,
        'enable_visualization': False,
        'enable_plotting': False,
        'dynamics_config': {
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.0,
            'cm_h': 0.0,
            'gravity': 9.81,
            'friction_az': 0.1,
            'friction_el': 0.1
        }
    }
    
    print("="*70)
    print("CONTROLLER COMPARISON - Sine Tracking")
    print("="*70)
    print(f"Target: ±20° sine wave, T=20s (0.05 Hz)")
    print(f"Friction: 0.1 Nm/(rad/s) on both axes")
    print()
    
    # Test 1: PID
    print("\n--- TEST 1: PID ---")
    config_pid = SimulationConfig(
        **base_config,
        use_feedback_linearization=False,
        coarse_controller_config={
            'kp': [3.257, 0.660],
            'ki': [10.232, 2.074],
            'kd': [0.1046599, 0.021709],
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            'enable_derivative': True
        }
    )
    
    runner_pid = DigitalTwinRunner(config_pid)
    results_pid = runner_pid.run_simulation(duration=10.0)
    
    tel_pid = results_pid['log_arrays']
    t = np.array(tel_pid['time'])
    q_az_pid = np.rad2deg(tel_pid['q_az'])
    q_el_pid = np.rad2deg(tel_pid['q_el'])
    target_az = np.rad2deg(tel_pid['target_az'])
    target_el = np.rad2deg(tel_pid['target_el'])
    
    err_az_pid = target_az - q_az_pid
    err_el_pid = target_el - q_el_pid
    
    print(f"  LOS RMS: {results_pid['los_error_rms']*1e6:.2f} µrad")
    print(f"  Az RMS Error: {np.std(np.deg2rad(err_az_pid))*1e6:.2f} µrad")
    print(f"  El RMS Error: {np.std(np.deg2rad(err_el_pid))*1e6:.2f} µrad")
    print(f"  Final Az Error: {err_az_pid[-1]:.4f}°")
    print(f"  Final El Error: {err_el_pid[-1]:.4f}°")
    
    # Test 2: FBL with friction feedforward
    print("\n--- TEST 2: FBL (with friction feedforward) ---")
    config_fbl = SimulationConfig(
        **base_config,
        use_feedback_linearization=True,
        feedback_linearization_config={
            'kp': [100.0, 100.0],
            'kd': [18.0, 18.0],
            'ki': [20.0, 20.0],
            'enable_integral': True,
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            'friction_az': 0.1,
            'friction_el': 0.1,
            'conditional_friction': False,
            'enable_disturbance_compensation': False,
        },
        ndob_config={'enable': False}
    )
    
    runner_fbl = DigitalTwinRunner(config_fbl)
    results_fbl = runner_fbl.run_simulation(duration=10.0)
    
    tel_fbl = results_fbl['log_arrays']
    q_az_fbl = np.rad2deg(tel_fbl['q_az'])
    q_el_fbl = np.rad2deg(tel_fbl['q_el'])
    
    err_az_fbl = target_az - q_az_fbl
    err_el_fbl = target_el - q_el_fbl
    
    print(f"  LOS RMS: {results_fbl['los_error_rms']*1e6:.2f} µrad")
    print(f"  Az RMS Error: {np.std(np.deg2rad(err_az_fbl))*1e6:.2f} µrad")
    print(f"  El RMS Error: {np.std(np.deg2rad(err_el_fbl))*1e6:.2f} µrad")
    print(f"  Final Az Error: {err_az_fbl[-1]:.4f}°")
    print(f"  Final El Error: {err_el_fbl[-1]:.4f}°")
    
    # Print sample tracking data
    print("\n--- Tracking Comparison (every 1 second) ---")
    print(f"{'t[s]':>6} {'tgt_az':>8} {'PID_az':>8} {'FBL_az':>8} {'PID_err':>8} {'FBL_err':>8}")
    print("-"*54)
    for i in range(0, len(t), 1000):  # Every 1 second
        print(f"{t[i]:6.1f} {target_az[i]:8.2f} {q_az_pid[i]:8.2f} {q_az_fbl[i]:8.2f} {err_az_pid[i]:8.4f} {err_az_fbl[i]:8.4f}")


if __name__ == '__main__':
    compare_controllers()
