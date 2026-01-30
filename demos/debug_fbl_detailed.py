#!/usr/bin/env python3
"""
Detailed FBL diagnostics - trace every computation.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner
)


def debug_fbl():
    """Run short FBL simulation with detailed output."""
    
    # Simple step response test
    config = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.010,
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=np.deg2rad(10.0),  # 10 degree step
        target_el=np.deg2rad(10.0),
        target_enabled=True,
        target_type='constant',  # Step response
        use_feedback_linearization=True,
        use_direct_state_feedback=False,
        enable_visualization=False,
        enable_plotting=False,
        dynamics_config={
            'pan_mass': 1,
            'tilt_mass': 0.5,
            'cm_r': 0.0,
            'cm_h': 0.0,
            'gravity': 9.81,
            'friction_az': 0.1,
            'friction_el': 0.1
        },
        feedback_linearization_config={
            'kp': [100.0, 100.0],
            'kd': [18.0, 18.0],
            'ki': [0.0, 0.0],
            'enable_integral': False,
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            # Disable friction feedforward - NDOB will estimate friction
            'friction_az': 0.0,
            'friction_el': 0.0,
            'conditional_friction': False,
            'enable_disturbance_compensation': False,
        },
        ndob_config={
            'enable': True,  # ENABLE NDOB to estimate friction
            'lambda_az': 100.0,  # Even faster bandwidth
            'lambda_el': 100.0,  # Same for elevation
            'd_max': 0.5  # Higher to allow full friction estimation
        }
    )
    
    print("="*70)
    print("FBL DETAILED DEBUG - Step Response")
    print("="*70)
    print(f"Target: Az=10°, El=10°")
    print(f"Gains: Kp={config.feedback_linearization_config['kp']}, Kd={config.feedback_linearization_config['kd']}")
    print()
    
    runner = DigitalTwinRunner(config)
    
    # Run simulation
    results = runner.run_simulation(duration=2.0)
    
    # Access log arrays
    tel = results['log_arrays']
    print(f"Available keys: {list(tel.keys())}")
    
    t = np.array(tel['time'])
    q_az = np.rad2deg(np.array(tel['q_az']))
    q_el = np.rad2deg(np.array(tel['q_el']))
    qd_az = np.rad2deg(np.array(tel['qd_az']))
    qd_el = np.rad2deg(np.array(tel['qd_el']))
    tau_az = np.array(tel['torque_az'])
    tau_el = np.array(tel['torque_el'])
    d_hat_az = np.array(tel['d_hat_ndob_az'])
    d_hat_el = np.array(tel['d_hat_ndob_el'])
    
    print("\nStep Response Data (sampled every 0.1s):")
    print("-"*110)
    print(f"{'t[s]':>8} {'q_az[°]':>10} {'q_el[°]':>10} {'qd_az[°/s]':>12} {'qd_el[°/s]':>12} {'tau_az[Nm]':>12} {'tau_el[Nm]':>12} {'d_az[Nm]':>10} {'d_el[Nm]':>10}")
    print("-"*110)
    
    for i in range(0, len(t), 100):  # Every 0.1s
        print(f"{t[i]:8.3f} {q_az[i]:10.4f} {q_el[i]:10.4f} {qd_az[i]:12.4f} {qd_el[i]:12.4f} {tau_az[i]:12.6f} {tau_el[i]:12.6f} {d_hat_az[i]:10.6f} {d_hat_el[i]:10.6f}")
    
    # Final values
    print("-"*80)
    print(f"\nFinal State:")
    print(f"  Az: position = {q_az[-1]:.4f}° (target = 10°), error = {10.0 - q_az[-1]:.4f}°")
    print(f"  El: position = {q_el[-1]:.4f}° (target = 10°), error = {10.0 - q_el[-1]:.4f}°")
    
    # Check expected dynamics
    print(f"\n\nExpected Dynamics Analysis:")
    print("-"*80)
    # With Kp=100, Kd=18: ωn=10 rad/s, ζ=0.9
    wn = np.sqrt(100)
    zeta = 18 / (2 * wn)
    print(f"Natural frequency: ωn = {wn:.1f} rad/s ({wn/(2*np.pi):.2f} Hz)")
    print(f"Damping ratio: ζ = {zeta:.2f}")
    
    # Expected settling time (2% criterion)
    ts = 4 / (zeta * wn)
    print(f"Expected settling time (2%): {ts:.2f} seconds")
    
    # Peak time
    wd = wn * np.sqrt(1 - zeta**2)
    if zeta < 1:
        tp = np.pi / wd
        print(f"Expected peak time: {tp:.3f} seconds")
        # Overshoot
        overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
        print(f"Expected overshoot: {overshoot:.1f}%")
    else:
        print("System is overdamped (no overshoot)")
    
    print(f"\n\nActual Peak Values:")
    az_max_idx = np.argmax(q_az)
    el_max_idx = np.argmax(q_el)
    print(f"  Az: peak = {q_az[az_max_idx]:.4f}° at t = {t[az_max_idx]:.3f}s")
    print(f"  El: peak = {q_el[el_max_idx]:.4f}° at t = {t[el_max_idx]:.3f}s")
    

if __name__ == '__main__':
    debug_fbl()
