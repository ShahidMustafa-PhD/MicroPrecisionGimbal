#!/usr/bin/env python3
"""Test different FBL configurations to diagnose poor tracking."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import SimulationConfig, DigitalTwinRunner

def test_config(name, fl_config, ndob_enable=False, ndob_lambda=100.0):
    """Run a test with given FBL config."""
    config = SimulationConfig(
        dt_sim=0.001, dt_coarse=0.01, dt_fine=0.001,
        log_period=0.001, seed=42,
        target_az=0.0, target_el=0.0,
        target_enabled=True, target_type='sine',
        target_amplitude=45.0, target_period=20.0,
        use_feedback_linearization=True,
        enable_visualization=False, enable_plotting=False,
        dynamics_config={
            'pan_mass': 1.0, 'tilt_mass': 0.5,
            'cm_r': 0.0, 'cm_h': 0.0, 'gravity': 9.81,
            'friction_az': 0.1, 'friction_el': 0.1
        },
        feedback_linearization_config=fl_config,
        ndob_config={
            'enable': ndob_enable,
            'lambda_az': ndob_lambda,
            'lambda_el': ndob_lambda,
            'd_max': 0.5
        }
    )
    
    runner = DigitalTwinRunner(config)
    results = runner.run_simulation(duration=5.0)
    rms = results['los_error_rms'] * 1e6
    print(f"  {name}: LOS RMS = {rms:.2f} µrad")
    return rms

print("=" * 70)
print("FBL CONFIGURATION DIAGNOSTIC TEST")
print("=" * 70)
print("\nTest: 45° sine wave, 20s period, 5s duration")
print("Plant: M=1.0kg, friction=0.1 Nm/(rad/s)\n")

# Test 1: No friction, no integral
print("TEST 1: FBL without friction/integral (baseline)")
test_config("Baseline", {
    'kp': [100.0, 100.0], 'kd': [20.0, 20.0], 'ki': [0.0, 0.0],
    'enable_integral': False,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.0, 'friction_el': 0.0
})

# Test 2: With friction feedforward
print("\nTEST 2: FBL with friction feedforward")
test_config("Friction FF", {
    'kp': [100.0, 100.0], 'kd': [20.0, 20.0], 'ki': [0.0, 0.0],
    'enable_integral': False,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.1, 'friction_el': 0.1
})

# Test 3: With integral action
print("\nTEST 3: FBL with integral action only")
test_config("Integral", {
    'kp': [100.0, 100.0], 'kd': [20.0, 20.0], 'ki': [20.0, 20.0],
    'enable_integral': True,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.0, 'friction_el': 0.0
})

# Test 4: With friction + integral
print("\nTEST 4: FBL with friction + integral")
test_config("Friction+Integral", {
    'kp': [100.0, 100.0], 'kd': [20.0, 20.0], 'ki': [20.0, 20.0],
    'enable_integral': True,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.1, 'friction_el': 0.1
})

# Test 5: With NDOB
print("\nTEST 5: FBL + NDOB")
test_config("NDOB", {
    'kp': [100.0, 100.0], 'kd': [20.0, 20.0], 'ki': [0.0, 0.0],
    'enable_integral': False,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.0, 'friction_el': 0.0
}, ndob_enable=True, ndob_lambda=100.0)

# Test 6: Higher gains
print("\nTEST 6: FBL with higher gains (Kp=400)")
test_config("High gains", {
    'kp': [400.0, 400.0], 'kd': [40.0, 40.0], 'ki': [50.0, 50.0],
    'enable_integral': True,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.1, 'friction_el': 0.1
})

# Test 7: Much higher gains
print("\nTEST 7: FBL with very high gains (Kp=900)")
test_config("Very high gains", {
    'kp': [900.0, 900.0], 'kd': [60.0, 60.0], 'ki': [100.0, 100.0],
    'enable_integral': True,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.1, 'friction_el': 0.1
})

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
