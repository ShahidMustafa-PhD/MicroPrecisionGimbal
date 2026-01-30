#!/usr/bin/env python3
"""Quick test to verify friction FF + NDOB behavior."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import SimulationConfig, DigitalTwinRunner

base_fbl_config = {
    'kp': [400.0, 400.0], 'kd': [40.0, 40.0], 'ki': [50.0, 50.0],
    'enable_integral': True,
    'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
    'friction_az': 0.1, 'friction_el': 0.1
}

def test_param(name, extra_params):
    fbl_config = base_fbl_config.copy()
    fbl_config.update(extra_params)
    
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
            'friction_az': 0.1, 'friction_el': 0.1
        },
        feedback_linearization_config=fbl_config,
        ndob_config={
            'enable': True,
            'lambda_az': 50.0, 'lambda_el': 50.0,
            'd_max': 0.5
        }
    )
    
    runner = DigitalTwinRunner(config)
    results = runner.run_simulation(duration=5.0)
    rms = results['los_error_rms'] * 1e6
    print(f"  {name}: {rms:.0f} µrad")
    return rms

print("=" * 60)
print("ISOLATING BROKEN PARAMETER")
print("=" * 60)

test_param("Baseline (no extras)", {})
test_param("+ conditional_friction=False", {'conditional_friction': False})
test_param("+ enable_disturbance_compensation=False", {'enable_disturbance_compensation': False})
test_param("+ enable_robust_term=False", {'enable_robust_term': False})
test_param("+ robust_eta", {'robust_eta': [0.01, 0.01]})
test_param("+ robust_lambda", {'robust_lambda': 5.0})

print("\n" + "=" * 60)
print("COMBINED TESTS")
print("=" * 60)
test_param("All extras combined", {
    'conditional_friction': False,
    'enable_disturbance_compensation': False,
    'enable_robust_term': False,
    'robust_eta': [0.01, 0.01],
    'robust_lambda': 5.0
})
