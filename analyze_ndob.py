#!/usr/bin/env python3
"""
NDOB Root Cause Analysis

This script isolates the NDOB behavior to understand why it performs worse
than simple friction feedforward.

Key observations:
1. FBL + friction feedforward: 2139 µrad (best)
2. FBL + NDOB: 7871 µrad (worse)
3. NDOB correctly estimates friction direction (verified)
4. NDOB becomes unstable at high bandwidth

Hypothesis:
- NDOB introduces phase lag because it's reactive (observes disturbance effect)
- Friction feedforward is proactive (compensates before disturbance acts)
- For dynamic tracking (sine waves), phase lag causes tracking error
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import SimulationConfig, DigitalTwinRunner

def run_test(name, friction_ff, ndob_enable, ndob_lambda=100.0, duration=5.0):
    """Run test with given config."""
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
        feedback_linearization_config={
            'kp': [400.0, 400.0], 'kd': [40.0, 40.0], 'ki': [50.0, 50.0],
            'enable_integral': True,
            'tau_max': [10.0, 10.0], 'tau_min': [-10.0, -10.0],
            'friction_az': friction_ff, 'friction_el': friction_ff
        },
        ndob_config={
            'enable': ndob_enable,
            'lambda_az': ndob_lambda, 'lambda_el': ndob_lambda,
            'd_max': 0.5
        }
    )
    runner = DigitalTwinRunner(config)
    results = runner.run_simulation(duration=duration)
    return results

print("=" * 70)
print("NDOB ROOT CAUSE ANALYSIS")
print("=" * 70)

# Test 1: FBL + Friction FF (baseline)
print("\n1. FBL + Friction FF (baseline)")
r1 = run_test("FBL+FF", friction_ff=0.1, ndob_enable=False)
rms1 = r1['los_error_rms'] * 1e6
print(f"   LOS RMS = {rms1:.0f} µrad")

# Test 2: FBL + NDOB only
print("\n2. FBL + NDOB only")
r2 = run_test("FBL+NDOB", friction_ff=0.0, ndob_enable=True, ndob_lambda=50.0)
rms2 = r2['los_error_rms'] * 1e6
print(f"   LOS RMS = {rms2:.0f} µrad")

# Test 3: FBL + BOTH (friction FF + NDOB)
print("\n3. FBL + BOTH (friction FF + NDOB)")
r3 = run_test("FBL+FF+NDOB", friction_ff=0.1, ndob_enable=True, ndob_lambda=50.0)
rms3 = r3['los_error_rms'] * 1e6
print(f"   LOS RMS = {rms3:.0f} µrad")

# Test 4: FBL + no friction compensation
print("\n4. FBL + no friction compensation (reference)")
r4 = run_test("FBL-only", friction_ff=0.0, ndob_enable=False)
rms4 = r4['los_error_rms'] * 1e6
print(f"   LOS RMS = {rms4:.0f} µrad")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print(f"""
Performance Ranking:
  1. FBL + Friction FF:    {rms1:.0f} µrad
  2. FBL + NDOB:           {rms2:.0f} µrad
  3. FBL + Both:           {rms3:.0f} µrad
  4. FBL (no compensation):{rms4:.0f} µrad
  
Interpretation:
  - Friction FF improves by: {(rms4-rms1)/rms4*100:.1f}% over no compensation
  - NDOB improves by:        {(rms4-rms2)/rms4*100:.1f}% over no compensation
  - Combined improves by:    {(rms4-rms3)/rms4*100:.1f}% over no compensation
""")

# Analyze NDOB disturbance estimates
print("\n" + "=" * 70)
print("NDOB PHASE LAG ANALYSIS")
print("=" * 70)

tel = r2['log_arrays']
t = np.array(tel['time'])
qd_az = np.array(tel['qd_az'])
d_hat_az = np.array(tel['d_hat_ndob_az'])

# Compute expected friction disturbance
d_friction = -0.1 * qd_az

# Compute phase difference using cross-correlation
from scipy import signal
correlation = np.correlate(d_hat_az - d_hat_az.mean(), 
                           d_friction - d_friction.mean(), mode='full')
lags = np.arange(-len(d_friction)+1, len(d_friction))
lag_samples = lags[np.argmax(correlation)]
lag_time = lag_samples * 0.001  # dt = 1ms

print(f"  NDOB lag behind true friction: {lag_time*1000:.1f} ms ({lag_samples} samples)")
print(f"  At ωn=20 rad/s, this corresponds to: {20.0 * lag_time * 180/np.pi:.1f}° phase lag")
print(f"  For 0.05 Hz sine (period=20s), this is: {360.0 * lag_time / 20.0:.3f}° phase error")
