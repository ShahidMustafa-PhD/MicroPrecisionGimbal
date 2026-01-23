"""
Compare velocity profiles across different command types
"""

import numpy as np
from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig, DigitalTwinRunner
)

def analyze_velocity(signal_type, duration):
    config = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.010,
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=np.deg2rad(5.0),
        target_el=np.deg2rad(2.5),
        target_enabled=True,
        target_type=signal_type,
        target_amplitude=5.0 if signal_type in ['sine', 'square'] else 0.0,
        target_period=2.0 if signal_type in ['sine', 'square'] else 1.0,
        use_feedback_linearization=True,
        use_direct_state_feedback=True,
        enable_visualization=False,
        enable_plotting=False,
        feedback_linearization_config={
            'kp': [450.0, 850.0],
            'kd': [20.0, 15.0],
            'friction_az': 0.1,
            'friction_el': 0.1
        },
        ndob_config={'enable': False}  # Pure FBL for fair comparison
    )
    
    runner = DigitalTwinRunner(config)
    result = runner.run_simulation(duration=duration)
    
    qd_az = np.array(result['log_arrays']['qd_az'])
    qd_deg = np.rad2deg(qd_az)
    
    return {
        'signal': signal_type,
        'max_vel': np.max(np.abs(qd_deg)),
        'rms_vel': np.sqrt(np.mean(qd_deg**2)),
        'p95_vel': np.percentile(np.abs(qd_deg), 95)
    }

print("="*80)
print("Velocity Profile Comparison")
print("="*80)
print()

results = []
results.append(analyze_velocity('constant', 2.0))
results.append(analyze_velocity('sine', 4.0))
results.append(analyze_velocity('square', 4.0))

print(f"{'Signal Type':<15} {'Max Vel':>12} {'RMS Vel':>12} {'95th %ile':>12}")
print("-"*80)
for res in results:
    print(f"{res['signal']:<15} {res['max_vel']:>10.1f}°/s {res['rms_vel']:>10.1f}°/s {res['p95_vel']:>10.1f}°/s")

print()
print("RECOMMENDATION:")
square_max = results[2]['max_vel']
sine_max = results[1]['max_vel']
print(f"  Sine wave velocity: {sine_max:.1f}°/s")
print(f"  Square wave velocity: {square_max:.1f}°/s (ratio: {square_max/sine_max:.1f}x)")
print()
print("  Suggested clipping limits:")
print(f"    Conservative: {sine_max*1.2:.1f}°/s (covers sine + 20% margin)")
print(f"    Aggressive: {sine_max*0.5:.1f}°/s (forces square wave smoothing)")
