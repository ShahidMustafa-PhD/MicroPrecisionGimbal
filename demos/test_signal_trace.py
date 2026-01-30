"""Trace full signal path through simulation with DIRECT STATE FEEDBACK."""
import numpy as np
from lasercom_digital_twin.core.simulation.simulation_runner import DigitalTwinRunner, SimulationConfig

config = SimulationConfig(
    dt_sim=0.001,
    dt_coarse=0.010,
    dt_fine=0.001,
    log_period=0.001,
    seed=42,
    target_az=np.deg2rad(5.0),
    target_el=np.deg2rad(3.0),
    target_enabled=True,
    use_feedback_linearization=True,
    use_direct_state_feedback=True,  # BYPASS EKF
    enable_visualization=False,
    real_time_factor=0.0,
    vibration_enabled=False,
    feedback_linearization_config={
        'kp': [50.0, 50.0], 'kd': [10.0, 10.0], 'ki': [5.0, 5.0],
        'enable_integral': True, 'tau_max': [1.0, 1.0], 'tau_min': [-1.0, -1.0],
        'friction_az': 0.1, 'friction_el': 0.1,
        'conditional_friction': True,  # NEW: Prevent friction from fighting controller
        'enable_robust_term': False,   # NEW: Optional sliding mode term
        'enable_disturbance_compensation': False
    },
    dynamics_config={
        'pan_mass': 0.5, 'tilt_mass': 0.25, 'cm_r': 0.0, 'cm_h': 0.0, 'gravity': 9.81,
        'friction_az': 0.1, 'friction_el': 0.1
    }
)

runner = DigitalTwinRunner(config)

print("=== Initial State ===")
print(f"q_az: {np.rad2deg(runner.q_az):.4f} deg")
print(f"q_el: {np.rad2deg(runner.q_el):.4f} deg")
print(f"Target: Az={np.rad2deg(config.target_az):.2f} deg, El={np.rad2deg(config.target_el):.2f} deg")

# Use run_single_step (correct order of operations)
for i in range(500):  # 500ms total
    runner.run_single_step()
    
    # Print at every 50ms
    if i > 0 and i % 50 == 0:
        print(f"\n=== t={runner.time*1000:.0f}ms ===")
        print(f"  True: Az={np.rad2deg(runner.q_az):.4f} deg, El={np.rad2deg(runner.q_el):.4f} deg")
        print(f"  True vel: Az={runner.qd_az:.4f} rad/s, El={runner.qd_el:.4f} rad/s")
        print(f"  tau_cmd: {runner.last_tau_cmd}")
        
        # Get controller internal state
        ctrl = runner.coarse_controller
        print(f"  Controller integral: {ctrl.integral}")
        print(f"  Last error: {np.rad2deg(ctrl.previous_error)} deg")
        
        print(f"  voltage_cmd: Az={runner.voltage_cmd_az:.4f}V, El={runner.voltage_cmd_el:.4f}V")
        print(f"  motor tau: Az={runner.state.torque_az:.6f} Nm, El={runner.state.torque_el:.6f} Nm")

print("\n=== Final Analysis ===")
print(f"Direction of q_az: {'toward target' if runner.q_az > 0 else 'away from target'}")
print(f"Direction of q_el: {'toward target' if runner.q_el > 0 else 'away from target'}")

