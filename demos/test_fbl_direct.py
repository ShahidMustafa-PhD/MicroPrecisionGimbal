"""Test FBL with direct state feedback (no EKF) to isolate the issue."""
import numpy as np
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
from lasercom_digital_twin.core.controllers.control_laws import FeedbackLinearizationController

# Create dynamics model
dynamics = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25, cm_r=0.0, cm_h=0.0, gravity=9.81)

# Create controller
fl_config = {
    'kp': [10.0, 10.0],
    'kd': [3.0, 3.0],
    'ki': [0.0, 0.0],  # No integral for simplicity
    'enable_integral': False,
    'tau_max': [1.0, 1.0],
    'tau_min': [-1.0, -1.0],
    'friction_az': 0.1,
    'friction_el': 0.1,
    'enable_disturbance_compensation': False
}
controller = FeedbackLinearizationController(fl_config, dynamics)

# Simulation parameters
dt = 0.001  # 1ms
duration = 2.0
target_az = np.deg2rad(5.0)
target_el = np.deg2rad(3.0)
friction = 0.1

# State variables
q = np.array([0.0, 0.0])
dq = np.array([0.0, 0.0])

print(f"Target: Az={np.rad2deg(target_az):.1f} deg, El={np.rad2deg(target_el):.1f} deg")
print(f"dt={dt*1000:.1f}ms, duration={duration}s")
print()

# Simulation loop - DIRECT STATE FEEDBACK (no EKF)
history = {'t': [], 'q_az': [], 'q_el': [], 'qd_az': [], 'qd_el': [], 'tau_az': [], 'tau_el': []}

for i in range(int(duration / dt)):
    t = i * dt
    
    # Create state estimate from TRUE state (perfect observer)
    state_estimate = {
        'theta_az': q[0],
        'theta_el': q[1],
        'theta_dot_az': dq[0],
        'theta_dot_el': dq[1],
        'dist_az': 0.0,
        'dist_el': 0.0
    }
    
    # Compute control
    q_ref = np.array([target_az, target_el])
    dq_ref = np.array([0.0, 0.0])
    tau_cmd, meta = controller.compute_control(q_ref, dq_ref, state_estimate, dt)
    
    # Apply friction to get net torque (same as plant does)
    tau_friction = friction * dq
    tau_net = tau_cmd - tau_friction
    
    # Forward dynamics: compute acceleration
    qdd = dynamics.compute_forward_dynamics(q, dq, tau_net)
    
    # Integrate (Euler)
    dq = dq + qdd * dt
    q = q + dq * dt
    
    # Log
    if i % 100 == 0:  # Every 100ms
        history['t'].append(t)
        history['q_az'].append(np.rad2deg(q[0]))
        history['q_el'].append(np.rad2deg(q[1]))
        history['qd_az'].append(dq[0])
        history['qd_el'].append(dq[1])
        history['tau_az'].append(tau_cmd[0])
        history['tau_el'].append(tau_cmd[1])

print("Time(s)  Az(deg)    El(deg)   Az_vel     El_vel    tau_az     tau_el")
print("-" * 75)
for i in range(len(history['t'])):
    print(f"{history['t'][i]:.1f}      {history['q_az'][i]:7.3f}   {history['q_el'][i]:7.3f}   "
          f"{history['qd_az'][i]:8.4f}   {history['qd_el'][i]:8.4f}   "
          f"{history['tau_az'][i]:8.5f}   {history['tau_el'][i]:8.5f}")

print()
print(f"Final position: Az={history['q_az'][-1]:.2f} deg, El={history['q_el'][-1]:.2f} deg")
print(f"Target: Az={np.rad2deg(target_az):.1f} deg, El={np.rad2deg(target_el):.1f} deg")
