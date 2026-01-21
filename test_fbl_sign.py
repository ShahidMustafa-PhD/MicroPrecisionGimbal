"""Test FBL controller sign convention."""
import numpy as np
from lasercom_digital_twin.core.controllers.control_laws import FeedbackLinearizationController
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics

dynamics = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25, cm_r=0.0, cm_h=0.0, gravity=9.81)

fl_config = {
    'kp': [50.0, 50.0], 'kd': [5.0, 5.0], 'ki': [5.0, 5.0],
    'enable_integral': True, 'tau_max': [1.0, 1.0], 'tau_min': [-1.0, -1.0],
    'friction_az': 0.1, 'friction_el': 0.1
}
controller = FeedbackLinearizationController(fl_config, dynamics)

# Test 1: position below target -> should give POSITIVE torque to accelerate toward target
q_ref = np.array([np.deg2rad(5), np.deg2rad(3)])  # Target: +5, +3 degrees
state_est = {'theta_az': 0.0, 'theta_el': 0.0, 'theta_dot_az': 0.0, 'theta_dot_el': 0.0, 'dist_az': 0.0, 'dist_el': 0.0}
tau, meta = controller.compute_control(q_ref, np.array([0.0, 0.0]), state_est, dt=0.01)
print(f"Test 1: pos=0, ref=+5 deg, error={np.rad2deg(meta['error'])} deg")
print(f"  -> tau={tau} Nm (should be POSITIVE to push toward +5)")

# Test 2: position above target -> should give NEGATIVE torque
state_est2 = {'theta_az': np.deg2rad(10), 'theta_el': np.deg2rad(10), 
              'theta_dot_az': 0.0, 'theta_dot_el': 0.0, 'dist_az': 0.0, 'dist_el': 0.0}
controller2 = FeedbackLinearizationController(fl_config, dynamics)  # Fresh controller
tau2, meta2 = controller2.compute_control(q_ref, np.array([0.0, 0.0]), state_est2, dt=0.01)
print(f"\nTest 2: pos=+10, ref=+5 deg, error={np.rad2deg(meta2['error'])} deg")
print(f"  -> tau={tau2} Nm (should be NEGATIVE to push toward +5)")

# Check what the dynamics says about the sign
print("\n--- Dynamics Model Check ---")
M = dynamics.get_mass_matrix(np.array([0, 0]))
G = dynamics.get_gravity_vector(np.array([0, 0]))
print(f"M(q=0) = {M}")
print(f"G(q=0) = {G}")

# Forward dynamics test: if M*qdd = tau, then qdd = M^{-1}*tau
# Positive tau should give positive qdd
test_tau = np.array([0.01, 0.01])
qdd = dynamics.compute_forward_dynamics(np.array([0, 0]), np.array([0, 0]), test_tau)
print(f"\nForward dynamics: tau={test_tau} -> qdd={qdd}")
print(f"  (positive tau should give positive qdd for standard convention)")
