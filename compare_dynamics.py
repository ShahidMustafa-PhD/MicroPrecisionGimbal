"""
Comparison of EKF dynamics: Simplified vs True Manipulator Equation

This script demonstrates the difference between the old simplified dynamics
and the new true manipulator equation implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


def simplified_dynamics(q, dq, tau, friction_az, friction_el, inertia_az, inertia_el, dist):
    """
    Old simplified dynamics (decoupled axes, no gravity).
    
    accel_az = (tau_az - friction_az * dq_az - dist_az) / inertia_az
    accel_el = (tau_el - friction_el * dq_el - dist_el) / inertia_el
    """
    theta_az, theta_el = q
    dq_az, dq_el = dq
    tau_az, tau_el = tau
    dist_az, dist_el = dist
    
    accel_az = (tau_az - friction_az * dq_az - dist_az) / inertia_az
    accel_el = (tau_el - friction_el * dq_el - dist_el) / inertia_el
    
    return np.array([accel_az, accel_el])


def true_dynamics(q, dq, tau, friction_az, friction_el, dist, gimbal):
    """
    True manipulator equation dynamics.
    
    M(q)q̈ + C(q,q̇)q̇ + G(q) = τ - friction + d
    """
    M = gimbal.get_mass_matrix(q)
    C = gimbal.get_coriolis_matrix(q, dq)
    G = gimbal.get_gravity_vector(q)
    
    friction = np.array([friction_az * dq[0], friction_el * dq[1]])
    tau_effective = tau - friction + dist
    
    rhs = tau_effective - (C @ dq) - G
    accel = np.linalg.solve(M, rhs)
    
    return accel


def simulate_trajectory(dynamics_func, q0, dq0, tau_profile, dt, n_steps, **kwargs):
    """Simulate gimbal trajectory using specified dynamics."""
    q_hist = np.zeros((n_steps, 2))
    dq_hist = np.zeros((n_steps, 2))
    
    q = q0.copy()
    dq = dq0.copy()
    
    for i in range(n_steps):
        q_hist[i] = q
        dq_hist[i] = dq
        
        tau = tau_profile(i * dt)
        accel = dynamics_func(q, dq, tau, **kwargs)
        
        # Euler integration
        dq = dq + accel * dt
        q = q + dq * dt
    
    return q_hist, dq_hist


def main():
    print("Comparing Simplified vs True Dynamics")
    print("=" * 80)
    
    # Parameters
    friction_az = 0.1
    friction_el = 0.1
    inertia_az = 1.0
    inertia_el = 1.0
    dist = np.array([0.0, 0.0])
    
    # Create GimbalDynamics instance
    gimbal = GimbalDynamics(
        pan_mass=0.5,
        tilt_mass=0.25,
        cm_r=0.002,
        cm_h=0.005,
        gravity=9.81
    )
    
    # Initial conditions
    q0 = np.array([0.0, np.pi/4])  # Start at 45 deg elevation
    dq0 = np.array([0.0, 0.0])
    
    # Control torque profile (small step input)
    def tau_profile(t):
        if t < 0.1:
            return np.array([0.01, 0.0])
        else:
            return np.array([0.0, 0.0])
    
    # Simulation parameters
    dt = 0.001  # 1 ms
    t_end = 1.0  # 1 second
    n_steps = int(t_end / dt)
    time = np.arange(n_steps) * dt
    
    print(f"Simulating {t_end} seconds with dt={dt} s")
    print(f"Initial state: θ_az={np.rad2deg(q0[0]):.1f}°, θ_el={np.rad2deg(q0[1]):.1f}°")
    
    # Simulate with simplified dynamics
    print("\nSimulating with simplified dynamics...")
    q_simple, dq_simple = simulate_trajectory(
        simplified_dynamics, q0, dq0, tau_profile, dt, n_steps,
        friction_az=friction_az, friction_el=friction_el,
        inertia_az=inertia_az, inertia_el=inertia_el, dist=dist
    )
    
    # Simulate with true dynamics
    print("Simulating with true manipulator dynamics...")
    q_true, dq_true = simulate_trajectory(
        true_dynamics, q0, dq0, tau_profile, dt, n_steps,
        friction_az=friction_az, friction_el=friction_el,
        dist=dist, gimbal=gimbal
    )
    
    # Compute differences
    q_diff = q_true - q_simple
    dq_diff = dq_true - dq_simple
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Max position difference (Az): {np.max(np.abs(q_diff[:, 0])):.6f} rad = {np.rad2deg(np.max(np.abs(q_diff[:, 0]))):.4f}°")
    print(f"Max position difference (El): {np.max(np.abs(q_diff[:, 1])):.6f} rad = {np.rad2deg(np.max(np.abs(q_diff[:, 1]))):.4f}°")
    print(f"Max velocity difference (Az): {np.max(np.abs(dq_diff[:, 0])):.6f} rad/s")
    print(f"Max velocity difference (El): {np.max(np.abs(dq_diff[:, 1])):.6f} rad/s")
    
    # Compute key effects
    M_initial = gimbal.get_mass_matrix(q0)
    M_final = gimbal.get_mass_matrix(q_true[-1])
    G_initial = gimbal.get_gravity_vector(q0)
    G_final = gimbal.get_gravity_vector(q_true[-1])
    
    print(f"\nInertia Matrix Evolution:")
    print(f"M(q0) = \n{M_initial}")
    print(f"M(q_final) = \n{M_final}")
    print(f"\nGravity Vector Evolution:")
    print(f"G(q0) = {G_initial}")
    print(f"G(q_final) = {G_final}")
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Azimuth position
    axes[0, 0].plot(time, np.rad2deg(q_simple[:, 0]), 'b--', label='Simplified', linewidth=2)
    axes[0, 0].plot(time, np.rad2deg(q_true[:, 0]), 'r-', label='True Dynamics', linewidth=2)
    axes[0, 0].set_ylabel('Azimuth [deg]')
    axes[0, 0].set_title('Azimuth Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Elevation position
    axes[0, 1].plot(time, np.rad2deg(q_simple[:, 1]), 'b--', label='Simplified', linewidth=2)
    axes[0, 1].plot(time, np.rad2deg(q_true[:, 1]), 'r-', label='True Dynamics', linewidth=2)
    axes[0, 1].set_ylabel('Elevation [deg]')
    axes[0, 1].set_title('Elevation Position')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Azimuth velocity
    axes[1, 0].plot(time, dq_simple[:, 0], 'b--', label='Simplified', linewidth=2)
    axes[1, 0].plot(time, dq_true[:, 0], 'r-', label='True Dynamics', linewidth=2)
    axes[1, 0].set_ylabel('Az Velocity [rad/s]')
    axes[1, 0].set_title('Azimuth Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Elevation velocity
    axes[1, 1].plot(time, dq_simple[:, 1], 'b--', label='Simplified', linewidth=2)
    axes[1, 1].plot(time, dq_true[:, 1], 'r-', label='True Dynamics', linewidth=2)
    axes[1, 1].set_ylabel('El Velocity [rad/s]')
    axes[1, 1].set_title('Elevation Velocity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Position differences
    axes[2, 0].plot(time, np.rad2deg(q_diff[:, 0]), 'g-', label='Az Difference', linewidth=2)
    axes[2, 0].plot(time, np.rad2deg(q_diff[:, 1]), 'm-', label='El Difference', linewidth=2)
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylabel('Position Error [deg]')
    axes[2, 0].set_title('Position Differences (True - Simplified)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Velocity differences
    axes[2, 1].plot(time, dq_diff[:, 0], 'g-', label='Az Difference', linewidth=2)
    axes[2, 1].plot(time, dq_diff[:, 1], 'm-', label='El Difference', linewidth=2)
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('Velocity Error [rad/s]')
    axes[2, 1].set_title('Velocity Differences (True - Simplified)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ekf_dynamics_comparison.png', dpi=150)
    print(f"\nPlot saved to: ekf_dynamics_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
