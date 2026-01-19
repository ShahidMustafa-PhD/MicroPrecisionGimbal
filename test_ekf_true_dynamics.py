"""
Test script to validate the EKF implementation with true manipulator dynamics.

This script verifies that:
1. The EKF predict() method correctly uses M(q), C(q,dq), G(q) from GimbalDynamics
2. The process Jacobian is computed with proper linearization
3. The covariance propagation is stable
"""

import numpy as np
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lasercom_digital_twin.core.estimators.state_estimator import PointingStateEstimator
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


def test_ekf_dynamics_consistency():
    """Test that EKF dynamics match GimbalDynamics forward dynamics."""
    
    print("=" * 80)
    print("Testing EKF with True Manipulator Dynamics")
    print("=" * 80)
    
    # Configuration for EKF
    config = {
        'initial_state': np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'initial_covariance': None,  # Use default
        'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-4, 1e-4],
        'measurement_noise_std': [2.4e-5, 2.4e-5, 1e-6, 1e-6, 1e-4, 1e-4],
        'inertia_az': 1.0,
        'inertia_el': 1.0,
        'friction_coeff_az': 0.1,
        'friction_coeff_el': 0.1,
        'qpd_sensitivity': 2000.0,
        'focal_length_m': 1.5,
        # GimbalDynamics parameters
        'pan_mass': 0.5,
        'tilt_mass': 0.25,
        'cm_r': 0.002,
        'cm_h': 0.005,
        'gravity': 9.81
    }
    
    # Initialize EKF
    ekf = PointingStateEstimator(config)
    
    # Initialize standalone GimbalDynamics for comparison
    gimbal = GimbalDynamics(
        pan_mass=config['pan_mass'],
        tilt_mass=config['tilt_mass'],
        cm_r=config['cm_r'],
        cm_h=config['cm_h'],
        gravity=config['gravity']
    )
    
    print("\n1. Testing Mass Matrix Computation")
    print("-" * 80)
    q_test = np.array([0.1, 0.2])
    M_gimbal = gimbal.get_mass_matrix(q_test)
    M_ekf = ekf.gimbal_dynamics.get_mass_matrix(q_test)
    
    print(f"Mass Matrix from GimbalDynamics:\n{M_gimbal}")
    print(f"Mass Matrix from EKF's GimbalDynamics:\n{M_ekf}")
    print(f"Matrix difference norm: {np.linalg.norm(M_gimbal - M_ekf):.2e}")
    
    if np.allclose(M_gimbal, M_ekf):
        print("✓ PASS: Mass matrices match")
    else:
        print("✗ FAIL: Mass matrices differ")
    
    print("\n2. Testing Coriolis Matrix Computation")
    print("-" * 80)
    dq_test = np.array([0.1, 0.05])
    C_gimbal = gimbal.get_coriolis_matrix(q_test, dq_test)
    C_ekf = ekf.gimbal_dynamics.get_coriolis_matrix(q_test, dq_test)
    
    print(f"Coriolis Matrix from GimbalDynamics:\n{C_gimbal}")
    print(f"Coriolis Matrix from EKF's GimbalDynamics:\n{C_ekf}")
    print(f"Matrix difference norm: {np.linalg.norm(C_gimbal - C_ekf):.2e}")
    
    if np.allclose(C_gimbal, C_ekf):
        print("✓ PASS: Coriolis matrices match")
    else:
        print("✗ FAIL: Coriolis matrices differ")
    
    print("\n3. Testing Gravity Vector Computation")
    print("-" * 80)
    G_gimbal = gimbal.get_gravity_vector(q_test)
    G_ekf = ekf.gimbal_dynamics.get_gravity_vector(q_test)
    
    print(f"Gravity Vector from GimbalDynamics: {G_gimbal}")
    print(f"Gravity Vector from EKF's GimbalDynamics: {G_ekf}")
    print(f"Vector difference norm: {np.linalg.norm(G_gimbal - G_ekf):.2e}")
    
    if np.allclose(G_gimbal, G_ekf):
        print("✓ PASS: Gravity vectors match")
    else:
        print("✗ FAIL: Gravity vectors differ")
    
    print("\n4. Testing EKF Prediction Step")
    print("-" * 80)
    
    # Control input
    tau = np.array([0.01, -0.005])
    dt = 0.001  # 1 ms time step
    
    # Store initial state
    x_initial = ekf.x_hat.copy()
    P_initial = ekf.P.copy()
    
    print(f"Initial state (first 6 elements): {x_initial[:6]}")
    print(f"Control torques: {tau}")
    print(f"Time step: {dt} s")
    
    # Run prediction
    ekf.predict(tau, dt)
    
    x_predicted = ekf.x_hat.copy()
    P_predicted = ekf.P.copy()
    
    print(f"Predicted state (first 6 elements): {x_predicted[:6]}")
    print(f"State change: {x_predicted[:6] - x_initial[:6]}")
    
    # Check that state changed (not stuck at initial)
    if not np.allclose(x_predicted, x_initial):
        print("✓ PASS: State propagated (not stuck)")
    else:
        print("✗ FAIL: State did not change")
    
    # Check covariance increased (prediction uncertainty)
    trace_initial = np.trace(P_initial)
    trace_predicted = np.trace(P_predicted)
    print(f"Covariance trace before: {trace_initial:.6e}")
    print(f"Covariance trace after: {trace_predicted:.6e}")
    
    if trace_predicted > trace_initial:
        print("✓ PASS: Covariance increased (as expected)")
    else:
        print("✗ FAIL: Covariance did not increase")
    
    print("\n5. Testing Process Jacobian Structure")
    print("-" * 80)
    
    # Compute Jacobian
    F = ekf._compute_process_jacobian(dt, dq_test[0], dq_test[1])
    
    print(f"Jacobian shape: {F.shape}")
    print(f"Jacobian (first 6x6 block):\n{F[:6, :6]}")
    
    # Check that Jacobian is close to identity (for small dt)
    F_deviation = F - np.eye(ekf.n_states)
    max_deviation = np.max(np.abs(F_deviation))
    print(f"Max deviation from identity: {max_deviation:.6e}")
    
    # For small dt, deviation should be small
    if max_deviation < 0.1:  # Reasonable threshold
        print("✓ PASS: Jacobian close to identity for small dt")
    else:
        print("⚠ WARNING: Large Jacobian deviation (may need smaller dt)")
    
    # Check coupling terms exist (non-zero off-diagonal elements)
    # Az velocity should depend on El angle (M(q) coupling)
    coupling_az_el = F[1, 3]  # ∂θ̇_az/∂θ_el
    print(f"Coupling term ∂θ̇_az/∂θ_el: {coupling_az_el:.6e}")
    
    if abs(coupling_az_el) > 1e-10:
        print("✓ PASS: Az-El coupling detected (nonlinear dynamics)")
    else:
        print("⚠ INFO: Weak Az-El coupling at this configuration")
    
    print("\n6. Testing Multiple Prediction Steps (Stability)")
    print("-" * 80)
    
    # Reset EKF
    ekf.reset()
    
    # Run multiple prediction steps
    n_steps = 100
    tau = np.array([0.001, -0.0005])
    dt = 0.001
    
    states_history = []
    for i in range(n_steps):
        ekf.predict(tau, dt)
        states_history.append(ekf.x_hat.copy())
    
    states_history = np.array(states_history)
    
    print(f"Ran {n_steps} prediction steps")
    print(f"Final state (first 6): {states_history[-1, :6]}")
    
    # Check for NaN or Inf
    if np.all(np.isfinite(states_history)):
        print("✓ PASS: All states finite (no numerical instability)")
    else:
        print("✗ FAIL: NaN or Inf detected")
    
    # Check covariance remains positive definite
    eigenvalues = np.linalg.eigvals(ekf.P)
    min_eigenvalue = np.min(eigenvalues)
    print(f"Minimum covariance eigenvalue: {min_eigenvalue:.6e}")
    
    if min_eigenvalue > 0:
        print("✓ PASS: Covariance remains positive definite")
    else:
        print("✗ FAIL: Covariance lost positive definiteness")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_ekf_dynamics_consistency()
