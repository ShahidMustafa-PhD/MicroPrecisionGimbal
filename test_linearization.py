#!/usr/bin/env python3
"""
Test script for GimbalDynamics.linearize() method validation.

This script performs comprehensive testing of the linearization:
1. Numerical accuracy verification
2. Coupling analysis at various operating points
3. Comparison with analytical derivatives (where applicable)
4. State-space model properties validation
"""

import numpy as np
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


def test_linearization_at_operating_point(gimbal: GimbalDynamics, q_op: np.ndarray, dq_op: np.ndarray):
    """Test linearization at a specific operating point."""
    print(f"\n{'='*80}")
    print(f"Testing linearization at q={q_op}, dq={dq_op}")
    print(f"{'='*80}")
    
    # Perform linearization
    A, B, C, D = gimbal.linearize(q_op, dq_op)
    
    print("\n1. STATE MATRIX A (4x4):")
    print("-" * 80)
    print(A)
    print(f"\nEigenvalues of A: {np.linalg.eigvals(A)}")
    
    # Check for instability
    eigs = np.linalg.eigvals(A)
    if np.any(np.real(eigs) > 0):
        print("⚠  WARNING: System is unstable (eigenvalues with positive real part)")
    else:
        print("✓  System is stable (all eigenvalues have negative/zero real part)")
    
    print("\n2. INPUT MATRIX B (4x2):")
    print("-" * 80)
    print(B)
    
    # Verify B structure: B = [0; M^{-1}]
    M_op = gimbal.get_mass_matrix(q_op)
    M_inv = np.linalg.inv(M_op)
    
    print("\n  Analytical M^{-1} (should match B[2:4,:]):")
    print(M_inv)
    print(f"\n  Difference: {np.linalg.norm(M_inv - B[2:4, :]):.2e}")
    
    if np.allclose(M_inv, B[2:4, :], atol=1e-5):
        print("✓  B matrix lower block matches M^{-1} (analytical)")
    else:
        print("⚠  B matrix lower block differs from M^{-1}")
    
    print("\n3. OUTPUT MATRIX C (2x4):")
    print("-" * 80)
    print(C)
    print("  Expected: [I_{2x2}, 0_{2x2}] for position measurement")
    
    C_expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    if np.allclose(C, C_expected):
        print("✓  C matrix is correct for position measurement")
    else:
        print("⚠  C matrix differs from expected")
    
    print("\n4. FEEDTHROUGH MATRIX D (2x2):")
    print("-" * 80)
    print(D)
    print("  Expected: 0_{2x2} for mechanical systems")
    
    if np.allclose(D, np.zeros((2, 2))):
        print("✓  D matrix is zero (no feedthrough)")
    else:
        print("⚠  D matrix is non-zero (unexpected)")
    
    print("\n5. COUPLING ANALYSIS:")
    print("-" * 80)
    
    # Extract coupling terms from A matrix
    # A has structure:
    # [  0       0      1      0  ]
    # [  0       0      0      1  ]
    # [ ∂a₁/∂q₁ ∂a₁/∂q₂ ∂a₁/∂v₁ ∂a₁/∂v₂ ]
    # [ ∂a₂/∂q₁ ∂a₂/∂q₂ ∂a₂/∂v₁ ∂a₂/∂v₂ ]
    
    # Position coupling
    pos_coupling_12 = A[2, 1]  # Effect of tilt position on pan acceleration
    pos_coupling_21 = A[3, 0]  # Effect of pan position on tilt acceleration
    
    # Velocity coupling
    vel_coupling_12 = A[2, 3]  # Effect of tilt velocity on pan acceleration
    vel_coupling_21 = A[3, 2]  # Effect of pan velocity on tilt acceleration
    
    print(f"  Position coupling:")
    print(f"    ∂ä_pan/∂q_tilt  = {pos_coupling_12:+.6f}")
    print(f"    ∂ä_tilt/∂q_pan  = {pos_coupling_21:+.6f}")
    
    print(f"\n  Velocity coupling:")
    print(f"    ∂ä_pan/∂v_tilt  = {vel_coupling_12:+.6f}")
    print(f"    ∂ä_tilt/∂v_pan  = {vel_coupling_21:+.6f}")
    
    # Overall coupling metric
    coupling_metric = (abs(pos_coupling_12) + abs(pos_coupling_21) + 
                      abs(vel_coupling_12) + abs(vel_coupling_21))
    
    print(f"\n  Total coupling metric: {coupling_metric:.6f}")
    
    if coupling_metric > 0.5:
        print("  ⚠  HIGH COUPLING: Consider MIMO control design or decoupling compensation")
    elif coupling_metric > 0.1:
        print("  ⚠  MODERATE COUPLING: Monitor performance, may need cross-coupling feedforward")
    else:
        print("  ✓  LOW COUPLING: Decentralized SISO control is acceptable")
    
    print("\n6. CONTROLLABILITY & OBSERVABILITY:")
    print("-" * 80)
    
    # Controllability matrix
    Qc = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    rank_c = np.linalg.matrix_rank(Qc)
    print(f"  Controllability matrix rank: {rank_c}/4")
    
    if rank_c == 4:
        print("  ✓  System is controllable")
    else:
        print("  ⚠  System is NOT fully controllable")
    
    # Observability matrix
    Qo = np.vstack([C, C @ A, C @ A @ A, C @ A @ A @ A])
    rank_o = np.linalg.matrix_rank(Qo)
    print(f"  Observability matrix rank: {rank_o}/4")
    
    if rank_o == 4:
        print("  ✓  System is observable")
    else:
        print("  ⚠  System is NOT fully observable")
    
    return A, B, C, D


def test_linearization_accuracy(gimbal: GimbalDynamics):
    """Test numerical accuracy of linearization via finite difference verification."""
    print(f"\n{'='*80}")
    print("NUMERICAL ACCURACY VERIFICATION")
    print(f"{'='*80}")
    
    # Operating point
    q_op = np.array([0.1, 0.2])
    dq_op = np.array([0.05, -0.03])
    
    # Get linearized model
    A, B, C, D = gimbal.linearize(q_op, dq_op)
    
    # Test prediction accuracy: compare linear vs nonlinear response to small perturbations
    print("\nTesting linearization prediction accuracy...")
    
    epsilon = 1e-4  # Small perturbation
    tau = np.array([0.01, -0.02])  # Small input torque
    
    # Nonlinear dynamics at operating point
    x_op = np.concatenate([q_op, dq_op])
    f_op = gimbal.state_space_derivative(0.0, x_op, tau)
    
    # Test perturbations
    perturbations = [
        ('q_pan', np.array([epsilon, 0, 0, 0])),
        ('q_tilt', np.array([0, epsilon, 0, 0])),
        ('dq_pan', np.array([0, 0, epsilon, 0])),
        ('dq_tilt', np.array([0, 0, 0, epsilon]))
    ]
    
    print("\nPerturbation tests:")
    for name, dx in perturbations:
        # Nonlinear response
        x_pert = x_op + dx
        f_pert = gimbal.state_space_derivative(0.0, x_pert, tau)
        df_nonlinear = f_pert - f_op
        
        # Linear approximation
        df_linear = A @ dx
        
        # Error
        error = np.linalg.norm(df_nonlinear - df_linear)
        relative_error = error / (np.linalg.norm(df_nonlinear) + 1e-10)
        
        print(f"  {name:10s}: error = {error:.2e}, relative = {relative_error:.2e}")
    
    print("\n✓  If relative errors < 1e-3, linearization is accurate for small perturbations")


def main():
    """Run comprehensive linearization tests."""
    print("\n" + "="*80)
    print("GIMBAL DYNAMICS LINEARIZATION VALIDATION")
    print("="*80)
    
    # Create gimbal instance
    gimbal = GimbalDynamics(
        pan_mass=0.5,
        tilt_mass=0.25,
        cm_r=0.0,
        cm_h=0.0,
        gravity=9.81
    )
    
    # Test 1: Upright position (q = [45°, 45°])
    test_linearization_at_operating_point(
        gimbal,
        q_op=np.array([45, 45])*np.pi/180,
        dq_op=np.array([0.0, 0.0])
    )
    
    # Test 2: High tilt angle (q = [0, 60°])
    test_linearization_at_operating_point(
        gimbal,
        q_op=np.array([0.0, np.deg2rad(60)]),
        dq_op=np.array([0.0, 0.0])
    )
    
    # Test 3: General configuration with velocity
    test_linearization_at_operating_point(
        gimbal,
        q_op=np.array([np.deg2rad(30), np.deg2rad(45)]),
        dq_op=np.array([0.1, -0.05])
    )
    
    # Test 4: Numerical accuracy
    test_linearization_accuracy(gimbal)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
