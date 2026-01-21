import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Any

class GimbalDynamics:
    """
    High-fidelity Lagrangian dynamics model for a 2-DOF Pan-Tilt gimbal system.
    
    System Description:
    - The system consists of a Pan frame (azimuth) and a Tilt frame (elevation).
    - Pan axis corresponds to the global Z-axis (yaw).
    - Tilt axis is perpendicular to the Pan axis.
    - Captures mass imbalances (static and dynamic) and coupled inertia terms.
    
    Lagrangian Formulation:
    M(q) * q_dd + C(q, q_d) * q_d + G(q) = tau
    
    where:
    - q: Joint position vector [pan_angle, tilt_angle] (rad)
    - q_d: Joint velocity vector (rad/s)
    - q_dd: Joint acceleration vector (rad/s^2)
    - M(q): Inertia Matrix (2x2)
    - C(q, q_d): Coriolis and Centrifugal Matrix (2x2)
    - G(q): Gravity Vector (2,)
    - tau: Control torques [tau_pan, tau_tilt] (Nm)
    
    Coordinates:
    - q[0]: Pan angle (rotation about Z_0)
    - q[1]: Tilt angle (rotation about Y_1, which is rotated by q[0])
    """

    def __init__(self, 
                 pan_mass: float = 0.5, 
                 tilt_mass: float = 0.25,
                 cm_r: float = 0.002,
                 cm_h: float = 0.005,
                 gravity: float = 9.81):
        """
        Initialize the Gimbal Dynamics model with physical parameters.

        Args:
            pan_mass (float): Mass of the pan assembly (moving azimuth part) [kg]
            tilt_mass (float): Mass of the tilt assembly (optical payload) [kg]
            cm_r (float): Longitudinal offset of Tilt CM from tilt axis [m]
            cm_h (float): Lateral offset of Tilt CM from tilt axis [m]
            gravity (float): Gravitational acceleration [m/s^2]. Default 9.81.
        """
        # Mass properties
        self.m1 = pan_mass
        self.m2 = tilt_mass
        self.g = gravity

        # Unbalance parameters (Tilt body CM offset)
        # We assume the tilt axis is Y. Visual axis is Z.
        # r is longitudinal (along Z), h is lateral (along X).
        self.r = cm_r
        self.h = cm_h

        # Pre-calculate constant products for efficiency
        # Moment arms
        self.m2_r = self.m2 * self.r
        self.m2_h = self.m2 * self.h
        
        # Approximate Inertia Tensors (Diagonal approximations for base structure)
        # Using a simple box model approximation for reasonable default moments of inertia
        # I = m * l^2 / 12 approximately. Assuming characteristic length ~0.1m
        l_char = 0.1
        i_xx = (1/12) * self.m1 * l_char**2
        i_yy = (1/12) * self.m1 * l_char**2
        i_zz = (1/12) * self.m1 * l_char**2
        
        self.dim_pan = 0.1 # 10cm radius approx
        self.dim_tilt = 0.08 # 8cm approx
        
        # Pan Inertia (I1) - Fixed to Pan Link
        self.I1_zz = 0.5 * self.m1 * (self.dim_pan ** 2) # Cylinder approx
        
        # Tilt Inertia (I2) - Fixed to Tilt Link
        # Defined around the TILT AXIS joint frame, including parallel axis theorem shifts if needed.
        # Assuming the inputs are principal moments at CM, we shift them or define them at the joint.
        # Here we define effective inertias at the joint for simplicity of the Lagrangian terms typically found in standard libraries.
        
        # I2_xx: Inertia about X axis (Lateral)
        # I2_yy: Inertia about Y axis (Tilt Axis)
        # I2_zz: Inertia about Z axis (Optical Axis)
        # Approximating a cylinder/box payload
        self.I2_xx = (1/12) * self.m2 * (self.dim_tilt**2 + (2*self.dim_tilt)**2) + self.m2 * (self.r**2)
        self.I2_yy = (1/12) * self.m2 * (self.dim_tilt**2 + (2*self.dim_tilt)**2) + self.m2 * (self.r**2 + self.h**2)
        self.I2_zz = 0.5 * self.m2 * self.dim_tilt**2 + self.m2 * (self.h**2)

        # Cross terms (Products of inertia) - assuming small/negligible for this simplified model
        # unless specifically requested. For a high-fidelity model, non-diagonal terms of I2 are relevant.
        # Let's keep Ixy, Iyz, Izx as small or zero for this standard setup unless implied by unbalance.
    # However, the physical unbalance r, h implies the Principal Axes are not aligned with Joint Axes.
    # We handle this by calculating the M(q) matrix explicitly with geometric transforms.

    def get_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        theta_pan, theta_tilt = q
        c2 = np.cos(theta_tilt)
        s2 = np.sin(theta_tilt)
    
    # 1. M11: Pan Axis Inertia
    # Projection of the tilt body inertia tensor onto the pan Z-axis
        I2_proj = self.I2_xx * (s2**2) + self.I2_zz * (c2**2)
    
    # CM position in Pan Frame (assuming tilt axis is at origin of pan frame)
    # x_pos is the lever arm for the Z-axis rotation
        x_pos = self.h * c2 + self.r * s2
        dist_sq_pan = x_pos**2
    
        m11 = self.I1_zz + I2_proj + self.m2 * dist_sq_pan
    
    # 2. M22: Tilt Axis Inertia (Parallel Axis Theorem applied here!)
    # I_pivot = I_cm + m * dist^2
        m22 = self.I2_yy + self.m2 * (self.r**2 + self.h**2)
    
    # 3. M12 / M21: Coupling (Dynamic Unbalance)
    # For LaserCom, we include the product of inertia term
    # z_pos = -h*s2 + r*c2
        m12 = self.m2 * (x_pos * (-self.h * s2 + self.r * c2))
        m21 = m12

        return np.array([[m11, m12], [m21, m22]])

    def get_coriolis_matrix(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Calculate the Coriolis and Centrifugal Matrix C(q, dq).
        
        Computed using Christoffel symbols of the first kind from M(q).
        C_kj = sum_i( c_ijk * dq_i )
        c_ijk = 0.5 * (dM_kj/dqi + dM_ki/dqj - dM_ij/dqk)
        
        Args:
            q (np.ndarray): Joint position vector.
            dq (np.ndarray): Joint velocity vector.

        Returns:
            np.ndarray: 2x2 Coriolis Matrix.
        """
        # Numerical differentiation or analytical? Analytical is faster/cleaner here given the relatively simple M.
        # Let's perform a lightweight symbolic-to-code derivation for M11 derivative.
        
        theta_tilt = q[1]
        d_theta_tilt = dq[1] # q2_dot
        
        c2 = np.cos(theta_tilt)
        s2 = np.sin(theta_tilt)
        
        # M11 = I1_zz + I2_xx*s^2 + I2_zz*c^2 + m2*(h*c + r*s)^2
        # dM11/dq2 needed. dM11/dq1 = 0.
        # dM22/dq = 0. M12 = 0.
        
        # Derivative of trig terms
        # d(s^2)/dq2 = 2s*c
        # d(c^2)/dq2 = -2s*c
        term1 = self.I2_xx * (2*s2*c2) - self.I2_zz * (2*s2*c2)
        
        # Derivative of (h*c + r*s)^2
        # = 2*(h*c + r*s) * (-h*s + r*c)
        term2 = self.m2 * 2 * (self.h * c2 + self.r * s2) * (self.r * c2 - self.h * s2)
        
        dM11_dq2 = term1 + term2
        
        # Christoffel Symbols
        # c111 = 0.5 * dM11/dq1 = 0
        # c121 = 0.5 * (dM11/dq2 + dM12/dq1 - dM12/dq1) = 0.5 * dM11_dq2
        # c211 = 0.5 * (dM11/dq2 + dM21/dq1 - dM11/dq2) ... wait.
        # c_ijk formula: 0.5 * (dM_kj/dqi + dM_ki/dqj - dM_ij/dqk)
        # Target: C * dq.  C_kj = sum_i c_ijk * dq_i. 
        # Row 1 (k=1):
        # C11 = c_111*dq1 + c_121*dq2
        # c_111 = 0.5 * dM11/dq1 = 0
        # c_121 = 0.5 * (dM11/dq2 + dM12/dq1 - dM12/dq1) = 0.5 * dM11_dq2
        # c_211 = 0.5 * (dM12/dq1 + dM12/dq1 - dM11/dq2) = -0.5 * dM11_dq2
        
        # Row 2 (k=2):
        # C21 = c_112*dq1 + c_122*dq2
        # c_112 = 0.5 * (dM11/dq2 + dM21/dq1 - dM11/dq2) ? No.
        # c_kij formula check:
        # C_kj = sum over i of Gamma_kij * dq_i
        # Gamma_ijk = 0.5 * (dM_ij/dqk + dM_ik/dqj - dM_jk/dqi) ?? No.
        # Standard: c_kj = sum_i 0.5 * (dM_kj/dqi + dM_ki/dqj - dM_ij/dqk) * dq_i
        
        # Let's simplify:
        # k=1 (Eq 1):
        # j=1: sum_i (dM11/dqi + dM1i/dq1 - dM_i1/dqk ... )
        # Using simpler form:
        # C11 = 0.5 * dM11/dq1 * dq1 + dM11/dq2 * dq2  (Wrong, C is matrix)
        # C matrix elements:
        # C11 = c111*q1' + c121*q2' = 0 + 0.5*dM11_dq2 * q2'
        # C12 = c211*q1' + c221*q2' = 0.5*dM11_dq2 * q1' + 0
        
        # C21 = c112*q1' + c122*q2' = -0.5*dM11_dq2 * q1' + 0
        # C22 = c212*q1' + c222*q2' = 0 + 0
        
        val = 0.5 * dM11_dq2
        
        # Note: factorization of C is not unique. This satisfies q' (M_dot - 2C) q = 0 skew symmetry check usually.
        # Consistent Matrix Form:
        C = np.array([
            [val * dq[1], val * dq[0]],
            [-val * dq[0], 0.0]
        ])
        
        return C

    def get_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate the Gravity Vector G(q).
        
        G(q) = dV/dq.
        
        Args:
            q (np.ndarray): Joint position vector.

        Returns:
            np.ndarray: 2 element Gravity vector [g_pan, g_tilt].
        """
        theta_tilt = q[1]
        
        # Gravity Torque Logic:
        # We assume q[1]=0 corresponds to the optical payload being horizontal.
        # The CM is offset by r (longitudinal, along optical axis) and h (lateral).
        # Gravity tries to pull the heavy side (r) down.
        # Torque = m * g * r * cos(theta) (Max at horizon, zero at vertical)
        # Torque due to h (if perp to r): m * g * h * sin(theta) (Zero at horizon, max at vertical)
        # Resulting G vector entry matches dV/dq.
        
        # Calculate restoring torque
        g_val = self.m2 * self.g * (self.r * np.cos(theta_tilt) - self.h * np.sin(theta_tilt))
        
        return np.array([0.0, g_val])

    def compute_forward_dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Compute joint accelerations q_dd given state and torque.
        
        Solves M(q) * q_dd = tau - C(q,dq)dq - G(q)
        
        Args:
            q (np.ndarray): Joint positions [rad]
            dq (np.ndarray): Joint velocities [rad/s]
            tau (np.ndarray): Input torques [Nm]

        Returns:
            np.ndarray: Joint accelerations [rad/s^2]
        """
        M = self.get_mass_matrix(q)
        C = self.get_coriolis_matrix(q, dq)
        G = self.get_gravity_vector(q)
        
        # Equation: M * q_dd + C * dq + G = tau
        # M * q_dd = tau - C * dq - G
        rhs = tau - (C @ dq) - G
        
        # Solve linear system for numerical stability (better than inv(M))
        q_dd = np.linalg.solve(M, rhs)
        
        return q_dd

    def state_space_derivative(self, t: float, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Compute derivative of state vector X = [q, dq] for ODE solvers.
        
        Args:
            t (float): Time (s) - required by standard ODE solvers
            state (np.ndarray): State vector [q1, q2, dq1, dq2]
            tau (np.ndarray): Control torque [tau1, tau2]

        Returns:
            np.ndarray: State derivative [dq1, dq2, ddq1, ddq2]
        """
        # Unpack state
        n = 2
        q = state[0:n]
        dq = state[n:2*n]
        
        # Compute acceleration
        ddq = self.compute_forward_dynamics(q, dq, tau)
        
        # Pack derivative
        d_state = np.concatenate([dq, ddq])
        
        return d_state

    def linearize(self, q_op: np.ndarray, dq_op: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the gimbal dynamics around an operating point to obtain state-space matrices.
        
        Mathematical Formulation:
        -------------------------
        The nonlinear dynamics are:
        
        $$M(q) \\ddot{q} + C(q, \\dot{q}) \\dot{q} + G(q) = \\tau$$
        
        Define state vector: $x = [q_1, q_2, \\dot{q}_1, \\dot{q}_2]^T \\in \\mathbb{R}^4$
        Define input vector: $u = [\\tau_1, \\tau_2]^T \\in \\mathbb{R}^2$
        
        The state-space form is:
        
        $$\\dot{x} = f(x, u)$$
        
        where:
        
        $$f(x, u) = \\begin{bmatrix} \\dot{q} \\\\ M(q)^{-1}[\\tau - C(q,\\dot{q})\\dot{q} - G(q)] \\end{bmatrix}$$
        
        Linearization around $(x_0, u_0)$:
        
        $$\\Delta \\dot{x} = A \\Delta x + B \\Delta u$$
        
        where:
        
        $$A = \\frac{\\partial f}{\\partial x}\\bigg|_{x_0, u_0}, \\quad B = \\frac{\\partial f}{\\partial u}\\bigg|_{x_0, u_0}$$
        
        The output equation (measuring joint positions):
        
        $$y = C x + D u$$
        
        where $C = [I_{2x2}, 0_{2x2}]$ maps states to measured positions.
        
        Jacobian Structure:
        ------------------
        
        $$A = \\begin{bmatrix}
        0_{2x2} & I_{2x2} \\\\
        \\frac{\\partial \\ddot{q}}{\\partial q} & \\frac{\\partial \\ddot{q}}{\\partial \\dot{q}}
        \\end{bmatrix}_{4x4}$$
        
        $$B = \\begin{bmatrix}
        0_{2x2} \\\\
        M(q_0)^{-1}
        \\end{bmatrix}_{4x2}$$
        
        Cross-Coupling Analysis:
        -----------------------
        The off-diagonal terms in A reveal dynamic coupling between Pan and Tilt axes.
        At high tilt angles (|q_2| > 45°), significant coupling may require:
        - Cross-coupling feedforward compensation
        - Decoupling control strategies
        - MIMO control design (H-infinity, LQR)
        
        Args:
            q_op (np.ndarray): Operating point for joint positions [rad] (2,)
            dq_op (np.ndarray): Operating point for joint velocities [rad/s] (2,)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - A (4x4): State matrix
                - B (4x2): Input matrix
                - C (2x4): Output matrix (measures positions)
                - D (2x2): Feedthrough matrix (zero for mechanical systems)
        
        Example:
            >>> gimbal = GimbalDynamics()
            >>> q0 = np.array([0.0, 0.0])  # Upright position
            >>> dq0 = np.array([0.0, 0.0])  # At rest
            >>> A, B, C, D = gimbal.linearize(q0, dq0)
            >>> print(f"A matrix shape: {A.shape}")
            A matrix shape: (4, 4)
        """
        # Validate inputs
        assert q_op.shape == (2,), f"q_op must be (2,), got {q_op.shape}"
        assert dq_op.shape == (2,), f"dq_op must be (2,), got {dq_op.shape}"
        
        # Operating point state and input
        x_op = np.concatenate([q_op, dq_op])

        # IMPORTANT: linearize about a dynamically consistent operating point.
        # For mechanical systems this means choosing u0 such that q̈=0 at (q0, q̇0):
        #   tau_op = C(q0,q̇0) q̇0 + G(q0)
        # (so that M(q0) q̈ = 0).
        tau_op = (self.get_coriolis_matrix(q_op, dq_op) @ dq_op) + self.get_gravity_vector(q_op)

        # Numerical differentiation parameters
        # 1e-7 is often too small for double-precision finite differences once nonlinear
        # trig + matrix solves are involved.
        epsilon = 1e-6
        
        # Initialize Jacobian matrices
        n_states = 4
        n_inputs = 2
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        
        # ====================================================================
        # COMPUTE A MATRIX: ∂f/∂x
        # ====================================================================
        # Use central difference: df/dx ≈ [f(x+ε) - f(x-ε)] / (2ε)
        
        for i in range(n_states):
            # Perturb state forward
            x_plus = x_op.copy()
            x_plus[i] += epsilon
            f_plus = self.state_space_derivative(0.0, x_plus, tau_op)
            
            # Perturb state backward
            x_minus = x_op.copy()
            x_minus[i] -= epsilon
            f_minus = self.state_space_derivative(0.0, x_minus, tau_op)
            
            # Central difference
            A[:, i] = (f_plus - f_minus) / (2.0 * epsilon)
        
        # ====================================================================
        # COMPUTE B MATRIX: ∂f/∂u
        # ====================================================================
        # B is simpler: upper half is zero, lower half is M(q)^{-1}
        
        for j in range(n_inputs):
            # Perturb input forward
            tau_plus = tau_op.copy()
            tau_plus[j] += epsilon
            f_plus = self.state_space_derivative(0.0, x_op, tau_plus)
            
            # Perturb input backward
            tau_minus = tau_op.copy()
            tau_minus[j] -= epsilon
            f_minus = self.state_space_derivative(0.0, x_op, tau_minus)
            
            # Central difference
            B[:, j] = (f_plus - f_minus) / (2.0 * epsilon)
        
        # ====================================================================
        # COMPUTE C MATRIX: Output equation y = [q1, q2]
        # ====================================================================
        # For position measurement: y = [q1, q2] = [I 0] * [q; dq]
        C = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # ====================================================================
        # COMPUTE D MATRIX: Feedthrough (zero for mechanical systems)
        # ====================================================================
        D = np.zeros((2, 2))
        
        # ====================================================================
        # ANALYTICAL VERIFICATION (Optional Sanity Check)
        # ====================================================================
        # For debugging: compute M^{-1} analytically and compare with B[2:4, :]
        M_op = self.get_mass_matrix(q_op)
        M_inv = np.linalg.inv(M_op)
        
        # B should have structure: [0; M^{-1}]
        analytical_B_lower = M_inv
        numerical_B_lower = B[2:4, :]
        
        if not np.allclose(analytical_B_lower, numerical_B_lower, atol=1e-5):
            print("WARNING: Numerical B matrix lower block differs from analytical M^{-1}")
            print(f"Analytical M^{{-1}}:\n{analytical_B_lower}")
            print(f"Numerical B[2:4,:]:\n{numerical_B_lower}")
        
        # ====================================================================
        # COUPLING ANALYSIS
        # ====================================================================
        # Check off-diagonal terms in A to assess coupling strength
        coupling_metric = np.abs(A[2, 3]) + np.abs(A[3, 2])  # Cross-velocity coupling
        
        if coupling_metric > 0.1:  # Arbitrary threshold
            print(f"INFO: Significant cross-axis coupling detected (metric: {coupling_metric:.3f})")
            print("Consider decoupling compensation or MIMO control design.")
        
        return A, B, C, D

if __name__ == "__main__":
    # Sanity Check
    print("Running GimbalDynamics Sanity Check...")
    
    gimbal = GimbalDynamics()
    
    # Test Configuration
    q_test = np.array([0.1, 0.5]) # Arbitrary angles
    dq_test = np.array([0.1, 0.0])
    tau_test = np.array([0.0, 0.0])
    
    # 1. Check Mass Matrix Positive Definiteness
    M = gimbal.get_mass_matrix(q_test)
    eigenvalues = np.linalg.eigvals(M)
    
    print(f"Mass Matrix at q={q_test}:\n{M}")
    print(f"Eigenvalues: {eigenvalues}")
    
    if np.all(eigenvalues > 0):
        print("PASS: Mass Matrix is Positive Definite.")
    else:
        print("FAIL: Mass Matrix is NOT Positive Definite.")
        
    # 2. Check Gravity at Horizon
    # If q=[0,0], r implies torque.
    G_horiz = gimbal.get_gravity_vector(np.array([0.0, 0.0]))
    print(f"Gravity Vector at Horizon: {G_horiz}")
    # Expected: [0, m2*g*r] approximately
    expected_g = gimbal.m2 * gimbal.g * gimbal.r
    if np.isclose(G_horiz[1], expected_g):
        print(f"PASS: Gravity torque matches expected mg*r ({expected_g:.4f}).")
    else:
        print(f"FAIL: Gravity torque {G_horiz[1]} != expected {expected_g}.")
        
    # 3. Forward Dynamics check
    accel = gimbal.compute_forward_dynamics(q_test, dq_test, tau_test)
    print(f"Forward Dynamics Accel: {accel}")
