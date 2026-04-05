import numpy as np
from typing import Tuple, Optional

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
    - q[1] = 0 assumes the optical payload (Z_2 axis) is pointing at the horizon.
    """

    def __init__(self, 
                 pan_mass: float = 0.5, 
                 tilt_mass: float = 0.25,
                 cm_r: float = 0.0,
                 cm_h: float = 0.0,
                 gravity: float = 9.81,
                 I1_matrix: Optional[np.ndarray] = None,
                 I2_matrix: Optional[np.ndarray] = None):
        """
        Initialize the Gimbal Dynamics model with physical parameters.

        Args:
            pan_mass (float): Mass of the pan assembly (moving azimuth part) [kg]
            tilt_mass (float): Mass of the tilt assembly (optical payload) [kg]
            cm_r (float): Longitudinal offset of Tilt CM from tilt axis [m]
            cm_h (float): Lateral offset of Tilt CM from tilt axis [m]
            gravity (float): Gravitational acceleration [m/s^2]. Default 9.81.
            I1_matrix (Optional[np.ndarray]): 3x3 Inertia tensor for the Pan assembly. 
                                              If provided, overrides default approximations.
            I2_matrix (Optional[np.ndarray]): 3x3 Inertia tensor for the Tilt assembly. 
                                              If provided, overrides default approximations.
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
        
        # Dimensions for default approximations
        self.dim_pan = 0.1 # 10cm radius approx
        self.dim_tilt = 0.08 # 8cm approx
        
        # --- PAN INERTIA CONFIGURATION ---
        if I1_matrix is not None:
            self.I1_zz = I1_matrix[2, 2]
        else:
            # Pan Inertia (I1) - Fixed to Pan Link
            self.I1_zz = 0.5 * self.m1 * (self.dim_pan ** 2) # Cylinder approx

        # --- TILT INERTIA CONFIGURATION ---
        if I2_matrix is not None:
            self.I2_xx = I2_matrix[0, 0]
            self.I2_yy = I2_matrix[1, 1]
            self.I2_zz = I2_matrix[2, 2]
        else:
            # Tilt Inertia (I2) - Fixed to Tilt Link
            # Defined around the TILT AXIS joint frame, including parallel axis theorem shifts if needed.
            self.I2_xx = (1/12) * self.m2 * (self.dim_tilt**2 + (2*self.dim_tilt)**2) + self.m2 * (self.r**2)
            self.I2_yy = (1/12) * self.m2 * (self.dim_tilt**2 + (2*self.dim_tilt)**2) + self.m2 * (self.r**2 + self.h**2)
            self.I2_zz = 0.5 * self.m2 * self.dim_tilt**2 + self.m2 * (self.h**2)

        # Cross terms (Products of inertia) - assuming small/negligible for this simplified model
        # unless specifically requested. For a high-fidelity model, non-diagonal terms of I2 are relevant.

    def get_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        theta_pan, theta_tilt = q
        c2 = np.cos(theta_tilt)
        s2 = np.sin(theta_tilt)
    
        # 1. M11: Pan Axis Inertia
        # FIXED KINEMATIC CONTRADICTION: At horizon (q2=0), the vertical Pan axis sees I2_xx.
        I2_proj = self.I2_xx * (c2**2) + self.I2_zz * (s2**2)
    
        # CM position in Pan Frame (assuming tilt axis is at origin of pan frame)
        # x_pos is the lever arm for the Z-axis rotation (horizontal distance from pan axis)
        x_pos = self.h * c2 + self.r * s2
        dist_sq_pan = x_pos**2
    
        m11 = self.I1_zz + I2_proj + self.m2 * dist_sq_pan
    
        # 2. M22: Tilt Axis Inertia (Parallel Axis Theorem applied here!)
        # I_pivot = I_cm + m * dist^2
        m22 = self.I2_yy + self.m2 * (self.r**2 + self.h**2)
    
        # 3. M12 / M21: Coupling (Dynamic Unbalance)
        # For LaserCom, we include the product of inertia term
        z_pos = -self.h * s2 + self.r * c2
        m12 = self.m2 * (x_pos * z_pos)
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
        theta_tilt = q[1]
        
        c2 = np.cos(theta_tilt)
        s2 = np.sin(theta_tilt)
        
        # M11 = I1_zz + I2_xx*c^2 + I2_zz*s^2 + m2*(h*c + r*s)^2
        # dM11/dq2 needed. dM11/dq1 = 0. dM22/dq = 0.
        
        # Derivative of trig terms (Updated to match corrected I2_proj)
        # d(c^2)/dq2 = -2s*c
        # d(s^2)/dq2 = 2s*c
        term1 = self.I2_xx * (-2*s2*c2) + self.I2_zz * (2*s2*c2)
        
        # Derivative of (h*c + r*s)^2
        # = 2*(h*c + r*s) * (-h*s + r*c)
        x_pos = self.h * c2 + self.r * s2
        z_pos = -self.h * s2 + self.r * c2
        term2 = self.m2 * 2 * x_pos * z_pos
        
        dM11_dq2 = term1 + term2
        
        # FIXED CORIOLIS INCOMPLETENESS: We must include the derivative of M12
        # M12 = m2 * x_pos * z_pos
        # dx_pos/dq2 = z_pos
        # dz_pos/dq2 = -x_pos
        dM12_dq2 = self.m2 * ((z_pos**2) - (x_pos**2))

        # Christoffel Symbols applied directly to Row 1 and Row 2 mapping
        # This exact mapping guarantees the skew-symmetry of (M_dot - 2C) = 0
        C11 = 0.5 * dM11_dq2 * dq[1]
        C12 = 0.5 * dM11_dq2 * dq[0] + dM12_dq2 * dq[1]
        
        C21 = -0.5 * dM11_dq2 * dq[0]
        C22 = 0.0
        
        C = np.array([
            [C11, C12],
            [C21, C22]
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
        # Torque is m * g * (horizontal lever arm distance from the Y tilt axis).
        # The horizontal lever arm in the global frame is x_pos = h*cos(q2) + r*sin(q2)
        
        # Calculate restoring torque (Unified with Mass Matrix frames)
        g_val = self.m2 * self.g * (self.h * np.cos(theta_tilt) + self.r * np.sin(theta_tilt))
        
        return np.array([0.0, g_val])

    # ... [compute_forward_dynamics, state_space_derivative, and linearize remain identical] ...
    # (Omitted below for brevity, but they plug directly in as before)
    
    def compute_forward_dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        M = self.get_mass_matrix(q)
        C = self.get_coriolis_matrix(q, dq)
        G = self.get_gravity_vector(q)
        rhs = tau - (C @ dq) - G
        return np.linalg.solve(M, rhs)


    def get_M_dot(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Calculates the exact analytical time derivative of the Mass Matrix.
        This is required for the mathematically pure NDOB formulation.
        """
        theta_tilt = q[1]
        dq_tilt = dq[1] # Elevation velocity
        
        c2 = np.cos(theta_tilt)
        s2 = np.sin(theta_tilt)
        
        # Exact analytical derivatives of M11 and M12 with respect to q2
        # (Copied directly from your get_coriolis_matrix math)
        term1 = self.I2_xx * (-2*s2*c2) + self.I2_zz * (2*s2*c2)
        
        x_pos = self.h * c2 + self.r * s2
        z_pos = -self.h * s2 + self.r * c2
        term2 = self.m2 * 2 * x_pos * z_pos
        
        dM11_dq2 = term1 + term2
        
        dM12_dq2 = self.m2 * ((z_pos**2) - (x_pos**2))
        
        # M22 is constant (inertia of tilt axis around itself doesn't change)
        dM22_dq2 = 0.0
        
        # Construct M_dot by multiplying the spatial derivatives by the joint velocity
        M_dot = np.array([
            [dM11_dq2 * dq_tilt, dM12_dq2 * dq_tilt],
            [dM12_dq2 * dq_tilt, dM22_dq2 * dq_tilt]
        ])
        
        return M_dot

if __name__ == "__main__":
    # Sanity Check
    print("Running GimbalDynamics Sanity Check...")
    
    # Optional Task 2 Validation: Initializing with custom matrices
    custom_I1 = np.diag([0.01, 0.01, 0.05])
    custom_I2 = np.diag([0.02, 0.03, 0.04])
    
    # It gracefully accepts the custom matrices without breaking positional defaults
    gimbal = GimbalDynamics(cm_r=0.05, cm_h=0.01, I1_matrix=custom_I1, I2_matrix=custom_I2)
    
    # Test Configuration
    q_test = np.array([0.1, 0.5]) # Arbitrary angles
    dq_test = np.array([0.1, 0.2])
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
        
    # 2. Check Skew-Symmetry (Crucial for FBL/DOB)
    # Approximating M_dot using finite differences to verify our C matrix analytical derivation
    dt = 1e-6
    M_next = gimbal.get_mass_matrix(q_test + dq_test * dt)
    M_dot = (M_next - M) / dt
    
    C = gimbal.get_coriolis_matrix(q_test, dq_test)
    skew_check = M_dot - 2 * C
    
    # If the matrix is skew-symmetric, adding it to its transpose should yield a zero matrix
    if np.allclose(skew_check + skew_check.T, 0, atol=1e-4):
        print("PASS: (M_dot - 2C) is skew-symmetric! Coriolis derivation is flawless.")
    else:
        print("FAIL: Skew-symmetry violated.")
        
    # 3. Forward Dynamics check
    accel = gimbal.compute_forward_dynamics(q_test, dq_test, tau_test)
    print(f"Forward Dynamics Accel: {accel}")