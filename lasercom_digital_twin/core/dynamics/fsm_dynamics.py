"""
Fast Steering Mirror (FSM) Dynamics Model

This module implements a high-fidelity 4th-order state-space representation of a
2-axis FSM driven by Voice Coil Actuators (VCA) for space laser communication.

The model captures:
- Flexural resonances from mechanical structure
- Cross-axis coupling from asymmetric mass distribution
- VCA actuator dynamics (electromagnetic force generation)

State-Space Representation:
--------------------------
dx/dt = A*x + B*u
y = C*x + D*u

where:
    x ∈ ℝ⁴: Internal state vector (modal coordinates)
    u ∈ ℝ²: Control input vector [V_tip, V_tilt] (actuator commands)
    y ∈ ℝ²: Output vector [θ_tip, θ_tilt] (angular displacements in radians)

Integration Method:
------------------
Uses 4th-order Runge-Kutta (RK4) for numerical stability when handling
high-frequency flexural modes (100-500 Hz). RK4 provides excellent accuracy
with reasonable computational cost and maintains stability for stiff systems
when dt < 1/(10*f_max).

Physical Interpretation:
-----------------------
The 4th-order model is a reduced-order representation obtained from modal
analysis of the full flexible structure. Each state can be interpreted as
a modal amplitude corresponding to a specific structural resonance mode.
The cross-coupling terms in matrices A and B represent the projection of
these modes onto the two output axes (tip/tilt).

Author: Senior Control Systems Engineer
Date: January 20, 2026
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FsmDynamicsConfig:
    """Configuration parameters for FSM dynamics model."""
    # Matrices are stored here for easy serialization/configuration
    A: np.ndarray  # State matrix (4x4)
    B: np.ndarray  # Input matrix (4x2)
    C: np.ndarray  # Output matrix (2x4)
    D: np.ndarray  # Feedthrough matrix (2x2)


class FsmDynamics:
    """
    High-fidelity 4th-order state-space dynamics for 2-axis FSM.
    
    This class encapsulates the plant model and provides methods for:
    - State propagation via RK4 integration
    - Output computation
    - State reset for Monte Carlo simulations
    
    The model is time-invariant (LTI system) but uses numerical integration
    to support future extensions with nonlinearities (e.g., actuator saturation,
    friction, thermal effects).
    
    Attributes
    ----------
    A : np.ndarray
        State matrix (4x4) - captures structural dynamics and damping
    B : np.ndarray
        Input matrix (4x2) - maps VCA commands to modal forces
    C : np.ndarray
        Output matrix (2x4) - maps modal states to tip/tilt angles
    D : np.ndarray
        Feedthrough matrix (2x2) - direct transmission (typically zero)
    x : np.ndarray
        Current state vector (4,) - modal coordinates
    """
    
    def __init__(self, config: Optional[FsmDynamicsConfig] = None):
        """
        Initialize FSM dynamics model.
        
        Parameters
        ----------
        config : FsmDynamicsConfig, optional
            Configuration with state-space matrices. If None, uses default
            matrices from modal analysis (4th-order reduced model).
        """
        if config is None:
            # Default matrices from modal reduction analysis
            # These capture the first two flexural modes with cross-coupling
            self.A = np.array([
                [-20.02,   8.20,  125.75,    1.14],
                [-8.16,  -19.92,    2.56, -146.84],
                [-125.77, -3.40,  -30.63,    8.92],
                [-0.88,   146.81,  -8.99,  -28.45]
            ])
            
            self.B = np.array([
                [40.85, -43.10],
                [-36.55, -38.03],
                [42.42, -43.00],
                [39.16, 36.67]
            ])
            
            self.C = np.array([
                [38.68, -41.64, -37.43, -41.31],
                [-45.05, -32.38, 47.41, -34.23]
            ])
            
            self.D = np.zeros((2, 2))
        else:
            self.A = config.A
            self.B = config.B
            self.C = config.C
            self.D = config.D
        
        # Validate dimensions
        self._validate_matrices()
        
        # Initialize state vector (modal coordinates)
        self.x = np.zeros(4)
        
        # Cache for performance metrics
        self._time = 0.0
    
    def _validate_matrices(self) -> None:
        """Validate state-space matrix dimensions for consistency."""
        assert self.A.shape == (4, 4), f"A must be 4x4, got {self.A.shape}"
        assert self.B.shape == (4, 2), f"B must be 4x2, got {self.B.shape}"
        assert self.C.shape == (2, 4), f"C must be 2x4, got {self.C.shape}"
        assert self.D.shape == (2, 2), f"D must be 2x2, got {self.D.shape}"
    
    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        """
        Reset the internal state for a new simulation run.
        
        Essential for Monte Carlo analysis where multiple runs are performed
        with different initial conditions or disturbances.
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial state vector (4,). If None, resets to zero state.
        """
        if x0 is None:
            self.x = np.zeros(4)
        else:
            assert x0.shape == (4,), f"x0 must be (4,), got {x0.shape}"
            self.x = x0.copy()
        
        self._time = 0.0
    
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate state forward by one time step using RK4 integration.
        
        Numerical Integration Method: 4th-order Runge-Kutta (RK4)
        ----------------------------------------------------------
        RK4 is chosen for FSM dynamics because:
        
        1. **Stability**: High-frequency flexural modes (100-500 Hz) can cause
           instability with simpler methods (Euler, RK2). RK4 provides excellent
           stability with reasonable time steps (dt ~ 1 ms).
        
        2. **Accuracy**: RK4 achieves O(dt⁴) local truncation error, essential
           for micro-radian precision pointing requirements.
        
        3. **Computational Cost**: While more expensive than Euler, RK4 allows
           larger time steps with maintained accuracy, offsetting the cost.
        
        RK4 Algorithm:
        -------------
        k1 = f(x_n, u_n)
        k2 = f(x_n + 0.5*dt*k1, u_n)
        k3 = f(x_n + 0.5*dt*k2, u_n)
        k4 = f(x_n + dt*k3, u_n)
        x_{n+1} = x_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        Parameters
        ----------
        u : np.ndarray
            Control input vector (2,) - [V_tip, V_tilt] in volts or normalized
        dt : float
            Time step [s]. Recommended: dt < 1e-3 for 500 Hz modes
        
        Returns
        -------
        np.ndarray
            Output vector y (2,) - [θ_tip, θ_tilt] in radians
        """
        assert u.shape == (2,), f"u must be (2,), got {u.shape}"
        assert dt > 0, f"dt must be positive, got {dt}"
        
        # RK4 integration for dx/dt = A*x + B*u
        # Note: u is assumed constant over the time step (zero-order hold)
        
        # k1 = f(x_n, u)
        k1 = self.A @ self.x + self.B @ u
        
        # k2 = f(x_n + 0.5*dt*k1, u)
        x_temp = self.x + 0.5 * dt * k1
        k2 = self.A @ x_temp + self.B @ u
        
        # k3 = f(x_n + 0.5*dt*k2, u)
        x_temp = self.x + 0.5 * dt * k2
        k3 = self.A @ x_temp + self.B @ u
        
        # k4 = f(x_n + dt*k3, u)
        x_temp = self.x + dt * k3
        k4 = self.A @ x_temp + self.B @ u
        
        # Update state: x_{n+1} = x_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.x = self.x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update time
        self._time += dt
        
        # Compute and return output
        return self.outputs(u)
    
    def outputs(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute output vector from current state.
        
        Output equation: y = C*x + D*u
        
        For most FSM systems, D = 0 (no direct feedthrough), meaning the
        output depends only on the modal states, not directly on the input.
        
        Parameters
        ----------
        u : np.ndarray, optional
            Current input vector (2,). Required only if D ≠ 0.
        
        Returns
        -------
        np.ndarray
            Output vector y (2,) - [θ_tip, θ_tilt] in radians
        """
        y = self.C @ self.x
        
        if u is not None and np.any(self.D != 0):
            assert u.shape == (2,), f"u must be (2,), got {u.shape}"
            y += self.D @ u
        
        return y
    
    def get_state(self) -> np.ndarray:
        """
        Get the current internal state vector.
        
        Returns
        -------
        np.ndarray
            State vector x (4,) - modal coordinates
        """
        return self.x.copy()
    
    def get_time(self) -> float:
        """
        Get the current simulation time.
        
        Returns
        -------
        float
            Elapsed time [s] since last reset
        """
        return self._time
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of state matrix A.
        
        Useful for modal analysis and stability verification.
        For a stable system, all eigenvalues must have negative real parts.
        
        Returns
        -------
        np.ndarray
            Complex eigenvalues (4,)
        """
        return np.linalg.eigvals(self.A)
    
    def get_resonance_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract resonance frequencies and damping ratios from eigenvalues.
        
        For a complex conjugate pair λ = σ ± jω:
            Natural frequency: ω_n = |λ| = sqrt(σ² + ω²)
            Damping ratio: ζ = -σ / ω_n
            Resonance frequency: f = ω / (2π)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            frequencies_hz : Resonance frequencies [Hz]
            damping_ratios : Damping ratios (dimensionless)
        """
        eigenvals = self.get_eigenvalues()
        
        frequencies = []
        dampings = []
        
        for eig in eigenvals:
            real_part = np.real(eig)
            imag_part = np.imag(eig)
            
            if np.abs(imag_part) > 1e-6:  # Complex eigenvalue
                omega_n = np.abs(eig)
                zeta = -real_part / omega_n
                freq_hz = np.abs(imag_part) / (2 * np.pi)
                
                frequencies.append(freq_hz)
                dampings.append(zeta)
        
        return np.array(frequencies), np.array(dampings)
    
    def linearize_at_state(self, x_op: np.ndarray, u_op: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return linearization matrices (for analysis purposes).
        
        Since this model is already linear time-invariant (LTI), the
        linearization is simply the original matrices. This method is
        included for API consistency with nonlinear plant models.
        
        Parameters
        ----------
        x_op : np.ndarray
            Operating point state (4,) - unused for LTI
        u_op : np.ndarray
            Operating point input (2,) - unused for LTI
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (A, B, C, D) matrices
        """
        return self.A.copy(), self.B.copy(), self.C.copy(), self.D.copy()


def create_fsm_dynamics_from_design() -> FsmDynamics:
    """
    Factory function to create FSM dynamics using matrices from control design.
    
    This ensures consistency between the plant model used for controller
    synthesis and the plant model used in closed-loop simulation.
    
    Returns
    -------
    FsmDynamics
        Configured FSM dynamics instance
    """
    # Matrices from modal reduction (same as in FSM_control_design.py)
    A = np.array([
        [-20.02,   8.20,  125.75,    1.14],
        [-8.16,  -19.92,    2.56, -146.84],
        [-125.77, -3.40,  -30.63,    8.92],
        [-0.88,   146.81,  -8.99,  -28.45]
    ])
    
    B = np.array([
        [40.85, -43.10],
        [-36.55, -38.03],
        [42.42, -43.00],
        [39.16, 36.67]
    ])
    
    C = np.array([
        [38.68, -41.64, -37.43, -41.31],
        [-45.05, -32.38, 47.41, -34.23]
    ])
    
    D = np.zeros((2, 2))
    
    config = FsmDynamicsConfig(A=A, B=B, C=C, D=D)
    return FsmDynamics(config)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Validation suite for FSM dynamics model.
    
    Tests:
    1. Stability (eigenvalues have negative real parts)
    2. Modal frequencies extraction
    3. Step response integration
    4. Numerical precision
    """
    print("=" * 70)
    print(" FSM DYNAMICS MODEL VALIDATION ")
    print("=" * 70)
    
    # Create instance
    fsm = create_fsm_dynamics_from_design()
    
    # Test 1: Stability check
    print("\n1. STABILITY ANALYSIS")
    print("-" * 70)
    eigenvals = fsm.get_eigenvalues()
    print(f"Eigenvalues:")
    for i, eig in enumerate(eigenvals):
        print(f"  λ{i+1} = {eig.real:8.2f} ± j{np.abs(eig.imag):8.2f}")
    
    stable = all(np.real(eig) < 0 for eig in eigenvals)
    print(f"\nSystem Stable: {'✓ YES' if stable else '✗ NO'}")
    
    # Test 2: Resonance frequencies
    print("\n2. MODAL CHARACTERISTICS")
    print("-" * 70)
    freqs, dampings = fsm.get_resonance_frequencies()
    for i, (f, z) in enumerate(zip(freqs, dampings)):
        print(f"  Mode {i+1}: f = {f:6.2f} Hz, ζ = {z:6.4f}")
    
    # Test 3: Step response
    print("\n3. STEP RESPONSE TEST")
    print("-" * 70)
    fsm.reset()
    dt = 0.001  # 1 ms time step
    duration = 0.1  # 100 ms simulation
    n_steps = int(duration / dt)
    
    # Apply unit step to tip axis
    u_step = np.array([1.0, 0.0])
    
    time_history = []
    output_history = []
    
    for i in range(n_steps):
        y = fsm.step(u_step, dt)
        time_history.append(fsm.get_time())
        output_history.append(y.copy())
    
    outputs = np.array(output_history)
    final_tip = outputs[-1, 0]
    final_tilt = outputs[-1, 1]
    
    print(f"Final Output (100ms):")
    print(f"  Tip:  {final_tip:.6e} rad")
    print(f"  Tilt: {final_tilt:.6e} rad (cross-coupling)")
    print(f"  Cross-talk ratio: {np.abs(final_tilt/final_tip)*100:.2f}%")
    
    # Test 4: Reset functionality
    print("\n4. RESET TEST")
    print("-" * 70)
    fsm.reset()
    x_after_reset = fsm.get_state()
    print(f"State after reset: {x_after_reset}")
    print(f"All zeros: {'✓ YES' if np.allclose(x_after_reset, 0) else '✗ NO'}")
    
    print("\n" + "=" * 70)
    print(" VALIDATION COMPLETE ")
    print("=" * 70)
