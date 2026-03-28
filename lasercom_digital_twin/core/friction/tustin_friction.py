import numpy as np
from typing import Union, Tuple

class SmoothedTustinFriction:
    """
    Smoothed Tustin (Stribeck) Friction Model for high-precision simulations.
    
    This class implements a continuous, memoryless friction model that captures:
    1. Coulomb friction (constant opposing torque)
    2. Static friction / Stiction (breakaway torque)
    3. The Stribeck effect (exponential drop in friction at low velocities)
    4. Viscous friction (linear damping)
    
    The standard signum function sgn(v) is replaced with tanh(v / v_epsilon) 
    to ensure Lipschitz continuity and prevent numerical chattering in fixed-step 
    ODE solvers (like Symplectic Euler or RK4) during zero-velocity crossings.
    
    Mathematical Formulation:
    tau_fric = [tau_c + (tau_s - tau_c) * exp(-(|v| / v_s)^alpha)] * tanh(v / v_epsilon) + b * v
    """

    def __init__(self, 
                 tau_c: Union[float, np.ndarray, list], 
                 tau_s: Union[float, np.ndarray, list], 
                 v_s: Union[float, np.ndarray, list], 
                 b: Union[float, np.ndarray, list], 
                 alpha: float = 2.0, 
                 v_epsilon: float = 1e-4):
        """
        Initialize the friction model parameters. 
        Parameters can be floats (single axis) or arrays (multi-axis).

        Args:
            tau_c: Coulomb friction magnitude [N*m]
            tau_s: Static friction magnitude (Stiction) [N*m]. Must be >= tau_c.
            v_s: Stribeck velocity threshold [rad/s].
            b: Viscous friction coefficient [N*m*s/rad].
            alpha: Exponent for the Stribeck curve decay. Usually 1.0 or 2.0.
            v_epsilon: Velocity scaling factor for the tanh smoothing [rad/s]. 
                       Smaller = sharper stick-slip, but requires smaller dt.
        """
        # Convert inputs to numpy arrays for vectorized operations
        self.tau_c = np.asarray(tau_c, dtype=float)
        self.tau_s = np.asarray(tau_s, dtype=float)
        self.v_s = np.asarray(v_s, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.alpha = float(alpha)
        self.v_epsilon = float(v_epsilon)

        # Validate physical constraints
        if np.any(self.tau_s < self.tau_c):
            raise ValueError("Physical violation: Static friction (tau_s) must be >= Coulomb friction (tau_c).")
        if np.any(self.v_s <= 0):
            raise ValueError("Stribeck velocity (v_s) must be strictly positive.")
        if self.v_epsilon <= 0:
            raise ValueError("Smoothing parameter (v_epsilon) must be strictly positive.")

    def compute_torque(self, velocity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the non-linear friction torque for a given velocity.

        Args:
            velocity: Current joint angular velocity [rad/s]. 
                      Can be a scalar or an array (e.g., [dq_az, dq_el]).

        Returns:
            Friction torque opposing the motion [N*m].
        """
        v = np.asarray(velocity, dtype=float)
        
        # 1. Calculate the Stribeck exponential decay
        # exp(-(|v| / v_s)^alpha)
        stribeck_decay = np.exp(-(np.abs(v) / self.v_s) ** self.alpha)
        
        # 2. Calculate the magnitude of the kinetic + static friction
        # tau_mag = tau_c + (tau_s - tau_c) * decay
        tau_mag = self.tau_c + (self.tau_s - self.tau_c) * stribeck_decay
        
        # 3. Apply the smoothed directional sign (tanh) and add viscous term
        tau_fric = tau_mag * np.tanh(v / self.v_epsilon) + self.b * v
        
        return tau_fric

    def __call__(self, velocity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Allows the instance to be called directly like a function.
        Example: torque = my_friction_model(current_velocity)
        """
        return self.compute_torque(velocity)