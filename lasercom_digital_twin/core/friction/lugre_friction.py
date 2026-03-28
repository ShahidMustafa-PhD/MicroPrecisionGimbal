import numpy as np
class LuGreFriction:
    """
    LuGre Dynamic Friction Model with internal bristle state.
    
    Unlike SmoothedTustinFriction (algebraic), this model has internal 
    dynamics: dz/dt = v - sigma_0 * |v| * z / g(v)
    
    The bristle state z must be integrated at every simulation timestep.
    """
    
    def __init__(self, 
                 sigma_0: np.ndarray,   # Bristle stiffness [N·m/rad]
                 sigma_1: np.ndarray,   # Bristle damping [N·m·s/rad]
                 sigma_2: np.ndarray,   # Viscous coefficient [N·m·s/rad]
                 tau_c: np.ndarray,     # Coulomb friction [N·m]
                 tau_s: np.ndarray,     # Static friction [N·m]
                 v_s: np.ndarray,       # Stribeck velocity [rad/s]
                 n_axes: int = 2):
        
        self.sigma_0 = np.asarray(sigma_0, dtype=float)
        self.sigma_1 = np.asarray(sigma_1, dtype=float)
        self.sigma_2 = np.asarray(sigma_2, dtype=float)
        self.tau_c = np.asarray(tau_c, dtype=float)
        self.tau_s = np.asarray(tau_s, dtype=float)
        self.v_s = np.asarray(v_s, dtype=float)
        
        # INTERNAL STATE: bristle deflection per axis
        self.z = np.zeros(n_axes)
        
        # Previous dz/dt for diagnostics
        self._dz_dt = np.zeros(n_axes)
    
    def g(self, velocity: np.ndarray) -> np.ndarray:
        """Stribeck function: g(v) = tau_c + (tau_s - tau_c) * exp(-(v/v_s)^2)"""
        return self.tau_c + (self.tau_s - self.tau_c) * np.exp(
            -(velocity / self.v_s) ** 2
        )
    
    def step(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance bristle state by dt and return friction torque.
        
        This MUST be called at every simulation timestep (not control timestep).
        The bristle dynamics can have natural frequencies in the kHz range
        (sqrt(sigma_0 / sigma_1)), so dt must be small enough to resolve them.
        
        Args:
            velocity: Joint angular velocity [rad/s] (n_axes,)
            dt: Integration timestep [s]
            
        Returns:
            Friction torque [N·m] (n_axes,)
        """
        v = np.asarray(velocity, dtype=float)
        
        # Bristle dynamics: dz/dt = v - sigma_0 * |v| * z / g(v)
        g_v = self.g(v)
        dz_dt = v - self.sigma_0 * np.abs(v) * self.z / g_v
        self._dz_dt = dz_dt.copy()
        
        # Integrate bristle state (Forward Euler — acceptable if dt is small)
        self.z += dz_dt * dt
        
        # Friction torque: tau_f = sigma_0 * z + sigma_1 * dz/dt + sigma_2 * v
        tau_fric = self.sigma_0 * self.z + self.sigma_1 * dz_dt + self.sigma_2 * v
        
        return tau_fric
    
    def get_bristle_state(self) -> np.ndarray:
        """Return current bristle deflection for logging."""
        return self.z.copy()
    
    def reset(self):
        """Reset bristle state to zero."""
        self.z = np.zeros_like(self.z)
        self._dz_dt = np.zeros_like(self._dz_dt)