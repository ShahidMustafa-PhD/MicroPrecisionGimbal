"""
Nonlinear Disturbance Observer (NDOB) for Gimbal Systems

This module implements a model-based disturbance observer for rejecting
unmodeled dynamics, friction, and external disturbances in the coarse
gimbal pointing loop.

Mathematical Formulation
------------------------
The NDOB estimates lumped disturbances acting on the gimbal system using
the known plant model and measured states.

**Plant Model (with disturbance):**

$$M(q)\\ddot{q} + C(q,\\dot{q})\\dot{q} + G(q) = \\tau + d$$

where $d \\in \\mathbb{R}^2$ is the lumped disturbance (friction, wind, cable drag, etc.)

**Observer Dynamics:**

$$\\dot{z} = -Lz + L\\left[C(q,\\dot{q})\\dot{q} + G(q) - \\tau_{applied} - p(q,\\dot{q})\\right]$$

**Auxiliary Function:**

$$p(q,\\dot{q}) = L M(q) \\dot{q}$$

**Disturbance Estimate:**

$$\\hat{d} = z + p(q,\\dot{q})$$

**Convergence:**
The estimation error $\\tilde{d} = d - \\hat{d}$ satisfies:

$$\\dot{\\tilde{d}} = -L\\tilde{d} + \\dot{d}$$

For constant or slowly-varying disturbances ($\\dot{d} \\approx 0$), the error
converges exponentially with time constant $\\tau = 1/\\lambda$ where $L = \\lambda I$.

References
----------
[1] Chen, W.H., "Disturbance Observer Based Control for Nonlinear Systems",
    IEEE/ASME Trans. Mechatronics, 2004.
[2] Sariyildiz, E., Ohnishi, K., "Stability and Robustness of Disturbance-
    Observer-Based Motion Control Systems", IEEE Trans. Ind. Electron., 2015.

Author: Senior Control Systems Engineer
Date: January 21, 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


@dataclass
class NDOBConfig:
    """
    Configuration for the Nonlinear Disturbance Observer.
    
    Attributes
    ----------
    lambda_az : float
        Observer bandwidth for azimuth axis [rad/s]. Higher = faster response
        but more noise sensitivity. Typical range: 20-100 rad/s.
    lambda_el : float
        Observer bandwidth for elevation axis [rad/s].
    d_max : float
        Saturation limit for disturbance estimate [Nm]. Prevents excessive
        compensation that could destabilize the system.
    enable : bool
        Master enable flag. If False, observer returns zero disturbance.
    """
    lambda_az: float = 40.0
    lambda_el: float = 40.0
    d_max: float = 5.0  # Max disturbance estimate [Nm]
    enable: bool = True


class NonlinearDisturbanceObserver:
    """
    Nonlinear Disturbance Observer (NDOB) for 2-DOF Gimbal Systems.
    
    This observer estimates lumped disturbances (friction, cable drag, wind,
    unmodeled dynamics) using the known gimbal dynamics model. The estimate
    can be used for feedforward compensation in the control loop.
    
    The observer is designed for use with the Feedback Linearization controller
    where the nominal dynamics are cancelled, leaving only the disturbance.
    
    Mathematical Basis
    ------------------
    Given the gimbal dynamics with disturbance:
    
    $$M(q)\\ddot{q} + C(q,\\dot{q})\\dot{q} + G(q) = \\tau + d$$
    
    The NDOB uses an auxiliary variable $z$ with dynamics:
    
    $$\\dot{z} = -Lz + L\\left[C\\dot{q} + G - \\tau - LM\\dot{q}\\right]$$
    
    The disturbance estimate is then:
    
    $$\\hat{d} = z + LM(q)\\dot{q}$$
    
    Implementation Notes
    -------------------
    - Uses Forward Euler integration for simplicity and causality
    - The applied torque from step (k-1) is used at step (k) for causality
    - Saturation prevents excessive compensation near singularities
    - The L matrix is diagonal (decoupled observer gains)
    
    Example
    -------
    >>> from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
    >>> from lasercom_digital_twin.core.n_dist_observer import NonlinearDisturbanceObserver, NDOBConfig
    >>> 
    >>> dynamics = GimbalDynamics()
    >>> config = NDOBConfig(lambda_az=40.0, lambda_el=40.0)
    >>> ndob = NonlinearDisturbanceObserver(dynamics, config)
    >>> 
    >>> # In control loop:
    >>> q = np.array([0.1, 0.2])
    >>> dq = np.array([0.01, 0.02])
    >>> tau_prev = np.array([0.5, 0.3])
    >>> d_hat = ndob.update(q, dq, tau_prev, dt=0.001)
    """
    
    def __init__(self, 
                 dynamics: 'GimbalDynamics',
                 config: Optional[NDOBConfig] = None):
        """
        Initialize the Nonlinear Disturbance Observer.
        
        Parameters
        ----------
        dynamics : GimbalDynamics
            Reference to the gimbal dynamics model providing M(q), C(q,dq), G(q).
        config : NDOBConfig, optional
            Observer configuration. Uses defaults if not provided.
        """
        self.dynamics = dynamics
        self.config = config if config is not None else NDOBConfig()
        
        # Observer state vector z ∈ R²
        self._z: np.ndarray = np.zeros(2)
        
        # Cached disturbance estimate
        self._d_hat: np.ndarray = np.zeros(2)
        
        # Diagonal gain matrix L = diag(λ_az, λ_el)
        self._L: np.ndarray = np.diag([self.config.lambda_az, self.config.lambda_el])
        
        # Initialization flag
        self._initialized: bool = False
        
        # Diagnostic: last computed values for debugging
        self._last_p: np.ndarray = np.zeros(2)
        self._last_z_dot: np.ndarray = np.zeros(2)
    
    @property
    def L(self) -> np.ndarray:
        """Get the current observer gain matrix."""
        return self._L.copy()
    
    @L.setter
    def L(self, value: np.ndarray) -> None:
        """
        Set the observer gain matrix (for live tuning).
        
        Parameters
        ----------
        value : np.ndarray
            2x2 diagonal gain matrix or 2-element vector of diagonal gains.
        """
        if value.shape == (2,):
            self._L = np.diag(value)
        elif value.shape == (2, 2):
            self._L = value.copy()
        else:
            raise ValueError(f"L must be (2,) or (2,2), got {value.shape}")
    
    def set_gains(self, lambda_az: float, lambda_el: float) -> None:
        """
        Update observer bandwidth gains (for live tuning).
        
        Parameters
        ----------
        lambda_az : float
            New azimuth bandwidth [rad/s]
        lambda_el : float
            New elevation bandwidth [rad/s]
        """
        self._L = np.diag([lambda_az, lambda_el])
        self.config.lambda_az = lambda_az
        self.config.lambda_el = lambda_el
    
    def reset(self) -> None:
        """
        Reset the observer state to zero.
        
        Call this when:
        - Initializing a new simulation
        - After large state discontinuities (mode switches)
        - When the observer estimate has diverged
        """
        self._z = np.zeros(2)
        self._d_hat = np.zeros(2)
        self._initialized = False
        self._last_p = np.zeros(2)
        self._last_z_dot = np.zeros(2)
    
    def _compute_auxiliary_p(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Compute the auxiliary function p(q, dq) = L * M(q) * dq.
        
        This function captures the momentum-like term that enables the
        observer to track disturbances without requiring acceleration
        measurements.
        
        Parameters
        ----------
        q : np.ndarray
            Joint positions [rad] (2,)
        dq : np.ndarray
            Joint velocities [rad/s] (2,)
        
        Returns
        -------
        np.ndarray
            Auxiliary variable p (2,)
        
        Mathematical Form
        -----------------
        $$p(q, \\dot{q}) = L M(q) \\dot{q}$$
        
        Note: This assumes M(q) is slowly varying compared to the observer
        bandwidth, which is valid for gimbal systems with bandwidth < 50 Hz.
        """
        M = self.dynamics.get_mass_matrix(q)
        p = self._L @ M @ dq
        return p
    
    def update(self, 
               q_meas: np.ndarray, 
               dq_meas: np.ndarray, 
               tau_applied: np.ndarray, 
               dt: float) -> np.ndarray:
        """
        Update the disturbance observer and return the estimate.
        
        This method should be called once per control cycle, AFTER the
        torque from the previous step has been applied.
        
        Parameters
        ----------
        q_meas : np.ndarray
            Measured joint positions from EKF [rad] (2,)
        dq_meas : np.ndarray
            Measured joint velocities from EKF [rad/s] (2,)
        tau_applied : np.ndarray
            Control torque applied at previous time step [Nm] (2,)
            IMPORTANT: Use τ(k-1) at step k for causality!
        dt : float
            Time step [s]
        
        Returns
        -------
        np.ndarray
            Estimated lumped disturbance d_hat [Nm] (2,)
        
        Observer Dynamics (Forward Euler)
        ---------------------------------
        $$z_{k+1} = z_k + dt \\cdot \\dot{z}_k$$
        
        where:
        $$\\dot{z} = -Lz + L\\left[C(q,\\dot{q})\\dot{q} + G(q) - \\tau - p(q,\\dot{q})\\right]$$
        
        Disturbance Estimate:
        $$\\hat{d} = z + p(q,\\dot{q})$$
        """
        # Early exit if disabled
        if not self.config.enable:
            self._d_hat = np.zeros(2)
            return self._d_hat.copy()
        
        # Validate inputs
        q = np.asarray(q_meas).flatten()[:2]
        dq = np.asarray(dq_meas).flatten()[:2]
        tau = np.asarray(tau_applied).flatten()[:2]
        
        # Get dynamics matrices at current state
        C = self.dynamics.get_coriolis_matrix(q, dq)
        G = self.dynamics.get_gravity_vector(q)
        
        # Compute auxiliary function p(q, dq) = L * M(q) * dq
        p = self._compute_auxiliary_p(q, dq)
        self._last_p = p.copy()
        
        # Compute Coriolis + gravity terms
        coriolis_term = C @ dq
        
        # Observer dynamics:
        # ż = -Lz + L[C*dq + G - τ - p]
        #   = -Lz + L*C*dq + L*G - L*τ - L*p
        inner_term = coriolis_term + G - tau - p
        z_dot = -self._L @ self._z + self._L @ inner_term
        self._last_z_dot = z_dot.copy()
        
        # Forward Euler integration
        self._z = self._z + dt * z_dot
        
        # Compute disturbance estimate: d_hat = z + p
        self._d_hat = self._z + p
        
        # Apply saturation for safety
        d_max = self.config.d_max
        self._d_hat = np.clip(self._d_hat, -d_max, d_max)
        
        self._initialized = True
        
        return self._d_hat.copy()
    
    def get_estimate(self) -> np.ndarray:
        """
        Get the current disturbance estimate without updating.
        
        Returns
        -------
        np.ndarray
            Last computed disturbance estimate [Nm] (2,)
        """
        return self._d_hat.copy()
    
    def get_state(self) -> np.ndarray:
        """
        Get the internal observer state z.
        
        Returns
        -------
        np.ndarray
            Observer state vector z (2,)
        """
        return self._z.copy()
    
    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information for debugging and tuning.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'z': Observer state
            - 'p': Auxiliary function value
            - 'z_dot': State derivative
            - 'd_hat': Disturbance estimate
            - 'L_diag': Diagonal of gain matrix
        """
        return {
            'z': self._z.copy(),
            'p': self._last_p.copy(),
            'z_dot': self._last_z_dot.copy(),
            'd_hat': self._d_hat.copy(),
            'L_diag': np.diag(self._L).copy(),
            'initialized': self._initialized
        }


def create_ndob_from_config(dynamics: 'GimbalDynamics', 
                            config_dict: Optional[dict] = None) -> NonlinearDisturbanceObserver:
    """
    Factory function to create NDOB from a configuration dictionary.
    
    Parameters
    ----------
    dynamics : GimbalDynamics
        Reference to gimbal dynamics model.
    config_dict : dict, optional
        Configuration dictionary with keys:
        - 'lambda_az': float (default 40.0)
        - 'lambda_el': float (default 40.0)
        - 'd_max': float (default 5.0)
        - 'enable': bool (default True)
    
    Returns
    -------
    NonlinearDisturbanceObserver
        Configured NDOB instance.
    """
    if config_dict is None:
        config_dict = {}
    
    config = NDOBConfig(
        lambda_az=config_dict.get('lambda_az', 40.0),
        lambda_el=config_dict.get('lambda_el', 40.0),
        d_max=config_dict.get('d_max', 5.0),
        enable=config_dict.get('enable', True)
    )
    
    return NonlinearDisturbanceObserver(dynamics, config)


# =============================================================================
# UNIT TEST / DEMO
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
    
    print("=" * 70)
    print("NONLINEAR DISTURBANCE OBSERVER - UNIT TEST")
    print("=" * 70)
    
    # Create dynamics and observer
    dynamics = GimbalDynamics(
        pan_mass=0.5,
        tilt_mass=0.25,
        cm_r=0.002,
        cm_h=0.005,
        gravity=9.81
    )
    
    config = NDOBConfig(lambda_az=40.0, lambda_el=40.0, d_max=5.0, enable=True)
    ndob = NonlinearDisturbanceObserver(dynamics, config)
    
    print(f"\nObserver Configuration:")
    print(f"  λ_az = {config.lambda_az} rad/s (τ = {1/config.lambda_az*1000:.1f} ms)")
    print(f"  λ_el = {config.lambda_el} rad/s (τ = {1/config.lambda_el*1000:.1f} ms)")
    print(f"  d_max = {config.d_max} Nm")
    
    # Simulate constant disturbance rejection
    print("\n" + "-" * 70)
    print("TEST: Constant Disturbance Estimation")
    print("-" * 70)
    
    # True disturbance (simulating friction + gravity offset)
    d_true = np.array([0.15, 0.08])  # Nm
    
    q = np.array([0.1, 0.3])  # rad
    dq = np.array([0.0, 0.0])  # At rest
    dt = 0.001  # 1 kHz
    
    # Simulate for 200 ms (8 time constants at λ=40)
    n_steps = 200
    d_hat_history = []
    
    for k in range(n_steps):
        # Simulate: tau_applied = 0 (open loop), disturbance present
        tau_applied = np.zeros(2)
        
        d_hat = ndob.update(q, dq, tau_applied, dt)
        d_hat_history.append(d_hat.copy())
    
    d_hat_final = d_hat_history[-1]
    
    print(f"\n  True disturbance:      d = [{d_true[0]:.4f}, {d_true[1]:.4f}] Nm")
    print(f"  Estimated (200 ms):    d̂ = [{d_hat_final[0]:.4f}, {d_hat_final[1]:.4f}] Nm")
    
    # Note: Without actual disturbance in dynamics, observer estimates nominal model error
    # In real simulation, the disturbance would manifest in the state evolution
    
    # Test live tuning
    print("\n" + "-" * 70)
    print("TEST: Live Tuning Interface")
    print("-" * 70)
    
    ndob.set_gains(lambda_az=60.0, lambda_el=60.0)
    print(f"  Updated gains: λ_az = {ndob.config.lambda_az}, λ_el = {ndob.config.lambda_el}")
    print(f"  New time constant: τ = {1/60.0*1000:.1f} ms")
    
    # Test reset
    ndob.reset()
    diag = ndob.get_diagnostics()
    print(f"\n  After reset: z = {diag['z']}, initialized = {diag['initialized']}")
    
    print("\n" + "=" * 70)
    print("NDOB UNIT TEST COMPLETE")
    print("=" * 70)
