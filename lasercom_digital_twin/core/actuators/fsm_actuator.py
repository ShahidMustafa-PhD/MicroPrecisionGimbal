"""
Fast Steering Mirror (FSM) Actuator Model

This module implements a high-fidelity model of voice-coil or piezo-driven 
fast steering mirrors used for fine pointing correction in laser communication 
terminals. The model captures second-order mechanical dynamics, hysteresis, 
and rate limiting effects.
"""

import numpy as np
from typing import Tuple


class FSMActuatorModel:
    """
    Fast Steering Mirror actuator model with hysteresis and rate limits.
    
    The FSM is modeled as a second-order linear system for each axis (tip/tilt):
        α̈ + 2ζωₙα̇ + ωₙ²α = ωₙ²α_cmd
    
    Non-ideal effects included:
    - Hysteresis: Rate-dependent position lag using a simplified Dahl model
    - Rate limits: Maximum achievable angular velocity
    """
    
    def __init__(self, config: dict):
        """
        Initialize the FSM actuator model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - 'omega_n': Natural frequency [rad/s]
            - 'zeta': Damping ratio [-]
            - 'alpha_max': Maximum tip angle [rad]
            - 'alpha_min': Minimum tip angle [rad]
            - 'beta_max': Maximum tilt angle [rad]
            - 'beta_min': Minimum tilt angle [rad]
            - 'rate_limit': Maximum angular rate [rad/s]
            - 'hysteresis_width': Hysteresis band width [rad]
            - 'hysteresis_gain': Hysteresis coupling gain [-]
        """
        # Second-order dynamics parameters
        self.omega_n: float = config.get('omega_n', 2000.0)  # Natural frequency [rad/s]
        self.zeta: float = config.get('zeta', 0.7)  # Damping ratio [-]
        
        # Position limits
        self.alpha_max: float = config.get('alpha_max', np.deg2rad(1.0))  # Tip max [rad]
        self.alpha_min: float = config.get('alpha_min', np.deg2rad(-1.0))  # Tip min [rad]
        self.beta_max: float = config.get('beta_max', np.deg2rad(1.0))  # Tilt max [rad]
        self.beta_min: float = config.get('beta_min', np.deg2rad(-1.0))  # Tilt min [rad]
        
        # Rate limits
        self.rate_limit: float = config.get('rate_limit', np.deg2rad(500.0))  # [rad/s]
        
        # Hysteresis parameters (simplified Dahl-like model)
        self.hysteresis_width: float = config.get('hysteresis_width', np.deg2rad(0.001))
        self.hysteresis_gain: float = config.get('hysteresis_gain', 0.15)
        
        # State variables for Tip axis (alpha)
        self.alpha: float = 0.0  # Position [rad]
        self.alpha_dot: float = 0.0  # Velocity [rad/s]
        self.alpha_hyst: float = 0.0  # Hysteresis state [rad]
        
        # State variables for Tilt axis (beta)
        self.beta: float = 0.0  # Position [rad]
        self.beta_dot: float = 0.0  # Velocity [rad/s]
        self.beta_hyst: float = 0.0  # Hysteresis state [rad]
        
    def _update_hysteresis(
        self, 
        position: float, 
        hyst_state: float, 
        cmd: float, 
        dt: float
    ) -> float:
        """
        Update hysteresis state using a simplified rate-dependent model.
        
        This implements a first-order approximation of the Dahl friction model:
            dh/dt = σ * (cmd - position - h) * |cmd_rate|
        
        where h is the hysteresis state variable.
        
        Parameters
        ----------
        position : float
            Current position [rad]
        hyst_state : float
            Current hysteresis state [rad]
        cmd : float
            Commanded position [rad]
        dt : float
            Time step [s]
            
        Returns
        -------
        float
            Updated hysteresis state [rad]
        """
        # Compute error with hysteresis
        error = float(cmd) - float(position) - float(hyst_state)
        
        # Rate-dependent hysteresis dynamics
        # Higher rates reduce hysteresis effect
        cmd_rate = abs(error / (float(dt) + 1e-10))  # Approximate command rate
        sigma = self.hysteresis_gain / (self.hysteresis_width + 1e-10)
        
        # Update hysteresis state
        import math
        dhyst_dt = sigma * error * math.tanh(cmd_rate / 10.0)  # Saturate at high rates
        hyst_state_new = float(hyst_state) + dhyst_dt * float(dt)
        
        # Limit hysteresis magnitude
        hyst_state_new = max(-self.hysteresis_width, min(hyst_state_new, self.hysteresis_width))
        
        return hyst_state_new
    
    def _step_axis(
        self,
        position: float,
        velocity: float,
        hyst_state: float,
        cmd: float,
        pos_min: float,
        pos_max: float,
        dt: float
    ) -> Tuple[float, float, float]:
        """
        Compute one time step for a single FSM axis.
        
        Parameters
        ----------
        position : float
            Current position [rad]
        velocity : float
            Current velocity [rad/s]
        hyst_state : float
            Current hysteresis state [rad]
        cmd : float
            Commanded position [rad]
        pos_min : float
            Minimum position limit [rad]
        pos_max : float
            Maximum position limit [rad]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tuple[float, float, float]
            (new_position, new_velocity, new_hyst_state)
        """
        # Update hysteresis state
        hyst_state_new = float(self._update_hysteresis(position, hyst_state, cmd, dt))
        
        # Effective command includes hysteresis lag
        cmd_eff = float(cmd) - hyst_state_new
        
        # Second-order dynamics: α̈ + 2ζωₙα̇ + ωₙ²α = ωₙ²α_cmd
        accel = self.omega_n**2 * (cmd_eff - float(position)) - 2 * self.zeta * self.omega_n * float(velocity)
        
        # Integrate velocity (Euler forward)
        velocity_new = float(velocity) + accel * dt
        
        # Apply rate limiting (ensure scalar)
        rate_lim = float(self.rate_limit)
        velocity_new = max(-rate_lim, min(velocity_new, rate_lim))
        
        # Integrate position
        position_new = float(position) + velocity_new * dt
        
        # Apply position limits (ensure scalar)
        position_new = max(float(pos_min), min(position_new, float(pos_max)))
        
        # If position hits limit, zero velocity
        if abs(position_new - float(pos_min)) < 1e-10 or abs(position_new - float(pos_max)) < 1e-10:
            velocity_new = 0.0
        
        return position_new, velocity_new, hyst_state_new
    
    def step(
        self, 
        alpha_cmd: float, 
        beta_cmd: float, 
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute one time step of the FSM dynamics for both axes.
        
        Parameters
        ----------
        alpha_cmd : float
            Commanded tip angle [rad]
        beta_cmd : float
            Commanded tilt angle [rad]
        dt : float
            Time step [s]
            
        Returns
        -------
        Tuple[float, float]
            (alpha_actual, beta_actual) - Actual tip/tilt angles [rad]
        """
        # Update tip axis (alpha)
        self.alpha, self.alpha_dot, self.alpha_hyst = self._step_axis(
            self.alpha,
            self.alpha_dot,
            self.alpha_hyst,
            alpha_cmd,
            self.alpha_min,
            self.alpha_max,
            dt
        )
        
        # Update tilt axis (beta)
        self.beta, self.beta_dot, self.beta_hyst = self._step_axis(
            self.beta,
            self.beta_dot,
            self.beta_hyst,
            beta_cmd,
            self.beta_min,
            self.beta_max,
            dt
        )
        
        return self.alpha, self.beta
    
    def reset(self) -> None:
        """
        Reset the FSM state to initial conditions.
        """
        self.alpha = 0.0
        self.alpha_dot = 0.0
        self.alpha_hyst = 0.0
        self.beta = 0.0
        self.beta_dot = 0.0
        self.beta_hyst = 0.0
    
    def get_state(self) -> dict:
        """
        Get the current internal state of the FSM model.
        
        Returns
        -------
        dict
            Dictionary containing current state variables
        """
        return {
            'alpha': self.alpha,
            'alpha_dot': self.alpha_dot,
            'alpha_hyst': self.alpha_hyst,
            'beta': self.beta,
            'beta_dot': self.beta_dot,
            'beta_hyst': self.beta_hyst
        }
