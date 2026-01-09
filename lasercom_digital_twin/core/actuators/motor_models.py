"""
Gimbal Motor Models for Coarse Pointing Assembly (CPA)

This module implements high-fidelity, reduced-order models of brushless DC (BLDC) 
and permanent magnet synchronous motors (PMSM) used in precision gimbal systems.
The models capture critical non-ideal physics including electrical dynamics, 
mechanical coupling, torque saturation, and cogging torque.
"""

import numpy as np
from typing import Tuple


class GimbalMotorModel:
    """
    Reduced-order BLDC/PMSM motor model for gimbal actuators.
    
    This model combines:
    - Electrical dynamics: First-order RL circuit approximation
    - Mechanical coupling: Torque production via current
    - Non-ideal effects: Saturation and cogging torque
    
    The electrical dynamics are modeled as:
        L * di/dt = V_cmd - R * i - K_e * omega
    
    The mechanical torque output is:
        tau_m = K_t * i + tau_cog(theta_m)
    
    where tau_cog is a periodic cogging torque disturbance.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the gimbal motor model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - 'R': Winding resistance [Ohm]
            - 'L': Winding inductance [H]
            - 'K_t': Torque constant [N·m/A]
            - 'K_e': Back-EMF constant [V·s/rad]
            - 'tau_max': Maximum output torque [N·m]
            - 'tau_min': Minimum output torque [N·m]
            - 'cogging_amplitude': Cogging torque amplitude [N·m]
            - 'cogging_frequency': Cogging frequency [cycles/rev]
        """
        # Electrical parameters
        self.R: float = config.get('R', 1.0)  # Winding resistance [Ohm]
        self.L: float = config.get('L', 0.001)  # Winding inductance [H]
        self.K_t: float = config.get('K_t', 0.1)  # Torque constant [N·m/A]
        self.K_e: float = config.get('K_e', 0.1)  # Back-EMF constant [V·s/rad]
        
        # Saturation limits
        self.tau_max: float = config.get('tau_max', 10.0)  # Max torque [N·m]
        self.tau_min: float = config.get('tau_min', -10.0)  # Min torque [N·m]
        
        # Cogging torque parameters
        self.cogging_amplitude: float = config.get('cogging_amplitude', 0.05)  # [N·m]
        self.cogging_frequency: int = config.get('cogging_frequency', 12)  # cycles/rev
        
        # State variables
        self.current: float = 0.0  # Motor current [A]
        
    def _compute_cogging_torque(self, theta_m: float) -> float:
        """
        Compute periodic cogging torque as a function of rotor angle.
        
        The cogging torque is modeled as a non-sinusoidal periodic disturbance
        using a Fourier series approximation with fundamental and third harmonic.
        
        Parameters
        ----------
        theta_m : float
            Mechanical rotor angle [rad]
            
        Returns
        -------
        float
            Cogging torque [N·m]
        """
        # Fundamental component
        tau_cog = self.cogging_amplitude * np.sin(self.cogging_frequency * theta_m)
        
        # Add third harmonic for non-sinusoidal shape (30% of fundamental)
        tau_cog += 0.3 * self.cogging_amplitude * np.sin(3 * self.cogging_frequency * theta_m)
        
        return tau_cog
    
    def step(
        self, 
        v_cmd: float, 
        omega_m: float, 
        theta_m: float, 
        dt: float
    ) -> float:
        """
        Compute one time step of the motor dynamics.
        
        This method integrates the electrical dynamics using Euler forward integration
        and computes the output torque including non-ideal effects.
        
        Parameters
        ----------
        v_cmd : float
            Commanded voltage to motor driver [V]
        omega_m : float
            Current mechanical angular velocity [rad/s]
        theta_m : float
            Current mechanical rotor angle [rad]
        dt : float
            Time step [s]
            
        Returns
        -------
        float
            Output motor torque [N·m]
        """
        # Electrical dynamics: di/dt = (V_cmd - R*i - K_e*omega) / L
        back_emf = self.K_e * omega_m
        numerator = v_cmd - self.R * self.current - back_emf
        # Prevent division by zero and numerical instability
        L_safe = max(self.L, 1e-6)
        di_dt = numerator / L_safe
        
        # Clamp di/dt to prevent overflow
        di_dt = np.clip(di_dt, -1e7, 1e7)
        
        # Integrate current (Euler forward)
        self.current += di_dt * dt
        
        # Clamp current to reasonable range
        self.current = np.clip(self.current, -1e3, 1e3)
        
        # Compute ideal torque from current
        tau_ideal = self.K_t * self.current
        
        # Add cogging torque disturbance
        tau_cog = self._compute_cogging_torque(theta_m)
        tau_total = tau_ideal + tau_cog
        
        # Apply saturation limits
        tau_output = np.clip(tau_total, self.tau_min, self.tau_max)
        
        return tau_output
    
    def reset(self) -> None:
        """
        Reset the motor state to initial conditions.
        """
        self.current = 0.0
    
    def get_state(self) -> dict:
        """
        Get the current internal state of the motor model.
        
        Returns
        -------
        dict
            Dictionary containing current state variables
        """
        return {
            'current': self.current
        }
