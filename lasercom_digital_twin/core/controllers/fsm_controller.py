"""
Fast Steering Mirror Controller

This module implements the high-bandwidth fine pointing controller for the
Fast Steering Mirror (FSM), providing precision tip/tilt correction in the
optical frame. The FSM controller operates at high bandwidth (100-1000 Hz)
to reject disturbances and sensor noise that the coarse gimbal cannot handle.

Control Architecture:
--------------------
The FSM controller receives:
1. Feedforward: Residual error from coarse loop (low frequency)
2. Feedback: QPD error signal (high frequency)

Output:
- FSM deflection command [rad] in tip/tilt coordinates

Key Features:
- PI control in optical frame (decoupled from mechanical roll)
- Saturation monitoring and handling
- Optional high-pass filtering for pure disturbance rejection mode
"""

import numpy as np
from typing import Tuple, Optional, Dict


class FSMController:
    """
    High-bandwidth PI controller for Fast Steering Mirror.
    
    This controller operates in the optical frame (O-frame), processing
    tip/tilt errors from the QPD and commanding FSM deflections to null
    the line-of-sight error. The controller is designed for high bandwidth
    (100-1000 Hz) to provide disturbance rejection beyond the capability
    of the coarse gimbal.
    
    Control Law:
    -----------
    e(t) = [e_tip, e_tilt]^T  (from QPD)
    
    u(t) = K_p * e(t) + K_i * ∫e(τ)dτ
    
    where u(t) is the FSM deflection command [rad].
    
    Optical Frame Operation:
    -----------------------
    The FSM operates independently of the mechanical gimbal's roll angle.
    The field rotation compensation (applied in coordinate transformations)
    ensures that the FSM's tip/tilt axes align with the detector frame.
    
    FSM Gain:
    --------
    The FSM actuator model includes a gain factor (typically 2×) that
    converts mirror angle to beam deflection. This controller outputs
    the desired beam deflection; the actuator model handles the conversion.
    """
    
    def __init__(self, config: dict):
        """
        Initialize FSM controller.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'kp': Proportional gains [rad/rad] (2-element: tip, tilt)
            - 'ki': Integral gains [rad/(rad·s)]
            - 'fsm_deflection_max': Max FSM angle [rad] (typically ±1 deg)
            - 'fsm_deflection_min': Min FSM angle [rad]
            - 'enable_feedforward': Use coarse loop residual [bool]
            - 'high_pass_filter_enabled': Pure disturbance rejection mode [bool]
            - 'high_pass_cutoff_hz': HPF cutoff frequency [Hz]
        """
        self.config = config
        
        # PI gains (2-DOF: tip and tilt)
        self.kp: np.ndarray = np.array(config.get('kp', [1.0, 1.0]))
        self.ki: np.ndarray = np.array(config.get('ki', [100.0, 100.0]))
        
        # FSM deflection limits (physical travel)
        self.fsm_max: np.ndarray = np.array(
            config.get('fsm_deflection_max', [np.deg2rad(1.0), np.deg2rad(1.0)])
        )
        self.fsm_min: np.ndarray = np.array(
            config.get('fsm_deflection_min', [-np.deg2rad(1.0), -np.deg2rad(1.0)])
        )
        
        # Feedforward enable
        self.enable_feedforward: bool = config.get('enable_feedforward', True)
        
        # High-pass filter for pure disturbance rejection
        self.use_high_pass: bool = config.get('high_pass_filter_enabled', False)
        self.hpf_cutoff_hz: float = config.get('high_pass_cutoff_hz', 0.1)
        self.hpf_cutoff_rad_s: float = 2 * np.pi * self.hpf_cutoff_hz
        
        # Controller state
        self.integral: np.ndarray = np.zeros(2)  # Integral term
        self.hpf_state: np.ndarray = np.zeros(2)  # High-pass filter state
        
        # Saturation tracking
        self.saturation_active: np.ndarray = np.zeros(2, dtype=bool)
        self.saturation_duration: np.ndarray = np.zeros(2)  # Cumulative time [s]
        
    def _apply_high_pass_filter(
        self,
        error: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Apply high-pass filter to error signal.
        
        This removes DC and low-frequency components, forcing the FSM to
        only correct high-frequency disturbances while the coarse loop
        handles steady-state tracking.
        
        First-order HPF:
            H(s) = s / (s + ωc)
        
        Discrete implementation:
            y[k] = α * (y[k-1] + x[k] - x[k-1])
            where α = τ / (τ + dt), τ = 1/ωc
        
        Parameters
        ----------
        error : np.ndarray
            Raw error signal [rad]
        dt : float
            Time step [s]
            
        Returns
        -------
        np.ndarray
            High-pass filtered error [rad]
        """
        if not self.use_high_pass:
            return error
        
        # Time constant
        tau = 1.0 / self.hpf_cutoff_rad_s
        alpha = tau / (tau + dt)
        
        # Discrete high-pass filter
        # Note: This is a simplified implementation
        # For production, use proper discrete filter design
        filtered_error = alpha * self.hpf_state + alpha * error
        self.hpf_state = filtered_error.copy()
        
        return filtered_error
    
    def compute_control(
        self,
        qpd_error: np.ndarray,
        dt: float,
        coarse_residual: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute FSM deflection command.
        
        Parameters
        ----------
        qpd_error : np.ndarray
            Pointing error from QPD [rad] (2-element: tip, tilt)
            This is the high-bandwidth feedback signal
        dt : float
            Time step [s]
        coarse_residual : Optional[np.ndarray]
            Residual error from coarse loop [rad] for feedforward
            If None, only uses QPD feedback
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            - command: FSM deflection command [rad] (2-element)
            - metadata: Dictionary with control signals and diagnostics
        """
        # Apply high-pass filter to error (if enabled)
        error_filtered = self._apply_high_pass_filter(qpd_error, dt)
        
        # Total error (feedback + feedforward)
        if self.enable_feedforward and coarse_residual is not None:
            # Feedforward from coarse loop helps prevent FSM saturation
            # during large tracking maneuvers
            total_error = error_filtered + coarse_residual
        else:
            total_error = error_filtered
        
        # Proportional term
        u_p = self.kp * total_error
        
        # Integral term
        u_i = self.ki * self.integral
        
        # Total control (before saturation)
        u_unsaturated = u_p + u_i
        
        # Apply FSM deflection limits
        u_saturated = np.clip(u_unsaturated, self.fsm_min, self.fsm_max)
        
        # Check saturation
        saturation_error = u_saturated - u_unsaturated
        self.saturation_active = np.abs(saturation_error) > 1e-9
        
        # Update saturation duration tracking
        for i in range(2):
            if self.saturation_active[i]:
                self.saturation_duration[i] += dt
            else:
                self.saturation_duration[i] = 0.0
        
        # Update integrator (with simple anti-windup: stop integration when saturated)
        for i in range(2):
            if not self.saturation_active[i]:
                self.integral[i] += total_error[i] * dt
            # Optionally: apply back-calculation anti-windup similar to coarse controller
        
        # Metadata for logging
        metadata = {
            'error_raw': qpd_error,
            'error_filtered': error_filtered,
            'total_error': total_error,
            'u_p': u_p,
            'u_i': u_i,
            'integral': self.integral.copy(),
            'saturated': self.saturation_active.copy(),
            'saturation_duration': self.saturation_duration.copy(),
            'feedforward_used': coarse_residual is not None and self.enable_feedforward
        }
        
        return u_saturated, metadata
    
    def is_saturated(self) -> bool:
        """
        Check if any FSM axis is currently saturated.
        
        Returns
        -------
        bool
            True if either axis is at physical limit
        """
        return np.any(self.saturation_active)
    
    def get_saturation_report(self) -> Dict:
        """
        Generate saturation diagnostic report.
        
        This method provides detailed information about FSM saturation,
        which is critical for system performance monitoring. Frequent
        saturation indicates:
        1. Coarse loop bandwidth is insufficient
        2. Disturbances exceed FSM authority
        3. Tracking maneuver is too aggressive
        
        Returns
        -------
        Dict
            Saturation diagnostics including duration and severity
        """
        return {
            'tip_saturated': self.saturation_active[0],
            'tilt_saturated': self.saturation_active[1],
            'tip_saturation_duration_s': self.saturation_duration[0],
            'tilt_saturation_duration_s': self.saturation_duration[1],
            'any_saturated': self.is_saturated(),
            'current_command': None  # Filled by calling code if needed
        }
    
    def reset(self) -> None:
        """
        Reset controller state to initial conditions.
        """
        self.integral = np.zeros(2)
        self.hpf_state = np.zeros(2)
        self.saturation_active = np.zeros(2, dtype=bool)
        self.saturation_duration = np.zeros(2)
    
    def get_state(self) -> Dict:
        """
        Get current controller state.
        
        Returns
        -------
        Dict
            Controller state variables
        """
        return {
            'integral': self.integral.copy(),
            'saturated': self.saturation_active.copy(),
            'saturation_duration': self.saturation_duration.copy(),
            'hpf_state': self.hpf_state.copy()
        }
