"""
Hierarchical Control Laws for Laser Communication Terminal

This module implements the two-level control architecture for precision pointing:
- Level 1: Coarse Gimbal Control (low bandwidth, large authority)
- Level 2: Fine FSM Control (high bandwidth, small authority)

The hierarchical structure ensures:
1. Coarse loop handles large-angle slewing and tracking
2. Fine loop provides high-bandwidth disturbance rejection
3. Proper decoupling prevents actuator saturation and interaction

Control Architecture:
--------------------
                [Target Ephemeris]
                        |
                        v
    +------- [Coarse PID Controller] -------+
    |                   |                    |
    |                   v                    |
    |          [Gimbal Motors]               |
    |                   |                    |
    |                   v                    |
    |            [Mechanical Gimbal]         |
    |                   |                    |
    |    +--------------+                    |
    |    |                                   |
    |    v                                   |
    | [Sensors: Encoders/Gyros]              |
    |    |                                   |
    |    v                                   |
    | [State Estimator/EKF] <----------------+
    |    |
    |    +-----------> [Residual Error] -----> [FSM PI Controller]
    |                                                    |
    |                                                    v
    +-----------------------------------> [FSM Actuator]
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ControllerState:
    """Container for controller internal state."""
    integral: np.ndarray
    previous_error: np.ndarray
    previous_output: np.ndarray
    saturation_flag: bool


class BaseController(ABC):
    """
    Abstract base class for all controllers.
    
    Defines the standard interface for control law implementation including
    initialization, state management, and step-wise computation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the controller.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with controller-specific parameters
        """
        self.config = config
        
    @abstractmethod
    def compute_control(
        self, 
        reference: np.ndarray, 
        measurement: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """
        Compute control command for one time step.
        
        Parameters
        ----------
        reference : np.ndarray
            Desired setpoint or trajectory
        measurement : np.ndarray
            Measured/estimated system state
        dt : float
            Time step [s]
            
        Returns
        -------
        np.ndarray
            Control command
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset controller to initial state.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict:
        """
        Get current controller state for logging/debugging.
        
        Returns
        -------
        Dict
            Dictionary containing controller state variables
        """
        pass


class CoarseGimbalController(BaseController):
    """
    Coarse gimbal PID controller for azimuth/elevation axes.
    
    This controller implements a robust PID control law for the mechanical
    gimbal, handling large-angle slewing and coarse tracking. Key features:
    
    - Independent PID control for Az and El axes
    - Back-calculation anti-windup to prevent integrator wind-up
    - Setpoint filtering for smooth reference tracking
    - Rate limiting on commanded torque
    - Decoupling feedforward to fine control loop
    
    Control Law:
    -----------
    e(t) = r(t) - y(t)
    
    u(t) = K_p * e(t) + K_i * ∫e(τ)dτ + K_d * de/dt
    
    Anti-Windup:
    -----------
    When actuator saturates (u_sat ≠ u), the integrator is modified:
    
    d(integral)/dt = e(t) + K_aw * (u_sat - u)
    
    where K_aw is the anti-windup gain (typically 1/K_i).
    """
    
    def __init__(self, config: dict):
        """
        Initialize coarse gimbal controller.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'kp': Proportional gains [N·m/rad] (2-element array for Az/El)
            - 'ki': Integral gains [N·m/(rad·s)]
            - 'kd': Derivative gains [N·m·s/rad]
            - 'tau_max': Maximum torque [N·m] (2-element array)
            - 'tau_min': Minimum torque [N·m] (2-element array)
            - 'tau_rate_limit': Max torque rate [N·m/s]
            - 'anti_windup_gain': Back-calculation gain (typically 1/ki)
            - 'derivative_filter_N': Derivative filter coefficient (typically 10-20)
            - 'setpoint_filter_wn': Setpoint filter natural frequency [rad/s]
            - 'setpoint_filter_zeta': Setpoint filter damping ratio
            - 'enable_derivative': Enable derivative term [bool]
        """
        super().__init__(config)
        
        # PID gains (2-DOF: Pan/Az and Tilt/El)
        # Default to optimized gains from linearization-based design
        # Pan axis: Higher inertia due to carrying tilt assembly
        # Tilt axis: Lower inertia, requires proportionally lower gains
        self.kp: np.ndarray = np.array(config.get('kp', [3.257, 0.661]))
        self.ki: np.ndarray = np.array(config.get('ki', [10.232, 2.078]))
        self.kd: np.ndarray = np.array(config.get('kd', [0.146599, 0.029709]))
        
        # Torque limits
        self.tau_max: np.ndarray = np.array(config.get('tau_max', [10.0, 10.0]))
        self.tau_min: np.ndarray = np.array(config.get('tau_min', [-10.0, -10.0]))
        self.tau_rate_limit: float = config.get('tau_rate_limit', 100.0)  # N·m/s
        
        # Anti-windup configuration (back-calculation method)
        # Default: Kb = 1/Ti = Ki/Kp for proper anti-windup scaling
        default_aw_gain = self.ki / (self.kp + 1e-10)
        self.anti_windup_gain: np.ndarray = np.array(
            config.get('anti_windup_gain', default_aw_gain)
        )
        
        # Derivative filter (low-pass to attenuate encoder noise)
        # Transfer function: D_filtered(s) = (Kd*s) / (1 + s/(N*wc))
        # Typical N = 10-20 for motion control
        self.derivative_filter_N: float = config.get('derivative_filter_N', 15.0)
        
        # Setpoint filter parameters
        self.use_setpoint_filter: bool = config.get('use_setpoint_filter', False)
        self.filter_wn: float = config.get('setpoint_filter_wn', 10.0)  # rad/s
        self.filter_zeta: float = config.get('setpoint_filter_zeta', 0.7)
        
        # Derivative control enable
        self.enable_derivative: bool = config.get('enable_derivative', True)
        
        # Controller state
        self.integral: np.ndarray = np.zeros(2)  # Integral term
        self.previous_error: np.ndarray = np.zeros(2)  # For derivative
        self.filtered_derivative: np.ndarray = np.zeros(2)  # Filtered D-term
        self.previous_output: np.ndarray = np.zeros(2)  # For rate limiting
        self.filtered_reference: np.ndarray = np.zeros(2)  # Filtered setpoint
        self.filter_state: np.ndarray = np.zeros(2)  # Filter state variable
        self.previous_reference: np.ndarray = np.zeros(2)  # For reference velocity calculation
        
        # Saturation tracking
        self.saturation_active: np.ndarray = np.zeros(2, dtype=bool)
        
        # PID component logging (for debugging)
        self.last_u_p: np.ndarray = np.zeros(2)
        self.last_u_i: np.ndarray = np.zeros(2)
        self.last_u_d: np.ndarray = np.zeros(2)
        
    def _apply_setpoint_filter(
        self,
        reference: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Apply second-order setpoint filter to reference command.
        
        The filter smooths rapid reference changes to prevent excitation of
        structural modes and reduce actuator stress.
        
        Filter: H(s) = ωn² / (s² + 2ζωn·s + ωn²)
        
        State-space form:
            dx/dt = [0, 1; -ωn², -2ζωn] * x + [0; ωn²] * r
            y = [1, 0] * x
        
        Parameters
        ----------
        reference : np.ndarray
            Raw reference command [rad]
        dt : float
            Time step [s]
            
        Returns
        -------
        np.ndarray
            Filtered reference [rad]
        """
        if not self.use_setpoint_filter:
            return reference
        
        # Second-order filter state-space (per axis)
        wn = self.filter_wn
        zeta = self.filter_zeta
        
        for i in range(2):
            # State: [position, velocity]
            # Simplified Euler integration for demonstration
            # In production, use RK4 or similar
            
            accel = wn**2 * (reference[i] - self.filtered_reference[i]) - \
                    2 * zeta * wn * self.filter_state[i]
            
            self.filter_state[i] += accel * dt
            self.filtered_reference[i] += self.filter_state[i] * dt
        
        return self.filtered_reference.copy()
    
    def _compute_anti_windup(
        self,
        error: np.ndarray,
        unsaturated_output: np.ndarray,
        saturated_output: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute anti-windup correction for integrator.
        
        Back-Calculation Method:
        -----------------------
        When the actuator saturates, feed back the difference between
        saturated and unsaturated outputs to prevent integrator wind-up.
        
        d(integral)/dt = e(t) + K_aw * (u_sat - u)
        
        This ensures the integrator "knows" about saturation and doesn't
        continue to accumulate error that cannot be corrected.
        
        Parameters
        ----------
        error : np.ndarray
            Current tracking error [rad]
        unsaturated_output : np.ndarray
            Control output before saturation [N·m]
        saturated_output : np.ndarray
            Control output after saturation [N·m]
        dt : float
            Time step [s]
            
        Returns
        -------
        np.ndarray
            Integrator update including anti-windup correction
        """
        # Standard integral update
        integral_update = error * dt
        
        # Anti-windup correction
        saturation_error = saturated_output - unsaturated_output
        anti_windup_correction = self.anti_windup_gain * saturation_error
        
        # Update saturation flags
        self.saturation_active = np.abs(saturation_error) > 1e-6
        
        # Combined update
        return integral_update + anti_windup_correction
    
    def compute_control(
        self,
        reference: np.ndarray,
        measurement: np.ndarray,
        dt: float,
        velocity_estimate: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute PID control command for gimbal axes.
        
        Parameters
        ----------
        reference : np.ndarray
            Desired position [rad] (2-element: Az, El)
        measurement : np.ndarray
            Measured/estimated position [rad] (2-element: Az, El)
        dt : float
            Time step [s]
        velocity_estimate : Optional[np.ndarray]
            Estimated velocity [rad/s] for derivative term
            If None, uses finite difference
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            - command: Commanded torque [N·m] (2-element)
            - metadata: Dictionary with control signals and diagnostics
        """
        # Apply setpoint filter
        reference_filtered = self._apply_setpoint_filter(reference, dt)
        
        # Compute tracking error
        error = reference_filtered - measurement
        
        # Proportional term
        u_p = self.kp * error
        
        # Integral term (using current integral state)
        u_i = self.ki * self.integral
        
        # Derivative term with low-pass filtering
        if self.enable_derivative:
            if velocity_estimate is not None:
                # Compute reference velocity from finite difference
                # CRITICAL: Must account for changing reference during slew maneuvers
                reference_velocity = (reference_filtered - self.previous_reference) / (dt + 1e-10)
                
                # Correct derivative: error_derivative = desired_vel - actual_vel
                error_derivative = reference_velocity - velocity_estimate
            else:
                # Finite difference (derivative on measurement, not error)
                error_derivative = (error - self.previous_error) / (dt + 1e-10)
            
            # Apply first-order low-pass filter to derivative term
            # Discrete filter: D[k] = α*D[k-1] + (1-α)*Kd*de/dt
            # where α = 1/(1 + N*wc*dt) and wc ≈ Ki/Kp (crossover freq)
            omega_c = self.ki / (self.kp + 1e-10)  # Approximate crossover
            alpha = 1.0 / (1.0 + self.derivative_filter_N * omega_c * dt)
            
            # Update filtered derivative
            raw_derivative = self.kd * error_derivative
            self.filtered_derivative = (alpha * self.filtered_derivative + 
                                       (1.0 - alpha) * raw_derivative)
            
            u_d = self.filtered_derivative
        else:
            u_d = np.zeros(2)
            error_derivative = np.zeros(2)
        
        # Total control (before saturation)
        u_unsaturated = u_p + u_i + u_d
        
        # Apply rate limiting
        if self.tau_rate_limit > 0:
            delta_u_max = self.tau_rate_limit * dt
            delta_u = u_unsaturated - self.previous_output
            delta_u = np.clip(delta_u, -delta_u_max, delta_u_max)
            u_rate_limited = self.previous_output + delta_u
        else:
            u_rate_limited = u_unsaturated
        
        # Apply torque saturation
        u_saturated = np.clip(u_rate_limited, self.tau_min, self.tau_max)
        
        # Update integrator with anti-windup
        integral_update = self._compute_anti_windup(
            error, u_rate_limited, u_saturated, dt
        )
        self.integral += integral_update
        
        # Update state
        self.previous_error = error.copy()
        self.previous_output = u_saturated.copy()
        self.previous_reference = reference_filtered.copy()
        
        # Store PID components for logging
        self.last_u_p = u_p.copy()
        self.last_u_i = u_i.copy()
        self.last_u_d = u_d.copy()
        
        # Metadata for logging and debugging
        metadata = {
            'error': error,
            'error_derivative': error_derivative,
            'u_p': u_p,
            'u_i': u_i,
            'u_d': u_d,
            'integral': self.integral.copy(),
            'saturated': self.saturation_active.copy(),
            'reference_filtered': reference_filtered,
            'gains': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        }
        
        return u_saturated, metadata
    
    def get_residual_error_for_fsm(
        self,
        reference: np.ndarray,
        measurement: np.ndarray
    ) -> np.ndarray:
        """
        Compute residual tracking error for FSM feedforward.
        
        This method provides the low-frequency residual error that the
        coarse loop cannot fully correct. The FSM uses this as a feedforward
        term to handle the steady-state portion, leaving only high-frequency
        disturbances for feedback correction.
        
        Decoupling Strategy:
        -------------------
        The FSM receives:
        1. Feedforward: Estimated steady-state error from coarse loop
        2. Feedback: High-frequency error from QPD
        
        This prevents FSM saturation during large tracking maneuvers.
        
        Parameters
        ----------
        reference : np.ndarray
            Target position [rad]
        measurement : np.ndarray
            Current position [rad]
            
        Returns
        -------
        np.ndarray
            Residual error [rad] to be passed to FSM controller
        """
        # Current tracking error
        error = reference - measurement
        
        # For proper decoupling, could low-pass filter this error
        # to extract only the slow component. For simplicity, return raw error.
        # In a full implementation, apply a low-pass filter here.
        
        return error
    
    def reset(self) -> None:
        """
        Reset controller state to initial conditions.
        """
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.filtered_derivative = np.zeros(2)
        self.previous_output = np.zeros(2)
        self.filtered_reference = np.zeros(2)
        self.filter_state = np.zeros(2)
        self.saturation_active = np.zeros(2, dtype=bool)
        self.last_u_p = np.zeros(2)
        self.last_u_i = np.zeros(2)
        self.last_u_d = np.zeros(2)
    
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
                'previous_error': self.previous_error.copy(),
                'filtered_derivative': self.filtered_derivative.copy(),
                'previous_output': self.previous_output.copy(),
                'saturated': self.saturation_active.copy(),
                'filtered_reference': self.filtered_reference.copy(),
                'u_p': self.last_u_p.copy(),
                'u_i': self.last_u_i.copy(),
                'u_d': self.last_u_d.copy()
            }


class FeedbackLinearizationController(BaseController):
    """
    Feedback Linearization Controller for a 2-DOF LaserCom Gimbal.
    
    This controller transforms the nonlinear dynamics into a linear system
    by canceling out M(q), C(q, dq), G(q), and estimated disturbances.
    
    Signal Flow:
    -----------
    1. Sensors (encoders, gyros, QPD) → Raw measurements
    2. State Estimator (EKF) → Filtered state + disturbance estimate
    3. This Controller → Uses filtered state for feedback linearization
    4. Output → Motor torque commands
    
    Control Law:
    -----------
    tau = M(q) * [ddq_ref + Kp*e + Kd*de] + C(q, dq)*dq + G(q) - d_hat
    
    where:
    - M(q): Inertia matrix from dynamics model
    - C(q, dq): Coriolis/centrifugal terms
    - G(q): Gravity torque
    - d_hat: Disturbance estimate from EKF
    - e = q_ref - q: Position error
    - de = dq_ref - dq: Velocity error
    """

    def __init__(self, config: dict, dynamics_model, ndob=None):
        """
        Initialize feedback linearization controller.
        
        Parameters
        ----------
        config : dict
            Controller configuration:
            - 'kp': Proportional gains [1/s²] (2-element for Az/El)
            - 'kd': Derivative gains [1/s] 
            - 'ki': Integral gains [1/s³] (optional, for robust tracking)
            - 'tau_max': Maximum torque [N·m]
            - 'tau_min': Minimum torque [N·m]
            - 'enable_integral': Enable integral action (default: False)
        dynamics_model : GimbalDynamics
            Instance of the physics model from gimbal_dynamics.py
            Must have methods: get_mass_matrix, get_coriolis_matrix, get_gravity_vector
        ndob : NonlinearDisturbanceObserver, optional
            Model-based observer for rejecting unmodeled disturbances.
        """
        super().__init__(config)
        self.dyn = dynamics_model
        self.ndob = ndob
        
        # PID Gains for the outer loop (linearized space)
        # Default gains are designed for critically damped response with 5ms motor lag
        # wn ~ 10 rad/s, zeta = 1 → kp = wn^2 = 100, kd = 2*zeta*wn = 20
        self.kp = np.array(config.get('kp', [100.0, 0.0]))
        self.kd = np.array(config.get('kd', [20.0, 0.0]))
        self.ki = np.array(config.get('ki', [10.0, 0.0]))
        self.enable_integral = config.get('enable_integral', False)
        
        # Friction compensation coefficients (must match plant friction!)
        # If the plant applies tau_net = tau_motor - D*dq, we must add D*dq to command
        self.friction_az = config.get('friction_az', 0.0)  # N·m·s/rad
        self.friction_el = config.get('friction_el', 0.0)  # N·m·s/rad
        
        # Conditional friction compensation (CRITICAL for stability)
        # When True, only compensate friction if velocity is in same direction as desired acceleration
        # This prevents friction feedforward from fighting the controller during transients
        self.conditional_friction = config.get('conditional_friction', True)
        
        # Robust/Sliding Mode Term (handles model uncertainty)
        # Adds a switching term: -eta * sign(s) where s = error_dot + lambda * error
        # This provides robustness to unmodeled dynamics and parameter variations
        self.enable_robust_term = config.get('enable_robust_term', False)
        self.robust_eta = np.array(config.get('robust_eta', [0.01, 0.01]))  # N·m switching gain
        self.robust_lambda = config.get('robust_lambda', 5.0)  # Sliding surface slope
        self.robust_epsilon = config.get('robust_epsilon', 0.01)  # Boundary layer for smooth switching
        
        # Disturbance compensation (from EKF estimates)
        # WARNING: EKF disturbance estimates may be noisy/biased during transients
        # Set to False for more robust behavior until EKF is properly tuned
        self.enable_disturbance_compensation = config.get('enable_disturbance_compensation', False)
        
        # Actuator limits
        self.tau_max = np.array(config.get('tau_max', [10.0, 10.0]))
        self.tau_min = np.array(config.get('tau_min', [-10.0, -10.0]))
        
        # State tracking
        self.previous_output = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.integral = np.zeros(2)

    def compute_control(
        self, 
        q_ref: np.ndarray, 
        dq_ref: np.ndarray,
        state_estimate: Dict[str, float],
        dt: float,
        ddq_ref: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute torque using Feedback Linearization.
        
        Signal Flow Explanation:
        -----------------------
        The state_estimate parameter comes from the Extended Kalman Filter (EKF)
        in the estimators folder. The EKF fuses:
        - Encoder measurements (absolute position)
        - Gyroscope data (angular velocity)
        - QPD measurements (fine pointing error)
        
        The EKF provides:
        - Filtered joint positions (theta_az, theta_el)
        - Filtered joint velocities (theta_dot_az, theta_dot_el)
        - Disturbance estimates (dist_az, dist_el)
        
        This controller uses these filtered states to:
        1. Cancel nonlinear dynamics (M, C, G terms)
        2. Compensate estimated disturbances
        3. Achieve linear closed-loop behavior
        
        Parameters
        ----------
        q_ref : np.ndarray
            Desired joint positions [rad] (2-element: Az, El)
        dq_ref : np.ndarray
            Desired joint velocities [rad/s]
        state_estimate : Dict
            Filtered state from EKF containing:
            - 'theta_az': Azimuth position [rad]
            - 'theta_el': Elevation position [rad]
            - 'theta_dot_az': Azimuth velocity [rad/s]
            - 'theta_dot_el': Elevation velocity [rad/s]
            - 'dist_az': Estimated disturbance torque on Az axis [N·m]
            - 'dist_el': Estimated disturbance torque on El axis [N·m]
        dt : float
            Time step [s]
        ddq_ref : Optional[np.ndarray]
            Desired joint accelerations [rad/s²]. If None, assumes zero.
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            - command: Commanded torque [N·m] (2-element)
            - metadata: Dict with intermediate signals for logging
        """
        # Handle default acceleration reference
        if ddq_ref is None:
            ddq_ref = np.zeros(2)
        
        # 1. Extract estimated states from EKF
        # The EKF in estimators/state_estimator.py provides these filtered estimates
        q = np.array([
            state_estimate['theta_az'], 
            state_estimate['theta_el']
        ])
        dq = np.array([
            state_estimate['theta_dot_az'], 
            state_estimate['theta_dot_el']
        ])
        
        # Disturbance estimates from EKF (only used if enabled)
        if self.enable_disturbance_compensation:
            d_hat = np.array([
                state_estimate['dist_az'], 
                state_estimate['dist_el']
            ])
        else:
            d_hat = np.zeros(2)  # Ignore EKF disturbance estimates

        # 2. Compute Linearizing Terms (Physics cancellation)
        # These methods are from GimbalDynamics in dynamics/gimbal_dynamics.py
        M = self.dyn.get_mass_matrix(q)
        C = self.dyn.get_coriolis_matrix(q, dq)
        G = self.dyn.get_gravity_vector(q)

        # 2.1 Update Nonlinear Disturbance Observer (NDOB)
        # Uses the dynamics model and measurements to estimate lumped disturbances
        # tau_prev is the torque applied during the previous step (k-1)
        d_hat_ndob = np.zeros(2)
        if self.ndob is not None:
            # Update observer and get estimate
            # dt is the control cycle duration
            d_hat_ndob = self.ndob.update(q, dq, self.previous_output, dt)

        # 3. Outer Loop: Define the desired acceleration in linearized space
        error = q_ref - q
        error_dot = dq_ref - dq
        
        # Update integral if enabled
        if self.enable_integral:
            self.integral += error * dt
            # Simple anti-windup: clamp integral
            integral_max = 1.0  # rad·s
            self.integral = np.clip(self.integral, -integral_max, integral_max)
        
        # Virtual control input (desired acceleration in linearized coordinates)
        # This is the "v" in the standard feedback linearization formulation
        # PID outer loop: v = Kp*e + Kd*de + Ki*∫e
        v = ddq_ref + self.kp * error + self.kd * error_dot
        if self.enable_integral:
            v += self.ki * self.integral

        # 4. Nonlinear Inverse Dynamics with Disturbance Compensation
        # Transform virtual control to actual torque by inverting the dynamics
        # tau = M*v + C*dq + G + D*dq - d_hat + u_robust
        # 
        # Explanation:
        # - M*v: Inertial torque needed for desired acceleration
        # - C*dq: Compensation for Coriolis/centrifugal effects
        # - G: Compensation for gravity torque
        # - D*dq: Compensation for viscous friction (CONDITIONAL - see below)
        # - d_hat: Feedforward compensation for estimated disturbances
        # - u_robust: Sliding mode term for robustness
        
        # Friction compensation with CONDITIONAL logic
        # Only compensate friction when velocity is in the same direction as desired control
        # This prevents friction feedforward from fighting the controller during transients
        # (e.g., when overshooting and trying to decelerate, friction helps slow down!)
        friction_coeff = np.array([self.friction_az, self.friction_el])
        
        if self.conditional_friction:
            # Compute desired acceleration direction from outer loop
            desired_accel_sign = np.sign(v)
            velocity_sign = np.sign(dq)
            
            # Only compensate friction if velocity and desired acceleration are aligned
            # If they oppose (overshoot scenario), let plant friction help slow down
            aligned = (desired_accel_sign * velocity_sign) >= 0
            friction_comp = np.where(aligned, friction_coeff * dq, np.zeros(2))
        else:
            # Standard friction compensation (always active)
            friction_comp = friction_coeff * dq
        
        # Robust/Sliding Mode Term for handling model uncertainties
        # Uses a smoothed sign function (tanh) to avoid chattering
        if self.enable_robust_term:
            # Sliding surface: s = error_dot + lambda * error
            s = error_dot + self.robust_lambda * error
            
            # Smoothed switching: tanh(s/epsilon) instead of sign(s)
            # This creates a boundary layer for continuous control
            u_robust = -self.robust_eta * np.tanh(s / self.robust_epsilon)
        else:
            u_robust = np.zeros(2)
        
        # Commanded torque (IMPROVED FORMULATION)
        # tau = M*v + C*dq + G + friction_comp - d_hat + d_hat_ndob + u_robust
        # Note: we ADD d_hat_ndob because the observer estimates the disturbance d
        # acting on the plant (tau + d = M*ddq + ...), so compensation is tau = ... - d_hat
        # Wait, usually it is tau_total = tau_nominal - d_hat.
        # Let's check NDOB sign. The NDOB module says: tau + d = M*ddq + ...
        # So tau = M*ddq + ... - d. Thus tau_cmd = (...) - d_hat.
        tau = M @ v + C @ dq + G + friction_comp - d_hat - d_hat_ndob + u_robust

        # 5. Apply Actuator Saturation
        u_saturated = np.clip(tau, self.tau_min, self.tau_max)
        
        # Update state
        self.previous_output = u_saturated.copy()
        self.previous_error = error.copy()
        
        # Metadata for logging and debugging
        metadata = {
            'error': error,
            'error_dot': error_dot,
            'v_signal': v,
            'M_matrix': M,
            'C_term': C @ dq,
            'G_term': G,
            'friction_comp': friction_comp,
            'u_robust': u_robust if self.enable_robust_term else np.zeros(2),
            'dist_compensated': d_hat,
            'tau_unsaturated': tau,
            'saturation_active': np.any(u_saturated != tau),
            'conditional_friction_active': self.conditional_friction
        }

        return u_saturated, metadata

    def reset(self) -> None:
        """Reset controller state."""
        self.previous_output = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.integral = np.zeros(2)

    def get_state(self) -> Dict:
        """Get current controller state."""
        return {
            'kp': self.kp.copy(),
            'kd': self.kd.copy(),
            'previous_output': self.previous_output.copy(),
            'previous_error': self.previous_error.copy()
        }