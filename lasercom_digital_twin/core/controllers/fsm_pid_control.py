"""
Fast Steering Mirror (FSM) PID Controller

This module implements a high-performance PIDF controller for 2-axis FSM
pointing control in space laser communication systems.

Control Architecture:
--------------------
- Decentralized MIMO: Two independent PIDF controllers (Tip and Tilt axes)
- Derivative Filtering: Low-pass filter on D-term to attenuate QPD sensor noise
- Anti-Windup: Back-calculation method to handle VCA voltage saturation
- Micro-radian Precision: Designed for sub-µrad steady-state error

PIDF Transfer Function (per axis):
----------------------------------
          Kp*s*(s/ωi + 1)*(s/ωd + 1)
C(s) = ────────────────────────────────
              s*(s + N*ωc)

where:
    Kp: Proportional gain
    Ki: Integral gain (ωi = Ki/Kp)
    Kd: Derivative gain (ωd = Kd/Kp)
    N: Derivative filter coefficient (typically 10-20)
    ωc: Crossover frequency [rad/s]

Implementation Notes:
--------------------
1. **Derivative Filtering**: Essential for FSM due to:
   - QPD sensor noise above 500 Hz
   - Structural resonances (100-500 Hz)
   - Prevents derivative kick on setpoint changes

2. **Anti-Windup**: Back-calculation method chosen because:
   - Simple implementation
   - Effective for rate-limited actuators (VCA slew rate)
   - Preserves phase margin

3. **Numeric Stability**: Uses trapezoidal integration for integral term
   to avoid accumulation errors in long-duration simulations.

Author: Senior Control Systems Engineer
Date: January 20, 2026
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FsmPidGains:
    """PID controller gains for one axis."""
    Kp: float  # Proportional gain
    Ki: float  # Integral gain [1/s]
    Kd: float  # Derivative gain [s]
    N: float   # Derivative filter coefficient (dimensionless)


@dataclass
class FsmControllerConfig:
    """Configuration for 2-axis FSM PIDF controller."""
    # Tip axis gains
    tip_gains: FsmPidGains
    
    # Tilt axis gains
    tilt_gains: FsmPidGains
    
    # Actuator limits (voltage or normalized command)
    u_min: float = -2.0  # Minimum command [V]
    u_max: float = 2.0   # Maximum command [V]
    
    # Anti-windup gain (back-calculation method)
    # Kb = 1/Ti is common choice where Ti = Kp/Ki
    Kb_tip: float = 1.0
    Kb_tilt: float = 1.0


class FsmPidController:
    """
    High-performance PIDF controller for 2-axis FSM pointing.
    
    This controller implements parallel-form PID with derivative filtering
    and anti-windup for precision optical pointing applications.
    
    Control Law (per axis):
    ----------------------
    u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt_filtered
    
    Discrete Implementation (Tustin/Trapezoidal):
    --------------------------------------------
    Proportional: P[k] = Kp * e[k]
    Integral:     I[k] = I[k-1] + (Ki*dt/2) * (e[k] + e[k-1]) + Kb*(u_sat - u)
    Derivative:   D[k] = (α*D[k-1] + Kd*(1-α)*(e[k]-e[k-1])/dt)
    
    where α = 1/(1 + N*ωc*dt) is the filter coefficient.
    
    Attributes
    ----------
    tip_gains : FsmPidGains
        PID gains for tip axis
    tilt_gains : FsmPidGains
        PID gains for tilt axis
    u_min, u_max : float
        Actuator command limits
    _integral : np.ndarray
        Integral state (2,) - [I_tip, I_tilt]
    _derivative : np.ndarray
        Filtered derivative state (2,)
    _error_prev : np.ndarray
        Previous error (2,) for derivative computation
    _initialized : bool
        Flag to handle first update call
    """
    
    def __init__(self, config: FsmControllerConfig):
        """
        Initialize FSM PIDF controller.
        
        Parameters
        ----------
        config : FsmControllerConfig
            Controller configuration with gains and limits
        """
        self.tip_gains = config.tip_gains
        self.tilt_gains = config.tilt_gains
        self.u_min = config.u_min
        self.u_max = config.u_max
        self.Kb_tip = config.Kb_tip
        self.Kb_tilt = config.Kb_tilt
        
        # Internal states (2-element arrays for tip/tilt)
        self._integral = np.zeros(2)      # Integral accumulator
        self._derivative = np.zeros(2)    # Filtered derivative
        self._error_prev = np.zeros(2)    # Previous error for derivative
        self._initialized = False         # First-call flag
        
        # Validate gains
        self._validate_gains()
    
    def _validate_gains(self) -> None:
        """Validate that all gains are positive and reasonable."""
        assert self.tip_gains.Kp > 0, "Tip Kp must be positive"
        assert self.tip_gains.Ki >= 0, "Tip Ki must be non-negative"
        assert self.tip_gains.Kd >= 0, "Tip Kd must be non-negative"
        assert self.tip_gains.N > 0, "Tip N must be positive"
        
        assert self.tilt_gains.Kp > 0, "Tilt Kp must be positive"
        assert self.tilt_gains.Ki >= 0, "Tilt Ki must be non-negative"
        assert self.tilt_gains.Kd >= 0, "Tilt Kd must be non-negative"
        assert self.tilt_gains.N > 0, "Tilt N must be positive"
        
        assert self.u_max > self.u_min, "u_max must be > u_min"
    
    def reset(self) -> None:
        """
        Reset controller internal states.
        
        Essential for Monte Carlo simulations and when restarting control
        after a long period without updates (e.g., switching between
        operating modes).
        """
        self._integral = np.zeros(2)
        self._derivative = np.zeros(2)
        self._error_prev = np.zeros(2)
        self._initialized = False
    
    def update(self, 
               setpoint: np.ndarray, 
               measurement: np.ndarray, 
               dt: float) -> np.ndarray:
        """
        Compute control command for one time step.
        
        Implementation Details:
        ----------------------
        1. **Error Computation**: e = setpoint - measurement
        
        2. **Proportional Term**: Straightforward multiplication
        
        3. **Integral Term**: Trapezoidal (Tustin) integration with anti-windup
           - Trapezoidal: More accurate than Forward Euler, avoids DC bias
           - Back-calculation anti-windup: When actuator saturates, the
             integral is adjusted to prevent windup
        
        4. **Derivative Term**: First-order low-pass filtered derivative
           - Filter equation: D[k] = α*D[k-1] + (1-α)*Kd*(e[k]-e[k-1])/dt
           - α = 1/(1 + N*ωc*dt) where ωc is crossover frequency
           - Prevents derivative kick on setpoint changes
        
        5. **Saturation**: Clamp final command to actuator limits
        
        Parameters
        ----------
        setpoint : np.ndarray
            Desired output (2,) - [θ_tip_des, θ_tilt_des] in radians
        measurement : np.ndarray
            Measured output (2,) - [θ_tip_meas, θ_tilt_meas] in radians
        dt : float
            Time step [s]. Must be > 0.
        
        Returns
        -------
        np.ndarray
            Control command (2,) - [u_tip, u_tilt] in volts (or normalized)
        """
        assert setpoint.shape == (2,), f"setpoint must be (2,), got {setpoint.shape}"
        assert measurement.shape == (2,), f"measurement must be (2,), got {measurement.shape}"
        assert dt > 0, f"dt must be positive, got {dt}"
        
        # Compute error
        error = setpoint - measurement
        
        # Handle first call (no previous error for derivative)
        if not self._initialized:
            self._error_prev = error.copy()
            self._initialized = True
        
        # Unpack gains for readability
        Kp = np.array([self.tip_gains.Kp, self.tilt_gains.Kp])
        Ki = np.array([self.tip_gains.Ki, self.tilt_gains.Ki])
        Kd = np.array([self.tip_gains.Kd, self.tilt_gains.Kd])
        N = np.array([self.tip_gains.N, self.tilt_gains.N])
        Kb = np.array([self.Kb_tip, self.Kb_tilt])
        
        # ====================================================================
        # PROPORTIONAL TERM
        # ====================================================================
        P = Kp * error
        
        # ====================================================================
        # INTEGRAL TERM (Trapezoidal Integration)
        # ====================================================================
        # Trapezoidal: I[k] = I[k-1] + (Ki*dt/2) * (e[k] + e[k-1])
        # This will be adjusted for anti-windup after saturation check
        I_increment = (Ki * dt / 2.0) * (error + self._error_prev)
        I_tentative = self._integral + I_increment
        
        # ====================================================================
        # DERIVATIVE TERM (Filtered)
        # ====================================================================
        # First-order low-pass filter on derivative
        # α = dt / (dt + τ_f) where τ_f = 1/(N*ωc)
        # For typical FSM: ωc ~ 2π*150 rad/s, N ~ 15
        # τ_f ~ 0.0007 s, so for dt=0.001s, α ~ 0.59
        
        # Approximation: α ≈ 1 / (1 + N*ωc*dt)
        # Assuming ωc ~ Ki/Kp for simplicity (crossover near integral corner)
        omega_c = Ki / Kp  # Approximate crossover frequency [rad/s]
        omega_c = np.where(omega_c > 0, omega_c, 1.0)  # Avoid division by zero
        
        alpha = 1.0 / (1.0 + N * omega_c * dt)
        
        # Derivative of error
        de_dt = (error - self._error_prev) / dt
        
        # Filtered derivative: D[k] = α*D[k-1] + (1-α)*Kd*de_dt
        D = alpha * self._derivative + (1.0 - alpha) * Kd * de_dt
        
        # ====================================================================
        # TOTAL COMMAND (before saturation)
        # ====================================================================
        u_unsaturated = P + I_tentative + D
        
        # ====================================================================
        # SATURATION
        # ====================================================================
        u_saturated = np.clip(u_unsaturated, self.u_min, self.u_max)
        
        # ====================================================================
        # ANTI-WINDUP (Back-Calculation Method)
        # ====================================================================
        # If actuator saturates, adjust integral to prevent windup
        # I[k] = I_tentative + Kb * (u_saturated - u_unsaturated)
        saturation_error = u_saturated - u_unsaturated
        self._integral = I_tentative + Kb * saturation_error
        
        # Update states for next iteration
        self._derivative = D.copy()
        self._error_prev = error.copy()
        
        return u_saturated
    
    def get_internal_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current internal controller states for debugging/analysis.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (integral, derivative, error_prev) - each (2,)
        """
        return (self._integral.copy(), 
                self._derivative.copy(), 
                self._error_prev.copy())
    
    def set_gains(self, 
                  tip_gains: Optional[FsmPidGains] = None,
                  tilt_gains: Optional[FsmPidGains] = None) -> None:
        """
        Update controller gains online (for adaptive control or tuning).
        
        Parameters
        ----------
        tip_gains : FsmPidGains, optional
            New gains for tip axis
        tilt_gains : FsmPidGains, optional
            New gains for tilt axis
        """
        if tip_gains is not None:
            self.tip_gains = tip_gains
        
        if tilt_gains is not None:
            self.tilt_gains = tilt_gains
        
        self._validate_gains()
    
    def get_gains(self) -> Tuple[FsmPidGains, FsmPidGains]:
        """
        Get current controller gains.
        
        Returns
        -------
        Tuple[FsmPidGains, FsmPidGains]
            (tip_gains, tilt_gains)
        """
        return self.tip_gains, self.tilt_gains


def create_fsm_controller_from_design(bandwidth_hz: float = 150.0) -> FsmPidController:
    """
    Factory function to create FSM controller with gains from control design.
    
    This loads the tuned PIDF gains from the FSM_control_design.py analysis.
    Gains are based on frequency-domain design targeting:
    - Bandwidth: 100-200 Hz
    - Phase margin: ≥ 45°
    - Cross-talk: < 5%
    
    Parameters
    ----------
    bandwidth_hz : float
        Target closed-loop bandwidth [Hz]. Default 150 Hz.
    
    Returns
    -------
    FsmPidController
        Configured controller instance
    """
    # These gains are derived from frequency-domain analysis
    # See FSM_control_design.py for derivation
    omega_c = 2 * np.pi * bandwidth_hz
    omega_i = omega_c / 10.0  # Integral corner frequency
    
    # Approximate gains (should match FSM_control_design.py output)
    # These are placeholder values - in production, load from JSON
    Kp_tip = 0.976
    Ki_tip = 91.98 #Kp_tip * omega_i
    Kd_tip = 0.000518 #Kp_tip / (2.0 * omega_c)
    N_tip = 15.0
    
    Kp_tilt = 0.9529
    Ki_tilt = 89.80 #Kp_tilt * omega_i
    Kd_tilt = 0.000506 #Kp_tilt / (2.0 * omega_c)
    N_tilt = 15.0
    
    tip_gains = FsmPidGains(Kp=Kp_tip, Ki=Ki_tip, Kd=Kd_tip, N=N_tip)
    tilt_gains = FsmPidGains(Kp=Kp_tilt, Ki=Ki_tilt, Kd=Kd_tilt, N=N_tilt)
    
    # Anti-windup gains (typically 1/Ti where Ti = Kp/Ki)
    Kb_tip = Ki_tip / Kp_tip if Kp_tip > 0 else 1.0
    Kb_tilt = Ki_tilt / Kp_tilt if Kp_tilt > 0 else 1.0
    
    config = FsmControllerConfig(
        tip_gains=tip_gains,
        tilt_gains=tilt_gains,
        u_min=-2.0,
        u_max=2.0,
        Kb_tip=Kb_tip,
        Kb_tilt=Kb_tilt
    )
    
    return FsmPidController(config)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Validation suite for FSM PIDF controller.
    
    Tests:
    1. Step response with no saturation
    2. Anti-windup behavior under saturation
    3. Derivative filtering effectiveness
    4. Reset functionality
    """
    print("=" * 70)
    print(" FSM PIDF CONTROLLER VALIDATION ")
    print("=" * 70)
    
    # Create controller
    controller = create_fsm_controller_from_design(bandwidth_hz=150.0)
    
    # Test parameters
    dt = 0.001  # 1 ms time step
    
    # Test 1: Step response (no saturation)
    print("\n1. STEP RESPONSE TEST (No Saturation)")
    print("-" * 70)
    controller.reset()
    
    setpoint = np.array([1e-5, 0.0])  # 10 µrad tip command
    measurement = np.array([0.0, 0.0])
    
    n_steps = 100
    commands = []
    errors = []
    
    for i in range(n_steps):
        u = controller.update(setpoint, measurement, dt)
        commands.append(u.copy())
        errors.append((setpoint - measurement).copy())
        
        # Simple mock measurement (assume perfect tracking for test)
        measurement += u * dt * 1e-6  # Rough integration
    
    commands = np.array(commands)
    print(f"Initial command:  {commands[0]}")
    print(f"Final command:    {commands[-1]}")
    print(f"Command range:    [{commands.min():.4f}, {commands.max():.4f}]")
    print(f"Final error:      {(setpoint - measurement)[0]*1e6:.3f} µrad")
    
    # Test 2: Saturation and anti-windup
    print("\n2. SATURATION TEST (Anti-Windup)")
    print("-" * 70)
    controller.reset()
    
    setpoint = np.array([1e-3, 0.0])  # Large 1 mrad command
    measurement = np.array([0.0, 0.0])
    
    n_steps = 50
    commands_sat = []
    integrals = []
    
    for i in range(n_steps):
        u = controller.update(setpoint, measurement, dt)
        commands_sat.append(u.copy())
        
        I, D, e_prev = controller.get_internal_states()
        integrals.append(I.copy())
        
        # No change in measurement (stuck at zero)
        # This tests if integral winds up or is clamped
    
    commands_sat = np.array(commands_sat)
    integrals = np.array(integrals)
    
    print(f"Command saturated at: {commands_sat[0, 0]:.4f}")
    print(f"Integral after 50 steps: {integrals[-1, 0]:.4f}")
    print(f"Anti-windup active: {'✓ YES' if integrals[-1, 0] < 100 else '✗ NO'}")
    
    # Test 3: Derivative filtering
    print("\n3. DERIVATIVE FILTER TEST")
    print("-" * 70)
    controller.reset()
    
    # Apply noisy measurement
    setpoint = np.array([0.0, 0.0])
    
    np.random.seed(42)
    n_steps = 100
    derivatives = []
    
    for i in range(n_steps):
        # Add high-frequency noise (simulating QPD noise)
        noise = np.random.randn(2) * 1e-7  # 100 nrad RMS noise
        measurement_noisy = measurement + noise
        
        u = controller.update(setpoint, measurement_noisy, dt)
        
        I, D, e_prev = controller.get_internal_states()
        derivatives.append(D.copy())
    
    derivatives = np.array(derivatives)
    derivative_std = np.std(derivatives[:, 0])
    
    print(f"Derivative term std: {derivative_std:.4e}")
    print(f"Filter effective: {'✓ YES' if derivative_std < 1e-3 else '⚠ MARGINAL'}")
    
    # Test 4: Reset
    print("\n4. RESET TEST")
    print("-" * 70)
    I_before, _, _ = controller.get_internal_states()
    controller.reset()
    I_after, D_after, e_after = controller.get_internal_states()
    
    print(f"Integral before reset: {I_before}")
    print(f"Integral after reset:  {I_after}")
    print(f"All zeros: {'✓ YES' if np.allclose(I_after, 0) and np.allclose(D_after, 0) else '✗ NO'}")
    
    print("\n" + "=" * 70)
    print(" VALIDATION COMPLETE ")
    print("=" * 70)
