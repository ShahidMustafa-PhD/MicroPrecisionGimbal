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
from dataclasses import dataclass, field


@dataclass
class FsmPidGains:
    """PID controller gains for one axis."""
    Kp: float        # Proportional gain
    Ki: float        # Integral gain [1/s]
    Kd: float        # Derivative gain [s]
    omega_f: float   # Derivative filter cutoff frequency [rad/s]

@dataclass
class NotchFilterConfig:
    """Configuration for a structural notch filter."""
    enabled: bool = False
    f_center_hz: float = 1000.0  # Center frequency of the notch
    zeta_zero: float = 0.01      # Depth of the notch (smaller = deeper)
    zeta_pole: float = 0.707      # Width of the notch (larger = wider)


@dataclass
class FsmControllerConfig:
    """Configuration for 2-axis FSM PIDF controller."""
    # Tip axis gains
    tip_gains: FsmPidGains
    
    # Tilt axis gains
    tilt_gains: FsmPidGains
    
    # Structural Filter
    notch_config: NotchFilterConfig = field(default_factory=NotchFilterConfig)
    
    # Actuator limits (voltage or normalized command)
    u_min: float = -50.0  # Minimum command [V] (Updated to PI limits)
    u_max: float = 50.0   # Maximum command [V]
    
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
        self.notch_config = config.notch_config
        self.u_min = config.u_min
        self.u_max = config.u_max
        self.Kb_tip = config.Kb_tip
        self.Kb_tilt = config.Kb_tilt
        
        # Internal states (2-element arrays for tip/tilt)
        self._integral = np.zeros(2)      # Integral accumulator
        self._derivative = np.zeros(2)    # Filtered derivative
        self._error_prev = np.zeros(2)    # Previous error for derivative
        self._initialized = False         # First-call flag
        
        # Biquad Notch Filter States: x = input, y = output. [tip, tilt]
        self._notch_x1 = np.zeros(2) # x[k-1]
        self._notch_x2 = np.zeros(2) # x[k-2]
        self._notch_y1 = np.zeros(2) # y[k-1]
        self._notch_y2 = np.zeros(2) # y[k-2]
        
        # Store dt to recompute notch coefficients only if dt changes
        self._last_dt = -1.0
        self._b = np.zeros(3)
        self._a = np.zeros(3)
        
        # Validate gains
        self._validate_gains()
    
    def _validate_gains(self) -> None:
        """Validate that all gains are positive and reasonable."""
        assert self.tip_gains.Kp > 0, "Tip Kp must be positive"
        assert self.tip_gains.Ki >= 0, "Tip Ki must be non-negative"
        assert self.tip_gains.Kd >= 0, "Tip Kd must be non-negative"
        assert self.tip_gains.omega_f > 0, "Tip omega_f must be positive"
        
        assert self.tilt_gains.Kp > 0, "Tilt Kp must be positive"
        assert self.tilt_gains.Ki >= 0, "Tilt Ki must be non-negative"
        assert self.tilt_gains.Kd >= 0, "Tilt Kd must be non-negative"
        assert self.tilt_gains.omega_f > 0, "Tilt omega_f must be positive"
        
        assert self.u_max > self.u_min, "u_max must be > u_min"

    def _compute_notch_coefficients(self, dt: float) -> None:
        """Compute digital biquad coefficients using Tustin transform with pre-warping."""
        if not self.notch_config.enabled:
            return
            
        wn = 2.0 * np.pi * self.notch_config.f_center_hz
        zz = self.notch_config.zeta_zero
        zp = self.notch_config.zeta_pole
        
        # Pre-warped frequency to ensure exact notch placement in discrete time
        W = np.tan(wn * dt / 2.0)
        
        # Bilinear transform denominators and numerators
        den = 1.0 + 2.0 * zp * W + W**2
        
        self._b[0] = (1.0 + 2.0 * zz * W + W**2) / den
        self._b[1] = (2.0 * (W**2 - 1.0)) / den
        self._b[2] = (1.0 - 2.0 * zz * W + W**2) / den
        
        # Note: a0 = 1.0 by definition
        self._a[1] = (2.0 * (W**2 - 1.0)) / den
        self._a[2] = (1.0 - 2.0 * zp * W + W**2) / den
        
        self._last_dt = dt
    
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
        self._notch_x1.fill(0.0)
        self._notch_x2.fill(0.0)
        self._notch_y1.fill(0.0)
        self._notch_y2.fill(0.0)
    
    def hold_integrator(self) -> None:
        """
        Hold integrator state (prevent accumulation).
        
        Call this when the beam is off-sensor to prevent integrator windup.
        The integrator value is frozen at its current value until normal
        updates resume. This prevents the FSM from "winding up" while
        waiting for coarse gimbal to bring the beam back on-sensor.
        
        Usage in simulation_runner.py:
            if not is_beam_on_sensor:
                self.fsm_pid.hold_integrator()
        """
        # Reset error_prev to current value to prevent derivative kick
        # when updates resume. Also sets initialized flag to False so
        # next update treats it as a fresh start for derivative.
        self._initialized = False
        # Zero the filtered derivative to prevent old state affecting new updates
        self._derivative = np.zeros(2)
        # Integrator keeps its current value (no reset, just no accumulation)
    
    def update(self, setpoint: np.ndarray, measurement: np.ndarray, dt: float) -> np.ndarray:
        if self.notch_config.enabled and dt != self._last_dt:
            self._compute_notch_coefficients(dt)
            
        error = setpoint - measurement
        
        if not self._initialized:
            self._error_prev = error.copy()
            self._initialized = True
            
        Kp = np.array([self.tip_gains.Kp, self.tilt_gains.Kp])
        Ki = np.array([self.tip_gains.Ki, self.tilt_gains.Ki])
        Kd = np.array([self.tip_gains.Kd, self.tilt_gains.Kd])
        omega_f = np.array([self.tip_gains.omega_f, self.tilt_gains.omega_f])
        
        # 1. Proportional Term
        P = Kp * error
        
        # 2. Filtered Derivative Term
        alpha = 1.0 / (1.0 + omega_f * dt)
        de_dt = (error - self._error_prev) / dt
        D = alpha * self._derivative + (1.0 - alpha) * Kd * de_dt
        
        # 3. CONDITIONAL INTEGRATION (The Fix for the Limit Cycle)
        # Calculate a rough estimate of the command to check for saturation
        u_estimate = P + D + self._integral
        
        # Determine if the actuator is saturated AND the error is pushing it deeper into saturation
        is_saturated_max = (u_estimate >= self.u_max) & (error > 0)
        is_saturated_min = (u_estimate <= self.u_min) & (error < 0)
        
        # Only add to the integrator if we are NOT saturated in the direction of the error
        I_increment = np.zeros(2)
        for i in range(2):
            if not (is_saturated_max[i] or is_saturated_min[i]):
                I_increment[i] = (Ki[i] * dt / 2.0) * (error[i] + self._error_prev[i])
                
        I_tentative = self._integral + I_increment
        
        # 4. Base PID Command
        u_pid = P + I_tentative + D
        
        # 4.5. Pre-Notch Saturation
        # CRITICAL FIX for bang-bang limit cycles:
        # A hard clipper (saturation) generates infinite frequencies. If we saturate AFTER 
        # the notch filter, the flat-topping acts as a hard step into the plant, completely 
        # bypassing the notch filter's protection and wildly exciting the 1000 Hz resonance.
        # We MUST saturate the PID output first, then filter the resulting square-wave
        # edges to remove the 1000 Hz spectral content, before sending it to the plant.
        u_pid_sat = np.clip(u_pid, self.u_min, self.u_max)
        
        # 5. Notch Filter
        if self.notch_config.enabled:
            u_notch = (self._b[0] * u_pid_sat + 
                       self._b[1] * self._notch_x1 + 
                       self._b[2] * self._notch_x2 - 
                       self._a[1] * self._notch_y1 - 
                       self._a[2] * self._notch_y2)
            
            self._notch_x2 = self._notch_x1.copy()
            self._notch_x1 = u_pid_sat.copy()
            self._notch_y2 = self._notch_y1.copy()
            self._notch_y1 = u_notch.copy()
        else:
            u_notch = u_pid_sat
            
        # 6. Final Hardware Protection Saturation
        # The notch step response may overshoot slightly (~5%), so we maintain a final
        # hard clamp to protect the physical amplifier.
        u_saturated = np.clip(u_notch, self.u_min, self.u_max)
        
        # 7. State Updates
        self._integral = I_tentative.copy()
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

def create_fsm_controller_from_design(json_path: str = "fsm_controller_gains.json") -> FsmPidController:
    """
    Factory function that dynamically loads the optimal FSM controller gains
    directly from the JSON file generated by the offline design script.
    """
    import numpy as np
    import json
    import os
    
    # 1. Check if the design file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Cannot find FSM design file: '{json_path}'. "
            f"Please run FSM_control_design.py first to generate the controller gains."
        )
        
    # 2. Load the JSON configuration
    with open(json_path, 'r') as f:
        design_data = json.load(f)
        
    # Extract the PIDF dictionary and Notch dictionary
    pidf = design_data['pidf_controller']
    notch = pidf['notch']
    
    # Calculate crossover frequency in rad/s to compute omega_f
    omega_c = 2.0 * np.pi * pidf['crossover_freq_hz']
    
    # 3. Dynamically populate Tip Gains
    tip_gains = FsmPidGains(
        Kp=pidf['Kp_tip'], 
        Ki=pidf['Ki_tip'], 
        Kd=pidf['Kd_tip'], 
        omega_f=pidf['N_tip'] * omega_c  # Convert N to rad/s
    )
    
    # 4. Dynamically populate Tilt Gains
    tilt_gains = FsmPidGains(
        Kp=pidf['Kp_tilt'], 
        Ki=pidf['Ki_tilt'], 
        Kd=pidf['Kd_tilt'], 
        omega_f=pidf['N_tilt'] * omega_c # Convert N to rad/s
    )
    
    # 5. Anti-windup gains (back-calculation: Kb = Ki/Kp)
    Kb_tip = tip_gains.Ki / tip_gains.Kp if tip_gains.Kp > 0 else 1.0
    Kb_tilt = tilt_gains.Ki / tilt_gains.Kp if tilt_gains.Kp > 0 else 1.0
    
    # 6. Dynamically build the Notch Filter Configuration
    notch_cfg = NotchFilterConfig(
        enabled=True, 
        f_center_hz=notch['f_center_hz'], 
        zeta_zero=notch['zeta_zero'], 
        zeta_pole=notch['zeta_pole']
    )
    
    # 7. Final Controller Assembly
    config = FsmControllerConfig(
        tip_gains=tip_gains,
        tilt_gains=tilt_gains,
        notch_config=notch_cfg,
        u_min=-50.0,  # Physical Amplifier Limits
        u_max=50.0,
        Kb_tip=Kb_tip,
        Kb_tilt=Kb_tilt
    )
    
    print(f"INFO: FSM Controller dynamically loaded from {json_path}")
    print(f"      Design Bandwidth: {pidf['crossover_freq_hz']} Hz")
    
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
