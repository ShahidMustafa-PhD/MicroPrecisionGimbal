"""
Fast Steering Mirror (FSM) Actuator Model

This module provides ``FSMActuatorModel``, a nonlinear wrapper around the
linear state-space plant in ``FsmDynamics``.  It adds the practical effects
that are vendor- and hardware-specific:

  • **Position saturation** at mechanical stroke limits (±θ_max)
  • **Angular-rate limiting** (slew-rate constraint from servo amplifier)
  • **Piezoelectric hysteresis** (simplified rate-dependent Dahl model,
    relevant for PZT actuators when strain-gauge feedback is disabled)

Usage
-----
>>> from lasercom_digital_twin.core.dynamics.fsm_dynamics import FsmDynamicsConfig
>>> from lasercom_digital_twin.core.actuators.fsm_actuator import FSMActuatorModel
>>>
>>> # Vendor preset
>>> fsm = FSMActuatorModel(FsmDynamicsConfig.pi_s330())
>>> tip, tilt = fsm.step(cmd_tip=0.5, cmd_tilt=0.0, dt=1e-3)
>>>
>>> # Legacy dict-based config (backward compatible)
>>> fsm = FSMActuatorModel({'omega_n': 2000, 'zeta': 0.7,
...                         'alpha_max': 0.017, 'alpha_min': -0.017,
...                         'beta_max': 0.017, 'beta_min': -0.017})
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict
import math

from lasercom_digital_twin.core.dynamics.fsm_dynamics import (
    FsmDynamics,
    FsmDynamicsConfig,
    FSMActuationType,
    create_fsm_from_vendor,
)


class FSMActuatorModel:
    """
    Fast Steering Mirror actuator with nonlinear effects.

    Wraps a linear ``FsmDynamics`` state-space plant and adds:
      - Dahl-type hysteresis (PZT)
      - Angular rate limiting
      - Hard position saturation at ±θ_max

    The model supports two construction modes:

    1. **Physics-based** (recommended): pass a ``FsmDynamicsConfig`` from one
       of the vendor presets.
    2. **Legacy dict** (backward compatible): pass the old-style dict with
       ``omega_n``, ``zeta``, ``alpha_max``, etc.

    Attributes
    ----------
    dynamics : FsmDynamics
        Underlying linear state-space model.
    alpha, beta : float
        Current actual tip / tilt angles [rad].
    alpha_dot, beta_dot : float
        Current angular velocities [rad/s].
    """

    def __init__(self, config: Union[FsmDynamicsConfig, dict]):
        """
        Initialise FSM actuator model.

        Parameters
        ----------
        config : FsmDynamicsConfig or dict
            If ``FsmDynamicsConfig``: full physics-based vendor configuration.
            If ``dict``: legacy configuration with keys ``omega_n``, ``zeta``,
            ``alpha_max``, ``alpha_min``, etc.
        """
        if isinstance(config, FsmDynamicsConfig):
            self._init_from_physics_config(config)
        elif isinstance(config, dict):
            self._init_from_legacy_dict(config)
        else:
            raise TypeError(
                f"config must be FsmDynamicsConfig or dict, got {type(config)}"
            )

    # ==================================================================
    #  Initialisation helpers
    # ==================================================================

    def _init_from_physics_config(self, config: FsmDynamicsConfig) -> None:
        """Initialise from a physics-based FsmDynamicsConfig."""
        # Build state-space if not already done
        if config.A is None:
            config.build_state_space()

        self.dynamics = FsmDynamics(config)
        self._config = config

        # Position limits (symmetric)
        self.alpha_max: float = config.theta_max
        self.alpha_min: float = -config.theta_max
        self.beta_max: float = config.theta_max
        self.beta_min: float = -config.theta_max

        # Rate limit
        self.rate_limit: float = config.rate_limit

        # Hysteresis (for PZT only, converted from % to radians)
        self.hysteresis_width: float = (
            config.hysteresis_percent / 100.0 * config.theta_max
            if config.hysteresis_percent > 0 else 0.0
        )
        self.hysteresis_gain: float = 0.15 if self.hysteresis_width > 0 else 0.0

        # Dynamics parameters (for legacy API queries)
        self.omega_n: float = 2.0 * np.pi * config.f_n
        self.zeta: float = config.zeta

        # Current state
        self.alpha: float = 0.0
        self.alpha_dot: float = 0.0
        self.alpha_hyst: float = 0.0
        self.beta: float = 0.0
        self.beta_dot: float = 0.0
        self.beta_hyst: float = 0.0

        self._use_state_space = True

    def _init_from_legacy_dict(self, config: dict) -> None:
        """
        Initialise from legacy dict config (backward compatibility).

        Falls back to the original 2nd-order Euler integration per axis
        so that existing tests continue to pass unchanged.
        """
        self.omega_n: float = config.get('omega_n', 2000.0)
        self.zeta: float = config.get('zeta', 0.7)

        self.alpha_max: float = config.get('alpha_max', np.deg2rad(1.0))
        self.alpha_min: float = config.get('alpha_min', np.deg2rad(-1.0))
        self.beta_max: float  = config.get('beta_max', np.deg2rad(1.0))
        self.beta_min: float  = config.get('beta_min', np.deg2rad(-1.0))

        self.rate_limit: float = config.get('rate_limit', np.deg2rad(500.0))

        self.hysteresis_width: float = config.get('hysteresis_width', np.deg2rad(0.001))
        self.hysteresis_gain: float  = config.get('hysteresis_gain', 0.15)

        self.alpha: float = 0.0
        self.alpha_dot: float = 0.0
        self.alpha_hyst: float = 0.0
        self.beta: float = 0.0
        self.beta_dot: float = 0.0
        self.beta_hyst: float = 0.0

        self._use_state_space = False
        self.dynamics = None
        self._config = None

    # ==================================================================
    #  Hysteresis model (Dahl-type, rate-dependent)
    # ==================================================================

    def _update_hysteresis(
        self,
        position: float,
        hyst_state: float,
        cmd: float,
        dt: float
    ) -> float:
        """
        Update hysteresis state using rate-dependent Dahl model.

            dh/dt = σ · (cmd − position − h) · tanh(|rate| / 10)

        Parameters
        ----------
        position : float
            Current position [rad].
        hyst_state : float
            Current hysteresis state [rad].
        cmd : float
            Commanded position [rad].
        dt : float
            Timestep [s].

        Returns
        -------
        float
            Updated hysteresis state [rad].
        """
        if self.hysteresis_width <= 0:
            return 0.0

        error = float(cmd) - float(position) - float(hyst_state)
        cmd_rate = abs(error / (float(dt) + 1e-10))
        sigma = self.hysteresis_gain / (self.hysteresis_width + 1e-10)

        dhyst_dt = sigma * error * math.tanh(cmd_rate / 10.0)
        hyst_new = float(hyst_state) + dhyst_dt * float(dt)

        # Clamp magnitude
        hyst_new = max(-self.hysteresis_width, min(hyst_new, self.hysteresis_width))
        return hyst_new

    # ==================================================================
    #  Per-axis legacy stepping (Euler, no state-space)
    # ==================================================================

    def _step_axis_legacy(
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
        Compute one timestep for a single axis using Euler integration.

        This is the original implementation kept for backward compatibility
        with legacy dict-based configs and existing test suites.
        """
        hyst_state_new = float(self._update_hysteresis(position, hyst_state, cmd, dt))
        cmd_eff = float(cmd) - hyst_state_new

        # 2nd-order dynamics:  α̈ + 2ζωnα̇ + ωn²α = ωn²·α_cmd
        accel = (self.omega_n**2 * (cmd_eff - float(position))
                 - 2.0 * self.zeta * self.omega_n * float(velocity))

        velocity_new = float(velocity) + accel * dt
        velocity_new = max(-float(self.rate_limit),
                           min(velocity_new, float(self.rate_limit)))

        position_new = float(position) + velocity_new * dt
        position_new = max(float(pos_min), min(position_new, float(pos_max)))

        if (abs(position_new - float(pos_min)) < 1e-10 or
                abs(position_new - float(pos_max)) < 1e-10):
            velocity_new = 0.0

        return position_new, velocity_new, hyst_state_new

    # ==================================================================
    #  State-space stepping (PZT / VCA / Reluctance)
    # ==================================================================

    def _step_state_space(
        self,
        alpha_cmd: float,
        beta_cmd: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Advance one timestep using FsmDynamics (RK4) + nonlinear post-processing.

        Steps:
          1. Apply hysteresis distortion to command (PZT only).
          2. Step the linear state-space model forward (RK4).
          3. Apply rate limiting and position saturation.
        """
        # 1. Hysteresis on commands
        self.alpha_hyst = self._update_hysteresis(
            self.alpha, self.alpha_hyst, alpha_cmd, dt)
        self.beta_hyst = self._update_hysteresis(
            self.beta, self.beta_hyst, beta_cmd, dt)

        cmd_eff_tip = alpha_cmd - self.alpha_hyst
        cmd_eff_tilt = beta_cmd - self.beta_hyst

        # 2. State-space step (RK4 inside FsmDynamics)
        u = np.array([cmd_eff_tip, cmd_eff_tilt])
        y = self.dynamics.step(u, dt)

        # 3. Rate limiting (compute effective velocity, clamp)
        new_alpha = float(y[0])
        new_beta = float(y[1])

        alpha_dot = (new_alpha - self.alpha) / dt if dt > 0 else 0.0
        beta_dot = (new_beta - self.beta) / dt if dt > 0 else 0.0

        rl = float(self.rate_limit)
        if abs(alpha_dot) > rl:
            new_alpha = self.alpha + np.sign(alpha_dot) * rl * dt
        if abs(beta_dot) > rl:
            new_beta = self.beta + np.sign(beta_dot) * rl * dt

        # 4. Position saturation
        new_alpha = max(float(self.alpha_min), min(new_alpha, float(self.alpha_max)))
        new_beta = max(float(self.beta_min), min(new_beta, float(self.beta_max)))

        # Update velocities (finite difference after clamping)
        self.alpha_dot = (new_alpha - self.alpha) / dt if dt > 0 else 0.0
        self.beta_dot = (new_beta - self.beta) / dt if dt > 0 else 0.0

        # Zero velocity at hard limits
        if abs(new_alpha - self.alpha_min) < 1e-10 or abs(new_alpha - self.alpha_max) < 1e-10:
            self.alpha_dot = 0.0
        if abs(new_beta - self.beta_min) < 1e-10 or abs(new_beta - self.beta_max) < 1e-10:
            self.beta_dot = 0.0

        self.alpha = new_alpha
        self.beta = new_beta

        return self.alpha, self.beta

    # ==================================================================
    #  Main step interface
    # ==================================================================

    def step(
        self,
        alpha_cmd: float,
        beta_cmd: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute one timestep of the FSM dynamics for both axes.

        Parameters
        ----------
        alpha_cmd : float
            Commanded tip angle [rad].
        beta_cmd : float
            Commanded tilt angle [rad].
        dt : float
            Timestep [s].

        Returns
        -------
        Tuple[float, float]
            (alpha_actual, beta_actual) — actual tip/tilt angles [rad].
        """
        if self._use_state_space:
            return self._step_state_space(alpha_cmd, beta_cmd, dt)
        else:
            # Legacy path (Euler per-axis)
            self.alpha, self.alpha_dot, self.alpha_hyst = self._step_axis_legacy(
                self.alpha, self.alpha_dot, self.alpha_hyst,
                alpha_cmd, self.alpha_min, self.alpha_max, dt
            )
            self.beta, self.beta_dot, self.beta_hyst = self._step_axis_legacy(
                self.beta, self.beta_dot, self.beta_hyst,
                beta_cmd, self.beta_min, self.beta_max, dt
            )
            return self.alpha, self.beta

    # ==================================================================
    #  Stroke utilisation & saturation
    # ==================================================================

    @property
    def stroke_max(self) -> float:
        """Maximum stroke per axis [rad]."""
        return self.alpha_max

    def get_stroke_utilization(self) -> np.ndarray:
        """
        Per-axis stroke utilisation as fraction of θ_max.

        Returns
        -------
        np.ndarray
            (2,) with values in [0, 1].
        """
        return np.array([
            abs(self.alpha) / self.alpha_max if self.alpha_max > 0 else 0.0,
            abs(self.beta) / self.beta_max if self.beta_max > 0 else 0.0,
        ])

    def is_saturated(self, threshold: float = 0.98) -> bool:
        """
        Check if either axis is at or beyond the stroke threshold.

        Parameters
        ----------
        threshold : float
            Fraction of θ_max (default 0.98).

        Returns
        -------
        bool
            True if any axis exceeds the threshold.
        """
        util = self.get_stroke_utilization()
        return bool(np.any(util >= threshold))

    # ==================================================================
    #  Reset & state access
    # ==================================================================

    def reset(self) -> None:
        """Reset FSM state to initial conditions (all zeros)."""
        self.alpha = 0.0
        self.alpha_dot = 0.0
        self.alpha_hyst = 0.0
        self.beta = 0.0
        self.beta_dot = 0.0
        self.beta_hyst = 0.0

        if self.dynamics is not None:
            self.dynamics.reset()

    def get_state(self) -> dict:
        """
        Return current internal state for logging / diagnostics.

        Returns
        -------
        dict
            Keys: alpha, alpha_dot, alpha_hyst, beta, beta_dot, beta_hyst,
            stroke_utilization_tip, stroke_utilization_tilt, is_saturated.
        """
        util = self.get_stroke_utilization()
        return {
            'alpha': self.alpha,
            'alpha_dot': self.alpha_dot,
            'alpha_hyst': self.alpha_hyst,
            'beta': self.beta,
            'beta_dot': self.beta_dot,
            'beta_hyst': self.beta_hyst,
            'stroke_utilization_tip': util[0],
            'stroke_utilization_tilt': util[1],
            'is_saturated': self.is_saturated(),
        }

    def get_config_info(self) -> dict:
        """
        Return a summary of the FSM configuration for logging.

        Returns
        -------
        dict
            Vendor, type, stroke, resonance, damping, states, etc.
        """
        if self._config is not None:
            return self._config.get_info()
        return {
            'vendor': 'Legacy',
            'type': 'legacy_2nd_order',
            'stroke_mrad': self.alpha_max * 1e3,
            'f_n_Hz': self.omega_n / (2.0 * np.pi),
            'omega_n_rad_s': self.omega_n,
            'zeta': self.zeta,
            'n_states': 2,
        }
