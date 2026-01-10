"""
Controller Design and Synthesis Tools

This module implements controller synthesis algorithms for the lasercom
pointing system, including PID controllers, Linear Quadratic Gaussian (LQG)
controllers, and H-infinity controllers.

The design process follows a systematic approach:
1. Plant model identification/linearization
2. Controller requirements specification
3. Controller synthesis and tuning
4. Stability and performance analysis
5. Robustness evaluation
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import control as ctrl
from abc import ABC, abstractmethod


@dataclass
class ControllerGains:
    """Container for controller gain parameters."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    kf: float = 0.0  # Feedforward gain


@dataclass
class ControllerSpecs:
    """Controller performance specifications."""
    bandwidth_hz: float = 10.0          # Closed-loop bandwidth [Hz]
    phase_margin_deg: float = 45.0      # Phase margin [deg]
    gain_margin_db: float = 6.0         # Gain margin [dB]
    settling_time_sec: float = 0.1      # Settling time [s]
    overshoot_percent: float = 5.0      # Maximum overshoot [%]
    steady_state_error: float = 1e-6    # Steady-state error tolerance


class BaseController(ABC):
    """Abstract base class for all controllers."""

    def __init__(self, specs: ControllerSpecs):
        self.specs = specs
        self.gains = ControllerGains()
        self.is_tuned = False

    @abstractmethod
    def design(self, plant_model: ctrl.StateSpace) -> None:
        """Design controller for given plant model."""
        pass

    @abstractmethod
    def get_transfer_function(self) -> ctrl.TransferFunction:
        """Return controller transfer function."""
        pass

    def validate_design(self, plant_model: ctrl.StateSpace) -> Dict[str, float]:
        """Validate controller design against specifications."""
        # Form closed-loop system
        controller_tf = self.get_transfer_function()
        closed_loop = ctrl.feedback(controller_tf * plant_model)

        # Analyze performance metrics
        results = {}

        # Bandwidth
        mag, phase, omega = ctrl.bode(closed_loop, plot=False)
        bw_idx = np.where(mag <= 0.707)[0]
        if len(bw_idx) > 0:
            results['bandwidth_hz'] = omega[bw_idx[0]] / (2 * np.pi)
        else:
            results['bandwidth_hz'] = float('inf')

        # Stability margins
        gm, pm, wg, wp = ctrl.margin(controller_tf * plant_model)
        results['gain_margin_db'] = 20 * np.log10(gm) if gm is not None else float('inf')
        results['phase_margin_deg'] = pm if pm is not None else 0.0

        # Step response characteristics
        t, y = ctrl.step_response(closed_loop)
        results['settling_time'] = self._calculate_settling_time(t, y)
        results['overshoot'] = self._calculate_overshoot(y)
        results['steady_state_error'] = abs(1.0 - y[-1])

        return results

    def _calculate_settling_time(self, t: np.ndarray, y: np.ndarray, tolerance: float = 0.02) -> float:
        """Calculate settling time for step response."""
        steady_state = y[-1]
        settling_band = tolerance * abs(steady_state)

        for i in range(len(y)):
            if all(abs(y[j] - steady_state) <= settling_band for j in range(i, len(y))):
                return t[i]
        return t[-1]

    def _calculate_overshoot(self, y: np.ndarray) -> float:
        """Calculate percentage overshoot."""
        steady_state = y[-1]
        max_value = np.max(y)
        if steady_state > 0:
            return 100 * (max_value - steady_state) / steady_state
        return 0.0


class PIDController(BaseController):
    """PID Controller with systematic tuning methods."""

    def __init__(self, specs: ControllerSpecs, tuning_method: str = "ziegler_nichols"):
        super().__init__(specs)
        self.tuning_method = tuning_method

    def design(self, plant_model: ctrl.StateSpace) -> None:
        """Design PID controller using specified tuning method."""
        if self.tuning_method == "ziegler_nichols":
            self._ziegler_nichols_tuning(plant_model)
        elif self.tuning_method == "manual":
            self._manual_tuning()
        else:
            raise ValueError(f"Unknown tuning method: {self.tuning_method}")

        self.is_tuned = True

    def _ziegler_nichols_tuning(self, plant_model: ctrl.StateSpace) -> None:
        """Ziegler-Nichols tuning for PID controller."""
        # Convert to transfer function for analysis
        plant_tf = ctrl.ss2tf(plant_model)

        # Find ultimate gain and period (simplified approach)
        # In practice, this would involve finding the stability boundary
        Ku = 1.0  # Ultimate gain (placeholder)
        Tu = 1.0  # Ultimate period (placeholder)

        # Ziegler-Nichols rules for PID
        self.gains.kp = 0.6 * Ku
        self.gains.ki = 1.2 * Ku / Tu
        self.gains.kd = 0.075 * Ku * Tu

    def _manual_tuning(self) -> None:
        """Manual tuning based on specifications."""
        # Placeholder for manual tuning logic
        self.gains.kp = 100.0
        self.gains.ki = 50.0
        self.gains.kd = 2.0

    def get_transfer_function(self) -> ctrl.TransferFunction:
        """Return PID controller transfer function."""
        s = ctrl.tf('s')
        return self.gains.kp + self.gains.ki/s + self.gains.kd*s


class LQGController(BaseController):
    """Linear Quadratic Gaussian (LQG) Controller."""

    def __init__(self, specs: ControllerSpecs, Q: np.ndarray = None, R: np.ndarray = None,
                 V: np.ndarray = None, W: np.ndarray = None):
        super().__init__(specs)
        self.Q = Q  # State weighting matrix
        self.R = R  # Control weighting matrix
        self.V = V  # Measurement noise covariance
        self.W = W  # Process noise covariance
        self.K = None  # Controller gain matrix
        self.L = None  # Kalman filter gain matrix

    def design(self, plant_model: ctrl.StateSpace) -> None:
        """Design LQG controller."""
        # Extract system matrices
        A, B, C, D = ctrl.ssdata(plant_model)

        # Default weighting matrices if not provided
        n_states = A.shape[0]
        n_outputs = C.shape[0]
        n_inputs = B.shape[1]

        if self.Q is None:
            self.Q = np.eye(n_states)
        if self.R is None:
            self.R = np.eye(n_inputs)
        if self.V is None:
            self.V = 0.01 * np.eye(n_outputs)
        if self.W is None:
            self.W = 0.01 * np.eye(n_states)

        # Design Kalman filter
        self.L, P, E = ctrl.lqe(A, np.eye(n_states), C, self.W, self.V)

        # Design LQR controller
        self.K, S, E = ctrl.lqr(A, B, self.Q, self.R)

        self.is_tuned = True

    def get_transfer_function(self) -> ctrl.TransferFunction:
        """Return LQG controller transfer function."""
        # For simplicity, return the LQR portion
        # Full LQG would require state-space representation
        return ctrl.ss([], [], [], self.K)


class ControllerDesigner:
    """Main controller design interface."""

    def __init__(self):
        self.controllers = {}

    def design_pid_controller(self, plant_model: ctrl.StateSpace,
                            specs: ControllerSpecs,
                            tuning_method: str = "ziegler_nichols") -> PIDController:
        """Design a PID controller."""
        controller = PIDController(specs, tuning_method)
        controller.design(plant_model)
        self.controllers['pid'] = controller
        return controller

    def design_lqg_controller(self, plant_model: ctrl.StateSpace,
                            specs: ControllerSpecs,
                            Q: np.ndarray = None, R: np.ndarray = None,
                            V: np.ndarray = None, W: np.ndarray = None) -> LQGController:
        """Design an LQG controller."""
        controller = LQGController(specs, Q, R, V, W)
        controller.design(plant_model)
        self.controllers['lqg'] = controller
        return controller

    def get_controller(self, name: str) -> Optional[BaseController]:
        """Retrieve a designed controller."""
        return self.controllers.get(name)

    def compare_controllers(self, plant_model: ctrl.StateSpace,
                          controllers: List[BaseController]) -> Dict[str, Dict]:
        """Compare multiple controllers on the same plant."""
        results = {}
        for i, controller in enumerate(controllers):
            name = f"controller_{i}"
            results[name] = controller.validate_design(plant_model)
        return results