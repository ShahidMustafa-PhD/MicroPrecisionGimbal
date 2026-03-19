"""
Mock simulation backend for GUI development.

Provides a lightweight `NdobaFbl` class that generates synthetic step data
matching the real `DigitalTwinRunner` signal schema.  Replace this import
with the production backend once the GUI skeleton is validated.

Signal schema mirrors: lasercom_digital_twin.core.simulation.simulation_runner.SimulationState
"""

import math
import time
import threading
from typing import Callable, Dict, Optional


class NdobaFbl:
    """
    Mock simulation engine for the dual-stage gimbal + FSM system.

    Accepts a subset of ``SimulationConfig`` kwargs so the GUI can pass its
    full config dict without modification.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.seed = self.config.get("seed", 42)

        # Read controller/disturbance flags for waveform shaping
        self._use_fbl   = self.config.get("use_feedback_linearization", False)
        self._use_ndob  = self.config.get("ndob_config", {}).get("enable", False)
        self._target_type = self.config.get("target_type", "constant")
        self._amplitude   = math.radians(self.config.get("target_amplitude", 1.0))  # → rad
        self._period      = self.config.get("target_period", 2.0)

        import numpy as np
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run_simulation(
        self,
        duration: float = 5.0,
        dt: float = 0.1,
        callback: Optional[Callable[[Dict], None]] = None,
        stop_flag: Optional[threading.Event] = None,
    ) -> Dict:
        if stop_flag is None:
            stop_flag = threading.Event()

        n_steps = int(duration / dt)
        step = 0

        for step in range(n_steps):
            if stop_flag.is_set():
                break

            t = step * dt
            step_data = self._generate_step_data(t, duration)

            if callback is not None:
                callback(step_data)

            time.sleep(dt)

        status = "cancelled" if stop_flag.is_set() else "completed"

        # Baseline noise root mean square bounds based on controller type
        if self._use_ndob:
            los_rms = 5.0e-6
            ss_az = 0.005  # rad (0.28 deg, easily PASS)
            sat = 0.0
            tau_rms = 0.05
            ts = 0.8
        elif self._use_fbl:
            los_rms = 18.0e-6
            ss_az = 0.012  # rad (0.68 deg, narrow PASS)
            sat = 2.0
            tau_rms = 0.08
            ts = 1.2
        else: # PID
            los_rms = 45.0e-6
            ss_az = 0.016  # rad (0.91 deg, FAIL)
            sat = 5.5
            tau_rms = 0.15
            ts = 2.4

        noise_factor = 1.0 + (self.rng.random() - 0.5) * 0.2

        if duration < 1.0:
            ss_az *= 5  # Heavily penalise short interrupted runs

        return {
            "status": status,
            "total_steps": step + 1,
            "final_time": (step + 1) * dt,
            "los_error_rms": los_rms * noise_factor,
            "los_error_final": los_rms * self.rng.random(),
            "ss_error_az": ss_az * noise_factor,
            "ss_error_el": ss_az * 0.8 * noise_factor,
            "settling_time_az": ts * noise_factor,
            "settling_time_el": ts * 0.9 * noise_factor,
            "torque_rms_az": tau_rms * noise_factor,
            "torque_rms_el": tau_rms * 0.5 * noise_factor,
            "fsm_saturation_pct": sat * noise_factor,
        }

    # ------------------------------------------------------------------ #
    #  Internal — synthetic step data generator                           #
    # ------------------------------------------------------------------ #

    def _target_signal(self, t: float) -> float:
        """Return target angle for the selected waveform at time *t*."""
        A = self._amplitude
        P = max(self._period, 1e-3)
        tt = self._target_type

        if tt == "sine":
            return A * math.sin(2 * math.pi * t / P)
        elif tt == "cosine":
            return A * math.cos(2 * math.pi * t / P)
        elif tt == "square":
            return A * (1.0 if math.sin(2 * math.pi * t / P) >= 0 else -1.0)
        elif tt == "hybridsig":
            # Slew for half the period then hold
            reach = math.radians(self.config.get("target_reachangle", 45.0))
            slew_t = P / 2.0
            frac = min(t / slew_t, 1.0)
            return reach * frac
        else:  # constant
            return 0.0

    def _generate_step_data(self, t: float, duration: float) -> Dict:
        noise = self.rng.normal(0, 1e-6)

        # Target signal
        tgt = self._target_signal(t)

        # FBL converges ~3× faster than PID; NDOB adds another 25% improvement
        if self._use_ndob:
            decay_k = 2.5
        elif self._use_fbl:
            decay_k = 2.0
        else:
            decay_k = 0.8

        decay = math.exp(-decay_k * t)

        q_az = tgt + 0.01 * decay * math.cos(2 * math.pi * 0.3 * t) + noise
        q_el = math.radians(self.config.get("target_el", 28.64)) + \
               0.01 * decay * math.sin(2 * math.pi * 0.2 * t) + noise

        los_x = 50e-6 * decay * math.sin(2 * math.pi * 1.0 * t) + noise
        los_y = 40e-6 * decay * math.cos(2 * math.pi * 1.2 * t) + noise

        # FSM correction
        fsm_tip   = -los_x * 0.95
        fsm_tilt  = -los_y * 0.95
        residual_x = los_x + fsm_tip
        residual_y = los_y + fsm_tilt

        progress_pct = min(int((t / duration) * 100), 100)

        return {
            "time":           round(t, 4),
            "q_az":           q_az,
            "q_el":           q_el,
            "qd_az":          0.2 * math.cos(2 * math.pi * 0.2 * t),
            "qd_el":         -0.15 * math.sin(2 * math.pi * 0.15 * t),
            "los_error_x":    los_x,
            "los_error_y":    los_y,
            "fsm_tip":        fsm_tip,
            "fsm_tilt":       fsm_tilt,
            "fsm_residual_x": residual_x,
            "fsm_residual_y": residual_y,
            "fsm_cmd_tip":    fsm_tip * 1.02,
            "fsm_cmd_tilt":   fsm_tilt * 1.02,
            "torque_az":      0.1 * math.sin(t),
            "torque_el":      0.05 * math.cos(t),
            "fsm_saturated":  False,
            "progress_pct":   progress_pct,
        }
