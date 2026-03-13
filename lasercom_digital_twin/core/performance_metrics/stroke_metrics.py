"""
Stroke Consumption Metrics — FSO Dual-Stage Gimbal Digital Twin
===============================================================

This module implements the three canonical "Stroke Consumption" benchmark
metrics that quantify the reliability of the spectral handover between the
coarse gimbal stage and the Fast Steering Mirror (FSM).

Physics Background
------------------
In any dual-stage tracking system, the FSM's mechanical stroke is a finite,
precious resource.  When the coarse stage fails to fully reject a disturbance,
the residual error propagates to the fine stage.  If this residual exceeds the
FSM's dynamic range, the fine stage saturates — effectively opening the control
loop and causing an immediate loss of FSO link.

Three benchmark metrics characterise this "Handover Bottleneck":

1.  **SCR_RMS — Root-Mean-Square Stroke Consumption Ratio** (Eq. 9 in the paper)
    Quantifies the continuous, time-averaged percentage of the total 
    dynamic range consumed. SCR_RMS provides a robust measure of the 
    sustained kinematic load demanded from the piezoelectric actuators, 
    heavily penalizing sustained sensor drop-outs while ignoring 
    micro-glitches.

2.  **S_bias — Bias Consumption** (Eq. 10)
    Quantifies the MEAN fraction of stroke continuously sacrificed to
    null the steady-state DC residual from the coarse stage.
    Persistent S_bias causes piezoelectric hysteresis drift and thermal
    buildup in the VCA drivers.

3.  **DSM — Dynamic Stroke Margin** (Eq. 11)
    The absolute remaining stroke capacity for high-frequency turbulence.
    A negative DSM guarantees FSM saturation during high-intensity gusts.

Separation of Concerns
-----------------------
This module is PURE COMPUTATION.  It contains no matplotlib, no file I/O,
and no simulation logic.  All visualization is delegated to the plotter.

Author
------
MicroPrecisionGimbal Digital Twin Team
Shahid Mustafa, PhD Research
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

# NumPy 2.0 renamed np.trapz → np.trapezoid; support both versions.
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_JITTER_CUTOFF_HZ: float = 50.0
"""Butterworth high-pass cutoff for separating jitter from DC bias [Hz].
   Frequencies above this are treated as 'high-frequency beam wander.'"""

_MIN_SAMPLES_FOR_STATS: int = 10
"""Minimum number of samples required to compute reliable statistics."""


# ---------------------------------------------------------------------------
# Result container (immutable by convention)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrokeMetricsResult:
    """
    Immutable container for Stroke Consumption benchmark results.

    All angular quantities are in radians unless a `_pct` or `_mdeg`
    suffix is present.

    Attributes
    ----------
    theta_max : float
        FSM mechanical stroke limit [rad].  Primary calibration constant.

    scr_tip / scr_tilt : float
        RMS Stroke Consumption Ratio [%] per axis.
        SCR_RMS = (1/Θ_max) * sqrt((1/T) * ∫|θ_fsm(t)|^2 dt) * 100

    s_bias_tip / s_bias_tilt : float
        Mean Bias Consumption [rad] per axis.
        S_bias = (1/T) ∫|θ_fsm(t)| dt  (trapezoidal rule over the full window)

    sigma_jitter_tip / sigma_jitter_tilt : float
        Standard deviation of the high-pass filtered FSM signal [rad].
        Represents true high-frequency atmospheric beam wander.

    dsm_tip / dsm_tilt : float
        Dynamic Stroke Margin [rad] per axis.
        DSM = θ_max − (S_bias + 3·σ_jitter)
        Positive DSM → safe; negative DSM → guaranteed link dropout.

    scr_timeseries_tip / scr_timeseries_tilt : np.ndarray
        Instantaneous stroke consumption [%] at every logged time step.
        Used for the time-domain panel of Fig 14.

    abs_fsm_tip / abs_fsm_tilt : np.ndarray
        |θ_fsm(t)| [rad] — functionally corrected. Used to plot 
        bias consumption visually.

    time : np.ndarray
        Time vector corresponding to the time-series outputs [s].
    """

    # Calibration
    theta_max: float

    # SCR — RMS ratios
    scr_tip: float
    scr_tilt: float

    # S_bias — Mean bias load
    s_bias_tip: float
    s_bias_tilt: float

    # Jitter — HF component standard deviation
    sigma_jitter_tip: float
    sigma_jitter_tilt: float

    # DSM — Remaining margin
    dsm_tip: float
    dsm_tilt: float

    # Time-series for plotting
    time: np.ndarray
    scr_timeseries_tip: np.ndarray
    scr_timeseries_tilt: np.ndarray
    abs_fsm_tip: np.ndarray
    abs_fsm_tilt: np.ndarray

    # ---------------------------------------------------------------------------
    # Derived convenience properties
    # ---------------------------------------------------------------------------
    @property
    def scr_tip_pct(self) -> float:
        """SCR Tip [%] — alias for readability."""
        return self.scr_tip

    @property
    def scr_tilt_pct(self) -> float:
        """SCR Tilt [%] — alias for readability."""
        return self.scr_tilt

    @property
    def dsm_tip_mrad(self) -> float:
        """DSM Tip converted to milliradians [mrad]."""
        return self.dsm_tip * 1e3

    @property
    def dsm_tilt_mrad(self) -> float:
        """DSM Tilt converted to milliradians [mrad]."""
        return self.dsm_tilt * 1e3

    @property
    def s_bias_tip_mrad(self) -> float:
        """S_bias Tip converted to milliradians [mrad]."""
        return self.s_bias_tip * 1e3

    @property
    def s_bias_tilt_mrad(self) -> float:
        """S_bias Tilt converted to milliradians [mrad]."""
        return self.s_bias_tilt * 1e3

    @property
    def is_link_safe_tip(self) -> bool:
        """True if the DSM on the Tip axis is strictly positive (no dropout risk)."""
        return self.dsm_tip > 0.0

    @property
    def is_link_safe_tilt(self) -> bool:
        """True if the DSM on the Tilt axis is strictly positive (no dropout risk)."""
        return self.dsm_tilt > 0.0

    def summary(self) -> str:
        """Return a human-readable summary string for console output."""
        lines = [
            "=" * 60,
            "  STROKE CONSUMPTION METRICS REPORT",
            "=" * 60,
            f"  θ_max (FSM stroke limit):  {self.theta_max * 1e3:.2f} mrad",
            "",
            "  ┌─────────────────────────────┬──────────┬──────────┐",
            "  │ Metric                      │   Tip    │   Tilt   │",
            "  ├─────────────────────────────┼──────────┼──────────┤",
            f"  │ SCR_RMS (RMS Load) [%]      │ {self.scr_tip:7.1f}  │ {self.scr_tilt:7.1f}  │",
            f"  │ S_bias (Mean Load) [mrad]   │ {self.s_bias_tip_mrad:7.3f}  │ {self.s_bias_tilt_mrad:7.3f}  │",
            f"  │ σ_jitter (HF wander) [mrad] │ {self.sigma_jitter_tip*1e3:7.4f}  │ {self.sigma_jitter_tilt*1e3:7.4f}  │",
            f"  │ DSM (Remaining margin)[mrad]│ {self.dsm_tip_mrad:7.3f}  │ {self.dsm_tilt_mrad:7.3f}  │",
            f"  │ Link Safe?                  │ {'✓ YES':^8} │ {'✓ YES' if self.is_link_safe_tilt else '✗ NO':^8} │",
            "  └─────────────────────────────┴──────────┴──────────┘",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric calculator
# ---------------------------------------------------------------------------
class StrokeMetrics:
    """
    Calculates RMS Stroke Consumption Ratio (SCR), Bias Consumption (S_bias),
    and Dynamic Stroke Margin (DSM) from FSM telemetry.

    This class is stateless between calls to :meth:`compute`.  A single
    instance can be reused across multiple result sets (PID / FBL / NDOB)
    with the same physical stroke limit.

    Parameters
    ----------
    theta_max : float
        Physical FSM mechanical stroke limit [rad].
        Default: 0.010 rad (10 mrad — PI S-330 specification).

    jitter_cutoff_hz : float, optional
        Butterworth high-pass cutoff frequency for separating high-frequency
        jitter σ from the low-frequency DC bias component [Hz].
        Default: 50 Hz.  Components above this frequency are classified as
        'atmospheric beam wander' for the DSM calculation.

    filter_order : int, optional
        Butterworth filter order.  Higher order → sharper transition band,
        but may produce Gibbs ringing near the cutoff.
        Default: 4 (trade-off between sharpness and numerical stability).
    """

    def __init__(
        self,
        theta_max: float = 0.010,
        jitter_cutoff_hz: float = _DEFAULT_JITTER_CUTOFF_HZ,
        filter_order: int = 4,
    ) -> None:
        if theta_max <= 0.0:
            raise ValueError(f"theta_max must be positive, got {theta_max}")
        if jitter_cutoff_hz <= 0.0:
            raise ValueError(f"jitter_cutoff_hz must be positive, got {jitter_cutoff_hz}")
        if filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {filter_order}")

        self.theta_max = theta_max
        self.jitter_cutoff_hz = jitter_cutoff_hz
        self.filter_order = filter_order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        time: np.ndarray,
        fsm_tip: np.ndarray,
        fsm_tilt: np.ndarray,
        dt: Optional[float] = None,
        link_active: Optional[np.ndarray] = None,
    ) -> StrokeMetricsResult:
        """
        Compute all three Stroke Consumption benchmark metrics.

        Parameters
        ----------
        time : array-like, shape (N,)
            Logged simulation time vector [s].
        fsm_tip : array-like, shape (N,)
            FSM tip mechanical angle [rad].
        fsm_tilt : array-like, shape (N,)
            FSM tilt mechanical angle [rad].
        dt : float, optional
            Nominal sampling interval [s].  If None, inferred from time array.
        link_active : array-like, shape (N,), optional
            Boolean array indicating whether the optical link was active
            (beam on QPD sensor) at each time step.  When ``False``, the
            FSM parks at 0° physically but the system has *functionally*
            exceeded its stroke limit.  This parameter enables the
            **Functional Saturation Override**: at every time step where
            ``link_active`` is ``False`` (and outside the settling window), 
            ``|θ_fsm|`` is replaced by ``theta_max`` so that SCR_RMS, 
            S_bias, and DSM correctly reflect 100 % stroke consumption 
            during sensor drop-outs.
        """
        time = np.asarray(time, dtype=float)
        fsm_tip = np.asarray(fsm_tip, dtype=float)
        fsm_tilt = np.asarray(fsm_tilt, dtype=float)

        self._validate_inputs(time, fsm_tip, fsm_tilt)

        # Infer dt from the time vector if not supplied
        if dt is None:
            dt = float(np.median(np.diff(time)))
            
        if dt <= 0.0:
            raise ValueError("Inferred time step (dt) is zero or negative. Time vector must be strictly increasing.")

        # ── Functional Saturation Override Setup ─────────────────────────
        abs_tip = np.abs(fsm_tip)
        abs_tilt = np.abs(fsm_tilt)
        
        # Create copies for our "Functional" data analysis
        abs_tip_functional = abs_tip.copy()
        abs_tilt_functional = abs_tilt.copy()

        if link_active is not None:
            link_active = np.asarray(link_active, dtype=bool)
            if link_active.shape != abs_tip.shape:
                raise ValueError("The 'link_active' array length must match the time/signal arrays.")
                
            # Distinguish between "Initial Slew" and "Failure Dropout"
            # has_acquired stays False until the first time link_active becomes True, then stays True forever.
            has_acquired = np.maximum.accumulate(link_active)
            
            # Mask out initial Handover Chatter (Settling Time)
            # We arm the failure detector ONLY after steady-state is reached (t >= 0.8s)
            is_steady_state = time >= 0.8
            
            # A failure is ONLY when the system has previously acquired the beam, 
            # currently lost it, AND is past the initial settling phase.
            is_failure_dropout = has_acquired & (~link_active) & is_steady_state
                
            # Override the parked 0 rad with the maximum stroke limit ONLY during a functional failure
            abs_tip_functional = np.where(is_failure_dropout, self.theta_max, abs_tip)
            abs_tilt_functional = np.where(is_failure_dropout, self.theta_max, abs_tilt)

        # ── SCR_RMS (Equation 9: RMS Stroke Consumption Ratio) ───────────
        # We calculate the Root Mean Square of the functional stroke array.
        # This heavily penalizes sustained dropouts while ignoring 1ms glitches.
        rms_tip = float(np.sqrt(np.mean(abs_tip_functional**2)))
        rms_tilt = float(np.sqrt(np.mean(abs_tilt_functional**2)))

        # Scalar metrics for the final printed report
        scr_tip = (rms_tip / self.theta_max) * 100.0
        scr_tilt = (rms_tilt / self.theta_max) * 100.0

        # Time-series arrays for Fig 14 top subplot (Instantaneous % consumed)
        scr_ts_tip = (abs_tip_functional / self.theta_max) * 100.0
        scr_ts_tilt = (abs_tilt_functional / self.theta_max) * 100.0

        # ── S_bias ───────────────────────────────────────────────────────
        T = time[-1] - time[0]
        s_bias_tip = float(_trapz(abs_tip_functional, x=time) / T) if T > 0 else 0.0
        s_bias_tilt = float(_trapz(abs_tilt_functional, x=time) / T) if T > 0 else 0.0

        # ── σ_jitter (High-pass filter, zero-phase) ───────────────────────
        # CRITICAL PHYSICS LOGIC: Use the RAW fsm_tip/tilt here! The override injects 
        # theta_max square-wave artifacts that would violently destabilize the Butterworth 
        # High-Pass filter and artificially inflate sigma_jitter far beyond physical reality.
        sigma_jitter_tip = self._compute_jitter_sigma(fsm_tip, dt)
        sigma_jitter_tilt = self._compute_jitter_sigma(fsm_tilt, dt)

        # ── DSM ──────────────────────────────────────────────────────────
        dsm_tip = self.theta_max - (s_bias_tip + 3.0 * sigma_jitter_tip)
        dsm_tilt = self.theta_max - (s_bias_tilt + 3.0 * sigma_jitter_tilt)

        return StrokeMetricsResult(
            theta_max=self.theta_max,
            scr_tip=scr_tip,
            scr_tilt=scr_tilt,
            s_bias_tip=s_bias_tip,
            s_bias_tilt=s_bias_tilt,
            sigma_jitter_tip=sigma_jitter_tip,
            sigma_jitter_tilt=sigma_jitter_tilt,
            dsm_tip=dsm_tip,
            dsm_tilt=dsm_tilt,
            time=time,
            scr_timeseries_tip=scr_ts_tip,
            scr_timeseries_tilt=scr_ts_tilt,
            abs_fsm_tip=abs_tip_functional,
            abs_fsm_tilt=abs_tilt_functional,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _compute_jitter_sigma(self, signal: np.ndarray, dt: float) -> float:
        """
        Isolate and compute the standard deviation of the high-frequency
        component of `signal` using a zero-phase Butterworth high-pass filter.
        """
        fs = 1.0 / dt
        nyquist = fs / 2.0
        normalized_cutoff = self.jitter_cutoff_hz / nyquist

        # Correct padlen calculation for filtfilt.
        padlen = 3 * (self.filter_order + 1)

        if normalized_cutoff >= 1.0 or len(signal) <= padlen:
            return float(np.std(signal))

        try:
            b, a = butter(self.filter_order, normalized_cutoff, btype='high', analog=False)
            hf_signal = filtfilt(b, a, signal)
            return float(np.std(hf_signal))
        except Exception:
            return float(np.std(signal))

    @staticmethod
    def _validate_inputs(
        time: np.ndarray,
        fsm_tip: np.ndarray,
        fsm_tilt: np.ndarray,
    ) -> None:
        """Raise ValueError for degenerate or corrupted inputs."""
        if len(time) < _MIN_SAMPLES_FOR_STATS:
            raise ValueError(
                f"Insufficient samples for statistics: got {len(time)}, "
                f"need at least {_MIN_SAMPLES_FOR_STATS}."
            )
        for name, arr in [("time", time), ("fsm_tip", fsm_tip), ("fsm_tilt", fsm_tilt)]:
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN detected in '{name}' array — numerically unstable simulation.")
            if np.any(np.isinf(arr)):
                raise ValueError(f"Inf detected in '{name}' array — divergent simulation state.")
        if not (len(fsm_tip) == len(fsm_tilt) == len(time)):
            raise ValueError(
                f"Array length mismatch: time={len(time)}, "
                f"fsm_tip={len(fsm_tip)}, fsm_tilt={len(fsm_tilt)}."
            )