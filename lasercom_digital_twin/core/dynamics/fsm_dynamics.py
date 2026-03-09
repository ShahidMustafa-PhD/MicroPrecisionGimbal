"""
Fast Steering Mirror (FSM) Dynamics Model — Vendor-Parameterised State-Space

This module implements physics-based state-space representations for three
classes of commercial Fast Steering Mirrors, each with fundamentally different
dynamic signatures:

    ┌─────────────┬────────────────────────┬────────────────────────┬─────────────────────────┐
    │             │  Piezoelectric (PZT)   │  Voice Coil (VCA)      │  Magnetic Reluctance    │
    ├─────────────┼────────────────────────┼────────────────────────┼─────────────────────────┤
    │ Vendor      │  Physik Instrumente    │  Newport FSM-300       │  CEDRAT M-FSM45         │
    │             │  S-330                 │                        │                         │
    │ Stroke      │  ±10 mrad              │  ±26 mrad (1.5°)       │  ±25 mrad               │
    │ 1st Res.    │  1 000 Hz              │  140 Hz                │  100 Hz                 │
    │ Damping ζ   │  0.02                  │  0.15                  │  0.08                   │
    │ DC Sens.    │  100 µrad/V            │  Current-driven        │  2 000 µrad/V           │
    │ BW (CL)     │  >800 Hz               │  ≈500 Hz               │  ≈300 Hz                │
    │ Sensor      │  Strain Gauge (SGS)    │  Optical PSD           │  Eddy-Current (ECS)     │
    │ States/axis │  2  (θ, θ̇)            │  3  (θ, θ̇, i)         │  2  (θ, θ̇)             │
    └─────────────┴────────────────────────┴────────────────────────┴─────────────────────────┘

Transfer Function Models
------------------------

PZT / Reluctance (2nd-order per axis):

    G(s) = K · ωn² / (s² + 2ζωn·s + ωn²)  ·  e^{-Td·s}

VCA (3rd-order per axis — coupled electro-mechanical):

    Electrical:  V = L·di/dt + R·i + Kb·θ̇
    Mechanical:  J·θ̈ + c·θ̇ + k·θ = Kt·i

    G(s) = Kt / [(Ls + R)(Js² + cs + k) + Kt·Kb·s]

If a current amplifier bypasses the L/R lag, the VCA reduces to 2nd-order.

Integration Method
------------------
4th-order Runge–Kutta (RK4) for all models.  Provides O(dt⁴) accuracy and
excellent stability for lightly-damped modes (ζ ≈ 0.02) when
dt < 1/(10·f_max).

References
----------
[1] PI S-330 datasheet, Physik Instrumente GmbH.
[2] Newport FSM-300 datasheet, Newport Corporation.
[3] CEDRAT M-FSM45 datasheet, CEDRAT Technologies.
[4] Chen, W.-H. et al., "Disturbance-Observer-Based Control," ISA Trans., 2015.

Author: Senior Control Systems Engineer
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Actuation-type enumeration
# ============================================================================

class FSMActuationType(Enum):
    """Actuation technology for the FSM."""
    PZT       = "pzt"          # Piezoelectric (Physik Instrumente style)
    VCA       = "vca"          # Voice-Coil Actuator (Newport style)
    RELUCTANCE = "reluctance"  # Magnetic Reluctance (CEDRAT style)


# ============================================================================
# FsmDynamicsConfig — physics-parameterised configuration
# ============================================================================

@dataclass
class FsmDynamicsConfig:
    """
    Complete physics-based configuration for an FSM plant model.

    Houses all mechanical, electrical, and nonlinear parameters needed to
    construct the state-space matrices A, B, C, D via ``build_state_space()``.

    The config also stores the **compiled** matrices after calling
    ``build_state_space()`` so that ``FsmDynamics`` can consume them directly.
    """

    # ------------------------------------------------------------------
    # Identity & type
    # ------------------------------------------------------------------
    vendor_name: str = "Generic"
    actuation_type: FSMActuationType = FSMActuationType.PZT
    sensor_type: str = "strain_gauge"        # 'strain_gauge', 'optical_psd', 'eddy_current'

    # ------------------------------------------------------------------
    # Mechanical parameters (common to all types)
    # ------------------------------------------------------------------
    theta_max: float = 0.010       # Maximum mechanical stroke per axis [rad]  (half-range)
    f_n: float = 500.0             # First mechanical resonance [Hz]
    zeta: float = 0.10             # Damping ratio [-]
    dc_sensitivity: float = 100e-6 # DC sensitivity [rad/V]  (PZT/Reluctance only)
    bandwidth_hz: float = 300.0    # Closed-loop bandwidth [Hz]  (informational)
    transport_delay: float = 0.0   # Pure transport delay Td [s]

    # Mechanical inertia & stiffness (used by VCA model)
    J: float = 1.0e-6              # Mirror inertia [kg·m²]
    c_mech: float = 0.0            # Viscous damping [N·m·s/rad]  (auto-filled from ζ if 0)
    k_mech: float = 0.0            # Torsional stiffness [N·m/rad] (auto-filled from ωn² if 0)

    # ------------------------------------------------------------------
    # Electrical parameters (VCA only)
    # ------------------------------------------------------------------
    L: float = 0.0                 # Coil inductance [H]
    R: float = 0.0                 # Coil resistance [Ω]
    K_t: float = 0.0              # Torque constant [N·m/A]
    K_b: float = 0.0              # Back-EMF constant [V·s/rad]
    use_current_amp: bool = False  # If True, bypass L/R → 2nd-order model

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    rate_limit: float = 100.0      # Maximum angular rate [rad/s]

    # ------------------------------------------------------------------
    # Hysteresis (PZT only, optional)
    # ------------------------------------------------------------------
    hysteresis_percent: float = 0.0   # Hysteresis as % of stroke (0–15%)

    # ------------------------------------------------------------------
    # Compiled state-space matrices (populated by build_state_space())
    # ------------------------------------------------------------------
    A: np.ndarray = field(default=None, repr=False)
    B: np.ndarray = field(default=None, repr=False)
    C: np.ndarray = field(default=None, repr=False)
    D: np.ndarray = field(default=None, repr=False)
    n_states: int = 4              # Total state dimension (set by build_state_space)
    n_states_per_axis: int = 2     # States per axis

    # ==================================================================
    #  Factory class methods — vendor presets
    # ==================================================================

    @classmethod
    def pi_s330(cls) -> 'FsmDynamicsConfig':
        """
        Physik Instrumente S-330 — Piezoelectric FSM.

        Characteristics:
          • ±10 mrad stroke
          • f_n = 1 000 Hz, ζ = 0.02  (very lightly damped)
          • 100 µrad/V DC sensitivity
          • >800 Hz closed-loop bandwidth (with strain-gauge feedback)
          • ~10-15% open-loop hysteresis (suppressed by SGS feedback)
          • Transport delay ≈ 50 µs (electronics)
        """
        cfg = cls(
            vendor_name="PI S-330",
            actuation_type=FSMActuationType.PZT,
            sensor_type="strain_gauge",
            theta_max=0.010,            # ±10 mrad
            f_n=1000.0,                 # Hz
            zeta=0.02,                  # Lightly damped flexure
            dc_sensitivity=100e-6,      # 100 µrad/V
            bandwidth_hz=800.0,
            transport_delay=50e-6,      # 50 µs
            rate_limit=200.0,           # rad/s
            hysteresis_percent=12.0,    # 12% open-loop (mitigated by SGS)
        )
        cfg.build_state_space()
        return cfg

    @classmethod
    def newport_fsm300(cls) -> 'FsmDynamicsConfig':
        """
        Newport FSM-300 — Voice-Coil Actuated FSM.

        Characteristics:
          • ±26 mrad (1.5°) stroke
          • f_n = 140 Hz  (mechanical resonance with payload)
          • ζ = 0.15  (moderate damping from eddy-currents)
          • 3rd-order model: coupled electrical (L, R) + mechanical (J, c, k)
          • Optical PSD position sensor
          • ≈500 Hz closed-loop bandwidth
          • Transport delay ≈ 200 µs

        Electrical parameters from system identification:
          L = 0.5 mH,  R = 3.2 Ω,  Kt = 0.012 N·m/A,  Kb = 0.012 V·s/rad
        """
        omega_n = 2.0 * np.pi * 140.0   # rad/s
        J = 5.0e-6                        # kg·m²  (mirror + mount assembly)
        k = J * omega_n**2                # torsional stiffness from ωn²·J
        c = 2.0 * 0.15 * omega_n * J     # viscous damping from 2·ζ·ωn·J

        cfg = cls(
            vendor_name="Newport FSM-300",
            actuation_type=FSMActuationType.VCA,
            sensor_type="optical_psd",
            theta_max=0.026,             # ±26 mrad
            f_n=140.0,
            zeta=0.15,
            dc_sensitivity=0.0,          # N/A — current-driven
            bandwidth_hz=500.0,
            transport_delay=200e-6,      # 200 µs
            J=J,
            c_mech=c,
            k_mech=k,
            L=0.5e-3,                    # 0.5 mH
            R=3.2,                       # Ω
            K_t=0.012,                   # N·m/A
            K_b=0.012,                   # V·s/rad  (≈ Kt for DC motors)
            use_current_amp=False,       # Full 3rd-order model
            rate_limit=50.0,             # rad/s
            hysteresis_percent=0.0,      # No piezo hysteresis
        )
        cfg.build_state_space()
        return cfg

    @classmethod
    def cedrat_mfsm45(cls) -> 'FsmDynamicsConfig':
        """
        CEDRAT Technologies M-FSM45 — Magnetic Reluctance FSM.

        Characteristics:
          • ±25 mrad stroke
          • f_n = 100 Hz, ζ = 0.08
          • 2 000 µrad/V DC sensitivity
          • Eddy-current sensor (ECS)
          • ≈300 Hz closed-loop bandwidth
          • Transport delay ≈ 100 µs
        """
        cfg = cls(
            vendor_name="CEDRAT M-FSM45",
            actuation_type=FSMActuationType.RELUCTANCE,
            sensor_type="eddy_current",
            theta_max=0.025,             # ±25 mrad
            f_n=100.0,
            zeta=0.08,
            dc_sensitivity=2000e-6,      # 2000 µrad/V
            bandwidth_hz=300.0,
            transport_delay=100e-6,      # 100 µs
            rate_limit=30.0,             # rad/s
            hysteresis_percent=0.0,
        )
        cfg.build_state_space()
        return cfg

    # ==================================================================
    #  State-space construction from physical parameters
    # ==================================================================

    def build_state_space(self) -> None:
        """
        Construct A, B, C, D matrices from physical parameters.

        For PZT / Reluctance (2nd-order per axis, 4 states total):
            State per axis:  x = [θ, θ̇]
            ẋ = [[  0,   1 ],  x  +  [   0  ] u
                 [-ωn², -2ζωn]]       [ K·ωn² ]

        For VCA (3rd-order per axis, 6 states total):
            State per axis:  x = [θ, θ̇, i]
            Derived from coupled electromechanical equations.
        """
        if self.actuation_type == FSMActuationType.VCA and not self.use_current_amp:
            self._build_vca_state_space()
        else:
            self._build_second_order_state_space()

    def _build_second_order_state_space(self) -> None:
        """
        Build 2nd-order state-space (PZT, Reluctance, or current-driven VCA).

        Per-axis transfer function:
            G(s) = K·ωn² / (s² + 2ζωn·s + ωn²)

        Controllable canonical form per axis:
            A_axis = [[  0,        1     ],
                      [-ωn²,    -2·ζ·ωn  ]]

            B_axis = [[   0    ],
                      [ K·ωn²  ]]

            C_axis = [[ 1,  0 ]]
            D_axis = [[ 0 ]]

        Two-axis composite (decoupled, no cross-coupling):
            A = block_diag(A_tip, A_tilt)         →  4×4
            B = block_diag(B_tip, B_tilt)         →  4×2
            C = block_diag(C_tip, C_tilt)         →  2×4
            D = zeros(2, 2)
        """
        omega_n = 2.0 * np.pi * self.f_n    # rad/s
        zeta = self.zeta

        # DC gain K
        if self.actuation_type == FSMActuationType.VCA and self.use_current_amp:
            # Current-driven VCA: K = Kt / k  (torque constant / stiffness)
            K = self.K_t / (self.J * omega_n**2) if self.J > 0 else self.dc_sensitivity
        else:
            K = self.dc_sensitivity              # rad/V for PZT and Reluctance

        # Per-axis matrices
        A_axis = np.array([
            [0.0,                   1.0           ],
            [-omega_n**2,          -2.0*zeta*omega_n]
        ])

        B_axis = np.array([
            [0.0],
            [K * omega_n**2]
        ])

        C_axis = np.array([[1.0, 0.0]])
        D_axis = np.array([[0.0]])

        # Two-axis composite (block-diagonal, assuming identical axes)
        self.A = np.block([
            [A_axis,            np.zeros((2, 2))],
            [np.zeros((2, 2)),  A_axis           ]
        ])

        self.B = np.block([
            [B_axis,            np.zeros((2, 1))],
            [np.zeros((2, 1)),  B_axis           ]
        ])

        self.C = np.block([
            [C_axis,            np.zeros((1, 2))],
            [np.zeros((1, 2)),  C_axis           ]
        ])

        self.D = np.zeros((2, 2))

        self.n_states_per_axis = 2
        self.n_states = 4

    def _build_vca_state_space(self) -> None:
        """
        Build 3rd-order state-space for VCA (voltage-driven, coupled L/R).

        Per-axis state vector:  x = [θ, θ̇, i]

        Equations of motion:
            θ̇  = x₂
            θ̈  = (Kt·i - c·θ̇ - k·θ) / J
            di/dt = (V - R·i - Kb·θ̇) / L

        State-space per axis:
            A_axis = [[    0,         1,         0       ],
                      [ -k/J,      -c/J,       Kt/J     ],
                      [    0,      -Kb/L,      -R/L      ]]

            B_axis = [[ 0 ],
                      [ 0 ],
                      [ 1/L ]]

            C_axis = [[ 1,  0,  0 ]]
            D_axis = [[ 0 ]]
        """
        omega_n = 2.0 * np.pi * self.f_n
        J = self.J
        # Auto-fill stiffness and damping if not explicitly set
        k = self.k_mech if self.k_mech > 0 else J * omega_n**2
        c = self.c_mech if self.c_mech > 0 else 2.0 * self.zeta * omega_n * J

        L = self.L
        R = self.R
        Kt = self.K_t
        Kb = self.K_b

        # Validate electrical parameters
        if L <= 0 or R <= 0 or Kt <= 0:
            raise ValueError(
                f"VCA model requires positive L ({L}), R ({R}), Kt ({Kt}). "
                "Set use_current_amp=True for current-driven 2nd-order model."
            )

        A_axis = np.array([
            [0.0,         1.0,          0.0    ],
            [-k/J,       -c/J,          Kt/J   ],
            [0.0,        -Kb/L,        -R/L    ]
        ])

        B_axis = np.array([
            [0.0  ],
            [0.0  ],
            [1.0/L]
        ])

        C_axis = np.array([[1.0, 0.0, 0.0]])
        D_axis = np.array([[0.0]])

        # Two-axis composite (block-diagonal)
        self.A = np.block([
            [A_axis,            np.zeros((3, 3))],
            [np.zeros((3, 3)),  A_axis           ]
        ])

        self.B = np.block([
            [B_axis,            np.zeros((3, 1))],
            [np.zeros((3, 1)),  B_axis           ]
        ])

        self.C = np.block([
            [C_axis,            np.zeros((1, 3))],
            [np.zeros((1, 3)),  C_axis           ]
        ])

        self.D = np.zeros((2, 2))

        self.n_states_per_axis = 3
        self.n_states = 6

    # ==================================================================
    #  Utility
    # ==================================================================

    def get_state_order(self) -> int:
        """Return total number of states in the 2-axis model."""
        return self.n_states

    def get_info(self) -> Dict:
        """Return a summary dictionary for logging / printing."""
        omega_n = 2.0 * np.pi * self.f_n
        return {
            'vendor': self.vendor_name,
            'type': self.actuation_type.value,
            'stroke_mrad': self.theta_max * 1e3,
            'f_n_Hz': self.f_n,
            'omega_n_rad_s': omega_n,
            'zeta': self.zeta,
            'bandwidth_Hz': self.bandwidth_hz,
            'n_states': self.n_states,
            'sensor': self.sensor_type,
            'transport_delay_us': self.transport_delay * 1e6,
        }


# ============================================================================
# FsmDynamics — state-space plant model
# ============================================================================

class FsmDynamics:
    """
    High-fidelity state-space dynamics for a 2-axis FSM.

    Supports 2nd-order (PZT, Reluctance) and 3rd-order (VCA) models with
    dynamic state dimension determined by the ``FsmDynamicsConfig``.

    This class handles:
      • State propagation via RK4 integration
      • Output computation (angular displacement)
      • Stroke utilisation and saturation queries
      • Position clamping at mechanical stroke limits
      • Eigenvalue / modal analysis

    Attributes
    ----------
    config : FsmDynamicsConfig
        Physics-based configuration (also stores compiled A, B, C, D).
    x : np.ndarray
        Current state vector (n_states,).
    """

    def __init__(self, config: Optional[FsmDynamicsConfig] = None):
        """
        Initialise FSM dynamics from a physics-based configuration.

        Parameters
        ----------
        config : FsmDynamicsConfig, optional
            If None, defaults to CEDRAT M-FSM45 (same resonances as
            the legacy 4th-order modal-reduction matrices, for backward
            compatibility).
        """
        if config is None:
            config = FsmDynamicsConfig.cedrat_mfsm45()

        # Ensure matrices are built
        if config.A is None:
            config.build_state_space()

        self.config = config

        # Pull matrices into local attributes for fast access
        self.A = config.A.copy()
        self.B = config.B.copy()
        self.C = config.C.copy()
        self.D = config.D.copy()

        self.n_states = config.n_states
        self.theta_max = config.theta_max

        # Validate
        self._validate_matrices()

        # State vector
        self.x = np.zeros(self.n_states)
        self._time = 0.0

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_matrices(self) -> None:
        """Validate state-space matrix dimensions for consistency."""
        n = self.n_states
        assert self.A.shape == (n, n), f"A must be {n}x{n}, got {self.A.shape}"
        assert self.B.shape == (n, 2), f"B must be {n}x2, got {self.B.shape}"
        assert self.C.shape == (2, n), f"C must be 2x{n}, got {self.C.shape}"
        assert self.D.shape == (2, 2), f"D must be 2x2, got {self.D.shape}"

    # ------------------------------------------------------------------
    #  State propagation
    # ------------------------------------------------------------------

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        """
        Reset internal state for a new simulation run.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial state vector (n_states,).  Defaults to zero.
        """
        if x0 is None:
            self.x = np.zeros(self.n_states)
        else:
            assert x0.shape == (self.n_states,), \
                f"x0 must be ({self.n_states},), got {x0.shape}"
            self.x = x0.copy()
        self._time = 0.0

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate state forward by one timestep using RK4 integration.

        Parameters
        ----------
        u : np.ndarray
            Control input (2,) — [V_tip, V_tilt] in volts (or normalised).
        dt : float
            Timestep [s].  Recommended: dt < 1/(10·f_n).

        Returns
        -------
        np.ndarray
            Output (2,) — [θ_tip, θ_tilt] in radians, clamped to ±θ_max.
        """
        assert u.shape == (2,), f"u must be (2,), got {u.shape}"
        assert dt > 0, f"dt must be positive, got {dt}"

        # RK4 with zero-order hold on u
        k1 = self.A @ self.x + self.B @ u
        k2 = self.A @ (self.x + 0.5 * dt * k1) + self.B @ u
        k3 = self.A @ (self.x + 0.5 * dt * k2) + self.B @ u
        k4 = self.A @ (self.x + dt * k3)        + self.B @ u

        self.x = self.x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self._time += dt

        # Compute output and enforce stroke limits
        y = self.outputs(u)
        y_clamped = np.clip(y, -self.theta_max, self.theta_max)

        # If output saturated, project state back to keep consistency
        if not np.allclose(y, y_clamped, atol=1e-12):
            # Simple projection: scale state so output equals clamped value
            # This avoids integrator wind-up inside the state-space model
            for axis in range(2):
                if abs(y[axis]) > self.theta_max:
                    # Zero the velocity of the saturated axis
                    axis_start = axis * self.config.n_states_per_axis
                    self.x[axis_start + 1] = 0.0  # velocity = 0 at limit
                    # CRITICAL FIX: The position state MUST be physically bounded!
                    # Otherwise RK4 integration allows the internal state to creep up 
                    # to the unbounded steady state, creating a massive artificial limit 
                    # cycle when the command reverses!
                    self.x[axis_start] = np.sign(y[axis]) * self.theta_max

        return y_clamped

    def outputs(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute output y = C·x + D·u.

        Parameters
        ----------
        u : np.ndarray, optional
            Current input (2,).  Required only if D ≠ 0.

        Returns
        -------
        np.ndarray
            Output (2,) — [θ_tip, θ_tilt] in radians.
        """
        y = self.C @ self.x
        if u is not None and np.any(self.D != 0):
            y += self.D @ u
        return y

    # ------------------------------------------------------------------
    #  Stroke utilisation & saturation
    # ------------------------------------------------------------------

    def get_stroke_utilization(self, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return per-axis stroke utilisation as fraction of θ_max.

        Parameters
        ----------
        y : np.ndarray, optional
            Output angles (2,).  If None, computed from current state.

        Returns
        -------
        np.ndarray
            (2,) with values in [0, 1].  1.0 = fully saturated.
        """
        if y is None:
            y = self.outputs()
        return np.abs(y) / self.theta_max

    def is_saturated(self, y: Optional[np.ndarray] = None,
                     threshold: float = 0.98) -> np.ndarray:
        """
        Check per-axis saturation status.

        Parameters
        ----------
        y : np.ndarray, optional
            Output angles (2,).  If None, computed from current state.
        threshold : float
            Fraction of θ_max above which stroke is considered saturated.

        Returns
        -------
        np.ndarray of bool
            (2,) — True if the axis is at or beyond the threshold.
        """
        util = self.get_stroke_utilization(y)
        return util >= threshold

    # ------------------------------------------------------------------
    #  State access
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return a copy of the current state vector."""
        return self.x.copy()

    def get_time(self) -> float:
        """Return elapsed simulation time since last reset [s]."""
        return self._time

    # ------------------------------------------------------------------
    #  Modal analysis
    # ------------------------------------------------------------------

    def get_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of A.

        For a stable system all real parts must be < 0.
        """
        return np.linalg.eigvals(self.A)

    def get_resonance_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract resonance frequencies [Hz] and damping ratios from eigenvalues.

        Returns
        -------
        frequencies_hz : np.ndarray
        damping_ratios : np.ndarray
        """
        eigenvals = self.get_eigenvalues()
        frequencies = []
        dampings = []

        processed = set()
        for i, eig in enumerate(eigenvals):
            if i in processed:
                continue
            imag = np.abs(np.imag(eig))
            if imag > 1e-6:
                omega_n = np.abs(eig)
                zeta = -np.real(eig) / omega_n
                freq_hz = imag / (2.0 * np.pi)
                frequencies.append(freq_hz)
                dampings.append(zeta)
                # Skip conjugate
                for j in range(i+1, len(eigenvals)):
                    if j not in processed and np.abs(eigenvals[j] - np.conj(eig)) < 1e-6:
                        processed.add(j)
                        break
            else:
                # Real eigenvalue (overdamped or electrical pole)
                frequencies.append(0.0)
                dampings.append(1.0)

        return np.array(frequencies), np.array(dampings)

    def linearize_at_state(self, x_op: np.ndarray, u_op: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (A, B, C, D).  Already LTI, so trivial."""
        return self.A.copy(), self.B.copy(), self.C.copy(), self.D.copy()


# ============================================================================
# Factory helpers
# ============================================================================

def create_fsm_from_vendor(vendor: str = "cedrat") -> FsmDynamics:
    """
    Create an FSM dynamics instance from a vendor preset string.

    Parameters
    ----------
    vendor : str
        One of ``'pi'``, ``'newport'``, ``'cedrat'`` (case-insensitive).

    Returns
    -------
    FsmDynamics
        Configured plant model.
    """
    key = vendor.strip().lower()
    presets = {
        'pi':       FsmDynamicsConfig.pi_s330,
        'pi_s330':  FsmDynamicsConfig.pi_s330,
        'newport':  FsmDynamicsConfig.newport_fsm300,
        'fsm300':   FsmDynamicsConfig.newport_fsm300,
        'cedrat':   FsmDynamicsConfig.cedrat_mfsm45,
        'mfsm45':   FsmDynamicsConfig.cedrat_mfsm45,
    }
    if key not in presets:
        raise ValueError(
            f"Unknown vendor '{vendor}'.  Choose from: {list(presets.keys())}"
        )
    config = presets[key]()
    return FsmDynamics(config)


def create_fsm_dynamics_from_design() -> FsmDynamics:
    """
    Backward-compatible factory — returns CEDRAT M-FSM45 model.

    This preserves the existing call-site in simulation_runner.py.
    """
    return create_fsm_from_vendor("cedrat")


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Validation suite: stability, modal analysis, and step response for
    all three vendor presets.
    """

    vendors = [
        ("PI S-330  (PZT)",        FsmDynamicsConfig.pi_s330),
        ("Newport FSM-300 (VCA)",  FsmDynamicsConfig.newport_fsm300),
        ("CEDRAT M-FSM45 (Rel.)",  FsmDynamicsConfig.cedrat_mfsm45),
    ]

    for label, factory in vendors:
        print("\n" + "=" * 72)
        print(f"  {label}")
        print("=" * 72)

        config = factory()
        fsm = FsmDynamics(config)
        info = config.get_info()

        print(f"\n  Vendor:           {info['vendor']}")
        print(f"  Type:             {info['type']}")
        print(f"  Stroke:           ±{info['stroke_mrad']:.1f} mrad")
        print(f"  Resonance:        {info['f_n_Hz']:.0f} Hz  (ωn = {info['omega_n_rad_s']:.0f} rad/s)")
        print(f"  Damping:          ζ = {info['zeta']:.3f}")
        print(f"  State dimension:  {info['n_states']}")
        print(f"  Sensor:           {info['sensor']}")

        # 1. Stability
        eigenvals = fsm.get_eigenvalues()
        stable = all(np.real(e) < 0 for e in eigenvals)
        print(f"\n  STABILITY: {'✓ STABLE' if stable else '✗ UNSTABLE'}")
        for i, eig in enumerate(eigenvals):
            print(f"    λ{i+1} = {eig.real:10.2f} ± j{abs(eig.imag):10.2f}")

        # 2. Modal characteristics
        freqs, dampings = fsm.get_resonance_frequencies()
        print(f"\n  MODES:")
        for i, (f, z) in enumerate(zip(freqs, dampings)):
            if f > 0:
                print(f"    Mode {i+1}: f = {f:8.2f} Hz,  ζ = {z:.4f}")
            else:
                print(f"    Pole {i+1}: real pole (overdamped / electrical)")

        # 3. Step response
        fsm.reset()
        dt = min(1e-4, 0.1 / info['f_n_Hz'])   # Ensure Nyquist-safe timestep
        duration = max(0.05, 5.0 / info['f_n_Hz'])  # ~5 cycles
        n_steps = int(duration / dt)

        u_step = np.array([1.0, 0.0])  # Unit step on tip axis
        times = []
        tips = []
        tilts = []
        for i in range(n_steps):
            y = fsm.step(u_step, dt)
            times.append(fsm.get_time())
            tips.append(y[0])
            tilts.append(y[1])

        tips = np.array(tips)
        tilts = np.array(tilts)

        peak_tip = np.max(np.abs(tips))
        final_tip = tips[-1]
        final_tilt = tilts[-1]
        cross_talk = abs(final_tilt / final_tip) * 100.0 if abs(final_tip) > 1e-15 else 0.0

        # Overshoot (percentage above final for underdamped)
        overshoot_pct = 0.0
        if abs(final_tip) > 1e-15:
            overshoot_pct = max(0.0, (peak_tip - abs(final_tip)) / abs(final_tip) * 100.0)

        print(f"\n  STEP RESPONSE (Tip, 1 V step):")
        print(f"    Duration:      {duration*1e3:.1f} ms  ({n_steps} steps at dt={dt*1e6:.0f} µs)")
        print(f"    Final tip:     {final_tip:.6e} rad  ({final_tip*1e3:.4f} mrad)")
        print(f"    Peak tip:      {peak_tip:.6e} rad")
        print(f"    Overshoot:     {overshoot_pct:.1f}%")
        print(f"    Cross-talk:    {cross_talk:.2f}%")

        # 4. Stroke utilisation check
        util = fsm.get_stroke_utilization()
        sat = fsm.is_saturated()
        print(f"\n  STROKE UTILISATION:")
        print(f"    Tip:  {util[0]*100:.1f}%   {'SATURATED' if sat[0] else 'OK'}")
        print(f"    Tilt: {util[1]*100:.1f}%   {'SATURATED' if sat[1] else 'OK'}")

    print("\n" + "=" * 72)
    print("  VALIDATION COMPLETE")
    print("=" * 72)
