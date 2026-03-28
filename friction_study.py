"""
friction_study.py
=================
Friction Model Parameter Tuning and Steady-State Matching Study
MicroPrecision Gimbal — Lasercom Digital Twin

PURPOSE
-------
This tool compares SmoothedTustinFriction (algebraic Stribeck) against the
LuGre dynamic friction model at steady state, and reveals their dynamic
differences (bristle transients).

STEADY-STATE EQUIVALENCE THEOREM
---------------------------------
When the LuGre bristle ODE settles (dz/dt = 0), the closed-form steady-state
torque is:

    tau_ss(v) = g(v) * sign(v)  +  sigma_2 * v

    where  g(v) = tau_c + (tau_s - tau_c) * exp(-(v/v_s)^2)

This is IDENTICAL to the Tustin formula with alpha = 2.0, tanh -> sign, b = sigma_2.

MATCHING RULES (edit the config block to satisfy all four):
    1.  LuGre  tau_c   ==  Tustin tau_c          (Coulomb level)
    2.  LuGre  tau_s   ==  Tustin tau_s          (static / stiction level)
    3.  LuGre  v_s     ==  Tustin v_s            (Stribeck knee velocity)
    4.  LuGre  sigma_2 ==  Tustin b              (viscous coefficient)
    5.  Tustin alpha   ==  2.0                   (LuGre Stribeck shape is Gaussian)

Parameters sigma_0 and sigma_1 affect ONLY the bristle transient dynamics
(rise time and damping of the LuGre response) — they have NO effect on SS.

USAGE
-----
    python friction_study.py

Author  : Dr. S. Shahid Mustafa
Project : MicroPrecision Gimbal — Lasercom Digital Twin
"""

# ===========================================================================
#  IMPORTS
# ===========================================================================
import sys
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Ensure package root is on path (same folder as this script)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasercom_digital_twin.core.friction.tustin_friction import SmoothedTustinFriction
from lasercom_digital_twin.core.friction.lugre_friction  import LuGreFriction

# ===========================================================================
#  USER CONFIGURATION  —  Edit ONLY this block
# ===========================================================================

# --------------- Axis identification (for plot labels only) ----------------
AXIS_LABEL = "Azimuth Gimbal"     # e.g. "Azimuth Gimbal" / "Elevation Gimbal"

# --------------- Tustin (Smoothed Stribeck) Parameters ---------------------
#
#   tau_fric = [tau_c + (tau_s - tau_c)*exp(-(|v|/v_s)^alpha)] * tanh(v/v_eps) + b*v
# for azimuth gimbal
if(1):
 TUSTIN = dict(
    tau_c     = 0.15,    # Coulomb friction magnitude     [N.m]   (plateau at high speed)
    tau_s     = 0.25,    # Static friction / stiction     [N.m]   (breakaway; must >= tau_c)
    v_s       = 0.05,    # Stribeck velocity threshold    [rad/s] (transition knee)
    b         = 0.02,    # Viscous friction coefficient   [N.m.s/rad]
    alpha     = 2.0,     # Stribeck exponent: 1=linear, 2=Gaussian. SET TO 2.0 to match LuGre.
    v_epsilon = 0.01,    # tanh smoothing half-width      [rad/s] (smaller = sharper stick-slip)
)
# for elevation gimbal
if(0):
 TUSTIN = dict(
    tau_c     = 0.10,    # Coulomb friction magnitude     [N.m]   (plateau at high speed)
    tau_s     = 0.18,    # Static friction / stiction     [N.m]   (breakaway; must >= tau_c)
    v_s       = 0.05,    # Stribeck velocity threshold    [rad/s] (transition knee)
    b         = 0.015,    # Viscous friction coefficient   [N.m.s/rad]
    alpha     = 2.0,     # Stribeck exponent: 1=linear, 2=Gaussian. SET TO 2.0 to match LuGre.
    v_epsilon = 0.01,    # tanh smoothing half-width      [rad/s] (smaller = sharper stick-slip)
)

# --------------- LuGre Dynamic Friction Parameters -------------------------
#
#   Bristle ODE : dz/dt = v - sigma_0 * |v| * z / g(v)
#   Friction    : tau   = sigma_0*z + sigma_1*(dz/dt) + sigma_2*v
#   Stribeck fn : g(v)  = tau_c + (tau_s - tau_c) * exp(-(v/v_s)^2)
#
#   Steady-state (dz/dt=0): tau_ss = g(v)*sign(v) + sigma_2*v
#
#for Azimuth gimbal
if(1):
 LUGRE = dict(
    sigma_0 = 1.0e4,   # Bristle stiffness              [N.m/rad]     — higher = faster transient
    sigma_1 = 1.0,     # Bristle micro-damping          [N.m.s/rad]   — higher = more overdamped
    sigma_2 = 0.02,    # Viscous coefficient            [N.m.s/rad]   — SET EQUAL TO Tustin b
    tau_c   = 0.15,    # Coulomb friction               [N.m]         — SET EQUAL TO Tustin tau_c
    tau_s   = 0.25,    # Static friction                [N.m]         — SET EQUAL TO Tustin tau_s
    v_s     = 0.05,    # Stribeck velocity              [rad/s]       — SET EQUAL TO Tustin v_s
)
#LUGRE = dict(
#    sigma_0 = 1.0e4,   # Bristle stiffness              [N.m/rad]     — higher = faster transient
#    sigma_1 = 1.0,     # Bristle micro-damping          [N.m.s/rad]   — higher = more overdamped
#    sigma_2 = 0.02,    # Viscous coefficient            [N.m.s/rad]   — SET EQUAL TO Tustin b
#    tau_c   = 0.15,    # Coulomb friction               [N.m]         — SET EQUAL TO Tustin tau_c
#    tau_s   = 0.25,    # Static friction                [N.m]         — SET EQUAL TO Tustin tau_s
#    v_s     = 0.05,    # Stribeck velocity              [rad/s]       — SET EQUAL TO Tustin v_s
#)
#for elevation gimbal
if(0):
 LUGRE = dict(
    sigma_0 = 1.0e8,   # Bristle stiffness              [N.m/rad]     — higher = faster transient
    sigma_1 = 0.8,     # Bristle micro-damping          [N.m.s/rad]   — higher = more overdamped
    sigma_2 = 0.015,    # Viscous coefficient            [N.m.s/rad]   — SET EQUAL TO Tustin b
    tau_c   = 0.10,    # Coulomb friction               [N.m]         — SET EQUAL TO Tustin tau_c
    tau_s   = 0.18,    # Static friction                [N.m]         — SET EQUAL TO Tustin tau_s
    v_s     = 0.05,    # Stribeck velocity              [rad/s]       — SET EQUAL TO Tustin v_s
)

# --------------- Velocity Sweep Settings -----------------------------------
V_MAX        = 0.5    # Max velocity for Stribeck curve sweep   [rad/s]
N_SWEEP      = 1000   # Number of velocity sample points in sweep
V_ZOOM_MAX   = 0.25   # Max velocity for Stribeck zoom panel    [rad/s]

# --------------- Transient Step-Response Settings --------------------------
# Velocities at which a step is applied from rest and LuGre is simulated
TRANSIENT_V_LIST = [0.01, 0.05, 0.10, 0.30, 0.50]   # [rad/s]
DT_TRANSIENT     = 1e-5    # Integration timestep  [s]  (keep << 2*g(v)/(sigma_0*|v|))
T_TRANSIENT      = 0.05    # Duration per transient [s]

# --------------- Hysteresis Simulation Settings ----------------------------
#
#   Velocity input: v(t) = HYST_AMP * sin(2*pi*f*t)
#   Run N_PRE_CYCLES to reach steady bristle state, then plot N_PLOT_CYCLES.
#   HYST_FREQS_HZ: list of excitation frequencies for the loop overlay plot.
#   HYST_COMPARE_F: single frequency used for LuGre-vs-Tustin comparison panel.
#
HYST_AMP        = 0.10       # Velocity amplitude A        [rad/s]
HYST_FREQS_HZ   = [1.0, 2.0, 5.0, 10.0, 20.0]   # Excitation frequencies [Hz]
HYST_COMPARE_F  = 5.0        # Frequency for panel (b) comparison [Hz]
HYST_N_PRE      = 5          # Warm-up cycles (discarded) for panel (a)/(b)
HYST_N_PLOT     = 1          # Cycles to plot (steady state)
HYST_DT         = 5e-5       # Bristle integration timestep for panels (a)/(b) [s]
HYST_DT_ENERGY  = 1e-4       # Coarser timestep used for energy-vs-freq sweep [s]
HYST_N_PRE_ENRG = 3          # Warm-up cycles for energy sweep (fewer = faster)

# --------------- NDOB Bandwidth -----------------------------------------------
#
#   NDOB pole: λ [rad/s].  f_NDOB = λ / (2π)
#   Used only for the bandwidth comparison row in the parameter table and
#   the energy-per-cycle plot in Figure 2.
#
NDOB_LAMBDA     = 90.0       # NDOB observer pole λ        [rad/s]

# --------------- Figure & Output Settings ----------------------------------
FIGURE_DPI   = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"   # matplotlib style
SAVE_FIGURE  = True        # Write PDF to disk?
_OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "figures_comparative", "friction")
SAVE_PATH    = os.path.join(_OUT_DIR, "fig1_stribeck_comparison.pdf")
SAVE_PATH_F2 = os.path.join(_OUT_DIR, "fig2_hysteresis_energy.pdf")
SAVE_PATH_F3 = os.path.join(_OUT_DIR, "fig3_transient_parametertable.pdf")

# ===========================================================================
#  END OF USER CONFIGURATION
# ===========================================================================


# ---------------------------------------------------------------------------
#  VALIDATION HELPERS
# ---------------------------------------------------------------------------

def _validate_params(tustin: dict, lugre: dict) -> list:
    """
    Check matching conditions and return a list of warning strings.
    Empty list = all conditions satisfied.
    """
    issues = []
    tol = 1e-9

    if abs(tustin['tau_c'] - lugre['tau_c']) > tol:
        issues.append(
            f"  [MISMATCH] tau_c: Tustin={tustin['tau_c']:.4f}  LuGre={lugre['tau_c']:.4f}  N.m"
        )
    if abs(tustin['tau_s'] - lugre['tau_s']) > tol:
        issues.append(
            f"  [MISMATCH] tau_s: Tustin={tustin['tau_s']:.4f}  LuGre={lugre['tau_s']:.4f}  N.m"
        )
    if abs(tustin['v_s'] - lugre['v_s']) > tol:
        issues.append(
            f"  [MISMATCH] v_s:   Tustin={tustin['v_s']:.4f}  LuGre={lugre['v_s']:.4f}  rad/s"
        )
    if abs(tustin['b'] - lugre['sigma_2']) > tol:
        issues.append(
            f"  [MISMATCH] viscous: Tustin b={tustin['b']:.4f}  LuGre sigma_2={lugre['sigma_2']:.4f}  N.m.s/rad"
        )
    if abs(tustin['alpha'] - 2.0) > tol:
        issues.append(
            f"  [WARNING]  Tustin alpha={tustin['alpha']:.2f} != 2.0  "
            f"(LuGre Stribeck is Gaussian; alpha!=2 introduces intentional shape difference)"
        )
    return issues


# ---------------------------------------------------------------------------
#  CURVE COMPUTATION
# ---------------------------------------------------------------------------

def compute_tustin_curve(v_array: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate Tustin friction torque over a velocity array."""
    model = SmoothedTustinFriction(
        tau_c=params['tau_c'],
        tau_s=params['tau_s'],
        v_s=params['v_s'],
        b=params['b'],
        alpha=params['alpha'],
        v_epsilon=params['v_epsilon'],
    )
    return model.compute_torque(v_array)


def compute_lugre_ss_analytical(v_array: np.ndarray, params: dict) -> np.ndarray:
    """
    Analytical LuGre steady-state torque (no simulation needed):
        tau_ss(v) = g(v) * sign(v) + sigma_2 * v
    Valid when dz/dt = 0.
    """
    g_v = (params['tau_c'] +
           (params['tau_s'] - params['tau_c']) *
           np.exp(-(v_array / params['v_s'])**2))
    # Use tanh for numerical continuity at v=0 (same idea as Tustin v_epsilon)
    sign_smooth = np.tanh(v_array / 1e-6)
    return g_v * sign_smooth + params['sigma_2'] * v_array


def compute_lugre_g_function(v_array: np.ndarray, params: dict) -> np.ndarray:
    """LuGre Stribeck function g(v) — envelope of the velocity-dependent friction limit."""
    return (params['tau_c'] +
            (params['tau_s'] - params['tau_c']) *
            np.exp(-(np.abs(v_array) / params['v_s'])**2))


def compute_tustin_envelope(v_array: np.ndarray, params: dict) -> np.ndarray:
    """Tustin friction envelope (magnitude, before sign application)."""
    return (params['tau_c'] +
            (params['tau_s'] - params['tau_c']) *
            np.exp(-(np.abs(v_array) / params['v_s'])**params['alpha']))


def _lugre_stable_nsub(v_scalar: float, params: dict, dt: float,
                        safety: float = 0.45) -> int:
    """
    Compute number of Forward Euler sub-steps needed for bristle stability.

    Stability condition: dt_sub < 2 * g(v) / (sigma_0 * |v|)
    Using `safety` factor (<0.5) for comfortable margin.
    """
    v_abs = abs(v_scalar) + 1e-12
    g_v   = (params['tau_c'] +
             (params['tau_s'] - params['tau_c']) *
             np.exp(-(v_scalar / params['v_s']) ** 2))
    dt_stable = safety * g_v / (params['sigma_0'] * v_abs)
    return max(1, int(np.ceil(dt / dt_stable)))


def compute_lugre_transient(v_step: float, params: dict,
                             dt: float, T: float):
    """
    Simulate LuGre bristle response to a velocity step from rest.
    Uses velocity-adaptive sub-stepping to guarantee Forward Euler stability.

    Returns
    -------
    t      : time array [s]
    tau    : friction torque array [N.m]
    z      : bristle state array [rad]
    tau_ss : analytical steady-state value [N.m]  (scalar)
    """
    n_steps = int(T / dt)
    model = LuGreFriction(
        sigma_0=np.array([params['sigma_0']]),
        sigma_1=np.array([params['sigma_1']]),
        sigma_2=np.array([params['sigma_2']]),
        tau_c=np.array([params['tau_c']]),
        tau_s=np.array([params['tau_s']]),
        v_s=np.array([params['v_s']]),
        n_axes=1,
    )
    v_arr  = np.array([float(v_step)])
    n_sub  = _lugre_stable_nsub(v_step, params, dt)
    dt_sub = dt / n_sub

    tau_out = np.empty(n_steps)
    z_out   = np.empty(n_steps)

    for i in range(n_steps):
        for _ in range(n_sub):
            tau_i = model.step(v_arr, dt_sub)[0]
        tau_out[i] = tau_i
        z_out[i]   = model.z[0]

    t_out = np.arange(n_steps) * dt

    # Analytical SS
    g_v    = params['tau_c'] + (params['tau_s'] - params['tau_c']) * np.exp(-(v_step / params['v_s'])**2)
    tau_ss = g_v * np.sign(v_step) + params['sigma_2'] * v_step

    return t_out, tau_out, z_out, tau_ss


def compute_matching_metrics(v_array: np.ndarray,
                              tau_tustin: np.ndarray,
                              tau_lugre_ss: np.ndarray,
                              v_deadzone: float = 1e-3) -> dict:
    """
    Compute quantitative matching quality metrics.
    Excludes the near-zero dead-zone where sign() is ill-defined.
    """
    mask      = np.abs(v_array) > v_deadzone
    diff      = tau_tustin[mask] - tau_lugre_ss[mask]
    ref_mag   = np.abs(tau_tustin[mask])
    rel_err   = np.abs(diff) / np.maximum(ref_mag, 1e-12)

    return dict(
        max_abs_err   = float(np.max(np.abs(diff))),
        rms_abs_err   = float(np.sqrt(np.mean(diff**2))),
        max_rel_err   = float(np.max(rel_err)) * 100.0,   # percent
        rms_rel_err   = float(np.sqrt(np.mean(rel_err**2))) * 100.0,
        n_eval        = int(np.sum(mask)),
    )


# ---------------------------------------------------------------------------
#  HYSTERESIS SIMULATION
# ---------------------------------------------------------------------------

def compute_hysteresis_loop(amplitude: float, freq_hz: float,
                             n_pre: int, n_plot: int,
                             dt: float, lugre_p: dict):
    """
    Simulate LuGre response to v(t) = amplitude * sin(2*pi*freq_hz*t).

    Runs n_pre cycles (warm-up, discarded), then records n_plot cycles.

    Returns
    -------
    v_plot   : velocity array over n_plot cycles [rad/s]
    tau_plot : LuGre friction torque over n_plot cycles [N.m]
    t_plot   : time array over n_plot cycles [s]
    energy   : energy dissipated over n_plot cycles [J]  (= integral of tau*v dt)
    """
    T_cycle   = 1.0 / freq_hz
    n_per     = max(1, int(round(T_cycle / dt)))
    dt_actual = T_cycle / n_per          # adjust dt to divide evenly into T

    n_warm  = n_pre  * n_per
    n_rec   = n_plot * n_per
    n_total = n_warm + n_rec

    model = LuGreFriction(
        sigma_0=np.array([lugre_p['sigma_0']]),
        sigma_1=np.array([lugre_p['sigma_1']]),
        sigma_2=np.array([lugre_p['sigma_2']]),
        tau_c  =np.array([lugre_p['tau_c']]),
        tau_s  =np.array([lugre_p['tau_s']]),
        v_s    =np.array([lugre_p['v_s']]),
        n_axes =1,
    )

    tau_rec = np.empty(n_rec)
    v_rec   = np.empty(n_rec)
    t_rec   = np.empty(n_rec)

    omega = 2.0 * np.pi * freq_hz
    for i in range(n_total):
        t_i = i * dt_actual
        v_i = amplitude * np.sin(omega * t_i)
        # Velocity-adaptive sub-stepping: recompute n_sub each outer step
        n_sub_i  = _lugre_stable_nsub(v_i, lugre_p, dt_actual)
        dt_sub_i = dt_actual / n_sub_i
        for _ in range(n_sub_i):
            tau_i = model.step(np.array([v_i]), dt_sub_i)[0]
        if i >= n_warm:
            k = i - n_warm
            v_rec[k]   = v_i
            tau_rec[k] = tau_i
            t_rec[k]   = t_i - n_warm * dt_actual

    energy = float(np.trapezoid(tau_rec * v_rec, t_rec))   # [N.m * rad/s * s = J]
    return v_rec, tau_rec, t_rec, energy


def compute_energy_vs_frequency(freqs_hz: list, amplitude: float,
                                 n_pre: int, n_plot: int, dt: float,
                                 lugre_p: dict, tustin_p: dict):
    """
    Compute energy dissipated per cycle by LuGre and Tustin over a range of
    excitation frequencies.

    Tustin is memoryless, so its energy is:
        E_tustin = integral_0^T  tau_tustin(v(t)) * v(t) dt
    This equals ∫ tau(v)·v dt — no frequency dependence beyond amplitude.

    Returns
    -------
    E_lugre_list  : list of LuGre energy per cycle [J]
    E_tustin_list : list of Tustin energy per cycle [J]
    """
    tustin_model = SmoothedTustinFriction(
        tau_c=tustin_p['tau_c'], tau_s=tustin_p['tau_s'],
        v_s=tustin_p['v_s'], b=tustin_p['b'],
        alpha=tustin_p['alpha'], v_epsilon=tustin_p['v_epsilon'],
    )

    E_lugre  = []
    E_tustin = []
    for f in freqs_hz:
        v_lg, _, t_lg, E_lg = compute_hysteresis_loop(
            amplitude, f, n_pre, n_plot, dt, lugre_p
        )
        # Tustin on the same velocity time-series
        tau_t = tustin_model.compute_torque(v_lg)
        E_t   = float(np.trapezoid(tau_t * v_lg, t_lg))
        E_lugre.append(E_lg)
        E_tustin.append(E_t)
    return E_lugre, E_tustin


# ---------------------------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------------------------

# Publication-grade color palette (colorblind-safe, IEEE/AIAA compatible)
_C = dict(
    tustin  = "#0072B2",   # deep blue   — Tustin / algebraic
    lugre   = "#D55E00",   # vermillion  — LuGre dynamic
    visc    = "#999999",   # mid-grey    — viscous reference
    env     = "#009E73",   # teal-green  — Stribeck envelope / g(v)
    resid   = "#CC79A7",   # mauve       — residual / error
    ref     = "#BBBBBB",   # light-grey  — neutral reference lines
)

def _despine(ax):
    """Remove top and right spines for clean publication look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)


def _guard_ylim(ax, min_span=0.01):
    """
    Prevent matplotlib ticker OverflowError when all plotted data is
    identically zero (zero y-range).  Ensures y-axis has at least min_span.
    Call after all ax.plot() calls on any panel whose data may be flat.
    """
    y0, y1 = ax.get_ylim()
    if abs(y1 - y0) < 1e-30:
        mid = (y0 + y1) / 2.0
        ax.set_ylim(mid - min_span / 2, mid + min_span / 2)


def _panel_label(ax, letter):
    """Place a bold panel label (a), (b)... centred below the x-axis."""
    ax.text(0.5, -0.22, f"({letter})",
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            va='top', ha='center',
            clip_on=False)



def build_figure(tustin_p: dict, lugre_p: dict,
                 issues: list, metrics: dict) -> plt.Figure:
    """Construct the 6-panel publication-quality comparison figure."""

    # ------------------------------------------------------------------
    #  Global rcParams — sets consistent typography for the whole figure
    # ------------------------------------------------------------------
    rc = {
        'font.family':         'serif',
        'mathtext.fontset':    'dejavuserif',
        'axes.labelsize':      9.5,
        'axes.titlesize':      9.5,
        'xtick.labelsize':     8.5,
        'ytick.labelsize':     8.5,
        'legend.fontsize':     8.0,
        'legend.framealpha':   0.92,
        'legend.edgecolor':    '#cccccc',
        'legend.handlelength': 2.0,
        'lines.linewidth':     1.8,
        'axes.linewidth':      0.8,
        'axes.grid':           True,
        'grid.color':          '#e0e0e0',
        'grid.linewidth':      0.5,
        'grid.linestyle':      '-',
        'figure.facecolor':    'white',
        'axes.facecolor':      'white',
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
    }
    with matplotlib.rc_context(rc):
        fig = _build_panels(tustin_p, lugre_p, issues, metrics)
    return fig


def _build_panels(tustin_p, lugre_p, issues, metrics):

    match_ok  = (len(issues) == 0)
    match_col = "#006400" if match_ok else "#8B0000"

    # ---- Figure and gridspec — 2×2, panels (a)–(d) only ----
    fig = plt.figure(figsize=(13.0, 8.0), dpi=FIGURE_DPI)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.62, wspace=0.38,
        left=0.08, right=0.96,
        top=0.93, bottom=0.10,
    )

    ax_main  = fig.add_subplot(gs[0, 0])
    ax_zoom  = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_env   = fig.add_subplot(gs[1, 1])

    # ---- Suptitle ----
    ss_status = "Steady-state match: confirmed" if match_ok \
                else "Steady-state match: FAILED — see parameter table"
    fig.suptitle(
        rf"Friction Model Stribeck Comparison — {AXIS_LABEL} $\;|\;$ {ss_status}",
        fontsize=10.5, fontweight='bold', color=match_col,
        y=0.975,
    )

    # ---- Pre-compute all curves ----
    v_full = np.linspace(-V_MAX, V_MAX, N_SWEEP)
    v_pos  = np.linspace(1e-4,   V_ZOOM_MAX, N_SWEEP // 2)
    v_env  = np.linspace(0.0,    V_MAX,      500)
    v_s    = float(tustin_p['v_s'])

    tau_t_full   = compute_tustin_curve(v_full, tustin_p)
    tau_lg_full  = compute_lugre_ss_analytical(v_full, lugre_p)
    tau_visc     = tustin_p['b'] * v_full
    residual_mNm = (tau_t_full - tau_lg_full) * 1e3

    tau_t_pos    = compute_tustin_curve(v_pos, tustin_p)
    tau_lg_pos   = compute_lugre_ss_analytical(v_pos, lugre_p)
    g_v_pos      = compute_lugre_g_function(v_pos, lugre_p)
    env_t_pos    = compute_tustin_envelope(v_pos, tustin_p)

    g_v_env  = compute_lugre_g_function(v_env, lugre_p)
    env_t_v  = compute_tustin_envelope(v_env, tustin_p)

    # ============================================================== #
    #  Panel (a) — Full bidirectional Stribeck curve
    # ============================================================== #
    ax = ax_main
    _despine(ax)

    ax.axhline(0, color='k', lw=0.5, zorder=1)
    ax.axvline(0, color='k', lw=0.5, zorder=1)

    # Coulomb / static reference levels (right-side only, labelled once)
    for yval, lbl in [(tustin_p['tau_c'],  r'$\tau_c$'),
                      (tustin_p['tau_s'],  r'$\tau_s$'),
                      (-tustin_p['tau_c'], ''),
                      (-tustin_p['tau_s'], '')]:
        ax.axhline(yval, color=_C['ref'], lw=0.7, ls='--', zorder=1)
        if lbl:
            ax.text(V_MAX * 1.01, yval, lbl,
                    fontsize=8, color='#555555', va='center', ha='left',
                    clip_on=False)

    # Stribeck velocity markers
    ax.axvline( v_s, color=_C['ref'], lw=0.7, ls='--', zorder=1)
    ax.axvline(-v_s, color=_C['ref'], lw=0.7, ls='--', zorder=1)
    ax.text(v_s + V_MAX * 0.015, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.04,
            r'$v_s$', fontsize=8, color='#777777', va='top')

    ax.plot(v_full, tau_visc,    color=_C['visc'],   lw=1.2, ls=(0,(4,3)), zorder=2,
            label=r'Viscous ($b\,\omega$)')
    ax.plot(v_full, tau_lg_full, color=_C['lugre'],  lw=2.2, ls='-',       zorder=4,
            label='LuGre (SS)')
    ax.plot(v_full, tau_t_full,  color=_C['tustin'], lw=1.8, ls='--',      zorder=5,
            label='Tustin')

    ax.set_xlabel(r'Angular velocity, $\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Friction torque, $\tau_f$ (N$\cdot$m)')
    ax.set_xlim(-V_MAX, V_MAX * 1.05)
    ax.legend(loc='lower right', handlelength=2.4)
    _panel_label(ax, 'a')

    # ============================================================== #
    #  Panel (b) — Stribeck region zoom (positive v)
    # ============================================================== #
    ax = ax_zoom
    _despine(ax)

    ax.fill_between(v_pos, g_v_pos, tustin_p['tau_c'],
                    alpha=0.10, color=_C['lugre'], zorder=1,
                    label=r'Stribeck excess above $\tau_c$')
    ax.fill_between(v_pos, 0, tustin_p['tau_c'],
                    alpha=0.06, color=_C['visc'], zorder=1)

    ax.axhline(tustin_p['tau_c'], color=_C['ref'],   lw=0.7, ls='--', zorder=2)
    ax.axhline(tustin_p['tau_s'], color=_C['ref'],   lw=0.7, ls='--', zorder=2)

    for mult, lbl in [(1, r'$v_s$'), (2, r'$2v_s$'), (3, r'$3v_s$')]:
        ax.axvline(mult * v_s, color=_C['ref'], lw=0.6, ls=':', zorder=1)
        ax.text(mult * v_s + V_ZOOM_MAX * 0.015, tustin_p['tau_c'] * 0.08,
                lbl, fontsize=7.5, color='#888888', va='bottom')

    ax.plot(v_pos, g_v_pos,   color=_C['env'],    lw=1.5, ls='-.',   zorder=3,
            label=r'LuGre $g(\omega)$ envelope')
    ax.plot(v_pos, env_t_pos, color=_C['tustin'], lw=1.3, ls=':',    zorder=3,
            label=r'Tustin envelope ($\alpha$=' + f"{tustin_p['alpha']:.0f}" + ')')
    ax.plot(v_pos, tau_lg_pos, color=_C['lugre'], lw=2.2, ls='-',    zorder=5,
            label='LuGre (SS torque)')
    ax.plot(v_pos, tau_t_pos,  color=_C['tustin'],lw=1.8, ls='--',   zorder=5,
            label='Tustin (torque)')

    # Label tau_c and tau_s at right edge
    ax.text(V_ZOOM_MAX * 1.01, tustin_p['tau_c'], r'$\tau_c$',
            fontsize=8, color='#555555', va='center', ha='left', clip_on=False)
    ax.text(V_ZOOM_MAX * 1.01, tustin_p['tau_s'], r'$\tau_s$',
            fontsize=8, color='#555555', va='center', ha='left', clip_on=False)

    ax.set_xlabel(r'Angular velocity, $\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Friction torque, $\tau_f$ (N$\cdot$m)')
    ax.set_xlim(0, V_ZOOM_MAX * 1.05)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', handlelength=2.2, fontsize=7.5)
    _panel_label(ax, 'b')

    # ============================================================== #
    #  Panel (c) — Steady-state residual
    # ============================================================== #
    ax = ax_resid
    _despine(ax)

    # Dead-zone shading (where tanh smoothing acts)
    dz = tustin_p['v_epsilon'] * 3.0
    ax.axvspan(-dz, dz, color='#FFF3CD', alpha=0.8, zorder=1,
               label=r'$\tanh$ smoothing zone ($\pm 3\,\varepsilon$)')

    ax.axhline(0, color='k', lw=0.8, ls='-', zorder=2)
    ax.fill_between(v_full, 0, residual_mNm,
                    color=_C['resid'], alpha=0.20, zorder=3)
    ax.plot(v_full, residual_mNm,
            color=_C['resid'], lw=1.8, zorder=4,
            label=r'$\tau_{\rm Tustin} - \tau_{\rm LuGre,\,SS}$')

    # Compact metric box — top right, no arrows
    rms_val = metrics['rms_abs_err'] * 1e3
    max_val = metrics['max_abs_err'] * 1e3
    box_txt = (f"Max $|e|$ = {max_val:.2f} mN$\\cdot$m\n"
               f"RMS $|e|$ = {rms_val:.2f} mN$\\cdot$m")
    ax.text(0.97, 0.96, box_txt,
            transform=ax.transAxes,
            fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', edgecolor='#cccccc', lw=0.8))

    ax.set_xlabel(r'Angular velocity, $\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Residual, $\Delta\tau_f$ (mN$\cdot$m)')
    ax.legend(loc='upper left', fontsize=7.5)
    _guard_ylim(ax)
    _panel_label(ax, 'c')

    # ============================================================== #
    #  Panel (d) — Stribeck envelope shape comparison
    # ============================================================== #
    ax = ax_env
    _despine(ax)

    diff_env_mNm = (env_t_v - g_v_env) * 1e3

    ax.axhline(lugre_p['tau_c'], color=_C['ref'], lw=0.7, ls='--', zorder=1)
    ax.axhline(lugre_p['tau_s'], color=_C['ref'], lw=0.7, ls='--', zorder=1)

    ax.plot(v_env, g_v_env,      color=_C['lugre'],  lw=2.0, ls='-',  zorder=4,
            label=r'LuGre $g(\omega)$')
    ax.plot(v_env, env_t_v,      color=_C['tustin'], lw=1.8, ls='--', zorder=5,
            label=r'Tustin envelope')
    ax.plot(v_env, diff_env_mNm / 1e3, color=_C['resid'], lw=1.2, ls=':', zorder=3,
            label=r'Envelope diff (N$\cdot$m)')

    # Labels at right edge
    ax.text(V_MAX * 1.01, lugre_p['tau_c'], r'$\tau_c$',
            fontsize=8, color='#555555', va='center', ha='left', clip_on=False)
    ax.text(V_MAX * 1.01, lugre_p['tau_s'], r'$\tau_s$',
            fontsize=8, color='#555555', va='center', ha='left', clip_on=False)

    ax.set_xlabel(r'Angular velocity, $\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Stribeck envelope (N$\cdot$m)')
    ax.set_xlim(0, V_MAX * 1.05)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7.5)
    _guard_ylim(ax)
    _panel_label(ax, 'd')

    return fig


# ---------------------------------------------------------------------------
#  FIGURE 3  —  Bristle Transient + Parameter Table
# ---------------------------------------------------------------------------

def build_figure3(tustin_p: dict, lugre_p: dict, metrics: dict) -> plt.Figure:
    """Panels (e) and (f) split into their own dedicated figure."""
    rc = {
        'font.family':         'serif',
        'mathtext.fontset':    'dejavuserif',
        'axes.labelsize':      9.5,
        'axes.titlesize':      9.5,
        'xtick.labelsize':     8.5,
        'ytick.labelsize':     8.5,
        'legend.fontsize':     8.0,
        'legend.framealpha':   0.92,
        'legend.edgecolor':    '#cccccc',
        'legend.handlelength': 2.0,
        'lines.linewidth':     1.8,
        'axes.linewidth':      0.8,
        'axes.grid':           True,
        'grid.color':          '#e0e0e0',
        'grid.linewidth':      0.5,
        'grid.linestyle':      '-',
        'figure.facecolor':    'white',
        'axes.facecolor':      'white',
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
    }
    with matplotlib.rc_context(rc):
        fig = _build_fig3_panels(tustin_p, lugre_p, metrics)
    return fig


def _build_fig3_panels(tustin_p, lugre_p, metrics):

    fig = plt.figure(figsize=(13.0, 6.5), dpi=FIGURE_DPI)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        hspace=0.0, wspace=0.38,
        left=0.07, right=0.97,
        top=0.90, bottom=0.18,
    )

    ax_trans = fig.add_subplot(gs[0, 0])
    ax_tbl   = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        rf"LuGre Bristle Transient Response and Parameter Matching Table — {AXIS_LABEL}",
        fontsize=10, fontweight='bold', y=0.97,
    )

    # ============================================================== #
    #  Panel (e) — LuGre bristle transient step responses
    # ============================================================== #
    ax = ax_trans
    _despine(ax)

    t_ms   = 1e3
    tau_mN = 1e3
    v_list = sorted(TRANSIENT_V_LIST)
    cmap   = matplotlib.colormaps['viridis'].resampled(len(v_list))

    g_vs  = lugre_p['tau_c'] + (lugre_p['tau_s'] - lugre_p['tau_c']) * np.exp(-1.0)
    tau_z = g_vs / (lugre_p['sigma_0'] * lugre_p['v_s'])

    for k, v_step in enumerate(v_list):
        col = cmap(k / max(len(v_list) - 1, 1))
        t, tau, _, tau_ss = compute_lugre_transient(
            v_step, lugre_p, DT_TRANSIENT, T_TRANSIENT
        )
        tau_t_equiv = float(compute_tustin_curve(np.array([v_step]), tustin_p)[0])

        ax.plot(t * t_ms, tau * tau_mN, color=col, lw=1.8, zorder=4,
                label=rf'$\omega={v_step:.2f}$ rad s$^{{-1}}$')
        ax.annotate('', xy=(T_TRANSIENT * t_ms, tau_ss * tau_mN),
                    xytext=(T_TRANSIENT * t_ms * 0.93, tau_ss * tau_mN),
                    arrowprops=dict(arrowstyle='-', color=col, lw=1.2))
        ax.axhline(tau_t_equiv * tau_mN, color=col, lw=0.8, ls=':', alpha=0.55, zorder=3)

    ax.axvspan(0, tau_z * t_ms, color='#DDEEFF', alpha=0.6, zorder=1,
               label=rf'$\tau_z$ at $v_s$ = {tau_z*t_ms:.2f} ms')

    from matplotlib.lines import Line2D as _L2D
    extra_handles = [
        _L2D([0], [0], color='k', lw=1.5, ls='-',  label='LuGre transient'),
        _L2D([0], [0], color='k', lw=0.8, ls=':', label='Tustin SS equiv.'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra_handles,
              labels=labels + ['LuGre transient', 'Tustin SS equiv.'],
              fontsize=7.0, loc='lower right', ncol=2, handlelength=1.8)

    ax.set_xlabel(r'Time (ms)')
    ax.set_ylabel(r'Friction torque (mN$\cdot$m)')
    ax.set_xlim(0, T_TRANSIENT * t_ms)
    _panel_label(ax, 'e')

    # ============================================================== #
    #  Panel (f) — Parameter matching table
    # ============================================================== #
    ax = ax_tbl
    ax.axis('off')

    def _chk(a, b, tol=1e-9):
        return "Yes" if abs(a - b) < tol else "No"

    def _fmt(x):
        return f"{x:.4g}" if abs(x) < 1000 else f"{x:.3e}"

    col_labels = ["Parameter", "Tustin", "LuGre", "Match"]
    rows = [
        [r"$\tau_c$ (N$\cdot$m)",               _fmt(tustin_p['tau_c']),     _fmt(lugre_p['tau_c']),   _chk(tustin_p['tau_c'],  lugre_p['tau_c'])],
        [r"$\tau_s$ (N$\cdot$m)",               _fmt(tustin_p['tau_s']),     _fmt(lugre_p['tau_s']),   _chk(tustin_p['tau_s'],  lugre_p['tau_s'])],
        [r"$v_s$ (rad s$^{-1}$)",               _fmt(tustin_p['v_s']),       _fmt(lugre_p['v_s']),     _chk(tustin_p['v_s'],    lugre_p['v_s'])],
        [r"$b\,/\,\sigma_2$ (N$\cdot$m s)",     _fmt(tustin_p['b']),         _fmt(lugre_p['sigma_2']), _chk(tustin_p['b'],      lugre_p['sigma_2'])],
        [r"$\alpha$ (Tustin only)",              _fmt(tustin_p['alpha']),     "2 (fixed)",              _chk(tustin_p['alpha'],  2.0)],
        [r"$\sigma_0$ (N$\cdot$m rad$^{-1}$)",  "—",                         _fmt(lugre_p['sigma_0']), "transient"],
        [r"$\sigma_1$ (N$\cdot$m s)",           "—",                         _fmt(lugre_p['sigma_1']), "transient"],
        [r"$\varepsilon$ (rad s$^{-1}$)",       _fmt(tustin_p['v_epsilon']), "—",                      "smoothing"],
        [r"$f_{\rm bristle}$ (Hz)",
         "—",
         f"{(1.0/(2*np.pi))*np.sqrt(lugre_p['sigma_0']/lugre_p['sigma_1']):.1f}",
         "dynamics"],
        [r"$f_{\rm NDOB}$ (Hz)",
         "—",
         "—",
         f"{NDOB_LAMBDA/(2*np.pi):.1f}"],
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='upper center',
        cellLoc='center',
        bbox=[0.0, 0.20, 1.0, 0.80],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    HDR = "#2B4C7E"
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor(HDR)
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.09)

    for i, row in enumerate(rows, start=1):
        for j in range(len(col_labels)):
            tbl[i, j].set_height(0.08)
            tbl[i, j].set_facecolor("#F7F9FC" if i % 2 == 0 else "white")
        val = row[3]
        if val == "Yes":
            tbl[i, 3].set_facecolor("#D4EDDA")
            tbl[i, 3].set_text_props(color="#155724", fontweight='bold')
        elif val == "No":
            tbl[i, 3].set_facecolor("#F8D7DA")
            tbl[i, 3].set_text_props(color="#721C24", fontweight='bold')
        else:
            tbl[i, 3].set_facecolor("#FFF3CD")
            tbl[i, 3].set_text_props(color="#856404")

    # Metrics box below table
    tau_z_vs = 1e3 * (lugre_p['tau_c'] + (lugre_p['tau_s'] - lugre_p['tau_c']) *
                      np.exp(-1.0)) / (lugre_p['sigma_0'] * lugre_p['v_s'])
    metrics_txt = (
        r"$\mathbf{Steady\text{-}state\ match\ metrics}$" + "\n"
        rf"Max $|\Delta\tau|$ = {metrics['max_abs_err']*1e3:.3f} mN$\cdot$m  ({metrics['max_rel_err']:.2f}%)" + "\n"
        rf"RMS $|\Delta\tau|$ = {metrics['rms_abs_err']*1e3:.3f} mN$\cdot$m  ({metrics['rms_rel_err']:.2f}%)" + "\n\n"
        r"$\mathbf{Bristle\ dynamics\ at\ }v_s$" + "\n"
        rf"$\tau_z = g(v_s)/(\sigma_0 v_s)$ = {tau_z_vs:.3f} ms" + "    "
        rf"$\omega_n = \sqrt{{\sigma_0}}$ = {np.sqrt(lugre_p['sigma_0']):.0f} rad s$^{{-1}}$"
    )
    ax.text(0.50, 0.16, metrics_txt,
            transform=ax.transAxes,
            fontsize=8.0, va='top', ha='center',
            linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.55',
                      facecolor='#F0F4FA', edgecolor='#B0BED0', lw=0.8))

    _panel_label(ax, 'f')

    return fig


# ---------------------------------------------------------------------------
#  FIGURE 2  —  Hysteresis & Energy Analysis
# ---------------------------------------------------------------------------

def build_figure2(tustin_p: dict, lugre_p: dict) -> plt.Figure:
    """
    Four-panel publication figure:
      (a)  Hysteresis loops at multiple excitation frequencies (LuGre only)
      (b)  Single-frequency loop: LuGre vs Tustin overlay
      (c)  Enclosed loop area / energy per cycle vs frequency
      (d)  Energy dissipation ratio  E_LuGre / E_Tustin vs frequency
           with NDOB and bristle bandwidth markers
    """

    rc = {
        'font.family':         'serif',
        'mathtext.fontset':    'dejavuserif',
        'axes.labelsize':      9.5,
        'axes.titlesize':      9.5,
        'xtick.labelsize':     8.5,
        'ytick.labelsize':     8.5,
        'legend.fontsize':     8.0,
        'legend.framealpha':   0.92,
        'legend.edgecolor':    '#cccccc',
        'legend.handlelength': 2.0,
        'lines.linewidth':     1.8,
        'axes.linewidth':      0.8,
        'axes.grid':           True,
        'grid.color':          '#e0e0e0',
        'grid.linewidth':      0.5,
        'grid.linestyle':      '-',
        'figure.facecolor':    'white',
        'axes.facecolor':      'white',
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
    }

    with matplotlib.rc_context(rc):
        fig = _build_fig2_panels(tustin_p, lugre_p)
    return fig


def _build_fig2_panels(tustin_p, lugre_p):

    # ---- Pre-compute all hysteresis data ----
    print("  [Fig 2] Simulating hysteresis loops ...", flush=True)

    # Panel (a): loop overlay at all HYST_FREQS_HZ
    loops = {}
    for f in HYST_FREQS_HZ:
        v_l, tau_l, _, E_l = compute_hysteresis_loop(
            HYST_AMP, f, HYST_N_PRE, HYST_N_PLOT, HYST_DT, lugre_p
        )
        loops[f] = (v_l, tau_l, E_l)

    # Panel (b): single-frequency comparison
    v_cmp, tau_lg_cmp, _, E_cmp = compute_hysteresis_loop(
        HYST_AMP, HYST_COMPARE_F, HYST_N_PRE, HYST_N_PLOT, HYST_DT, lugre_p
    )
    tustin_model = SmoothedTustinFriction(
        tau_c=tustin_p['tau_c'], tau_s=tustin_p['tau_s'],
        v_s=tustin_p['v_s'], b=tustin_p['b'],
        alpha=tustin_p['alpha'], v_epsilon=tustin_p['v_epsilon'],
    )
    tau_t_cmp = tustin_model.compute_torque(v_cmp)

    # Panels (c) & (d): energy vs frequency over a log-spaced grid
    # Use coarser dt and fewer pre-cycles — sufficient for energy trend curves.
    f_dense  = np.logspace(np.log10(0.5), np.log10(50.0), 25)
    E_lg_arr, E_t_arr = compute_energy_vs_frequency(
        list(f_dense), HYST_AMP, HYST_N_PRE_ENRG, HYST_N_PLOT,
        HYST_DT_ENERGY, lugre_p, tustin_p
    )
    E_lg_arr = np.array(E_lg_arr)
    E_t_arr  = np.array(E_t_arr)

    # Bristle and NDOB frequencies
    f_bristle = (1.0 / (2.0 * np.pi)) * np.sqrt(lugre_p['sigma_0'] / lugre_p['sigma_1'])
    f_ndob    = NDOB_LAMBDA / (2.0 * np.pi)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(13.0, 10.0), dpi=FIGURE_DPI)
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.48, wspace=0.36,
        left=0.09, right=0.96,
        top=0.93, bottom=0.08,
    )
    ax_hyst  = fig.add_subplot(gs[0, 0])
    ax_comp  = fig.add_subplot(gs[0, 1])
    ax_eng   = fig.add_subplot(gs[1, 0])
    ax_ratio = fig.add_subplot(gs[1, 1])

    # Explicit colorblind-safe palette for hysteresis loops —
    # avoids plasma's yellow endpoint which is unsuitable for publication.
    _LOOP_COLORS = ["#0D47A1", "#0072B2", "#009E73", "#D55E00", "#7B2D8B"]
    # Extend automatically if HYST_FREQS_HZ has more than 5 entries
    if len(HYST_FREQS_HZ) > len(_LOOP_COLORS):
        _extra = matplotlib.colormaps['tab10'].resampled(len(HYST_FREQS_HZ))
        _LOOP_COLORS = [_extra(i) for i in range(len(HYST_FREQS_HZ))]
    cmap_loops = _LOOP_COLORS

    # ============================================================== #
    #  Panel (a) — Hysteresis loops at multiple frequencies
    # ============================================================== #
    ax = ax_hyst
    _despine(ax)

    ax.axhline(0, color='k', lw=0.5, zorder=1)
    ax.axvline(0, color='k', lw=0.5, zorder=1)

    for k, f in enumerate(HYST_FREQS_HZ):
        col = cmap_loops[k]
        v_l, tau_l, E_l = loops[f]
        ax.plot(v_l, tau_l * 1e3, color=col, lw=1.6, zorder=3,
                label=rf'$f$ = {f:.0f} Hz  ($\Delta E$ = {abs(E_l)*1e3:.2f} mJ)')

    # Tustin (no hysteresis — single-valued curve)
    v_sw = np.linspace(-HYST_AMP * 1.05, HYST_AMP * 1.05, 600)
    tau_t_sw = tustin_model.compute_torque(v_sw)
    ax.plot(v_sw, tau_t_sw * 1e3, color='#333333', lw=1.2, ls='--',
            zorder=4, label='Tustin (memoryless)')

    ax.set_xlabel(r'Angular velocity, $\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Friction torque, $\tau_f$ (mN$\cdot$m)')
    ax.legend(loc='upper left', fontsize=7.2, handlelength=1.8)
    _panel_label(ax, 'a')

    # inset note
    ax.text(0.97, 0.03,
            "Loop area $=$ energy\ndissipated per cycle",
            transform=ax.transAxes, fontsize=7.5,
            ha='right', va='bottom', color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFBE6',
                      edgecolor='#CCBB88', lw=0.7))

    # ============================================================== #
    #  Panel (b) — LuGre vs Tustin at one frequency
    # ============================================================== #
    ax = ax_comp
    _despine(ax)

    ax.axhline(0, color='k', lw=0.5, zorder=1)
    ax.axvline(0, color='k', lw=0.5, zorder=1)

    # Fill the enclosed LuGre loop area
    ax.fill(v_cmp, tau_lg_cmp * 1e3,
            color=_C['lugre'], alpha=0.12, zorder=2)

    ax.plot(v_cmp, tau_lg_cmp * 1e3, color=_C['lugre'], lw=2.0, zorder=4,
            label=rf'LuGre  ($f$ = {HYST_COMPARE_F:.0f} Hz)')
    ax.plot(v_cmp, tau_t_cmp * 1e3,  color=_C['tustin'], lw=1.6, ls='--', zorder=5,
            label='Tustin (single-valued)')

    # Annotate energy enclosed
    ax.text(0.97, 0.03,
            rf"$\Delta E_{{cycle}}$ = {abs(E_cmp)*1e3:.3f} mJ",
            transform=ax.transAxes, fontsize=8.0,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#cccccc', lw=0.8))

    ax.set_xlabel(r'Angular velocity, $\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Friction torque, $\tau_f$ (mN$\cdot$m)')
    ax.legend(loc='upper left', fontsize=8.0)
    _panel_label(ax, 'b')

    # ============================================================== #
    #  Panel (c) — Energy dissipated per cycle vs frequency
    # ============================================================== #
    ax = ax_eng
    _despine(ax)

    ax.semilogx(f_dense, np.abs(E_lg_arr) * 1e3,
                color=_C['lugre'],  lw=2.0, label='LuGre')
    ax.semilogx(f_dense, np.abs(E_t_arr)  * 1e3,
                color=_C['tustin'], lw=1.6, ls='--', label='Tustin')

    # Bandwidth markers
    for fval, lbl, col in [
        (f_bristle, rf'$f_{{\rm bristle}}$ = {f_bristle:.1f} Hz', '#E07000'),
        (f_ndob,    rf'$f_{{\rm NDOB}}$ = {f_ndob:.1f} Hz',       '#006080'),
    ]:
        ax.axvline(fval, color=col, lw=1.2, ls=':', zorder=3)
        ax.text(fval * 1.07, 0.97, lbl,
                transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=col, va='top', ha='left',
                rotation=90)

    ax.set_xlabel(r'Excitation frequency, $f$ (Hz)')
    ax.set_ylabel(r'Energy dissipated per cycle (mJ)')
    ax.legend(loc='upper right', fontsize=8.0)
    _guard_ylim(ax)
    _panel_label(ax, 'c')

    # ============================================================== #
    #  Panel (d) — Energy ratio LuGre / Tustin  +  bandwidth markers
    # ============================================================== #
    ax = ax_ratio
    _despine(ax)

    ratio = np.abs(E_lg_arr) / np.maximum(np.abs(E_t_arr), 1e-30)

    ax.semilogx(f_dense, ratio, color=_C['resid'], lw=2.0,
                label=r'$E_{\rm LuGre}/E_{\rm Tustin}$')
    ax.axhline(1.0, color='k', lw=0.8, ls='--', label='Ratio = 1  (identical energy)')

    # Shaded "NDOB can compensate" region (f < f_ndob)
    ax.axvspan(f_dense[0], f_ndob, color='#D0F0D0', alpha=0.35, zorder=1,
               label=r'NDOB bandwidth ($f < f_{\rm NDOB}$)')

    # Bandwidth markers
    for fval, lbl, col in [
        (f_bristle, rf'$f_{{\rm bristle}}$ = {f_bristle:.1f} Hz', '#E07000'),
        (f_ndob,    rf'$f_{{\rm NDOB}}$ = {f_ndob:.1f} Hz',       '#006080'),
    ]:
        ax.axvline(fval, color=col, lw=1.2, ls=':', zorder=4)
        ax.text(fval * 1.07, 0.97, lbl,
                transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=col, va='top', ha='left',
                rotation=90)

    # Bandwidth mismatch annotation
    ax.annotate(
        'Bandwidth\nmismatch',
        xy=((f_bristle + f_ndob) / 2, ratio[np.argmin(np.abs(f_dense - (f_bristle + f_ndob) / 2))]),
        xytext=(f_bristle * 0.5, ratio.max() * 0.7),
        fontsize=7.5, color='#333333',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.9,
                        connectionstyle='arc3,rad=0.25'),
    )

    ax.set_xlabel(r'Excitation frequency, $f$ (Hz)')
    ax.set_ylabel(r'Energy ratio $E_{\rm LuGre}/E_{\rm Tustin}$')
    ax.legend(loc='upper right', fontsize=7.5, handlelength=1.8)
    _guard_ylim(ax)
    _panel_label(ax, 'd')

    # ---- Suptitle ----
    fig.suptitle(
        rf"LuGre Hysteresis and Energy Analysis — {AXIS_LABEL}"
        rf"$\;|\;$ $A$ = {HYST_AMP:.2f} rad s$^{{-1}}$"
        rf"$\;|\;$ $f_{{\rm bristle}}$ = {f_bristle:.1f} Hz"
        rf"$\;|\;$ $f_{{\rm NDOB}}$ = {f_ndob:.1f} Hz",
        fontsize=10, fontweight='bold', y=0.975,
    )

    return fig


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  FRICTION MODEL STEADY-STATE MATCHING STUDY")
    print(f"  Axis: {AXIS_LABEL}")
    print("=" * 70)

    # ---- Validate ----
    issues = _validate_params(TUSTIN, LUGRE)
    if issues:
        print("\n[!] PARAMETER MISMATCH DETECTED — SS curves will NOT coincide:")
        for msg in issues:
            print(msg)
    else:
        print("\n[OK] All matching conditions satisfied — SS curves should coincide.")

    # ---- Compute curves for metrics ----
    v_eval     = np.linspace(-V_MAX, V_MAX, N_SWEEP)
    tau_tustin = compute_tustin_curve(v_eval, TUSTIN)
    tau_lg_ss  = compute_lugre_ss_analytical(v_eval, LUGRE)
    metrics    = compute_matching_metrics(v_eval, tau_tustin, tau_lg_ss)

    print(f"\n  Max absolute residual : {metrics['max_abs_err']*1e3:.4f} mN.m")
    print(f"  RMS absolute residual : {metrics['rms_abs_err']*1e3:.4f} mN.m")
    print(f"  Max relative error    : {metrics['max_rel_err']:.4f} %")
    print(f"  RMS relative error    : {metrics['rms_rel_err']:.4f} %")

    os.makedirs(_OUT_DIR, exist_ok=True)

    # ---- Build Figure 1 — Stribeck comparison (panels a–d) ----
    print("\n  Building Figure 1 (Stribeck comparison, panels a-d) ...")
    fig1 = build_figure(TUSTIN, LUGRE, issues, metrics)
    if SAVE_FIGURE:
        fig1.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
        print(f"  Saved -> {SAVE_PATH}")

    # ---- Build Figure 2 — Hysteresis & energy ----
    print("\n  Building Figure 2 (Hysteresis & energy analysis) ...")
    fig2 = build_figure2(TUSTIN, LUGRE)
    if SAVE_FIGURE:
        fig2.savefig(SAVE_PATH_F2, dpi=300, bbox_inches='tight')
        print(f"  Saved -> {SAVE_PATH_F2}")

    # ---- Build Figure 3 — Bristle transient + parameter table (panels e–f) ----
    print("\n  Building Figure 3 (Bristle transient & parameter table, panels e-f) ...")
    fig3 = build_figure3(TUSTIN, LUGRE, metrics)
    if SAVE_FIGURE:
        fig3.savefig(SAVE_PATH_F3, dpi=300, bbox_inches='tight')
        print(f"  Saved -> {SAVE_PATH_F3}")

    plt.show()
    print("\n  Done.")


if __name__ == "__main__":
    main()
