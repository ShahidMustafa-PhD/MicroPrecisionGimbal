#!/usr/bin/env python3
"""
Friction Model Comparative Study: Viscous vs Tustin vs LuGre under FBL+NDOB Control
====================================================================================

This script runs three sequential high-fidelity simulations of the MicroPrecision
gimbal digital twin.  Every simulation uses the *same* Feedback-Linearization +
Nonlinear Disturbance Observer (FBL+NDOB) controller, identical initial
conditions, identical trajectories, identical sensors, and identical
environmental disturbances.  The only variable that changes between the three
runs is the plant-side friction model:

    1) Simple viscous friction       (memoryless, linear in velocity)
    2) Smoothed Tustin / Stribeck    (memoryless, nonlinear in velocity)
    3) Dynamic LuGre bristle model   (history-dependent, quadrant glitches)

The goal is to expose the sensitivity of the NDOB-augmented feedback
linearization controller to the structural complexity of the friction model.
Because the plant friction is the only degree of freedom, every residual
performance difference can be attributed to the plant/observer model mismatch
introduced by that particular friction description.

This file is standalone: it does NOT modify any existing module and does NOT
subclass the existing ResearchComparisonPlotter.  A dedicated
FrictionComparisonPlotter class lives in this file and reproduces the eight
plot types that the user requested:

    - Tracking error (log scale, with handover thresholds)
    - Position tracking (Az, El vs command)
    - FSM performance (tip, tilt, post-FSM residual LOS)
    - Internal controller signals (v_virtual, tau_unsaturated, d_hat_NDOB)
    - Instantaneous stroke consumption
    - Stroke margin summary bar chart
    - Benchmark table
    - Friction compensation vs true friction torque

All figures are saved as 300 DPI vector PDFs to
    figures_comparative/friction_comparative/
and are also displayed interactively to the user at the end of the run.

Author : Dr. S. Shahid Mustafa
Date   : April 10, 2026
"""

from __future__ import annotations

import copy
import csv
import datetime
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality matplotlib defaults (LaTeX typography, STIX math)
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams['mathtext.fontset']          = 'stix'
matplotlib.rcParams['font.family']               = 'STIXGeneral'
matplotlib.rcParams['font.size']                 = 12
matplotlib.rcParams['axes.labelsize']            = 12
matplotlib.rcParams['axes.titlesize']            = 14
matplotlib.rcParams['xtick.labelsize']           = 10
matplotlib.rcParams['ytick.labelsize']           = 10
matplotlib.rcParams['legend.fontsize']           = 10
matplotlib.rcParams['figure.titlesize']          = 16
matplotlib.rcParams['figure.dpi']                = 100
matplotlib.rcParams['savefig.dpi']               = 300
matplotlib.rcParams['axes.grid']                 = False
matplotlib.rcParams['axes.axisbelow']            = True

# Make the local package importable when the script is invoked from any CWD
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner,
)
from lasercom_digital_twin.core.performance_metrics import (
    StrokeMetrics,
    StrokeMetricsResult,
)

# Re-use the Stribeck / steady-state comparison machinery from friction_study.py.
# That script is __main__-guarded, so importing it only pulls in the helper
# functions and constants — nothing is executed on import.
from friction_study import (
    build_figure                as _fs_build_stribeck_figure,
    compute_tustin_curve        as _fs_compute_tustin_curve,
    compute_lugre_ss_analytical as _fs_compute_lugre_ss,
    compute_matching_metrics    as _fs_compute_matching_metrics,
    _validate_params            as _fs_validate_params,
    V_MAX                       as _FS_V_MAX,
    N_SWEEP                     as _FS_N_SWEEP,
)


# =============================================================================
# Color scheme for the three friction models
# =============================================================================
# Chosen for colorblind safety and strong luminance contrast in B&W prints.
class FrictionColors:
    VISCOUS = '#1f77b4'   # Blue   — simplest, linear baseline
    TUSTIN  = '#ff7f0e'   # Orange — memoryless Stribeck
    LUGRE   = '#2ca02c'   # Green  — dynamic bristle model
    GROUND  = '#9467bd'   # Purple — applied-friction ground truth
    LIMIT   = '#d62728'   # Red    — saturation / threshold lines
    TARGET  = '#000000'   # Black  — command trace


# Colors for the disturbance-torque figure (axis and component decomposition)
class DisturbancePlotColors:
    AZIMUTH   = '#1f77b4'   # Blue
    ELEVATION = '#d62728'   # Red
    WIND      = '#17becf'   # Teal  — Dryden gust
    VIBRATION = '#8c564b'   # Brown — structural modes


FRICTION_LABELS = {
    'viscous': 'FBL+NDOB (Viscous)',
    'tustin':  'FBL+NDOB (Tustin)',
    'lugre':   'FBL+NDOB (LuGre)',
}

FRICTION_COLOR_MAP = {
    'viscous': FrictionColors.VISCOUS,
    'tustin':  FrictionColors.TUSTIN,
    'lugre':   FrictionColors.LUGRE,
}

FRICTION_ORDER: Tuple[str, str, str] = ('viscous', 'tustin', 'lugre')


# =============================================================================
# NDOB sweep persistence
# =============================================================================
# Every run of this script logs the NDOB tuning (lambda_az, lambda_el, d_max)
# together with the eleven benchmark metrics shown in Fig 7 of
# FrictionComparisonPlotter, for each of the three friction models. The CSV
# accumulates across runs so that the bar-chart history below grows richer
# with every experiment. Stored outside figures_*/ so it survives `git clean`
# and isn't swept up by the figures gitignore.
NDOB_SWEEP_DIR        = Path('ndob_sweep_results')
NDOB_SWEEP_LOG_PATH   = NDOB_SWEEP_DIR / 'ndob_sweep_log.csv'
NDOB_SWEEP_PLOTS_DIR  = NDOB_SWEEP_DIR / 'plots'

# (csv_key, display_name, unit, value_format) — one entry per Fig 7 row.
FIG7_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    ('post_fsm_residual_tip_urad',  'Post-FSM Residual Tip',  r'$\mu$rad', '.2f'),
    ('post_fsm_residual_tilt_urad', 'Post-FSM Residual Tilt', r'$\mu$rad', '.2f'),
    ('los_rms_urad',                'LOS Error RMS',          r'$\mu$rad', '.2f'),
    ('torque_rms_az_nm',            'Torque RMS Az',          r'N$\cdot$m', '.4f'),
    ('torque_rms_el_nm',            'Torque RMS El',          r'N$\cdot$m', '.4f'),
    ('scr_rms_tip_pct',             'SCR$_{rms}$ Tip',        r'%',        '.1f'),
    ('scr_rms_tilt_pct',            'SCR$_{rms}$ Tilt',       r'%',        '.1f'),
    ('s_bias_tip_mrad',             'S$_{bias}$ Tip',         r'mrad',     '.3f'),
    ('s_bias_tilt_mrad',            'S$_{bias}$ Tilt',        r'mrad',     '.3f'),
    ('dsm_tip_mrad',                'DSM Tip',                r'mrad',     '.3f'),
    ('dsm_tilt_mrad',               'DSM Tilt',               r'mrad',     '.3f'),
)


def _ndob_csv_fieldnames() -> List[str]:
    """Column order for the NDOB sweep CSV — header + per-model metric blocks."""
    fields = ['timestamp', 'enable', 'lambda_az', 'lambda_el', 'd_max']
    for fm in FRICTION_ORDER:
        for key, _name, _unit, _fmt in FIG7_METRICS:
            fields.append(f'{fm}_{key}')
    return fields


def append_ndob_experiment_log(
    csv_path: Path,
    ndob_cfg: Dict,
    metrics_per_model: Dict[str, Dict[str, float]],
) -> int:
    """Append one experiment row to the NDOB sweep CSV.

    Parameters
    ----------
    csv_path
        Destination CSV. Created (with header) on first call.
    ndob_cfg
        The ``SimulationConfig.ndob_config`` dict actually used for the run.
    metrics_per_model
        ``{friction_model: {metric_key: value}}`` as produced by
        ``FrictionComparisonPlotter._collect_fig7_metrics``.

    Returns
    -------
    int
        Zero-based index of the newly appended row (i.e. the experiment number
        minus one).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ndob_csv_fieldnames()

    row: Dict[str, object] = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'enable':    int(bool(ndob_cfg.get('enable', False))),
        'lambda_az': float(ndob_cfg.get('lambda_az', float('nan'))),
        'lambda_el': float(ndob_cfg.get('lambda_el', float('nan'))),
        'd_max':     float(ndob_cfg.get('d_max',     float('nan'))),
    }
    for fm in FRICTION_ORDER:
        per_model = metrics_per_model.get(fm, {})
        for key, _name, _unit, _fmt in FIG7_METRICS:
            row[f'{fm}_{key}'] = float(per_model.get(key, float('nan')))

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    with csv_path.open('r', encoding='utf-8') as fh:
        n_data_rows = max(0, sum(1 for _ in fh) - 1)
    return n_data_rows - 1


def _read_ndob_sweep_log(csv_path: Path) -> List[Dict[str, str]]:
    """Read every row from the NDOB sweep CSV. Returns [] if the file is missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open('r', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def build_ndob_sweep_history_figures(
    csv_path: Path,
    output_dir: Path | None = None,
    save: bool = True,
) -> Dict[str, plt.Figure]:
    """Build one annotated bar chart per Fig 7 metric from the sweep CSV.

    Each figure shows three bars per experiment (viscous / Tustin / LuGre)
    with the corresponding NDOB tuning ($\\lambda_{az}$, $\\lambda_{el}$,
    $d_{max}$) annotated below the group. The figures grow richer as more
    experiments are appended to the CSV.

    Parameters
    ----------
    csv_path
        Path to the NDOB sweep CSV produced by ``append_ndob_experiment_log``.
    output_dir
        Where to save PDFs. Defaults to ``NDOB_SWEEP_PLOTS_DIR``.
    save
        If True, write a PDF for every metric figure.

    Returns
    -------
    dict
        ``{metric_key: matplotlib.Figure}``. Empty if the CSV is missing or empty.
    """
    rows = _read_ndob_sweep_log(csv_path)
    if not rows:
        print(f"[NDOB sweep] No data at {csv_path} — history plot skipped.")
        return {}

    # Sort experiments left-to-right by ascending (lambda_az, lambda_el) so
    # lower NDOB gains sit on the left and the bars march upward in the gain.
    # d_max breaks ties between otherwise-identical tunings.
    def _sort_key(r: Dict[str, str]) -> Tuple[float, float, float]:
        try:
            return (float(r['lambda_az']),
                    float(r['lambda_el']),
                    float(r['d_max']))
        except (KeyError, ValueError):
            return (float('inf'), float('inf'), float('inf'))

    rows = sorted(rows, key=_sort_key)

    if output_dir is None:
        output_dir = NDOB_SWEEP_PLOTS_DIR
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    n_exp = len(rows)
    x = np.arange(n_exp)
    bar_w = 0.26
    fig_width = float(min(22.0, max(8.0, 1.7 * n_exp + 3.0)))

    figs: Dict[str, plt.Figure] = {}

    for key, name, unit, fmt in FIG7_METRICS:
        fig, ax = plt.subplots(figsize=(fig_width, 6.2))

        for i, fm in enumerate(FRICTION_ORDER):
            try:
                vals = [float(r[f'{fm}_{key}']) for r in rows]
            except (KeyError, ValueError):
                vals = [float('nan')] * n_exp
            offset = (i - 1) * bar_w
            bars = ax.bar(
                x + offset, vals, bar_w,
                color=FRICTION_COLOR_MAP[fm],
                label=FRICTION_LABELS[fm],
                edgecolor='black', linewidth=0.6,
                alpha=0.92,
            )
            for b, v in zip(bars, vals):
                if not np.isfinite(v):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height(),
                    format(v, fmt),
                    ha='center', va='bottom',
                    fontsize=7.5, rotation=0,
                )

        tick_labels = []
        for i, r in enumerate(rows):
            try:
                la = float(r['lambda_az'])
                le = float(r['lambda_el'])
                dm = float(r['d_max'])
                lab = (
                    f'#{i + 1}\n'
                    rf'$\lambda_{{az}}$={la:.1f}'  '\n'
                    rf'$\lambda_{{el}}$={le:.1f}'  '\n'
                    rf'$d_{{max}}$={dm:.2f}'
                )
            except (KeyError, ValueError):
                lab = f'#{i + 1}'
            tick_labels.append(lab)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel(f'{name} [{unit}]', fontsize=12, fontweight='bold')
        ax.set_xlabel('NDOB experiment (annotated with tuning)',
                      fontsize=12, fontweight='bold')
        ax.set_title(
            f'NDOB Sweep History — {name}   (n = {n_exp})',
            fontsize=14, fontweight='bold', pad=10,
        )
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='best', framealpha=0.92, fontsize=10)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.10 * (ymax - ymin if ymax > ymin else 1.0))

        fig.tight_layout()

        if save:
            out_path = output_dir / f'ndob_sweep_{key}.pdf'
            try:
                fig.savefig(
                    str(out_path), format='pdf', dpi=300,
                    bbox_inches='tight', facecolor='white',
                )
                print(f"  [OK] {out_path.name}")
            except Exception as exc:
                print(f"  [ERROR] Failed to save {out_path.name}: {exc}")

        figs[f'ndob_sweep_{key}'] = fig

    return figs


def build_scr_tables_pdf(
    csv_path: Path,
    output_path: Path | None = None,
) -> None:
    """Build a professional multi-page PDF with SCR tables grouped by (λ_az, λ_el).

    Each unique (lambda_az, lambda_el) pair gets its own full-page table with
    columns for d_max, SCR_rms Tip [%] and SCR_rms Tilt [%] per friction
    model. Tables are split across pages for clarity and readability.

    Parameters
    ----------
    csv_path
        Path to the NDOB sweep CSV.
    output_path
        Where to save the PDF. Defaults to ``NDOB_SWEEP_DIR / 'scr_tables.pdf'``.
    """
    rows = _read_ndob_sweep_log(csv_path)
    if not rows:
        print("[SCR tables] No data — skipped.")
        return

    if output_path is None:
        output_path = NDOB_SWEEP_DIR / 'scr_tables.pdf'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Group rows by (lambda_az, lambda_el), sorted ascending ──────────
    from collections import OrderedDict
    groups: Dict[Tuple[float, float], List[Dict[str, str]]] = OrderedDict()
    for r in sorted(rows, key=lambda r: (
        float(r.get('lambda_az', 'inf')),
        float(r.get('lambda_el', 'inf')),
        float(r.get('d_max', 'inf')),
    )):
        pair = (float(r['lambda_az']), float(r['lambda_el']))
        groups.setdefault(pair, []).append(r)

    n_groups = len(groups)

    # Use PdfPages for multi-page output
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(str(output_path)) as pdf:
        page_num = 0
        for page_idx, ((laz, lel), grp_rows) in enumerate(groups.items()):
            fig = plt.figure(figsize=(11.0, 8.5))
            ax = fig.add_subplot(111)
            ax.axis('off')

            # ── Title block ────────────────────────────────────────────
            title_text = (
                rf'$\lambda_{{az}}$ = {laz:.1f},  '
                rf'$\lambda_{{el}}$ = {lel:.1f}'
            )
            subtitle_text = f'Stroke Consumption Ratio (SCR) Analysis'
            fig.text(0.5, 0.96, title_text,
                     ha='center', fontsize=18, fontweight='bold')
            fig.text(0.5, 0.92, subtitle_text,
                     ha='center', fontsize=13, style='italic', color='#555555')

            # ── Build table data ───────────────────────────────────────
            col_labels = [
                r'$d_{{\max}}$',
                'Viscous\nSCR Tip',
                'Viscous\nSCR Tilt',
                'Tustin\nSCR Tip',
                'Tustin\nSCR Tilt',
                'LuGre\nSCR Tip',
                'LuGre\nSCR Tilt',
            ]
            cell_text = []
            for r in grp_rows:
                def _v(fm, key):
                    try:
                        val = float(r[f'{fm}_{key}'])
                        return f'{val:.1f}%'
                    except (KeyError, ValueError):
                        return '—'

                cell_text.append([
                    format(float(r.get('d_max', 'nan')), '.3f'),
                    _v('viscous', 'scr_rms_tip_pct'),
                    _v('viscous', 'scr_rms_tilt_pct'),
                    _v('tustin',  'scr_rms_tip_pct'),
                    _v('tustin',  'scr_rms_tilt_pct'),
                    _v('lugre',   'scr_rms_tip_pct'),
                    _v('lugre',   'scr_rms_tilt_pct'),
                ])

            # ── Create matplotlib table ────────────────────────────────
            col_widths = [0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]
            tbl = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                colWidths=col_widths,
                loc='center',
                cellLoc='center',
                bbox=[0.05, 0.15, 0.90, 0.75],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1.0, 2.3)

            # ── Style header and alternating rows ──────────────────────
            header_colors = {
                0: '#2d3e50',
                1: FrictionColors.VISCOUS,  2: FrictionColors.VISCOUS,
                3: FrictionColors.TUSTIN,   4: FrictionColors.TUSTIN,
                5: FrictionColors.LUGRE,    6: FrictionColors.LUGRE,
            }
            for (row_i, col_i), cell in tbl.get_celld().items():
                if row_i == 0:
                    cell.set_facecolor(header_colors.get(col_i, '#2d3e50'))
                    cell.set_text_props(color='white', fontweight='bold',
                                        fontsize=11)
                    cell.set_height(0.08)
                else:
                    if row_i % 2 == 0:
                        cell.set_facecolor('#f5f5f5')
                    else:
                        cell.set_facecolor('#ffffff')
                    cell.set_height(0.062)
                cell.set_edgecolor('#cccccc')
                cell.set_linewidth(0.8)

            # ── Footer info ────────────────────────────────────────────
            footer_text = (
                f'Page {page_idx + 1} of {n_groups}  |  '
                f'{len(grp_rows)} experiment(s)  |  '
                'Generated by friction_controller_comparison.py'
            )
            fig.text(0.5, 0.03, footer_text,
                     ha='center', fontsize=9, color='#888888',
                     style='italic')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            page_num += 1

    print(f"  [OK] {output_path.name}  ({n_groups} page(s), "
          f"{sum(len(g) for g in groups.values())} experiment(s))")
    return


# =============================================================================
# Simulation driver
# =============================================================================
def _build_config(
    friction_model: str,
    *,
    signal_type: str,
    target_az_deg: float,
    target_el_deg: float,
    target_amplitude: float,
    target_period: float,
    target_reachangle: float,
    duration: float,
    env_disturbance_enabled: bool,
    env_disturbance_cfg: Dict,
    viscous_cfg: Dict,
    tustin_cfg: Dict,
    lugre_cfg: Dict,
) -> SimulationConfig:
    """
    Build a SimulationConfig for a single FBL+NDOB run with the requested
    plant friction model.  All three runs share identical controller gains,
    NDOB bandwidth, sensors, and disturbance configuration — only the
    friction selection flags and the friction dictionaries change.
    """
    friction_model = friction_model.lower()
    if friction_model not in ('viscous', 'tustin', 'lugre'):
        raise ValueError(
            f"friction_model must be 'viscous', 'tustin', or 'lugre' — got '{friction_model}'"
        )

    use_tustin = (friction_model == 'tustin')
    use_lugre  = (friction_model == 'lugre')

    config = SimulationConfig(
        dt_sim=0.0001,
        dt_coarse=0.002,
        dt_fine=0.00001,
        dt_qpd=0.00001,
        log_period=0.001,
        seed=42,
        target_az=np.deg2rad(target_az_deg),
        target_el=np.deg2rad(target_el_deg),
        target_enabled=True,
        target_type=signal_type,
        target_amplitude=target_amplitude,
        target_period=target_period,
        target_reachangle=target_reachangle,
        # --- FBL+NDOB controller (identical for all three friction runs) ---
        use_feedback_linearization=True,
        use_direct_state_feedback=False,
        # --- Plant friction selection (the single variable under study) ---
        use_tustin_friction=use_tustin,
        use_lugre_friction=use_lugre,
        tustin_config=tustin_cfg,
        lugre_config=lugre_cfg,
        viscous_config=viscous_cfg,
        # --- Visualisation and logging ---
        enable_visualization=False,
        enable_plotting=False,           # We drive our own plotting
        real_time_factor=0.0,
        vibration_enabled=False,
        # --- Environmental disturbances (injected into plant only) ---
        environmental_disturbance_enabled=env_disturbance_enabled,
        environmental_disturbance_config=env_disturbance_cfg,
        # --- Feedback linearization inner-loop gains ---
        feedback_linearization_config={
            'kp': [400.0, 800.0],
            'kd': [40.0, 60.0],
            'ki': [50.0, 50.0],
            'enable_integral': False,
            'tau_max': [1.0, 0.7],
            'tau_min': [-1.0, -0.7],
            'conditional_friction': True,
            'enable_disturbance_compensation': False,
        },
        # --- NDOB (enabled, identical tuning for all runs) ---
        ndob_config={
            'enable': True,
            'lambda_az': 110.0,
            'lambda_el': 100.0,
            'd_max': 0.5,
        },
        dynamics_config={
            'pan_mass': 1.0,
            'tilt_mass': 0.5,
            'cm_r': 0.0,
            'cm_h': 0.0,
            'gravity': 9.81,
        },
        qpd_config={'linear_range': 0.008},
        coarse_controller_config={
            'kp': [3.514, 1.320],
            'ki': [15.464, 4.148],
            'kd': [0.293, 0.059418],
            'tau_max': [1.0, 0.7],
            'tau_min': [-1.0, -0.7],
            'anti_windup_gain': 1.0,
            'tau_rate_limit': 50.0,
            'enable_derivative': True,
        },
    )
    return config


def run_friction_comparison(
    signal_type: str = 'hybridsig',
    disturbance_config: Dict | None = None,
) -> Dict[str, Dict]:
    """
    Execute three FBL+NDOB simulations with viscous / Tustin / LuGre friction.

    Returns
    -------
    Dict[str, Dict]
        A dictionary keyed by friction model name ('viscous', 'tustin', 'lugre')
        where each value is the full results dictionary produced by
        DigitalTwinRunner.run_simulation().
    """
    print("\n" + "=" * 80)
    print("FRICTION MODEL COMPARATIVE STUDY — FBL+NDOB Controller")
    print("=" * 80)
    print(f"Signal Type     : {signal_type.upper()}")
    print("Controller      : Feedback Linearization + NDOB (fixed across runs)")
    print("Study Variable  : Plant friction model (viscous / Tustin / LuGre)")
    print("Objective       : Isolate the effect of friction-model complexity on")
    print("                  NDOB-augmented FBL tracking performance.")
    print("=" * 80 + "\n")

    # Common trajectory settings — identical for every run
    target_az_deg     = 0.0
    target_el_deg     = 0.0
    duration          = 5.0
    target_amplitude  = 30.0     # [deg]
    target_period     = 15.0     # [s]
    target_reachangle = 5.0      # [deg]

    # ── Environmental disturbance block (identical for every run) ───────────
    env_disturbance_enabled = disturbance_config is not None
    env_disturbance_cfg = {
        'seed': 42,
        'wind': {
            'enabled': False,
            'start_time': 5.0,
            'mean_velocity': 5.0,
            'turbulence_intensity': 0.15,
            'scale_length': 200.0,
            'direction_deg': 45.0,
            'gimbal_area': 0.02,
            'gimbal_arm': 0.15,
        },
        'vibration': {
            'enabled': False,
            'start_time': 2.0,
            'modal_frequencies': [15.0, 45.0, 80.0],
            'modal_dampings': [0.02, 0.015, 0.01],
            'modal_amplitudes': [1e-3, 5e-4, 2e-4],
            'inertia_coupling': 0.1,
        },
        'structural_noise': {
            'enabled': False,
            'std': 0.005,
            'freq_low': 100.0,
            'freq_high': 500.0,
        },
    }
    if disturbance_config:
        for sec in ('wind', 'vibration', 'structural_noise'):
            if sec in disturbance_config:
                env_disturbance_cfg[sec].update(disturbance_config[sec])

    # ── Friction model parameter dictionaries ───────────────────────────────
    # All three runs use these exact dictionaries.  The SimulationConfig flags
    # select which one is actually applied to the plant for a given run.
    # Steady-state Coulomb / Stribeck values are matched between Tustin and
    # LuGre so that any observed difference is attributable solely to LuGre's
    # bristle dynamics, not to a change in Coulomb level.
    viscous_cfg = {
        'friction_az': 0.09,
        'friction_el': 0.075,
    }

    tustin_cfg = {
        'tau_s_az': 0.25, 'tau_s_el': 0.18,
        'tau_c_az': 0.15, 'tau_c_el': 0.10,
        'v_s_az':  0.05,  'v_s_el':  0.05,
        'b_az':    0.02,  'b_el':    0.015,
        'v_epsilon': 0.005,
        'nominal_tau_s_az': 0.25, 'nominal_tau_s_el': 0.18,
        'nominal_tau_c_az': 0.15, 'nominal_tau_c_el': 0.10,
        'nominal_v_s_az':  0.05,  'nominal_v_s_el':  0.05,
        'nominal_v_epsilon': 0.005,
        'nominal_friction_noise_pct': 0.20,
    }

    lugre_cfg = {
        'sigma_0_az': 1.0e4, 'sigma_0_el': 8.0e3,
        'sigma_1_az': 1.0,   'sigma_1_el': 0.8,
        'sigma_2_az': 0.02,  'sigma_2_el': 0.015,
        'tau_c_az':   0.15,  'tau_c_el':   0.10,
        'tau_s_az':   0.25,  'tau_s_el':   0.18,
        'v_s_az':     0.05,  'v_s_el':     0.05,
        'lugre_max_dt': 1e-4,
    }

    print("Test Conditions:")
    print(f"  - Target Base   : Az={target_az_deg:.1f}°, El={target_el_deg:.1f}°")
    print(f"  - Signal Type   : {signal_type}")
    if signal_type != 'constant':
        print(f"  - Amplitude     : ±{target_amplitude:.1f}°")
        print(f"  - Period        : {target_period:.1f} s")
    if signal_type == 'hybridsig':
        print(f"  - Reach Angle   : {target_reachangle:.1f}°")
    print(f"  - Duration      : {duration:.1f} s")
    print(f"  - Env. Disturb. : {'ENABLED' if env_disturbance_enabled else 'disabled'}")
    print()

    results: Dict[str, Dict] = {}
    ndob_cfg_used: Dict | None = None

    for idx, friction_model in enumerate(FRICTION_ORDER, start=1):
        print("-" * 80)
        print(f"RUN {idx}/3 — FBL+NDOB with {friction_model.upper()} friction")
        print("-" * 80)

        cfg = _build_config(
            friction_model,
            signal_type=signal_type,
            target_az_deg=target_az_deg,
            target_el_deg=target_el_deg,
            target_amplitude=target_amplitude,
            target_period=target_period,
            target_reachangle=target_reachangle,
            duration=duration,
            env_disturbance_enabled=env_disturbance_enabled,
            env_disturbance_cfg=copy.deepcopy(env_disturbance_cfg),
            viscous_cfg=copy.deepcopy(viscous_cfg),
            tustin_cfg=copy.deepcopy(tustin_cfg),
            lugre_cfg=copy.deepcopy(lugre_cfg),
        )

        if ndob_cfg_used is None:
            ndob_cfg_used = dict(cfg.ndob_config)

        runner = DigitalTwinRunner(cfg)
        res = runner.run_simulation(duration=duration)
        results[friction_model] = res

        los_rms = res['los_error_rms'] * 1e6
        print(f"[OK] {friction_model.capitalize():<7s} run complete — "
              f"LOS RMS = {los_rms:.2f} µrad\n")

    # ── Textual performance summary ─────────────────────────────────────────
    print("=" * 96)
    print("FRICTION MODEL PERFORMANCE SUMMARY  (FBL+NDOB controller, all runs)")
    print("=" * 96)
    header = (
        f"{'Metric':<38} "
        f"{'Viscous':<16} {'Tustin':<16} {'LuGre':<16}"
    )
    print(header)
    print("-" * len(header))

    def _fmt(results_dict: Dict, key: str, scale: float = 1.0, fmt: str = '.3f') -> str:
        val = results_dict.get(key, np.nan) * scale
        return f"{val:{fmt}}"

    for label, key, scale, fmt in [
        ('LOS Error RMS   [µrad]',   'los_error_rms',   1e6, '.2f'),
        ('LOS Error Final [µrad]',   'los_error_final', 1e6, '.2f'),
        ('Torque RMS Az   [N·m]',    'torque_rms_az',   1.0, '.4f'),
        ('Torque RMS El   [N·m]',    'torque_rms_el',   1.0, '.4f'),
    ]:
        row = f"{label:<38} "
        for fm in FRICTION_ORDER:
            row += f"{_fmt(results[fm], key, scale, fmt):<16}"
        print(row)
    print("=" * 96 + "\n")

    # ── Build Tustin / LuGre scalar dictionaries for the Stribeck figure ────
    # friction_study.build_figure expects per-axis scalar dicts; we expose the
    # Azimuth parameter set (the Elevation set can be plotted by re-running
    # with ``axis_label='Elevation'`` and swapping the *_el keys below).
    tustin_p = {
        'tau_c':     tustin_cfg['tau_c_az'],
        'tau_s':     tustin_cfg['tau_s_az'],
        'v_s':       tustin_cfg['v_s_az'],
        'b':         tustin_cfg['b_az'],
        'alpha':     2.0,
        'v_epsilon': tustin_cfg['v_epsilon'],
    }
    lugre_p = {
        'sigma_0': lugre_cfg['sigma_0_az'],
        'sigma_1': lugre_cfg['sigma_1_az'],
        'sigma_2': lugre_cfg['sigma_2_az'],
        'tau_c':   lugre_cfg['tau_c_az'],
        'tau_s':   lugre_cfg['tau_s_az'],
        'v_s':     lugre_cfg['v_s_az'],
    }

    # ── Generate publication-quality comparative figures ────────────────────
    print("Generating publication-quality comparative plots ...")
    plotter = FrictionComparisonPlotter(
        save_figures=True,
        show_figures=True,
        output_dir=Path('figures_comparative') / 'friction_comparative',
    )
    plotter.plot_all(
        results,
        tustin_p=tustin_p,
        lugre_p=lugre_p,
        axis_label='Azimuth Gimbal',
        ndob_cfg=ndob_cfg_used,
        ndob_log_path=NDOB_SWEEP_LOG_PATH,
        ndob_history_dir=NDOB_SWEEP_PLOTS_DIR,
    )

    return results


# =============================================================================
# FRICTION COMPARISON PLOTTER
# =============================================================================
class FrictionComparisonPlotter:
    """
    Publication-quality plotter that compares FBL+NDOB performance under three
    different plant friction models (viscous / Tustin / LuGre).

    This class is deliberately self-contained: it does not inherit from the
    existing ResearchComparisonPlotter because the data model differs — here
    every trace on every figure belongs to the *same* controller, with the
    friction model as the only distinguishing feature.

    Figures generated (all saved as vector PDFs):
      Fig 1  — Tracking error (2×1, log scale, with FSM stroke thresholds)
      Fig 2  — Position tracking (2×1, Az / El with command overlay)
      Fig 3  — FSM performance (3×1, tip, tilt, post-FSM residual LOS)
      Fig 4  — Internal signals (3×1, v_virtual, tau_unsaturated, d_hat_NDOB)
      Fig 5  — Instantaneous stroke consumption (2×1)
      Fig 6  — Stroke margin summary (bar chart)
      Fig 7  — Benchmark table (text figure)
      Fig 8  — Friction compensation vs true applied friction (2×1, twin-axis)
    """

    # Visual parameters (match the house style used elsewhere in the project)
    LINEWIDTH       = 2.0
    ALPHA_PRIMARY   = 0.9
    GRID_ALPHA      = 0.3
    GRID_LS         = ':'
    LEGEND_FONTSIZE = 12
    AXIS_FONTSIZE   = 13
    TITLE_FONTSIZE  = 14
    SUPTITLE_FSIZE  = 15

    FIG_SIZE_2x1  = (11.0, 8.5)
    FIG_SIZE_3x1  = (11.0, 10.5)
    FIG_SIZE_1x1  = (11.0, 7.0)

    def __init__(
        self,
        save_figures: bool = True,
        show_figures: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        self.save_figures = save_figures
        self.show_figures = show_figures
        self.output_dir = Path(output_dir) if output_dir is not None else \
            Path('figures_comparative') / 'friction_comparative'
        if self.save_figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figures: Dict[str, plt.Figure] = {}
        self._results: Dict[str, Dict] = {}
        self._times:   Dict[str, np.ndarray] = {}
        self._tustin_p: Dict | None = None
        self._lugre_p:  Dict | None = None
        self._axis_label: str = 'Azimuth'

    # ────────────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────────────
    def plot_all(
        self,
        results: Dict[str, Dict],
        tustin_p: Dict | None = None,
        lugre_p: Dict | None = None,
        axis_label: str = 'Azimuth',
        ndob_cfg: Dict | None = None,
        ndob_log_path: Path | None = None,
        ndob_history_dir: Path | None = None,
    ) -> Dict[str, plt.Figure]:
        """Generate every figure, save to disk, and (optionally) display.

        Parameters
        ----------
        results : dict
            Per-friction-model results returned by ``DigitalTwinRunner``.
        tustin_p, lugre_p : dict, optional
            Tustin and LuGre parameter dictionaries in the format expected by
            ``friction_study.build_figure`` (scalar keys: tau_c, tau_s, v_s,
            b/sigma_2, alpha, v_epsilon, sigma_0, sigma_1).  When both are
            provided, Figure 10 (Stribeck steady-state comparison) is also
            generated.
        axis_label : str
            Axis identifier used in the Stribeck figure suptitle.
        """
        for fm in FRICTION_ORDER:
            if fm not in results:
                raise KeyError(f"Missing results for friction model '{fm}'")

        self._results = results
        self._times = {fm: np.asarray(results[fm]['log_arrays']['time'])
                       for fm in FRICTION_ORDER}
        self._tustin_p = tustin_p
        self._lugre_p  = lugre_p
        self._axis_label = axis_label

        self.figures['fig1_tracking_error']      = self._plot_tracking_error()
        self.figures['fig2_position_tracking']   = self._plot_position_tracking()
        self.figures['fig3_fsm_performance']     = self._plot_fsm_performance()
        self.figures['fig4_internal_signals']    = self._plot_internal_signals()
        self.figures['fig5_stroke_consumption']  = self._plot_stroke_consumption()
        self.figures['fig6_stroke_margin']       = self._plot_stroke_margin_summary()
        self.figures['fig7_benchmark_table']     = self._plot_benchmark_table()
        self.figures['fig8_friction_comp']       = self._plot_friction_compensation()
        self.figures['fig9_disturbance_torques'] = self._plot_disturbance_torques()
        if tustin_p is not None and lugre_p is not None:
            self.figures['fig10_stribeck_comparison'] = self._plot_stribeck_comparison()

        if self.save_figures:
            self._save_all_figures()

        # ── NDOB sweep persistence + history bar charts ─────────────────
        # When called with an ndob_cfg + log path, append this run's
        # benchmark metrics to the CSV and (re)build one bar chart per
        # Fig 7 row from the full accumulated history. The history figures
        # are saved to their own directory and remain open so the
        # subsequent ``plt.show()`` displays them alongside Fig 1–10.
        self._ndob_history_figs: Dict[str, plt.Figure] = {}
        if ndob_cfg is not None and ndob_log_path is not None:
            try:
                metrics_for_log = self._collect_fig7_metrics()
                exp_idx = append_ndob_experiment_log(
                    ndob_log_path, ndob_cfg, metrics_for_log,
                )
                print(
                    f"\n[NDOB sweep] Logged experiment #{exp_idx + 1} to "
                    f"{Path(ndob_log_path).resolve()}"
                )
                print("[NDOB sweep] Building history bar charts ...")
                self._ndob_history_figs = build_ndob_sweep_history_figures(
                    ndob_log_path,
                    output_dir=ndob_history_dir,
                    save=self.save_figures,
                )
                print("[NDOB sweep] Building SCR summary table ...")
                self._scr_table_fig = build_scr_tables_pdf(
                    ndob_log_path,
                )
            except Exception as exc:
                print(f"[NDOB sweep] Skipped — {type(exc).__name__}: {exc}")

        if self.show_figures:
            print("\nDisplaying comparative figures (close the windows to exit) ...")
            plt.show()

        return self.figures

    # ────────────────────────────────────────────────────────────────────
    # Helper utilities
    # ────────────────────────────────────────────────────────────────────
    def _friction_iter(self):
        """Yield (friction_name, time, log_arrays, color, label) for each run."""
        for fm in FRICTION_ORDER:
            log = self._results[fm]['log_arrays']
            yield (
                fm,
                self._times[fm],
                log,
                FRICTION_COLOR_MAP[fm],
                FRICTION_LABELS[fm],
            )

    @staticmethod
    def _get_array(log: Dict, key: str, t: np.ndarray) -> np.ndarray:
        arr = log.get(key, None)
        if arr is None:
            return np.zeros_like(t)
        return np.asarray(arr)

    @staticmethod
    def _get_residual(log: Dict, key: str) -> np.ndarray:
        """Post-FSM residual error with safe fallback to los_error - 2·fsm."""
        arr = log.get(key, None)
        if arr is not None:
            return np.asarray(arr)
        axis = 'x' if 'x' in key else 'y'
        los = np.asarray(log.get(f'los_error_{axis}', np.array([0.0])))
        fsm_key = 'fsm_tip' if axis == 'x' else 'fsm_tilt'
        fsm = np.asarray(log.get(fsm_key, np.array([0.0])))
        n = min(len(los), len(fsm))
        return los[:n] - 2.0 * fsm[:n]

    def _compute_stroke_metrics(self, theta_max: float) -> Dict[str, StrokeMetricsResult]:
        calculator = StrokeMetrics(
            theta_max=theta_max,
            jitter_cutoff_hz=50.0,
            filter_order=4,
        )
        out: Dict[str, StrokeMetricsResult] = {}
        for fm in FRICTION_ORDER:
            log = self._results[fm]['log_arrays']
            t = np.asarray(log['time'])
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 1e-4
            link_active = log.get('is_beam_on_sensor', None)
            if link_active is not None:
                link_active = np.asarray(link_active, dtype=bool)
            out[fm] = calculator.compute(
                time=t,
                fsm_tip=np.asarray(log['fsm_tip']),
                fsm_tilt=np.asarray(log['fsm_tilt']),
                dt=dt,
                link_active=link_active,
            )
        return out

    def _get_stroke_limit(self) -> float:
        # Use the NDOB-LuGre run as reference (any run works — limit is static).
        log = self._results['lugre']['log_arrays']
        arr = log.get('fsm_stroke_limit_rad', None)
        if arr is not None and len(arr) > 0:
            return float(arr[0])
        return 0.010

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 1 — Tracking error (log scale, 2×1)
    # ────────────────────────────────────────────────────────────────────
    def _plot_tracking_error(self) -> plt.Figure:
        fig, (ax_az, ax_el) = plt.subplots(
            2, 1, figsize=self.FIG_SIZE_2x1, sharex=True,
        )

        max_err_az = 0.0
        max_err_el = 0.0

        for fm, t, log, color, label in self._friction_iter():
            q_az = self._get_array(log, 'q_az', t)
            q_el = self._get_array(log, 'q_el', t)
            tgt_az = self._get_array(log, 'target_az', t)
            tgt_el = self._get_array(log, 'target_el', t)

            err_az = np.rad2deg(np.abs(q_az - tgt_az))
            err_el = np.rad2deg(np.abs(q_el - tgt_el))
            # Log scale — replace zeros with a tiny floor
            err_az = np.maximum(err_az, 1e-6)
            err_el = np.maximum(err_el, 1e-6)

            ax_az.plot(t, err_az, color=color, linewidth=self.LINEWIDTH,
                       alpha=self.ALPHA_PRIMARY, label=label)
            ax_el.plot(t, err_el, color=color, linewidth=self.LINEWIDTH,
                       alpha=self.ALPHA_PRIMARY, label=label)

            max_err_az = max(max_err_az, float(np.max(err_az)))
            max_err_el = max(max_err_el, float(np.max(err_el)))

        # FSM stroke limit shading (use LuGre run's limit array)
        limit_rad = self._get_array(
            self._results['lugre']['log_arrays'],
            'fsm_stroke_limit_rad',
            self._times['lugre'],
        )
        if np.allclose(limit_rad, 0.0):
            limit_rad = np.full_like(self._times['lugre'], self._get_stroke_limit())
        limit_deg = np.rad2deg(limit_rad)
        t_ref = self._times['lugre']

        for ax, max_err in ((ax_az, max_err_az), (ax_el, max_err_el)):
            ax.plot(t_ref, limit_deg, color=FrictionColors.LIMIT,
                    linewidth=self.LINEWIDTH * 1.2, linestyle='--',
                    label='FSM Stroke / QPD Limit')
            top = max(float(limit_deg[0]) * 5.0, max_err * 2.0, 1e-2)
            ax.fill_between(t_ref, 1e-6, limit_deg,
                            color='lightgreen', alpha=0.12,
                            label='LOS Acquired (QPD Active)')
            ax.fill_between(t_ref, limit_deg, top,
                            color='lightcoral', alpha=0.12,
                            label='Out of QPD Range')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4, top=top)
            ax.grid(True, which='both', alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FONTSIZE, ncol=2)

        ax_az.set_ylabel('Azimuth Error [deg]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_el.set_ylabel('Elevation Error [deg]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_el.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')

        fig.suptitle(
            'Tracking Error — FBL+NDOB under Viscous / Tustin / LuGre Friction',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 2 — Position tracking (2×1)
    # ────────────────────────────────────────────────────────────────────
    def _plot_position_tracking(self) -> plt.Figure:
        fig, (ax_az, ax_el) = plt.subplots(
            2, 1, figsize=self.FIG_SIZE_2x1, sharex=True,
        )

        command_plotted = False
        for fm, t, log, color, label in self._friction_iter():
            q_az = np.rad2deg(self._get_array(log, 'q_az', t))
            q_el = np.rad2deg(self._get_array(log, 'q_el', t))
            ax_az.plot(t, q_az, color=color, linewidth=self.LINEWIDTH,
                       alpha=self.ALPHA_PRIMARY, label=label)
            ax_el.plot(t, q_el, color=color, linewidth=self.LINEWIDTH,
                       alpha=self.ALPHA_PRIMARY, label=label)

            if not command_plotted:
                cmd_az = np.rad2deg(self._get_array(log, 'target_az', t))
                cmd_el = np.rad2deg(self._get_array(log, 'target_el', t))
                ax_az.plot(t, cmd_az, color=FrictionColors.TARGET,
                           linewidth=self.LINEWIDTH * 0.9, linestyle='--',
                           alpha=0.7, label='Command')
                ax_el.plot(t, cmd_el, color=FrictionColors.TARGET,
                           linewidth=self.LINEWIDTH * 0.9, linestyle='--',
                           alpha=0.7, label='Command')
                command_plotted = True

        ax_az.set_ylabel('Azimuth Angle [deg]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_el.set_ylabel('Elevation Angle [deg]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_el.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        for ax in (ax_az, ax_el):
            ax.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax.legend(loc='best', fontsize=self.LEGEND_FONTSIZE, ncol=2)

        fig.suptitle(
            'Position Tracking — FBL+NDOB under Viscous / Tustin / LuGre Friction',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 3 — FSM performance (3×1)
    # ────────────────────────────────────────────────────────────────────
    def _plot_fsm_performance(self) -> plt.Figure:
        fig, (ax_tip, ax_tilt, ax_res) = plt.subplots(
            3, 1, figsize=self.FIG_SIZE_3x1, sharex=True,
        )
        to_urad = 1e6

        rms_summary: Dict[str, Tuple[float, float]] = {}

        for fm, t, log, color, label in self._friction_iter():
            fsm_tip  = self._get_array(log, 'fsm_tip',  t) * to_urad
            fsm_tilt = self._get_array(log, 'fsm_tilt', t) * to_urad

            ax_tip.plot(t, fsm_tip,   color=color, linewidth=self.LINEWIDTH,
                        alpha=self.ALPHA_PRIMARY, label=label)
            ax_tilt.plot(t, fsm_tilt, color=color, linewidth=self.LINEWIDTH,
                         alpha=self.ALPHA_PRIMARY, label=label)

            res_x = self._get_residual(log, 'fsm_residual_error_x') * to_urad
            res_y = self._get_residual(log, 'fsm_residual_error_y') * to_urad
            # Time/residual length mismatch guard
            n = min(len(t), len(res_x), len(res_y))
            ax_res.plot(t[:n], res_x[:n], color=color, linewidth=self.LINEWIDTH,
                        alpha=self.ALPHA_PRIMARY, label=f'{label} — X')
            ax_res.plot(t[:n], res_y[:n], color=color, linewidth=self.LINEWIDTH,
                        alpha=self.ALPHA_PRIMARY, linestyle='--',
                        label=f'{label} — Y')

            if n > 10:
                steady_start = t[n - 1] * 0.5
                mask = t[:n] >= steady_start
                if mask.sum() > 0:
                    rms_x = float(np.sqrt(np.mean(res_x[:n][mask] ** 2)))
                    rms_y = float(np.sqrt(np.mean(res_y[:n][mask] ** 2)))
                    rms_summary[fm] = (rms_x, rms_y)

        # FSM stroke limit overlay
        limit_rad = self._get_array(
            self._results['lugre']['log_arrays'],
            'fsm_stroke_limit_rad',
            self._times['lugre'],
        )
        if np.allclose(limit_rad, 0.0):
            limit_rad = np.full_like(self._times['lugre'], self._get_stroke_limit())
        limit_urad = limit_rad * to_urad
        t_ref = self._times['lugre']
        for ax in (ax_tip, ax_tilt):
            ax.plot(t_ref,  limit_urad, color=FrictionColors.LIMIT,
                    linewidth=self.LINEWIDTH, linestyle='--', alpha=0.6,
                    label='Stroke Limit')
            ax.plot(t_ref, -limit_urad, color=FrictionColors.LIMIT,
                    linewidth=self.LINEWIDTH, linestyle='--', alpha=0.6)

        ax_tip.set_ylabel('FSM Tip [µrad]',  fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_tilt.set_ylabel('FSM Tilt [µrad]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_res.set_ylabel('Residual LOS [µrad]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_res.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')

        ax_res.axhspan(-100.0, 100.0, alpha=0.12, color='green',
                       label='±100 µrad Target')
        ax_res.axhline(0.0, color='black', linewidth=self.LINEWIDTH * 0.6,
                       linestyle='--', alpha=0.5)

        for ax in (ax_tip, ax_tilt, ax_res):
            ax.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FONTSIZE - 1, ncol=3)

        if rms_summary:
            parts = []
            for fm in FRICTION_ORDER:
                if fm in rms_summary:
                    rx, ry = rms_summary[fm]
                    parts.append(f'{fm.capitalize()} X/Y: {rx:.2f}/{ry:.2f}')
            ax_res.set_title(
                'Steady-state post-FSM RMS [µrad] — ' + '  |  '.join(parts),
                fontsize=self.TITLE_FONTSIZE - 1, fontweight='bold',
            )

        fig.suptitle(
            'FSM Performance — FBL+NDOB under Viscous / Tustin / LuGre Friction',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 4 — Internal signals (virtual ctrl, commanded torque, d_hat)
    # ────────────────────────────────────────────────────────────────────
    def _plot_internal_signals(self) -> plt.Figure:
        fig, (ax_v, ax_tau, ax_dhat) = plt.subplots(
            3, 1, figsize=self.FIG_SIZE_3x1, sharex=True,
        )

        for fm, t, log, color, label in self._friction_iter():
            v_az = self._get_array(log, 'v_virtual_az', t)
            v_el = self._get_array(log, 'v_virtual_el', t)
            ax_v.plot(t, v_az, color=color, linewidth=self.LINEWIDTH,
                      alpha=self.ALPHA_PRIMARY, label=f'{label} — Az')
            ax_v.plot(t, v_el, color=color, linewidth=self.LINEWIDTH,
                      alpha=self.ALPHA_PRIMARY, linestyle='--',
                      label=f'{label} — El')

            tau_az = self._get_array(log, 'tau_unsaturated_az', t)
            if np.allclose(tau_az, 0.0):
                tau_az = self._get_array(log, 'torque_az', t)
            tau_el = self._get_array(log, 'tau_unsaturated_el', t)
            if np.allclose(tau_el, 0.0):
                tau_el = self._get_array(log, 'torque_el', t)
            ax_tau.plot(t, tau_az, color=color, linewidth=self.LINEWIDTH,
                        alpha=self.ALPHA_PRIMARY, label=f'{label} — Az')
            ax_tau.plot(t, tau_el, color=color, linewidth=self.LINEWIDTH,
                        alpha=self.ALPHA_PRIMARY, linestyle='--',
                        label=f'{label} — El')

            d_hat_az = self._get_array(log, 'd_hat_ndob_az', t)
            d_hat_el = self._get_array(log, 'd_hat_ndob_el', t)
            ax_dhat.plot(t, d_hat_az, color=color, linewidth=self.LINEWIDTH,
                         alpha=self.ALPHA_PRIMARY, label=f'{label} — Az')
            ax_dhat.plot(t, d_hat_el, color=color, linewidth=self.LINEWIDTH,
                         alpha=self.ALPHA_PRIMARY, linestyle='--',
                         label=f'{label} — El')

        ax_tau.axhline( 1.0, color=FrictionColors.LIMIT, linewidth=self.LINEWIDTH,
                       linestyle=':', alpha=0.6, label='Saturation')
        ax_tau.axhline(-1.0, color=FrictionColors.LIMIT, linewidth=self.LINEWIDTH,
                       linestyle=':', alpha=0.6)

        ax_v.set_ylabel(r'Virtual Control $v$ [rad/s$^{2}$]',
                        fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_tau.set_ylabel(r'Commanded Torque $\tau$ [N$\cdot$m]',
                          fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_dhat.set_ylabel(r'NDOB Estimate $\hat{d}$ [N$\cdot$m]',
                           fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_dhat.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')

        for ax in (ax_v, ax_tau, ax_dhat):
            ax.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax.axhline(0.0, color='black', linewidth=self.LINEWIDTH * 0.5,
                       linestyle='--', alpha=0.4)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FONTSIZE - 2, ncol=3)

        fig.suptitle(
            r'Internal Control Signals ($v$, $\tau$, $\hat{d}_{\mathrm{NDOB}}$) — '
            r'Viscous / Tustin / LuGre',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 5 — Instantaneous stroke consumption
    # ────────────────────────────────────────────────────────────────────
    def _plot_stroke_consumption(self) -> plt.Figure:
        theta_max = self._get_stroke_limit()
        try:
            metrics = self._compute_stroke_metrics(theta_max)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=self.FIG_SIZE_2x1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'Stroke metrics unavailable:\n{exc}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, (ax_tip, ax_tilt) = plt.subplots(
            2, 1, figsize=self.FIG_SIZE_2x1, sharex=True,
        )

        max_tip = 110.0
        max_tilt = 110.0
        for fm in FRICTION_ORDER:
            m = metrics[fm]
            color = FRICTION_COLOR_MAP[fm]
            label = FRICTION_LABELS[fm]
            ax_tip.plot(m.time, m.scr_timeseries_tip, color=color,
                        linewidth=self.LINEWIDTH, alpha=self.ALPHA_PRIMARY,
                        label=label)
            ax_tilt.plot(m.time, m.scr_timeseries_tilt, color=color,
                         linewidth=self.LINEWIDTH, alpha=self.ALPHA_PRIMARY,
                         label=label)
            max_tip  = max(max_tip,  float(np.max(m.scr_timeseries_tip)))
            max_tilt = max(max_tilt, float(np.max(m.scr_timeseries_tilt)))

        for ax, ymax in ((ax_tip, max_tip), (ax_tilt, max_tilt)):
            ax.axhline(100.0, color=FrictionColors.LIMIT, linewidth=1.5,
                       linestyle=':', label='Saturation Limit')
            ax.axhspan(100.0, ymax * 1.05, alpha=0.07, color='red')
            ax.axhspan(0.0,    80.0,       alpha=0.05, color='green')
            ax.set_ylim(0.0, ymax * 1.05)
            ax.set_ylabel('Utilisation Ratio [%]',
                          fontsize=self.AXIS_FONTSIZE, fontweight='bold')
            ax.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FONTSIZE, ncol=2)

        ax_tip.set_title(
            r'(a) Azimuth Axis (Tip) — $|\theta_{\mathrm{FSM}}(t)|/\Theta_{\max}$',
            fontsize=self.TITLE_FONTSIZE, fontweight='bold',
        )
        ax_tilt.set_title(
            r'(b) Elevation Axis (Tilt) — $|\theta_{\mathrm{FSM}}(t)|/\Theta_{\max}$',
            fontsize=self.TITLE_FONTSIZE, fontweight='bold',
        )
        ax_tilt.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')

        fig.suptitle(
            'Instantaneous FSM Stroke Utilisation — FBL+NDOB friction study',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 6 — Stroke margin summary (bar chart)
    # ────────────────────────────────────────────────────────────────────
    def _plot_stroke_margin_summary(self) -> plt.Figure:
        theta_max = self._get_stroke_limit()
        try:
            metrics = self._compute_stroke_metrics(theta_max)
        except Exception as exc:
            fig, ax = plt.subplots(figsize=self.FIG_SIZE_1x1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'Stroke summary unavailable:\n{exc}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=self.FIG_SIZE_1x1)

        labels = [FRICTION_LABELS[fm] for fm in FRICTION_ORDER]
        scr_tip  = np.array([metrics[fm].scr_tip  for fm in FRICTION_ORDER])
        scr_tilt = np.array([metrics[fm].scr_tilt for fm in FRICTION_ORDER])
        colors   = [FRICTION_COLOR_MAP[fm] for fm in FRICTION_ORDER]

        x = np.arange(len(labels))
        bar_w = 0.35
        bars_tip  = ax.bar(x - bar_w / 2, scr_tip,  bar_w,
                           color=colors, alpha=0.9, edgecolor='black',
                           linewidth=1.2, label='Tip (Az)')
        bars_tilt = ax.bar(x + bar_w / 2, scr_tilt, bar_w,
                           color=colors, alpha=0.55, edgecolor='black',
                           linewidth=1.2, hatch='///', label='Tilt (El)')
        ax.axhline(100.0, color=FrictionColors.LIMIT, linewidth=2.0,
                   linestyle='--', label='Saturation Boundary (100%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax.set_ylabel('RMS Stroke Consumption Ratio [%]',
                      fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax.set_title('RMS Stroke Consumption Ratio (SCR) — friction-model comparison',
                     fontsize=self.TITLE_FONTSIZE, fontweight='bold')
        for bar in list(bars_tip) + list(bars_tilt):
            h = bar.get_height()
            ax.annotate(
                f'{h:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold',
                color='black' if h <= 100.0 else '#cc0000',
            )
        ax.grid(True, axis='y', alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
        ax.legend(loc='upper left', fontsize=self.LEGEND_FONTSIZE)

        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 7 — Benchmark table
    # ────────────────────────────────────────────────────────────────────
    def _collect_fig7_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute every metric tabulated in Fig 7, per friction model.

        Single source of truth for both ``_plot_benchmark_table`` and the
        NDOB sweep CSV logger. Raises if the stroke metrics cannot be
        computed (caller decides whether to fall back).
        """
        theta_max = self._get_stroke_limit()
        stroke = self._compute_stroke_metrics(theta_max)

        to_urad = 1e6
        out: Dict[str, Dict[str, float]] = {}
        for fm in FRICTION_ORDER:
            log = self._results[fm]['log_arrays']
            t = self._times[fm]
            res_x = self._get_residual(log, 'fsm_residual_error_x') * to_urad
            res_y = self._get_residual(log, 'fsm_residual_error_y') * to_urad
            n = min(len(t), len(res_x), len(res_y))
            if n > 10:
                steady = t[n - 1] * 0.5
                mask = t[:n] >= steady
                if mask.sum() > 0:
                    tip_rms  = float(np.sqrt(np.mean(res_x[:n][mask] ** 2)))
                    tilt_rms = float(np.sqrt(np.mean(res_y[:n][mask] ** 2)))
                else:
                    tip_rms, tilt_rms = 0.0, 0.0
            else:
                tip_rms, tilt_rms = 0.0, 0.0

            out[fm] = {
                'post_fsm_residual_tip_urad':  tip_rms,
                'post_fsm_residual_tilt_urad': tilt_rms,
                'los_rms_urad':                float(self._results[fm].get('los_error_rms', 0.0) * 1e6),
                'torque_rms_az_nm':            float(self._results[fm].get('torque_rms_az', 0.0)),
                'torque_rms_el_nm':            float(self._results[fm].get('torque_rms_el', 0.0)),
                'scr_rms_tip_pct':             float(stroke[fm].scr_tip),
                'scr_rms_tilt_pct':            float(stroke[fm].scr_tilt),
                's_bias_tip_mrad':             float(stroke[fm].s_bias_tip_mrad),
                's_bias_tilt_mrad':            float(stroke[fm].s_bias_tilt_mrad),
                'dsm_tip_mrad':                float(stroke[fm].dsm_tip_mrad),
                'dsm_tilt_mrad':               float(stroke[fm].dsm_tilt_mrad),
            }
        return out

    def _plot_benchmark_table(self) -> plt.Figure:
        try:
            metrics = self._collect_fig7_metrics()
        except Exception as exc:
            fig, ax = plt.subplots(figsize=self.FIG_SIZE_1x1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'Benchmark table unavailable:\n{exc}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=self.FIG_SIZE_1x1)
        ax.set_axis_off()

        col_labels = ['Metric',
                      FRICTION_LABELS['viscous'],
                      FRICTION_LABELS['tustin'],
                      FRICTION_LABELS['lugre']]

        table_rows = []
        for key, name, unit, fmt in FIG7_METRICS:
            row = [f'{name} [{unit}]']
            for fm in FRICTION_ORDER:
                row.append(format(metrics[fm][key], fmt))
            table_rows.append(row)

        col_widths = [0.30, 0.22, 0.22, 0.22]
        tbl = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            colWidths=col_widths,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.8)

        header_colors = {
            1: FrictionColors.VISCOUS,
            2: FrictionColors.TUSTIN,
            3: FrictionColors.LUGRE,
        }
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                if col == 0:
                    cell.set_facecolor('#2d3e50')
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor(header_colors.get(col, '#2d3e50'))
                    cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#f5f5f5')

        ax.set_title(
            'Benchmark Results — FBL+NDOB under Viscous / Tustin / LuGre Friction',
            fontsize=self.TITLE_FONTSIZE, fontweight='bold', pad=12,
        )
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 8 — Friction compensation vs true friction (twin-axis velocity)
    # ────────────────────────────────────────────────────────────────────
    def _plot_friction_compensation(self) -> plt.Figure:
        fig, (ax_az, ax_el) = plt.subplots(
            2, 1, figsize=self.FIG_SIZE_2x1, sharex=True,
        )
        vel_color = '#e67e22'

        ax_az_v = ax_az.twinx()
        ax_el_v = ax_el.twinx()

        ground_truth_plotted = {'az': False, 'el': False}

        for fm, t, log, color, label in self._friction_iter():
            fc_az  = self._get_array(log, 'friction_comp_az', t)
            fc_el  = self._get_array(log, 'friction_comp_el', t)
            tau_az = self._get_array(log, 'tau_friction_az',  t)
            tau_el = self._get_array(log, 'tau_friction_el',  t)
            qd_az  = self._get_array(log, 'qd_az',            t)
            qd_el  = self._get_array(log, 'qd_el',            t)

            # Plant-applied friction (ground truth) — plot once per axis with
            # a dashed line, using the *true* model that was active for that
            # run.  To keep the legend compact we plot the LuGre run's ground
            # truth only (LuGre is the richest reference).
            if fm == 'lugre':
                ax_az.plot(t, tau_az, color=FrictionColors.GROUND,
                           linewidth=self.LINEWIDTH, linestyle='--',
                           alpha=self.ALPHA_PRIMARY,
                           label=r'$\tau_{\mathrm{friction}}$ (LuGre Applied)')
                ax_el.plot(t, tau_el, color=FrictionColors.GROUND,
                           linewidth=self.LINEWIDTH, linestyle='--',
                           alpha=self.ALPHA_PRIMARY,
                           label=r'$\tau_{\mathrm{friction}}$ (LuGre Applied)')
                ground_truth_plotted['az'] = True
                ground_truth_plotted['el'] = True

            ax_az.plot(t, fc_az, color=color, linewidth=self.LINEWIDTH,
                       alpha=self.ALPHA_PRIMARY,
                       label=rf'$\hat{{\tau}}_{{\mathrm{{comp}}}}$ ({label})')
            ax_el.plot(t, fc_el, color=color, linewidth=self.LINEWIDTH,
                       alpha=self.ALPHA_PRIMARY,
                       label=rf'$\hat{{\tau}}_{{\mathrm{{comp}}}}$ ({label})')

            ax_az_v.plot(t, np.rad2deg(qd_az), color=color,
                         linewidth=self.LINEWIDTH * 0.7, linestyle=':',
                         alpha=0.55)
            ax_el_v.plot(t, np.rad2deg(qd_el), color=color,
                         linewidth=self.LINEWIDTH * 0.7, linestyle=':',
                         alpha=0.55)

        ax_az.set_ylabel(r'Azimuth Torque [N$\cdot$m]',
                         fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_el.set_ylabel(r'Elevation Torque [N$\cdot$m]',
                         fontsize=self.AXIS_FONTSIZE, fontweight='bold')
        ax_el.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')

        ax_az_v.set_ylabel(r'$\dot{q}_{\mathrm{az}}$ [deg/s]',
                           color=vel_color, fontsize=self.AXIS_FONTSIZE)
        ax_el_v.set_ylabel(r'$\dot{q}_{\mathrm{el}}$ [deg/s]',
                           color=vel_color, fontsize=self.AXIS_FONTSIZE)
        ax_az_v.tick_params(axis='y', labelcolor=vel_color)
        ax_el_v.tick_params(axis='y', labelcolor=vel_color)

        for ax in (ax_az, ax_el):
            ax.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax.axhline(0.0, color='black', linewidth=self.LINEWIDTH * 0.5,
                       linestyle='--', alpha=0.4)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FONTSIZE - 1, ncol=2)

        fig.suptitle(
            'Friction Compensation vs Plant Friction Torque — FBL+NDOB friction study',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 9 — Environmental disturbance torques (reference run)
    # ────────────────────────────────────────────────────────────────────
    def _plot_disturbance_torques(self) -> plt.Figure:
        """
        Plot the environmental disturbance torques injected into the plant.

        All three friction runs share the same disturbance seed and
        configuration, so the disturbance signals are identical across runs.
        We use the LuGre run as the reference and report total disturbance
        (Az, El) plus its wind / vibration decomposition in a 2×2 layout —
        mirroring ``ResearchComparisonPlotter._plot_disturbance_torques``.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

        lw = self.LINEWIDTH
        log = self._results['lugre']['log_arrays']
        has_disturbance = (
            'tau_disturbance_az' in log
            and np.std(log['tau_disturbance_az']) > 1e-10
        )

        if has_disturbance:
            t = np.asarray(log['time'])
            tau_d_az = np.asarray(log['tau_disturbance_az']) * 1000.0
            tau_d_el = np.asarray(log['tau_disturbance_el']) * 1000.0
            wind_az  = np.asarray(log.get('wind_torque_az',      np.zeros_like(t))) * 1000.0
            wind_el  = np.asarray(log.get('wind_torque_el',      np.zeros_like(t))) * 1000.0
            vib_az   = np.asarray(log.get('vibration_torque_az', np.zeros_like(t))) * 1000.0
            vib_el   = np.asarray(log.get('vibration_torque_el', np.zeros_like(t))) * 1000.0

            props = dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.9, edgecolor='lightgray')

            # (a) Total Disturbance — Azimuth
            ax1.plot(t, tau_d_az, color=DisturbancePlotColors.AZIMUTH,
                     linewidth=lw * 0.8, alpha=0.9)
            ax1.fill_between(t, tau_d_az, alpha=0.15,
                             color=DisturbancePlotColors.AZIMUTH)
            ax1.axhline(0.0, color='black', linewidth=lw * 1.2, alpha=0.8)
            mean_az = float(np.mean(tau_d_az))
            std_az  = float(np.std(tau_d_az))
            ax1.axhline(mean_az, color='gray', linewidth=lw, linestyle='--',
                        alpha=0.9, label=f'Mean: {mean_az:.2f} mN·m')
            ax1.axhline(mean_az + 3 * std_az, color='orange', linewidth=lw,
                        linestyle=':', alpha=0.8, label=r'$\pm 3\sigma$ Bound')
            ax1.axhline(mean_az - 3 * std_az, color='orange', linewidth=lw,
                        linestyle=':', alpha=0.8)
            ax1.set_ylabel('Torque [mN·m]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
            ax1.text(0.02, 0.95, 'Total Disturbance — Azimuth',
                     transform=ax1.transAxes, fontsize=self.LEGEND_FONTSIZE,
                     fontweight='bold', va='top', bbox=props)
            ax1.legend(loc='lower right', fontsize=self.LEGEND_FONTSIZE - 1, framealpha=0.95)
            ax1.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax1.set_xlim([t[0], t[-1]])

            # (b) Total Disturbance — Elevation
            ax2.plot(t, tau_d_el, color=DisturbancePlotColors.ELEVATION,
                     linewidth=lw * 0.8, alpha=0.9)
            ax2.fill_between(t, tau_d_el, alpha=0.15,
                             color=DisturbancePlotColors.ELEVATION)
            ax2.axhline(0.0, color='black', linewidth=lw * 1.2, alpha=0.8)
            mean_el = float(np.mean(tau_d_el))
            std_el  = float(np.std(tau_d_el))
            ax2.axhline(mean_el, color='gray', linewidth=lw, linestyle='--',
                        alpha=0.9, label=f'Mean: {mean_el:.2f} mN·m')
            ax2.axhline(mean_el + 3 * std_el, color='orange', linewidth=lw,
                        linestyle=':', alpha=0.8, label=r'$\pm 3\sigma$ Bound')
            ax2.axhline(mean_el - 3 * std_el, color='orange', linewidth=lw,
                        linestyle=':', alpha=0.8)
            ax2.text(0.02, 0.95, 'Total Disturbance — Elevation',
                     transform=ax2.transAxes, fontsize=self.LEGEND_FONTSIZE,
                     fontweight='bold', va='top', bbox=props)
            ax2.legend(loc='lower right', fontsize=self.LEGEND_FONTSIZE - 1, framealpha=0.95)
            ax2.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax2.set_xlim([t[0], t[-1]])

            # (c) Components — Azimuth
            ax3.plot(t, wind_az, color=DisturbancePlotColors.WIND,
                     linewidth=lw, label='Dryden Wind Gust', alpha=0.9)
            ax3.plot(t, vib_az, color=DisturbancePlotColors.VIBRATION,
                     linewidth=lw * 0.7, label='Structural Vibration', alpha=0.7)
            ax3.axhline(0.0, color='black', linewidth=lw * 1.2, alpha=0.8)
            ax3.set_ylabel('Torque [mN·m]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
            ax3.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
            ax3.text(0.02, 0.95, 'Disturbance Components — Azimuth',
                     transform=ax3.transAxes, fontsize=self.LEGEND_FONTSIZE,
                     fontweight='bold', va='top', bbox=props)
            ax3.legend(loc='lower right', fontsize=self.LEGEND_FONTSIZE - 1, framealpha=0.95)
            ax3.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax3.set_xlim([t[0], t[-1]])

            # (d) Components — Elevation
            ax4.plot(t, wind_el, color=DisturbancePlotColors.WIND,
                     linewidth=lw, label='Dryden Wind Gust', alpha=0.9)
            ax4.plot(t, vib_el, color=DisturbancePlotColors.VIBRATION,
                     linewidth=lw * 0.7, label='Structural Vibration', alpha=0.7)
            ax4.axhline(0.0, color='black', linewidth=lw * 1.2, alpha=0.8)
            ax4.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
            ax4.text(0.02, 0.95, 'Disturbance Components — Elevation',
                     transform=ax4.transAxes, fontsize=self.LEGEND_FONTSIZE,
                     fontweight='bold', va='top', bbox=props)
            ax4.legend(loc='lower right', fontsize=self.LEGEND_FONTSIZE - 1, framealpha=0.95)
            ax4.grid(True, alpha=self.GRID_ALPHA, linestyle=self.GRID_LS)
            ax4.set_xlim([t[0], t[-1]])
        else:
            for ax, label_text in zip(
                [ax1, ax2, ax3, ax4],
                ['Total Disturbance — Azimuth',
                 'Total Disturbance — Elevation',
                 'Wind & Vibration — Azimuth',
                 'Wind & Vibration — Elevation'],
            ):
                ax.text(0.02, 0.95, label_text, transform=ax.transAxes,
                        fontsize=self.LEGEND_FONTSIZE, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.9, edgecolor='lightgray'))
                ax.text(0.5, 0.5,
                        'Environmental Disturbances Disabled\n\n'
                        'Enable with:\n'
                        '  environmental_disturbance_enabled=True\n'
                        '  environmental_disturbance_config={...}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=self.AXIS_FONTSIZE, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
                if ax in (ax3, ax4):
                    ax.set_xlabel('Time [s]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')
                if ax in (ax1, ax3):
                    ax.set_ylabel('Torque [mN·m]', fontsize=self.AXIS_FONTSIZE, fontweight='bold')

        fig.suptitle(
            'Environmental Disturbance Torques — Reference (LuGre Run)',
            fontsize=self.SUPTITLE_FSIZE, fontweight='bold',
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    # ────────────────────────────────────────────────────────────────────
    # FIGURE 10 — Stribeck steady-state comparison (reused from friction_study)
    # ────────────────────────────────────────────────────────────────────
    def _plot_stribeck_comparison(self) -> plt.Figure:
        """
        Generate the 2×2 Stribeck / steady-state comparison figure that lives
        in ``friction_study._build_panels``.  This is a *theoretical* plot of
        Tustin vs LuGre vs viscous friction curves for the parameter sets
        actually applied to the plant during the three comparative runs.

        Panels:
            (a) Full bidirectional Stribeck curve (viscous / LuGre / Tustin)
            (b) Positive-velocity zoom with g(ω) and Tustin envelopes
            (c) Steady-state residual τ_Tustin − τ_LuGre,SS
            (d) Stribeck envelope shape comparison

        Requires that ``tustin_p`` and ``lugre_p`` were passed to ``plot_all``.
        """
        if self._tustin_p is None or self._lugre_p is None:
            fig, ax = plt.subplots(figsize=self.FIG_SIZE_1x1)
            ax.set_axis_off()
            ax.text(0.5, 0.5,
                    'Stribeck comparison unavailable\n'
                    '(tustin_p / lugre_p not supplied to plot_all).',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=self.AXIS_FONTSIZE)
            return fig

        # Validate matching conditions and compute residual metrics.
        issues = _fs_validate_params(self._tustin_p, self._lugre_p)
        v_eval     = np.linspace(-_FS_V_MAX, _FS_V_MAX, _FS_N_SWEEP)
        tau_tustin = _fs_compute_tustin_curve(v_eval, self._tustin_p)
        tau_lg_ss  = _fs_compute_lugre_ss(v_eval, self._lugre_p)
        metrics    = _fs_compute_matching_metrics(v_eval, tau_tustin, tau_lg_ss)

        # Delegate to friction_study's publication-grade builder.
        fig = _fs_build_stribeck_figure(
            self._tustin_p, self._lugre_p, issues, metrics,
        )
        match_ok = (len(issues) == 0)
        ss_status = ('Steady-state match: confirmed' if match_ok
                     else 'Steady-state match: FAILED — see parameter table')
        fig.suptitle(
            f'Friction Model Stribeck Comparison — {self._axis_label}  |  {ss_status}',
            fontsize=self.SUPTITLE_FSIZE - 2, fontweight='bold',
            color=('#006400' if match_ok else '#8B0000'),
            y=0.985,
        )
        return fig

    # ────────────────────────────────────────────────────────────────────
    # Save all generated figures as publication-quality PDFs
    # ────────────────────────────────────────────────────────────────────
    def _save_all_figures(self) -> None:
        print(f"\nSaving {len(self.figures)} figures to {self.output_dir.absolute()} ...")
        for name, fig in self.figures.items():
            path = self.output_dir / f'{name}.pdf'
            try:
                fig.savefig(
                    str(path),
                    format='pdf',
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    metadata={
                        'Title':   name.replace('_', ' ').title(),
                        'Author':  'MicroPrecisionGimbal — Friction Study',
                        'Subject': 'FBL+NDOB friction-model comparative study',
                        'Creator': 'friction_controller_comparison.py',
                    },
                )
                if hasattr(fig.canvas, 'flush_events'):
                    fig.canvas.flush_events()
                time.sleep(0.01)
                print(f"  [OK] {path.name}")
            except Exception as exc:
                print(f"  [ERROR] Failed to save {path.name}: {exc}")
        print(f"[OK] Saved {len(self.figures)} PDF figures "
              f"(300 DPI, vector, journal-ready).")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Environmental disturbance configuration — identical for every run
    # ------------------------------------------------------------------
    example_disturbance_config = {
        'wind': {
            'enabled': False,
            'scale_length': 200.0,
            'turbulence_intensity': 0.25,
            'mean_velocity': 8.0,
            'direction_deg': 45.0,
            'start_time': 2.5,
        },
        'vibration': {
            'enabled': True,
            'modal_frequencies': [15.0, 45.0, 80.0],
            'modal_dampings':    [0.02, 0.015, 0.01],
            'modal_amplitudes':  [2e-3, 7e-4, 4e-4],
            'inertia_coupling':  0.1,
            'start_time': 0.0,
        },
        'structural_noise': {
            'enabled': True,
            'std':      0.01,
            'freq_low': 100.0,
            'freq_high': 500.0,
        },
        'seed': 42,
    }

    # Signal type alternatives: 'constant', 'square', 'sine', 'cosine', 'hybridsig'
    run_friction_comparison(
        signal_type='hybridsig',
        disturbance_config=example_disturbance_config,
    )
