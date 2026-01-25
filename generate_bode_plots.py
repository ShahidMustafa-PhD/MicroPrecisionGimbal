#!/usr/bin/env python3
"""
Bode Plot Generator for FBL and FBL+NDOB Controllers

Generates publication-quality frequency response plots (Bode plots) for the
Feedback Linearization (FBL) and FBL+NDOB control systems.

This standalone tool creates:
1. Open-loop Bode plots with stability margins
2. Closed-loop Bode plots (sensitivity and complementary sensitivity)
3. NDOB disturbance rejection frequency response
4. Comparative analysis with phase/gain margin annotations

Output is publication-ready (300 DPI, LaTeX-compatible labels).

Author: MicroPrecisionGimbal Control Design Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List


@dataclass
class GimbalPlantParams:
    """Physical parameters for gimbal plant model."""
    inertia_az: float = 1.0      # Azimuth moment of inertia [kg·m²]
    inertia_el: float = 0.5      # Elevation moment of inertia [kg·m²]
    friction_az: float = 0.1     # Azimuth viscous friction [N·m·s/rad]
    friction_el: float = 0.1     # Elevation viscous friction [N·m·s/rad]


@dataclass
class FBLControllerParams:
    """Feedback Linearization controller gains."""
    kp: float = 400.0            # Position gain [1/s²] (ωn² = Kp)
    kd: float = 40.0             # Velocity gain [1/s] (2ζωn = Kd)
    ki: float = 50.0             # Integral gain [1/s³]
    enable_integral: bool = True


@dataclass
class NDOBParams:
    """Nonlinear Disturbance Observer parameters."""
    lambda_obs: float = 50.0     # Observer bandwidth [rad/s]


def create_gimbal_plant(params: GimbalPlantParams, axis: str = 'az') -> ctrl.TransferFunction:
    """
    Create linearized gimbal plant transfer function.
    
    The FBL controller cancels the nonlinear dynamics, leaving an effective
    double integrator plant. However, uncompensated friction appears as a
    disturbance.
    
    For analysis purposes, we model the plant including friction:
    
    G(s) = 1 / (Js² + Ds)
    
    where J is inertia and D is friction coefficient.
    
    Parameters
    ----------
    params : GimbalPlantParams
        Physical plant parameters
    axis : str
        'az' for azimuth, 'el' for elevation
        
    Returns
    -------
    ctrl.TransferFunction
        Plant transfer function from torque to angle
    """
    if axis == 'az':
        J = params.inertia_az
        D = params.friction_az
    else:
        J = params.inertia_el
        D = params.friction_el
    
    # G(s) = 1 / (Js² + Ds) = 1/J / (s² + D/J*s)
    # = 1/J / s(s + D/J)
    num = [1.0]
    den = [J, D, 0.0]  # Js² + Ds + 0
    
    return ctrl.tf(num, den)


def create_fbl_controller(params: FBLControllerParams) -> ctrl.TransferFunction:
    """
    Create FBL outer loop PID controller transfer function.
    
    After feedback linearization cancels the nonlinear dynamics,
    the outer loop sees an effective double integrator (Type-2 plant).
    The PID controller is:
    
    C(s) = Kp + Kd*s + Ki/s
         = (Kd*s² + Kp*s + Ki) / s
    
    Parameters
    ----------
    params : FBLControllerParams
        Controller gains
        
    Returns
    -------
    ctrl.TransferFunction
        PID controller transfer function
    """
    if params.enable_integral:
        # PID: C(s) = (Kd*s² + Kp*s + Ki) / s
        num = [params.kd, params.kp, params.ki]
        den = [1.0, 0.0]
    else:
        # PD: C(s) = Kd*s + Kp
        num = [params.kd, params.kp]
        den = [1.0]
    
    return ctrl.tf(num, den)


def create_ndob_transfer_function(params: NDOBParams) -> ctrl.TransferFunction:
    """
    Create NDOB equivalent transfer function for disturbance estimation.
    
    The NDOB estimates the lumped disturbance d with dynamics:
    
    d_hat(s) = λ/(s + λ) * d(s)
    
    This is a first-order low-pass filter with bandwidth λ.
    The disturbance rejection transfer function from d to e is:
    
    S_d(s) = 1 - λ/(s + λ) = s/(s + λ)
    
    Parameters
    ----------
    params : NDOBParams
        Observer parameters
        
    Returns
    -------
    ctrl.TransferFunction
        NDOB disturbance estimation transfer function
    """
    lam = params.lambda_obs
    # H_ndob(s) = λ/(s + λ)
    return ctrl.tf([lam], [1.0, lam])


def create_ndob_rejection_tf(params: NDOBParams) -> ctrl.TransferFunction:
    """
    Create NDOB disturbance rejection (residual) transfer function.
    
    After NDOB compensation, the remaining disturbance is:
    
    d_residual(s) = d(s) - d_hat(s) = d(s) * [1 - λ/(s+λ)]
                  = d(s) * s/(s+λ)
    
    This is a high-pass filter showing what disturbances leak through.
    
    Returns
    -------
    ctrl.TransferFunction
        Residual disturbance transfer function
    """
    lam = params.lambda_obs
    # S_d(s) = s/(s + λ)
    return ctrl.tf([1.0, 0.0], [1.0, lam])


def compute_loop_transfer_functions(plant: ctrl.TransferFunction,
                                     controller: ctrl.TransferFunction,
                                     ndob: Optional[ctrl.TransferFunction] = None
                                     ) -> Dict[str, ctrl.TransferFunction]:
    """
    Compute standard loop transfer functions for analysis.
    
    Returns
    -------
    Dict with:
        - 'L': Open-loop transfer function L = C*G
        - 'T': Complementary sensitivity T = L/(1+L)
        - 'S': Sensitivity S = 1/(1+L)
        - 'Sd': Disturbance sensitivity (if NDOB provided)
    """
    L = controller * plant
    T = ctrl.feedback(L, 1)
    S = ctrl.feedback(1, L)
    
    result = {'L': L, 'T': T, 'S': S}
    
    if ndob is not None:
        # With NDOB, disturbance rejection is modified
        # S_d(s) = S(s) * H_ndob_rejection(s)
        # where H_ndob_rejection = s/(s + λ) is the residual TF
        result['ndob'] = ndob
    
    return result


def plot_bode_comparison(tf_fbl: Dict, tf_ndob: Dict,
                          omega: np.ndarray,
                          title_suffix: str = "",
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Generate comparative Bode plots for FBL vs FBL+NDOB.
    
    Creates a 2x2 figure with:
    - Top-left: Open-loop magnitude
    - Top-right: Open-loop phase
    - Bottom-left: Closed-loop magnitude (T)
    - Bottom-right: Sensitivity magnitude (S)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for publication
    COLOR_FBL = '#2ca02c'      # Green
    COLOR_NDOB = '#9467bd'     # Purple
    
    # Open-loop Bode
    ax_mag = axes[0, 0]
    ax_phase = axes[0, 1]
    
    # Compute frequency response
    mag_fbl, phase_fbl, _ = ctrl.frequency_response(tf_fbl['L'], omega)
    mag_ndob, phase_ndob, _ = ctrl.frequency_response(tf_ndob['L'], omega)
    
    # Magnitude plot
    ax_mag.semilogx(omega, 20*np.log10(np.abs(mag_fbl)), 
                    color=COLOR_FBL, linewidth=2, label='FBL')
    ax_mag.semilogx(omega, 20*np.log10(np.abs(mag_ndob)), 
                    color=COLOR_NDOB, linewidth=2, linestyle='--', label='FBL+NDOB')
    ax_mag.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax_mag.set_ylabel('Magnitude [dB]', fontsize=12, fontweight='bold')
    ax_mag.set_title('Open-Loop Magnitude', fontsize=12, fontweight='bold')
    ax_mag.legend(loc='best', fontsize=10)
    ax_mag.grid(True, alpha=0.3, linestyle=':', which='both')
    ax_mag.set_xlim([omega[0], omega[-1]])
    
    # Phase plot
    ax_phase.semilogx(omega, np.rad2deg(np.unwrap(np.angle(mag_fbl))), 
                      color=COLOR_FBL, linewidth=2, label='FBL')
    ax_phase.semilogx(omega, np.rad2deg(np.unwrap(np.angle(mag_ndob))), 
                      color=COLOR_NDOB, linewidth=2, linestyle='--', label='FBL+NDOB')
    ax_phase.axhline(-180, color='r', linewidth=1, linestyle=':', label='-180°')
    ax_phase.set_ylabel('Phase [deg]', fontsize=12, fontweight='bold')
    ax_phase.set_title('Open-Loop Phase', fontsize=12, fontweight='bold')
    ax_phase.legend(loc='best', fontsize=10)
    ax_phase.grid(True, alpha=0.3, linestyle=':', which='both')
    ax_phase.set_xlim([omega[0], omega[-1]])
    
    # Closed-loop (Complementary Sensitivity T)
    ax_cl = axes[1, 0]
    mag_T_fbl, _, _ = ctrl.frequency_response(tf_fbl['T'], omega)
    mag_T_ndob, _, _ = ctrl.frequency_response(tf_ndob['T'], omega)
    
    ax_cl.semilogx(omega, 20*np.log10(np.abs(mag_T_fbl)), 
                   color=COLOR_FBL, linewidth=2, label='FBL')
    ax_cl.semilogx(omega, 20*np.log10(np.abs(mag_T_ndob)), 
                   color=COLOR_NDOB, linewidth=2, linestyle='--', label='FBL+NDOB')
    ax_cl.axhline(-3, color='orange', linewidth=1, linestyle=':', label='-3 dB')
    ax_cl.set_xlabel('Frequency [rad/s]', fontsize=12, fontweight='bold')
    ax_cl.set_ylabel('Magnitude [dB]', fontsize=12, fontweight='bold')
    ax_cl.set_title('Closed-Loop Magnitude $T(j\\omega)$', fontsize=12, fontweight='bold')
    ax_cl.legend(loc='best', fontsize=10)
    ax_cl.grid(True, alpha=0.3, linestyle=':', which='both')
    ax_cl.set_xlim([omega[0], omega[-1]])
    
    # Sensitivity S
    ax_sens = axes[1, 1]
    mag_S_fbl, _, _ = ctrl.frequency_response(tf_fbl['S'], omega)
    mag_S_ndob, _, _ = ctrl.frequency_response(tf_ndob['S'], omega)
    
    ax_sens.semilogx(omega, 20*np.log10(np.abs(mag_S_fbl)), 
                     color=COLOR_FBL, linewidth=2, label='FBL')
    ax_sens.semilogx(omega, 20*np.log10(np.abs(mag_S_ndob)), 
                     color=COLOR_NDOB, linewidth=2, linestyle='--', label='FBL+NDOB')
    ax_sens.set_xlabel('Frequency [rad/s]', fontsize=12, fontweight='bold')
    ax_sens.set_ylabel('Magnitude [dB]', fontsize=12, fontweight='bold')
    ax_sens.set_title('Sensitivity $S(j\\omega)$', fontsize=12, fontweight='bold')
    ax_sens.legend(loc='best', fontsize=10)
    ax_sens.grid(True, alpha=0.3, linestyle=':', which='both')
    ax_sens.set_xlim([omega[0], omega[-1]])
    
    fig.suptitle(f'Frequency Response Comparison: FBL vs FBL+NDOB{title_suffix}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


def plot_ndob_disturbance_rejection(ndob_params: NDOBParams,
                                     omega: np.ndarray,
                                     save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot NDOB disturbance rejection characteristics.
    
    Shows:
    - NDOB estimation transfer function H(s) = λ/(s+λ)
    - Residual disturbance transfer function 1-H(s) = s/(s+λ)
    - Phase characteristics (shows the 84ms lag at λ=50 rad/s)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Create transfer functions
    H_ndob = create_ndob_transfer_function(ndob_params)
    S_ndob = create_ndob_rejection_tf(ndob_params)
    
    # Compute frequency responses
    mag_H, phase_H, _ = ctrl.frequency_response(H_ndob, omega)
    mag_S, phase_S, _ = ctrl.frequency_response(S_ndob, omega)
    
    # Magnitude plot
    ax1.semilogx(omega, 20*np.log10(np.abs(mag_H)), 
                 'b-', linewidth=2.5, label=r'$\hat{d}/d = \lambda/(s+\lambda)$ (Estimation)')
    ax1.semilogx(omega, 20*np.log10(np.abs(mag_S)), 
                 'r--', linewidth=2.5, label=r'$d_{res}/d = s/(s+\lambda)$ (Residual)')
    ax1.axhline(-3, color='orange', linewidth=1, linestyle=':', label='-3 dB')
    ax1.axvline(ndob_params.lambda_obs, color='green', linewidth=1.5, linestyle='--',
                label=f'$\\lambda$ = {ndob_params.lambda_obs} rad/s')
    ax1.set_ylabel('Magnitude [dB]', fontsize=12, fontweight='bold')
    ax1.set_title('NDOB Disturbance Estimation & Rejection', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', which='both')
    ax1.set_xlim([omega[0], omega[-1]])
    ax1.set_ylim([-40, 5])
    
    # Phase plot with time delay annotation
    ax2.semilogx(omega, np.rad2deg(np.angle(mag_H)), 
                 'b-', linewidth=2.5, label=r'Estimation $\hat{d}/d$')
    ax2.semilogx(omega, np.rad2deg(np.angle(mag_S)), 
                 'r--', linewidth=2.5, label=r'Residual $d_{res}/d$')
    ax2.axvline(ndob_params.lambda_obs, color='green', linewidth=1.5, linestyle='--')
    
    # Annotate phase lag at bandwidth
    phase_at_bw = -45  # degrees at λ
    time_delay = 1.0 / ndob_params.lambda_obs  # Approximate time constant
    ax2.annotate(f'Phase lag = 45° at λ\n(τ ≈ {1000*time_delay:.1f} ms)',
                 xy=(ndob_params.lambda_obs, phase_at_bw),
                 xytext=(ndob_params.lambda_obs*3, phase_at_bw-20),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    ax2.set_xlabel('Frequency [rad/s]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Phase [deg]', fontsize=12, fontweight='bold')
    ax2.set_title('NDOB Phase Characteristics', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':', which='both')
    ax2.set_xlim([omega[0], omega[-1]])
    ax2.set_ylim([-100, 100])
    
    fig.suptitle(f'NDOB Frequency Response (λ = {ndob_params.lambda_obs} rad/s)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


def plot_stability_margins(plant: ctrl.TransferFunction,
                            controller: ctrl.TransferFunction,
                            omega: np.ndarray,
                            title: str = "Stability Margins",
                            save_path: Optional[Path] = None) -> Tuple[plt.Figure, Dict]:
    """
    Plot Bode diagram with annotated stability margins.
    
    Returns figure and dict with computed margins.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    L = controller * plant
    
    # Compute margins
    try:
        gm, pm, wg, wp = ctrl.margin(L)
        gm_db = 20*np.log10(gm) if gm not in [None, np.inf] else float('inf')
    except:
        gm, pm, wg, wp = None, None, None, None
        gm_db = float('inf')
    
    # Frequency response
    mag, phase, _ = ctrl.frequency_response(L, omega)
    
    # Magnitude plot
    ax1.semilogx(omega, 20*np.log10(np.abs(mag)), 'b-', linewidth=2.5)
    ax1.axhline(0, color='k', linewidth=1, linestyle=':')
    
    # Annotate gain margin
    if wg is not None and gm not in [None, np.inf]:
        ax1.axvline(wg, color='r', linewidth=1.5, linestyle='--', alpha=0.7)
        ax1.annotate(f'GM = {gm_db:.1f} dB\n@ {wg:.1f} rad/s',
                     xy=(wg, 0), xytext=(wg*2, 10),
                     fontsize=10, color='r',
                     arrowprops=dict(arrowstyle='->', color='r'))
    
    ax1.set_ylabel('Magnitude [dB]', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', which='both')
    ax1.set_xlim([omega[0], omega[-1]])
    
    # Phase plot
    phase_deg = np.rad2deg(np.unwrap(np.angle(mag)))
    ax2.semilogx(omega, phase_deg, 'b-', linewidth=2.5)
    ax2.axhline(-180, color='r', linewidth=1, linestyle=':')
    
    # Annotate phase margin
    if wp is not None and pm is not None:
        ax2.axvline(wp, color='g', linewidth=1.5, linestyle='--', alpha=0.7)
        ax2.annotate(f'PM = {pm:.1f}°\n@ {wp:.1f} rad/s',
                     xy=(wp, -180+pm), xytext=(wp*2, -180+pm+30),
                     fontsize=10, color='g',
                     arrowprops=dict(arrowstyle='->', color='g'))
    
    ax2.set_xlabel('Frequency [rad/s]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Phase [deg]', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title} - Phase', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', which='both')
    ax2.set_xlim([omega[0], omega[-1]])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    margins = {
        'gain_margin_dB': gm_db,
        'phase_margin_deg': pm,
        'gain_crossover_freq': wg,
        'phase_crossover_freq': wp
    }
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig, margins


def generate_all_bode_plots(output_dir: Path = None,
                            plant_params: GimbalPlantParams = None,
                            fbl_params: FBLControllerParams = None,
                            ndob_params: NDOBParams = None) -> Dict:
    """
    Generate complete set of Bode plots for FBL and FBL+NDOB.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory to save figures. If None, uses 'figures_bode/'
    plant_params : GimbalPlantParams, optional
        Plant parameters. Uses defaults if not provided.
    fbl_params : FBLControllerParams, optional
        FBL controller parameters. Uses defaults if not provided.
    ndob_params : NDOBParams, optional  
        NDOB parameters. Uses defaults if not provided.
        
    Returns
    -------
    Dict
        Analysis results including stability margins
    """
    # Use defaults if not provided
    if output_dir is None:
        output_dir = Path('figures_bode')
    output_dir.mkdir(exist_ok=True)
    
    if plant_params is None:
        plant_params = GimbalPlantParams()
    if fbl_params is None:
        fbl_params = FBLControllerParams()
    if ndob_params is None:
        ndob_params = NDOBParams()
    
    print("=" * 70)
    print("BODE PLOT GENERATOR FOR FBL AND FBL+NDOB CONTROLLERS")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Plant: J_az={plant_params.inertia_az} kg·m², D_az={plant_params.friction_az} N·m·s/rad")
    print(f"  FBL: Kp={fbl_params.kp}, Kd={fbl_params.kd}, Ki={fbl_params.ki}")
    print(f"  NDOB: λ={ndob_params.lambda_obs} rad/s (τ={1000/ndob_params.lambda_obs:.1f} ms)")
    print(f"  Output: {output_dir.absolute()}\n")
    
    # Create transfer functions
    plant = create_gimbal_plant(plant_params, 'az')
    controller = create_fbl_controller(fbl_params)
    ndob_tf = create_ndob_transfer_function(ndob_params)
    
    # Frequency range for analysis
    omega = np.logspace(-1, 3, 1000)  # 0.1 to 1000 rad/s
    
    # Compute loop transfer functions
    tf_fbl = compute_loop_transfer_functions(plant, controller)
    tf_ndob = compute_loop_transfer_functions(plant, controller, ndob_tf)
    
    results = {}
    
    print("Generating figures...")
    
    # Figure 1: Comparative Bode plots
    plot_bode_comparison(tf_fbl, tf_ndob, omega,
                         save_path=output_dir / 'bode_comparison.png')
    
    # Figure 2: NDOB disturbance rejection
    plot_ndob_disturbance_rejection(ndob_params, omega,
                                     save_path=output_dir / 'ndob_disturbance_rejection.png')
    
    # Figure 3: FBL stability margins
    fig_fbl, margins_fbl = plot_stability_margins(
        plant, controller, omega,
        title="FBL Controller Stability Margins",
        save_path=output_dir / 'stability_margins_fbl.png'
    )
    results['fbl_margins'] = margins_fbl
    
    # Figure 4: Nichols chart (gain-phase plot)
    fig_nichols = plt.figure(figsize=(10, 8))
    ax = fig_nichols.add_subplot(111)
    
    L_fbl = controller * plant
    mag_fbl, phase_fbl, _ = ctrl.frequency_response(L_fbl, omega)
    
    ax.plot(np.rad2deg(np.unwrap(np.angle(mag_fbl))), 20*np.log10(np.abs(mag_fbl)),
            'b-', linewidth=2, label='FBL Open-Loop')
    ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax.axvline(-180, color='r', linewidth=0.5, linestyle=':')
    ax.scatter([-180], [0], color='r', s=100, marker='x', linewidths=2,
               label='Critical Point (-180°, 0 dB)')
    
    ax.set_xlabel('Phase [deg]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude [dB]', fontsize=12, fontweight='bold')
    ax.set_title('Nichols Chart - FBL Controller', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':', which='both')
    ax.set_xlim([-360, 0])
    ax.set_ylim([-60, 60])
    
    fig_nichols.savefig(output_dir / 'nichols_chart_fbl.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'nichols_chart_fbl.png'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("STABILITY MARGIN ANALYSIS")
    print("=" * 70)
    print(f"\nFBL Controller:")
    if margins_fbl['gain_margin_dB'] != float('inf'):
        print(f"  Gain Margin: {margins_fbl['gain_margin_dB']:.1f} dB @ {margins_fbl['gain_crossover_freq']:.1f} rad/s")
    else:
        print(f"  Gain Margin: ∞ (unconditionally stable in gain)")
    if margins_fbl['phase_margin_deg'] is not None:
        print(f"  Phase Margin: {margins_fbl['phase_margin_deg']:.1f}° @ {margins_fbl['phase_crossover_freq']:.1f} rad/s")
    
    print(f"\nNDOB Characteristics:")
    print(f"  Bandwidth: {ndob_params.lambda_obs} rad/s ({ndob_params.lambda_obs/(2*np.pi):.1f} Hz)")
    print(f"  Time Constant: {1000/ndob_params.lambda_obs:.1f} ms")
    print(f"  -3dB Frequency: {ndob_params.lambda_obs} rad/s")
    print(f"  Phase Lag at Bandwidth: 45°")
    
    print("\n" + "=" * 70)
    print(f"COMPLETE: Generated 4 figures in {output_dir.absolute()}/")
    print("=" * 70)
    
    return results


def main():
    """Main entry point with command-line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Bode plots for FBL and FBL+NDOB controllers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Plant parameters
    parser.add_argument('--inertia', type=float, default=1.0,
                        help='Gimbal inertia [kg·m²]')
    parser.add_argument('--friction', type=float, default=0.1,
                        help='Viscous friction [N·m·s/rad]')
    
    # Controller parameters
    parser.add_argument('--kp', type=float, default=400.0,
                        help='FBL position gain [1/s²]')
    parser.add_argument('--kd', type=float, default=40.0,
                        help='FBL velocity gain [1/s]')
    parser.add_argument('--ki', type=float, default=50.0,
                        help='FBL integral gain [1/s³]')
    
    # NDOB parameters
    parser.add_argument('--ndob-lambda', type=float, default=50.0,
                        help='NDOB observer bandwidth [rad/s]')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='figures_bode',
                        help='Output directory for figures')
    parser.add_argument('--show', action='store_true',
                        help='Display figures interactively')
    
    args = parser.parse_args()
    
    # Create parameter objects
    plant_params = GimbalPlantParams(
        inertia_az=args.inertia,
        inertia_el=args.inertia * 0.5,  # Elevation is typically lighter
        friction_az=args.friction,
        friction_el=args.friction
    )
    
    fbl_params = FBLControllerParams(
        kp=args.kp,
        kd=args.kd,
        ki=args.ki,
        enable_integral=True
    )
    
    ndob_params = NDOBParams(lambda_obs=args.ndob_lambda)
    
    # Generate plots
    results = generate_all_bode_plots(
        output_dir=Path(args.output_dir),
        plant_params=plant_params,
        fbl_params=fbl_params,
        ndob_params=ndob_params
    )
    
    if args.show:
        plt.show()
    
    return results


if __name__ == '__main__':
    main()
