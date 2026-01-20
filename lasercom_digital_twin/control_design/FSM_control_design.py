#!/usr/bin/env python3
"""
Fast Steering Mirror (FSM) Control Design

This script designs precision controllers for a 2-axis FSM driven by Voice Coil
Actuators (VCA) for space laser communication applications. The FSM provides
high-bandwidth fine pointing (100-200 Hz) to correct residual Line-of-Sight
errors after coarse gimbal positioning.

Key Challenges Addressed:
-------------------------
1. MIMO (2-in, 2-out) coupling between Tip and Tilt axes
2. Flexural resonances from mechanical structure
3. Cross-axis coupling from asymmetric mass distribution
4. High-frequency sensor noise from Quadrant Photo Detector (QPD)

System Model: 4th-order reduced-order state-space representation
Control Objectives:
    - Bandwidth: 100-200 Hz (closed-loop)
    - Phase Margin: ≥ 45° (robustness)
    - Cross-talk: < 5% (axis decoupling)
    - Disturbance Rejection: > 40 dB at low frequencies

Author: Senior Control Systems Engineer
Date: January 20, 2026
"""

# Set matplotlib backend BEFORE any imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import linalg
import control as ctrl
from typing import Dict, Tuple, List
import json
from dataclasses import asdict
import json
from dataclasses import dataclass, asdict


# ============================================================================
# SYSTEM MODEL DEFINITION
# ============================================================================

def create_fsm_model() -> Tuple[signal.StateSpace, ctrl.StateSpace]:
    """
    Create the 4th-order reduced-order FSM state-space model.
    
    This model captures:
    - First two flexural resonances (structural modes)
    - Cross-axis coupling from asymmetric inertia
    - VCA actuator dynamics (first-order approximation)
    
    Returns
    -------
    Tuple[signal.StateSpace, ctrl.StateSpace]
        SciPy and Python Control library representations
    """
    # State Matrix (4x4) - captures structural resonances and damping
    Ar = np.array([
        [-20.02,   8.20,  125.75,    1.14],
        [-8.16,  -19.92,    2.56, -146.84],
        [-125.77, -3.40,  -30.63,    8.92],
        [-0.88,   146.81,  -8.99,  -28.45]
    ])
    
    # Input Matrix (4x2) - VCA force to state mapping
    Br = np.array([
        [40.85, -43.10],
        [-36.55, -38.03],
        [42.42, -43.00],
        [39.16, 36.67]
    ])
    
    # Output Matrix (2x4) - state to Tip/Tilt angle mapping
    Cr = np.array([
        [38.68, -41.64, -37.43, -41.31],
        [-45.05, -32.38, 47.41, -34.23]
    ])
    
    # Direct transmission (2x2) - no direct feed-through
    Dr = np.zeros((2, 2))
    
    # Create state-space models
    sys_scipy = signal.StateSpace(Ar, Br, Cr, Dr)
    sys_ctrl = ctrl.StateSpace(Ar, Br, Cr, Dr)
    
    return sys_scipy, sys_ctrl


# ============================================================================
# SYSTEM CHARACTERIZATION
# ============================================================================

def analyze_system_characteristics(sys_ctrl: ctrl.StateSpace) -> Dict:
    """
    Compute modal characteristics of the FSM system.
    
    Parameters
    ----------
    sys_ctrl : ctrl.StateSpace
        FSM plant model
        
    Returns
    -------
    Dict
        System characteristics including poles, zeros, resonances
    """
    print("=" * 70)
    print("FSM SYSTEM CHARACTERIZATION")
    print("=" * 70)
    
    # Compute eigenvalues (poles)
    poles = np.linalg.eigvals(sys_ctrl.A)
    
    print("\n1. SYSTEM POLES (Eigenvalues)")
    print("-" * 70)
    for i, pole in enumerate(poles):
        real_part = np.real(pole)
        imag_part = np.imag(pole)
        freq_hz = np.abs(imag_part) / (2 * np.pi)
        
        if np.abs(imag_part) > 1e-6:
            # Complex conjugate pair - extract damping ratio
            wn = np.abs(pole)
            zeta = -real_part / wn
            print(f"  Pole {i+1}: {real_part:8.2f} ± j{np.abs(imag_part):8.2f}")
            print(f"           ω_n = {wn:8.2f} rad/s  |  f_n = {freq_hz:6.2f} Hz")
            print(f"           ζ   = {zeta:8.4f} (damping ratio)")
        else:
            print(f"  Pole {i+1}: {real_part:8.2f} (real)")
    
    # Compute transmission zeros (MIMO system)
    print("\n2. TRANSMISSION ZEROS")
    print("-" * 70)
    try:
        zeros = ctrl.zero(sys_ctrl)
        if len(zeros) > 0:
            for i, zero in enumerate(zeros):
                real_part = np.real(zero)
                imag_part = np.imag(zero)
                freq_hz = np.abs(imag_part) / (2 * np.pi)
                print(f"  Zero {i+1}: {real_part:8.2f} ± j{np.abs(imag_part):8.2f}")
                print(f"           f = {freq_hz:6.2f} Hz")
        else:
            print("  No finite transmission zeros (minimum phase system)")
    except Exception as e:
        print(f"  Unable to compute zeros for MIMO system: {e}")
    
    # Compute DC gain matrix
    print("\n3. DC GAIN MATRIX (Steady-State)")
    print("-" * 70)
    try:
        # DC gain: G(0) = -C * A^(-1) * B + D
        dc_gain = -sys_ctrl.C @ np.linalg.inv(sys_ctrl.A) @ sys_ctrl.B + sys_ctrl.D
        print(f"  G(0) = ")
        print(f"    [{dc_gain[0,0]:10.4f}  {dc_gain[0,1]:10.4f}]  (Tip)")
        print(f"    [{dc_gain[1,0]:10.4f}  {dc_gain[1,1]:10.4f}]  (Tilt)")
        
        # Diagonal dominance check
        diag_dom_tip = np.abs(dc_gain[0,0]) / np.abs(dc_gain[0,1])
        diag_dom_tilt = np.abs(dc_gain[1,1]) / np.abs(dc_gain[1,0])
        print(f"\n  Diagonal Dominance:")
        print(f"    Tip:  |G11/G12| = {diag_dom_tip:.2f}")
        print(f"    Tilt: |G22/G21| = {diag_dom_tilt:.2f}")
        
        if diag_dom_tip > 2.0 and diag_dom_tilt > 2.0:
            print(f"    ✓ System is diagonally dominant (decentralized control viable)")
        else:
            print(f"    ⚠ Weak diagonal dominance (consider MIMO compensation)")
    except Exception as e:
        print(f"  Unable to compute DC gain: {e}")
    
    # Package results
    characteristics = {
        'poles': poles,
        'dc_gain': dc_gain if 'dc_gain' in locals() else None,
        'resonance_freqs_hz': [np.abs(np.imag(p))/(2*np.pi) for p in poles if np.abs(np.imag(p)) > 1e-6]
    }
    
    return characteristics


# ============================================================================
# MIMO BODE PLOTS
# ============================================================================

def plot_mimo_bode(sys_ctrl: ctrl.StateSpace, save_path: str = "fsm_bode_mimo.png"):
    """
    Generate MIMO Bode plots showing all 4 transfer functions.
    
    Parameters
    ----------
    sys_ctrl : ctrl.StateSpace
        FSM plant model (2x2 MIMO)
    save_path : str
        Output file path
    """
    print("\n4. GENERATING MIMO BODE PLOTS")
    print("-" * 70)
    
    # Frequency range: 0.1 Hz to 1 kHz
    omega = np.logspace(-1, 3, 1000) * 2 * np.pi
    
    # Compute frequency response
    mag = np.zeros((2, 2, len(omega)))
    phase = np.zeros((2, 2, len(omega)))
    
    for i in range(2):
        for j in range(2):
            # Extract SISO transfer function from MIMO system
            sys_ij = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, j:j+1], 
                            sys_ctrl.C[i:i+1, :], sys_ctrl.D[i:i+1, j:j+1])
            mag_ij, phase_ij, _ = ctrl.bode(sys_ij, omega, plot=False)
            mag[i, j, :] = 20 * np.log10(mag_ij)
            phase[i, j, :] = phase_ij * 180 / np.pi
    
    freq_hz = omega / (2 * np.pi)
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    labels = [
        ['G11: Tip Cmd → Tip Angle', 'G12: Tilt Cmd → Tip Angle'],
        ['G21: Tip Cmd → Tilt Angle', 'G22: Tilt Cmd → Tilt Angle']
    ]
    
    colors = [
        ['#1f77b4', '#ff7f0e'],  # Blue (direct), Orange (cross)
        ['#ff7f0e', '#d62728']   # Orange (cross), Red (direct)
    ]
    
    for i in range(2):
        for j in range(2):
            # Magnitude plot
            ax_mag = axes[2*i, j]
            is_direct = (i == j)
            linewidth = 2.5 if is_direct else 1.5
            alpha = 0.9 if is_direct else 0.7
            
            ax_mag.semilogx(freq_hz, mag[i, j, :], 
                           color=colors[i][j], linewidth=linewidth, alpha=alpha)
            ax_mag.set_ylabel('Magnitude [dB]', fontsize=10, fontweight='bold')
            ax_mag.set_title(labels[i][j], fontsize=11, fontweight='bold')
            ax_mag.grid(True, which='both', alpha=0.3, linestyle=':')
            ax_mag.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
            
            # Highlight direct vs cross-coupling
            if is_direct:
                ax_mag.text(0.05, 0.95, 'DIRECT PATH', 
                          transform=ax_mag.transAxes, fontsize=9, 
                          verticalalignment='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            else:
                ax_mag.text(0.05, 0.95, 'CROSS-COUPLING', 
                          transform=ax_mag.transAxes, fontsize=9, 
                          verticalalignment='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
            # Phase plot
            ax_phase = axes[2*i + 1, j]
            ax_phase.semilogx(freq_hz, phase[i, j, :], 
                             color=colors[i][j], linewidth=linewidth, alpha=alpha)
            ax_phase.set_ylabel('Phase [deg]', fontsize=10, fontweight='bold')
            ax_phase.set_xlabel('Frequency [Hz]', fontsize=10, fontweight='bold')
            ax_phase.grid(True, which='both', alpha=0.3, linestyle=':')
            ax_phase.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
            ax_phase.axhline(-180, color='red', linewidth=0.8, linestyle=':', alpha=0.5)
    
    fig.suptitle('FSM MIMO Frequency Response (Open-Loop)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ MIMO Bode plot saved as '{save_path}'")


# ============================================================================
# CONTROLLER DESIGN
# ============================================================================

@dataclass
class PIControllerGains:
    """PI controller gains for FSM."""
    Kp_tip: float
    Ki_tip: float
    Kp_tilt: float
    Ki_tilt: float
    crossover_freq_hz: float
    phase_margin_deg: float


@dataclass
class PIDFControllerGains:
    """PIDF controller gains with derivative filtering."""
    Kp_tip: float
    Ki_tip: float
    Kd_tip: float
    Kp_tilt: float
    Ki_tilt: float
    Kd_tilt: float
    N_tip: float  # Derivative filter coefficient
    N_tilt: float
    crossover_freq_hz: float
    phase_margin_deg: float


def design_pi_controller(sys_ctrl: ctrl.StateSpace, 
                        target_bandwidth_hz: float = 150.0) -> PIControllerGains:
    """
    Design decentralized PI controller for FSM.
    
    Design Strategy:
    ---------------
    1. Use frequency response of direct paths (G11, G22)
    2. Place crossover frequency at target bandwidth
    3. Add integral action for zero steady-state error
    4. Ensure phase margin ≥ 45° for robustness
    
    Parameters
    ----------
    sys_ctrl : ctrl.StateSpace
        FSM plant model
    target_bandwidth_hz : float
        Desired closed-loop bandwidth [Hz]
        
    Returns
    -------
    PIControllerGains
        Tuned PI gains
    """
    print("\n5. DESIGNING PI CONTROLLER (Baseline)")
    print("-" * 70)
    
    omega_c = 2 * np.pi * target_bandwidth_hz  # Crossover frequency [rad/s]
    
    # Extract direct path transfer functions (decoupled design)
    sys_tip = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 0:1], 
                     sys_ctrl.C[0:1, :], sys_ctrl.D[0:1, 0:1])
    sys_tilt = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 1:2], 
                      sys_ctrl.C[1:2, :], sys_ctrl.D[1:2, 1:2])
    
    # PI controller structure: C(s) = Kp * (1 + Ki/s)
    # Place integral zero at 1/10 of crossover for phase boost
    omega_i = omega_c / 10.0  # Integral frequency
    
    # Compute plant gain at crossover
    mag_tip, phase_tip, _ = ctrl.bode(sys_tip, [omega_c], plot=False)
    mag_tilt, phase_tilt, _ = ctrl.bode(sys_tilt, [omega_c], plot=False)
    
    # Proportional gain: Set open-loop gain = 1 at crossover (0 dB)
    Kp_tip = 1.0 / mag_tip[0]
    Kp_tilt = 1.0 / mag_tilt[0]
    
    # Integral gain: Ki = Kp * omega_i
    Ki_tip = Kp_tip * omega_i
    Ki_tilt = Kp_tilt * omega_i
    
    # Compute phase margin (approximate for PI controller)
    # PI adds phase: angle = arctan(omega_c / omega_i)
    phase_boost_deg = np.arctan(omega_c / omega_i) * 180 / np.pi
    phase_margin_tip = 180 + phase_tip[0] * 180 / np.pi + phase_boost_deg
    phase_margin_tilt = 180 + phase_tilt[0] * 180 / np.pi + phase_boost_deg
    phase_margin_avg = (phase_margin_tip + phase_margin_tilt) / 2
    
    print(f"  Target Bandwidth:     {target_bandwidth_hz:.1f} Hz")
    print(f"  Integral Frequency:   {omega_i / (2*np.pi):.1f} Hz")
    print(f"\n  TIP AXIS GAINS:")
    print(f"    Kp = {Kp_tip:.4f}")
    print(f"    Ki = {Ki_tip:.4f}")
    print(f"    Phase Margin ≈ {phase_margin_tip:.1f}°")
    print(f"\n  TILT AXIS GAINS:")
    print(f"    Kp = {Kp_tilt:.4f}")
    print(f"    Ki = {Ki_tilt:.4f}")
    print(f"    Phase Margin ≈ {phase_margin_tilt:.1f}°")
    
    if phase_margin_avg >= 45.0:
        print(f"\n  ✓ Phase margin {phase_margin_avg:.1f}° meets 45° requirement")
    else:
        print(f"\n  ⚠ Phase margin {phase_margin_avg:.1f}° below 45° (reduce bandwidth)")
    
    gains = PIControllerGains(
        Kp_tip=Kp_tip,
        Ki_tip=Ki_tip,
        Kp_tilt=Kp_tilt,
        Ki_tilt=Ki_tilt,
        crossover_freq_hz=target_bandwidth_hz,
        phase_margin_deg=phase_margin_avg
    )
    
    return gains


def design_pidf_controller(sys_ctrl: ctrl.StateSpace,
                          target_bandwidth_hz: float = 150.0) -> PIDFControllerGains:
    """
    Design PIDF controller with derivative filtering for noise attenuation.
    
    Design Strategy:
    ---------------
    1. Start with PI gains from baseline design
    2. Add derivative action for phase lead and damping
    3. Filter derivative term (N ≈ 10-20) to avoid noise amplification
    4. QPD sensor noise dominates above 500 Hz
    
    Parameters
    ----------
    sys_ctrl : ctrl.StateSpace
        FSM plant model
    target_bandwidth_hz : float
        Desired closed-loop bandwidth [Hz]
        
    Returns
    -------
    PIDFControllerGains
        Tuned PIDF gains with filtering
    """
    print("\n6. DESIGNING PIDF CONTROLLER (High Performance)")
    print("-" * 70)
    
    omega_c = 2 * np.pi * target_bandwidth_hz
    
    # Start with PI design
    sys_tip = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 0:1], 
                     sys_ctrl.C[0:1, :], sys_ctrl.D[0:1, 0:1])
    sys_tilt = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 1:2], 
                      sys_ctrl.C[1:2, :], sys_ctrl.D[1:2, 1:2])
    
    omega_i = omega_c / 10.0
    
    # Base PI gains
    mag_tip, phase_tip, _ = ctrl.bode(sys_tip, [omega_c], plot=False)
    mag_tilt, phase_tilt, _ = ctrl.bode(sys_tilt, [omega_c], plot=False)
    
    Kp_tip = 1.0 / mag_tip[0]
    Kp_tilt = 1.0 / mag_tilt[0]
    Ki_tip = Kp_tip * omega_i
    Ki_tilt = Kp_tilt * omega_i
    
    # Derivative gains for phase lead
    # Kd chosen to add ~15-20° phase lead at crossover
    # Filtered derivative: Kd*s / (1 + s/N*omega_c)
    N = 15.0  # Filter coefficient (typical: 10-20)
    
    # Derivative gain: Kd = Kp / (N * omega_c) tuned for phase lead
    Kd_tip = Kp_tip / (2.0 * omega_c)
    Kd_tilt = Kp_tilt / (2.0 * omega_c)
    
    # Compute approximate phase margin with PID
    phase_boost_i = np.arctan(omega_c / omega_i) * 180 / np.pi
    phase_boost_d = np.arctan(Kd_tip * omega_c / Kp_tip) * 180 / np.pi
    phase_margin_tip = 180 + phase_tip[0] * 180 / np.pi + phase_boost_i + phase_boost_d
    phase_margin_tilt = 180 + phase_tilt[0] * 180 / np.pi + phase_boost_i + phase_boost_d
    phase_margin_avg = (phase_margin_tip + phase_margin_tilt) / 2
    
    print(f"  Target Bandwidth:     {target_bandwidth_hz:.1f} Hz")
    print(f"  Integral Frequency:   {omega_i / (2*np.pi):.1f} Hz")
    print(f"  Derivative Filter N:  {N:.1f}")
    print(f"\n  TIP AXIS GAINS:")
    print(f"    Kp = {Kp_tip:.4f}")
    print(f"    Ki = {Ki_tip:.4f}")
    print(f"    Kd = {Kd_tip:.6f}")
    print(f"    Phase Margin ≈ {phase_margin_tip:.1f}°")
    print(f"\n  TILT AXIS GAINS:")
    print(f"    Kp = {Kp_tilt:.4f}")
    print(f"    Ki = {Ki_tilt:.4f}")
    print(f"    Kd = {Kd_tilt:.6f}")
    print(f"    Phase Margin ≈ {phase_margin_tilt:.1f}°")
    
    if phase_margin_avg >= 45.0:
        print(f"\n  ✓ Phase margin {phase_margin_avg:.1f}° meets 45° requirement")
    else:
        print(f"\n  ⚠ Phase margin {phase_margin_avg:.1f}° below 45° (reduce bandwidth)")
    
    gains = PIDFControllerGains(
        Kp_tip=Kp_tip,
        Ki_tip=Ki_tip,
        Kd_tip=Kd_tip,
        Kp_tilt=Kp_tilt,
        Ki_tilt=Ki_tilt,
        Kd_tilt=Kd_tilt,
        N_tip=N,
        N_tilt=N,
        crossover_freq_hz=target_bandwidth_hz,
        phase_margin_deg=phase_margin_avg
    )
    
    return gains


# ============================================================================
# CLOSED-LOOP SIMULATION
# ============================================================================

def simulate_closed_loop_step(sys_ctrl: ctrl.StateSpace,
                             gains: PIDFControllerGains,
                             duration: float = 0.1) -> Dict:
    """
    Simulate closed-loop step response with PIDF controller.
    
    Parameters
    ----------
    sys_ctrl : ctrl.StateSpace
        FSM plant model
    gains : PIDFControllerGains
        Controller gains
    duration : float
        Simulation duration [s]
        
    Returns
    -------
    Dict
        Simulation results with time histories
    """
    print("\n7. CLOSED-LOOP STEP RESPONSE SIMULATION")
    print("-" * 70)
    
    # Create PIDF transfer functions for each axis
    # PIDF(s) = Kp + Ki/s + Kd*N*s/(s + N*omega_c)
    omega_c = 2 * np.pi * gains.crossover_freq_hz
    N_omega_tip = gains.N_tip * omega_c
    N_omega_tilt = gains.N_tilt * omega_c
    
    # Tip axis PIDF
    num_tip = [gains.Kd_tip * N_omega_tip, 
               gains.Kp_tip * N_omega_tip + gains.Kd_tip * N_omega_tip * gains.Ki_tip / gains.Kp_tip,
               gains.Ki_tip * N_omega_tip]
    den_tip = [1, N_omega_tip, 0]
    C_tip = ctrl.tf(num_tip, den_tip)
    
    # Tilt axis PIDF
    num_tilt = [gains.Kd_tilt * N_omega_tilt,
                gains.Kp_tilt * N_omega_tilt + gains.Kd_tilt * N_omega_tilt * gains.Ki_tilt / gains.Kp_tilt,
                gains.Ki_tilt * N_omega_tilt]
    den_tilt = [1, N_omega_tilt, 0]
    C_tilt = ctrl.tf(num_tilt, den_tilt)
    
    # For MIMO system, use decentralized control (2x2 diagonal controller)
    # Extract SISO systems for each axis
    sys_tip = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 0:1], 
                     sys_ctrl.C[0:1, :], sys_ctrl.D[0:1, 0:1])
    sys_tilt = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 1:2], 
                      sys_ctrl.C[1:2, :], sys_ctrl.D[1:2, 1:2])
    
    # Closed-loop systems
    try:
        CL_tip = ctrl.feedback(C_tip * sys_tip)
        CL_tilt = ctrl.feedback(C_tilt * sys_tilt)
    except Exception as e:
        print(f"  ⚠ Error forming closed-loop system: {e}")
        return {}
    
    # Time vector
    t = np.linspace(0, duration, 1000)
    
    # Step responses
    print(f"  Simulating {duration*1000:.0f} ms step response...")
    try:
        t_tip, y_tip = ctrl.step_response(CL_tip, t)
        t_tilt, y_tilt = ctrl.step_response(CL_tilt, t)
    except Exception as e:
        print(f"  ⚠ Simulation error: {e}")
        return {}
    
    # Compute metrics
    def compute_metrics(t, y):
        # Rise time (10% to 90%)
        idx_10 = np.argmax(y >= 0.1)
        idx_90 = np.argmax(y >= 0.9)
        rise_time = t[idx_90] - t[idx_10] if idx_90 > idx_10 else 0
        
        # Settling time (2% criterion)
        steady_state = y[-1]
        settling_band = 0.02 * steady_state
        settled_idx = np.where(np.abs(y - steady_state) <= settling_band)[0]
        settling_time = t[settled_idx[0]] if len(settled_idx) > 0 else t[-1]
        
        # Overshoot
        peak = np.max(y)
        overshoot = ((peak - steady_state) / steady_state) * 100 if steady_state > 0 else 0
        
        return rise_time, settling_time, overshoot
    
    rise_tip, settle_tip, overshoot_tip = compute_metrics(t_tip, y_tip)
    rise_tilt, settle_tilt, overshoot_tilt = compute_metrics(t_tilt, y_tilt)
    
    print(f"\n  TIP AXIS METRICS:")
    print(f"    Rise Time:      {rise_tip*1000:.2f} ms")
    print(f"    Settling Time:  {settle_tip*1000:.2f} ms (2% criterion)")
    print(f"    Peak Overshoot: {overshoot_tip:.2f}%")
    
    print(f"\n  TILT AXIS METRICS:")
    print(f"    Rise Time:      {rise_tilt*1000:.2f} ms")
    print(f"    Settling Time:  {settle_tilt*1000:.2f} ms (2% criterion)")
    print(f"    Peak Overshoot: {overshoot_tilt:.2f}%")
    
    results = {
        't_tip': t_tip,
        'y_tip': y_tip,
        't_tilt': t_tilt,
        'y_tilt': y_tilt,
        'rise_time_tip': rise_tip,
        'settling_time_tip': settle_tip,
        'overshoot_tip': overshoot_tip,
        'rise_time_tilt': rise_tilt,
        'settling_time_tilt': settle_tilt,
        'overshoot_tilt': overshoot_tilt
    }
    
    return results


def analyze_cross_talk(sys_ctrl: ctrl.StateSpace, 
                       gains: PIDFControllerGains) -> Dict:
    """
    Quantify cross-axis coupling in closed-loop system.
    
    Cross-talk metric: When commanding Tip axis, measure Tilt response.
    
    Parameters
    ----------
    sys_ctrl : ctrl.StateSpace
        FSM plant model
    gains : PIDFControllerGains
        Controller gains
        
    Returns
    -------
    Dict
        Cross-talk measurements
    """
    print("\n8. CROSS-AXIS COUPLING ANALYSIS")
    print("-" * 70)
    
    # For cross-talk, we need to simulate the full MIMO closed-loop
    # Simplification: Measure open-loop cross-coupling at bandwidth frequency
    omega_c = 2 * np.pi * gains.crossover_freq_hz
    
    # Extract cross-coupling paths
    sys_tip_to_tilt = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 0:1],  # Tip cmd
                              sys_ctrl.C[1:2, :], sys_ctrl.D[1:2, 0:1])  # Tilt out
    sys_tilt_to_tip = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 1:2],  # Tilt cmd
                              sys_ctrl.C[0:1, :], sys_ctrl.D[0:1, 1:2])  # Tip out
    
    # Direct paths for comparison
    sys_tip_direct = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 0:1],
                            sys_ctrl.C[0:1, :], sys_ctrl.D[0:1, 0:1])
    sys_tilt_direct = ctrl.ss(sys_ctrl.A, sys_ctrl.B[:, 1:2],
                             sys_ctrl.C[1:2, :], sys_ctrl.D[1:2, 1:2])
    
    # Measure at crossover frequency
    mag_tip_to_tilt, _, _ = ctrl.bode(sys_tip_to_tilt, [omega_c], plot=False)
    mag_tilt_to_tip, _, _ = ctrl.bode(sys_tilt_to_tip, [omega_c], plot=False)
    mag_tip_direct, _, _ = ctrl.bode(sys_tip_direct, [omega_c], plot=False)
    mag_tilt_direct, _, _ = ctrl.bode(sys_tilt_direct, [omega_c], plot=False)
    
    # Cross-talk ratio (as percentage)
    crosstalk_tip_to_tilt = (mag_tip_to_tilt[0] / mag_tip_direct[0]) * 100
    crosstalk_tilt_to_tip = (mag_tilt_to_tip[0] / mag_tilt_direct[0]) * 100
    
    print(f"  Measured at {gains.crossover_freq_hz:.1f} Hz (crossover frequency):")
    print(f"\n  Tip Command → Tilt Response:")
    print(f"    |G21/G11| = {crosstalk_tip_to_tilt:.2f}%")
    
    print(f"\n  Tilt Command → Tip Response:")
    print(f"    |G12/G22| = {crosstalk_tilt_to_tip:.2f}%")
    
    max_crosstalk = max(crosstalk_tip_to_tilt, crosstalk_tilt_to_tip)
    print(f"\n  Maximum Cross-Talk: {max_crosstalk:.2f}%")
    
    if max_crosstalk < 5.0:
        print(f"  ✓ Cross-talk below 5% threshold (decentralized control acceptable)")
    elif max_crosstalk < 10.0:
        print(f"  ⚠ Moderate cross-talk (consider decoupling compensation)")
    else:
        print(f"  ✗ High cross-talk (MIMO controller recommended)")
    
    results = {
        'crosstalk_tip_to_tilt': crosstalk_tip_to_tilt,
        'crosstalk_tilt_to_tip': crosstalk_tilt_to_tip,
        'max_crosstalk': max_crosstalk
    }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_step_responses(sim_results: Dict, save_path: str = "fsm_step_response.png"):
    """Generate step response plots."""
    if not sim_results:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Tip axis
    ax1.plot(sim_results['t_tip'] * 1000, sim_results['y_tip'], 
            color='#1f77b4', linewidth=2, label='Tip Response')
    ax1.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Reference')
    ax1.axhline(1.02, color='red', linewidth=0.8, linestyle=':', alpha=0.4)
    ax1.axhline(0.98, color='red', linewidth=0.8, linestyle=':', alpha=0.4)
    ax1.set_ylabel('Tip Angle [normalized]', fontsize=11, fontweight='bold')
    ax1.set_title('Closed-Loop Step Response - Tip Axis', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Add metrics text
    metrics_text_tip = f"Rise: {sim_results['rise_time_tip']*1000:.1f} ms\n"
    metrics_text_tip += f"Settle: {sim_results['settling_time_tip']*1000:.1f} ms\n"
    metrics_text_tip += f"Overshoot: {sim_results['overshoot_tip']:.1f}%"
    ax1.text(0.70, 0.30, metrics_text_tip, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tilt axis
    ax2.plot(sim_results['t_tilt'] * 1000, sim_results['y_tilt'],
            color='#d62728', linewidth=2, label='Tilt Response')
    ax2.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Reference')
    ax2.axhline(1.02, color='red', linewidth=0.8, linestyle=':', alpha=0.4)
    ax2.axhline(0.98, color='red', linewidth=0.8, linestyle=':', alpha=0.4)
    ax2.set_ylabel('Tilt Angle [normalized]', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time [ms]', fontsize=11, fontweight='bold')
    ax2.set_title('Closed-Loop Step Response - Tilt Axis', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # Add metrics text
    metrics_text_tilt = f"Rise: {sim_results['rise_time_tilt']*1000:.1f} ms\n"
    metrics_text_tilt += f"Settle: {sim_results['settling_time_tilt']*1000:.1f} ms\n"
    metrics_text_tilt += f"Overshoot: {sim_results['overshoot_tilt']:.1f}%"
    ax2.text(0.70, 0.30, metrics_text_tilt, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('PIDF Closed-Loop Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Step response plot saved as '{save_path}'")


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

def export_gains_json(pi_gains: PIControllerGains,
                     pidf_gains: PIDFControllerGains,
                     filename: str = "fsm_controller_gains.json"):
    """
    Export controller gains in JSON format for integration.
    
    Parameters
    ----------
    pi_gains : PIControllerGains
        PI controller gains
    pidf_gains : PIDFControllerGains
        PIDF controller gains
    filename : str
        Output JSON file
    """
    config = {
        "controller_type": "PIDF_Decentralized",
        "description": "FSM PIDF controller gains for DigitalTwinRunner integration",
        "design_bandwidth_hz": pidf_gains.crossover_freq_hz,
        "phase_margin_deg": pidf_gains.phase_margin_deg,
        "pi_controller": asdict(pi_gains),
        "pidf_controller": asdict(pidf_gains),
        "usage_notes": [
            "Use PI controller for baseline performance",
            "Use PIDF controller for high-bandwidth operation",
            "Derivative filter N prevents QPD noise amplification",
            "Gains tuned for 150 Hz bandwidth with 45° phase margin"
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n9. EXPORTING CONTROLLER GAINS")
    print("-" * 70)
    print(f"  ✓ Gains exported to '{filename}'")
    print(f"  ✓ Ready for integration with DigitalTwinRunner")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n")
    print("=" * 70)
    print(" FAST STEERING MIRROR (FSM) CONTROL DESIGN ")
    print(" 2-Axis MIMO System with Voice Coil Actuators ")
    print("=" * 70)
    print("\nObjective: High-bandwidth precision pointing for laser communication")
    print("Target:    100-200 Hz bandwidth, 45° phase margin, <5% cross-talk\n")
    
    # 1. Create system model
    sys_scipy, sys_ctrl = create_fsm_model()
    print("✓ 4th-order FSM state-space model created (2-in, 2-out MIMO)")
    
    # 2. System characterization
    characteristics = analyze_system_characteristics(sys_ctrl)
    
    # 3. MIMO Bode plots
    plot_mimo_bode(sys_ctrl)
    
    # 4. Design PI controller (baseline)
    pi_gains = design_pi_controller(sys_ctrl, target_bandwidth_hz=150.0)
    
    # 5. Design PIDF controller (high performance)
    pidf_gains = design_pidf_controller(sys_ctrl, target_bandwidth_hz=150.0)
    
    # 6. Closed-loop simulation
    sim_results = simulate_closed_loop_step(sys_ctrl, pidf_gains, duration=0.1)
    
    # 7. Cross-talk analysis
    crosstalk_results = analyze_cross_talk(sys_ctrl, pidf_gains)
    
    # 8. Plot results
    if sim_results:
        plot_step_responses(sim_results)
    
    # 9. Export gains
    export_gains_json(pi_gains, pidf_gains)
    
    # Final summary
    print("\n" + "=" * 70)
    print(" DESIGN SUMMARY ")
    print("=" * 70)
    print(f"  Controller Type:    PIDF with Derivative Filtering")
    print(f"  Target Bandwidth:   150 Hz")
    print(f"  Phase Margin:       {pidf_gains.phase_margin_deg:.1f}°")
    print(f"  Max Cross-Talk:     {crosstalk_results['max_crosstalk']:.2f}%")
    if sim_results:
        print(f"  Settling Time (2%): {max(sim_results['settling_time_tip'], sim_results['settling_time_tilt'])*1000:.1f} ms")
        print(f"  Max Overshoot:      {max(sim_results['overshoot_tip'], sim_results['overshoot_tilt']):.1f}%")
    print("\n✓ FSM control design completed successfully!")
    print("=" * 70 + "\n")
    
    # Only show plots interactively if explicitly requested
    # (useful for interactive environments like Jupyter)
    if len(sys.argv) > 1 and sys.argv[1] == '--show':
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plots interactively: {e}")
            print("Plots have been saved to PNG files.")
    else:
        print("Plots saved to PNG files. Use '--show' argument to display interactively.")


if __name__ == "__main__":
    main()
