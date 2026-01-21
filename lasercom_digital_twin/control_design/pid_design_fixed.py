#!/usr/bin/env python3
"""
FIXED Controller Design for Double-Integrator Gimbal Plant

This module provides corrected PID synthesis for gimbal systems where
dynamics are acceleration-based (Type-2 plant), not position-based.

Key Fix: Recognize that gimbal plant is:
    G(s) = 1/(M*s²)  (double integrator from torque to position)
    
Not:
    G(s) = K/(s+a)   (first-order position response)

Author: Senior Control Systems Engineer
Date: January 21, 2026
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


@dataclass
class ControllerGains:
    """PID gain container."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


def design_pid_for_double_integrator(
    gimbal: GimbalDynamics,
    q_op: np.ndarray = None,
    dq_op: np.ndarray = None,
    bandwidth_hz: float = 5.0,
    damping_ratio: float = 0.707
) -> Dict:
    """
    Design PID controller for double-integrator gimbal plant.
    
    Mathematical Basis:
    ------------------
    Plant Model (acceleration-based):
        tau → q_ddot → integrate → q_dot → integrate → q
        Transfer function: G(s) = 1/(M*s²)
    
    Closed-Loop Target:
        T(s) = (Kp + Kd*s) / (M*s² + Kd*s + Kp)
    
    Standard Form:
        T(s) = ωn² / (s² + 2ζωn*s + ωn²)
    
    Matching Coefficients:
        ωn² = Kp/M  →  Kp = M * ωn²
        2ζωn = Kd/M  →  Kd = 2*ζ*M*ωn = 2*ζ*sqrt(M*Kp)
    
    Integral Term:
        Ki added at ωn/10 for disturbance rejection without affecting transient
    
    Args:
        gimbal: GimbalDynamics instance
        q_op: Operating point positions [rad]
        dq_op: Operating point velocities [rad/s]
        bandwidth_hz: Desired closed-loop bandwidth [Hz]
        damping_ratio: Target damping ratio (0.707 = critically damped)
    
    Returns:
        Dictionary with gains and analysis
    
    Example:
        >>> gimbal = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25)
        >>> result = design_pid_for_double_integrator(gimbal, bandwidth_hz=5.0)
        >>> print(f"Kp_pan: {result['gains_pan'].kp:.1f}")
    """
    if q_op is None:
        q_op = np.array([0.0, 0.0])
    if dq_op is None:
        dq_op = np.array([0.0, 0.0])
    
    print("=" * 80)
    print("PID DESIGN FOR DOUBLE-INTEGRATOR GIMBAL PLANT")
    print("=" * 80)
    print(f"Operating Point: q={q_op}, dq={dq_op}")
    print(f"Target Bandwidth: {bandwidth_hz:.1f} Hz")
    print(f"Damping Ratio: {damping_ratio:.3f}")
    print()
    
    # Linearize to extract inertia matrix
    print("Step 1: Linearizing to extract inertia...")
    A, B, C, D = gimbal.linearize(q_op, dq_op)
    
    # B matrix structure: [0, 0; M^-1]
    # Extract M^-1 from lower 2x2 block
    M_inv = B[2:, :]
    M = np.linalg.inv(M_inv)
    
    M_pan = M[0, 0]
    M_tilt = M[1, 1]
    
    print(f"  Pan inertia: {M_pan:.6f} kg·m²")
    print(f"  Tilt inertia: {M_tilt:.6f} kg·m²")
    
    # Check for coupling
    coupling_ratio = abs(M[0, 1]) / M_pan if M_pan > 0 else 0
    print(f"  Coupling: {coupling_ratio*100:.2f}% (off-diagonal/diagonal)")
    print()
    
    # Design gains using double-integrator formulas
    print("Step 2: Computing PID gains for double-integrator...")
    omega_n = 2 * np.pi * bandwidth_hz  # Natural frequency [rad/s]
    
    # Pan axis gains
    Kp_pan = M_pan * (omega_n ** 2)
    Kd_pan = 2 * damping_ratio * np.sqrt(M_pan * Kp_pan)
    Ki_pan = Kp_pan * omega_n / 10.0  # Integral corner at ωn/10
    
    # Tilt axis gains
    Kp_tilt = M_tilt * (omega_n ** 2)
    Kd_tilt = 2 * damping_ratio * np.sqrt(M_tilt * Kp_tilt)
    Ki_tilt = Kp_tilt * omega_n / 10.0
    
    print(f"  Pan Gains:")
    print(f"    Kp = {Kp_pan:.3f} N·m/rad")
    print(f"    Ki = {Ki_pan:.3f} N·m/(rad·s)")
    print(f"    Kd = {Kd_pan:.6f} N·m·s/rad")
    print()
    print(f"  Tilt Gains:")
    print(f"    Kp = {Kp_tilt:.3f} N·m/rad")
    print(f"    Ki = {Ki_tilt:.3f} N·m/(rad·s)")
    print(f"    Kd = {Kd_tilt:.6f} N·m·s/rad")
    print()
    
    # Predicted performance
    print("Step 3: Predicted closed-loop performance...")
    
    # Natural frequency check
    wn_pan_check = np.sqrt(Kp_pan / M_pan)
    wn_tilt_check = np.sqrt(Kp_tilt / M_tilt)
    
    print(f"  Pan axis:")
    print(f"    Natural frequency: {wn_pan_check/(2*np.pi):.2f} Hz")
    print(f"    Rise time (0-100%): {1.8/wn_pan_check*1000:.1f} ms")
    print(f"    Settling time (2%): {4.0/(damping_ratio*wn_pan_check)*1000:.1f} ms")
    
    overshoot = 100 * np.exp(-damping_ratio * np.pi / np.sqrt(1 - damping_ratio**2))
    print(f"    Overshoot: {overshoot:.1f}%")
    print()
    
    print(f"  Tilt axis:")
    print(f"    Natural frequency: {wn_tilt_check/(2*np.pi):.2f} Hz")
    print(f"    Rise time (0-100%): {1.8/wn_tilt_check*1000:.1f} ms")
    print(f"    Settling time (2%): {4.0/(damping_ratio*wn_tilt_check)*1000:.1f} ms")
    print(f"    Overshoot: {overshoot:.1f}%")
    print()
    
    # Warnings
    print("Step 4: Design validation...")
    
    if coupling_ratio > 0.1:
        print(f"  ⚠ WARNING: Significant coupling detected ({coupling_ratio*100:.1f}%)")
        print("    Consider MIMO controller or cross-coupling feedforward")
    
    if bandwidth_hz > 20:
        print(f"  ⚠ WARNING: High bandwidth ({bandwidth_hz} Hz)")
        print("    Verify actuator can provide required torque rates")
    
    print("  ✓ Design complete")
    print()
    
    print("=" * 80)
    print("USAGE IN SIMULATION")
    print("=" * 80)
    print("""
config = SimulationConfig(
    coarse_controller_config={{
        'kp': [{Kp_pan:.3f}, {Kp_tilt:.3f}],
        'ki': [{Ki_pan:.3f}, {Ki_tilt:.3f}],
        'kd': [{Kd_pan:.6f}, {Kd_tilt:.6f}],
        'anti_windup_gain': 1.0,
        'tau_rate_limit': 50.0
    }}
)
""".format(
        Kp_pan=Kp_pan, Kp_tilt=Kp_tilt,
        Ki_pan=Ki_pan, Ki_tilt=Ki_tilt,
        Kd_pan=Kd_pan, Kd_tilt=Kd_tilt
    ))
    print("=" * 80)
    
    return {
        'gains_pan': ControllerGains(kp=Kp_pan, ki=Ki_pan, kd=Kd_pan),
        'gains_tilt': ControllerGains(kp=Kp_tilt, ki=Ki_tilt, kd=Kd_tilt),
        'M': M,
        'bandwidth_hz': bandwidth_hz,
        'damping_ratio': damping_ratio,
        'natural_freq_hz': omega_n / (2 * np.pi),
        'coupling_ratio': coupling_ratio
    }


if __name__ == "__main__":
    """
    Standalone execution: Design corrected PID gains for gimbal.
    """
    print("\n" + "=" * 80)
    print("CORRECTED PID CONTROLLER DESIGN TOOL")
    print("Recognizes gimbal as Type-2 (double integrator) plant")
    print("=" * 80 + "\n")
    
    # Create gimbal instance
    gimbal = GimbalDynamics(
        pan_mass=0.5,
        tilt_mass=0.25,
        cm_r=0.002,
        cm_h=0.0005,
        gravity=9.81
    )
    
    # Design for 5 Hz bandwidth (typical coarse stage)
    result = design_pid_for_double_integrator(
        gimbal=gimbal,
        q_op=np.array([0.0, 0.0]),
        dq_op=np.array([0.0, 0.0]),
        bandwidth_hz=5.0,
        damping_ratio=0.707
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL (INCORRECT) DESIGN")
    print("=" * 80)
    print(f"\nOriginal gains (from linearization-based design):")
    print(f"  Pan:  Kp=3.257,  Ki=10.232,  Kd=0.104")
    print(f"  Tilt: Kp=0.661,  Ki=2.078,   Kd=0.021")
    print(f"\nCorrected gains (double-integrator design):")
    print(f"  Pan:  Kp={result['gains_pan'].kp:.3f},  Ki={result['gains_pan'].ki:.3f},  Kd={result['gains_pan'].kd:.3f}")
    print(f"  Tilt: Kp={result['gains_tilt'].kp:.3f},  Ki={result['gains_tilt'].ki:.3f},  Kd={result['gains_tilt'].kd:.3f}")
    print(f"\nScaling factor: ~{result['gains_pan'].kp/3.257:.0f}x higher (as expected for Type-2 plant)")
    print("\n" + "=" * 80 + "\n")
