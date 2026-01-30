#!/usr/bin/env python3
"""
Standalone EKF Qualification Test - No pytest required.

This script validates the EKF's ability to track low-frequency sine waves,
specifically targeting the 0.1 Hz (10-second period) case that was failing.

Run with: python test_ekf_standalone.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.estimators.state_estimator import (
    PointingStateEstimator,
    StateIndex,
    MeasurementIndex
)


def test_ekf_sine_wave(freq_hz: float, amplitude_deg: float, duration_s: float, verbose: bool = True):
    """
    Test EKF tracking of a sinusoidal signal.
    
    Parameters
    ----------
    freq_hz : float
        Sine wave frequency [Hz]
    amplitude_deg : float
        Sine wave amplitude [degrees]
    duration_s : float
        Test duration [seconds]
    verbose : bool
        Print detailed results
        
    Returns
    -------
    dict
        Test results including RMS errors and pass/fail status
    """
    # Create EKF with TUNED parameters for low-frequency tracking
    config = {
        'initial_state': np.zeros(10),
        'pan_mass': 0.5,
        'tilt_mass': 0.25,
        'friction_coeff_az': 0.1,
        'friction_coeff_el': 0.1,
        # Low R to trust sensors (ideal sensor conditions)
        'measurement_noise_std': [1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2],
        # High disturbance Q to absorb model errors
        'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-1, 1e-1]
    }
    ekf = PointingStateEstimator(config)
    
    amplitude_rad = np.deg2rad(amplitude_deg)
    omega = 2 * np.pi * freq_hz
    dt = 0.001  # 1 ms timestep
    n_steps = int(duration_s / dt)
    
    # Skip first 20% for settling
    settle_steps = int(0.2 * n_steps)
    
    errors_pos = []
    errors_vel = []
    true_positions = []
    est_positions = []
    times = []
    
    for i in range(n_steps):
        t = i * dt
        
        # True sinusoidal motion
        true_pos = amplitude_rad * np.sin(omega * t)
        true_vel = amplitude_rad * omega * np.cos(omega * t)
        
        # Perfect sensor measurements (ideal conditions)
        measurements = {
            'theta_az_enc': true_pos,
            'theta_el_enc': true_pos,
            'theta_dot_az_gyro': true_vel,
            'theta_dot_el_gyro': true_vel,
            'nes_x_qpd': 0.0,
            'nes_y_qpd': 0.0
        }
        
        # No control torque (pure tracking test)
        ekf.step(np.zeros(2), measurements, dt)
        
        fused = ekf.get_fused_state()
        
        # Record data
        times.append(t)
        true_positions.append(true_pos)
        est_positions.append(fused['theta_az'])
        
        # Record errors after settling
        if i >= settle_steps:
            errors_pos.append(abs(fused['theta_az'] - true_pos))
            errors_vel.append(abs(fused['theta_dot_az'] - true_vel))
    
    # Compute RMS errors
    rms_pos = np.sqrt(np.mean(np.array(errors_pos)**2))
    rms_vel = np.sqrt(np.mean(np.array(errors_vel)**2))
    max_pos_error = np.max(np.array(errors_pos))
    
    # Requirements
    max_rms_pos = 100e-6  # 100 µrad
    max_rms_vel = 1e-3    # 1 mrad/s
    
    passed = rms_pos < max_rms_pos and rms_vel < max_rms_vel
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EKF Sine Wave Tracking Test: {freq_hz} Hz ({1/freq_hz:.1f}s period)")
        print(f"{'='*60}")
        print(f"  Amplitude: {amplitude_deg}°")
        print(f"  Duration: {duration_s}s")
        print(f"  Timestep: {dt*1000}ms")
        print(f"  Settle time: {settle_steps * dt}s")
        print(f"\n  Results:")
        print(f"    Position RMS error: {rms_pos*1e6:.2f} µrad (limit: {max_rms_pos*1e6:.0f})")
        print(f"    Velocity RMS error: {rms_vel*1e3:.2f} mrad/s (limit: {max_rms_vel*1e3:.0f})")
        print(f"    Max position error: {max_pos_error*1e6:.2f} µrad")
        print(f"    3-sigma violations: {ekf.innovation_violation_count}")
        print(f"\n  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {
        'freq_hz': freq_hz,
        'rms_pos': rms_pos,
        'rms_vel': rms_vel,
        'max_pos_error': max_pos_error,
        'violations': ekf.innovation_violation_count,
        'passed': passed,
        'times': np.array(times),
        'true_positions': np.array(true_positions),
        'est_positions': np.array(est_positions)
    }


def test_ekf_static_convergence(verbose: bool = True):
    """Test EKF converges to correct static position."""
    config = {
        'initial_state': np.zeros(10),  # Start at zero
        'pan_mass': 0.5,
        'tilt_mass': 0.25,
        'friction_coeff_az': 0.1,
        'friction_coeff_el': 0.1,
        'measurement_noise_std': [1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2],
        'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-1, 1e-1]
    }
    ekf = PointingStateEstimator(config)
    
    # True static position
    true_pos = np.deg2rad(5.0)  # 5 degrees
    dt = 0.001
    n_steps = 2000  # 2 seconds
    
    for _ in range(n_steps):
        measurements = {
            'theta_az_enc': true_pos,
            'theta_el_enc': true_pos,
            'theta_dot_az_gyro': 0.0,
            'theta_dot_el_gyro': 0.0,
            'nes_x_qpd': 0.0,
            'nes_y_qpd': 0.0
        }
        ekf.step(np.zeros(2), measurements, dt)
    
    fused = ekf.get_fused_state()
    pos_error_az = abs(fused['theta_az'] - true_pos)
    
    passed = pos_error_az < 100e-6  # 100 µrad
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EKF Static Convergence Test")
        print(f"{'='*60}")
        print(f"  True position: {np.rad2deg(true_pos):.2f}°")
        print(f"  Estimated position: {np.rad2deg(fused['theta_az']):.4f}°")
        print(f"  Position error: {pos_error_az*1e6:.2f} µrad")
        print(f"\n  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {'passed': passed, 'error': pos_error_az}


def test_q_r_tuning():
    """Print Q and R matrices to verify tuning."""
    config = {
        'initial_state': np.zeros(10),
        'measurement_noise_std': [1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2],
        'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-1, 1e-1]
    }
    ekf = PointingStateEstimator(config)
    
    print(f"\n{'='*60}")
    print(f"EKF Covariance Matrix Analysis")
    print(f"{'='*60}")
    
    print("\nQ diagonal (process noise variances):")
    q_diag = np.diag(ekf.Q)
    state_names = ['θ_Az', 'θ̇_Az', 'b_Az', 'θ_El', 'θ̇_El', 'b_El', 'φ_roll', 'φ̇_roll', 'd_Az', 'd_El']
    for i, name in enumerate(state_names):
        print(f"  {name:10s}: {q_diag[i]:.2e}")
    
    print("\nR diagonal (measurement noise variances):")
    r_diag = np.diag(ekf.R)
    meas_names = ['enc_Az', 'enc_El', 'gyro_Az', 'gyro_El', 'qpd_x', 'qpd_y']
    for i, name in enumerate(meas_names):
        print(f"  {name:10s}: {r_diag[i]:.2e}")
    
    print("\nP diagonal (initial state covariances):")
    p_diag = np.diag(ekf.P)
    for i, name in enumerate(state_names):
        print(f"  {name:10s}: {p_diag[i]:.2e}")
    
    # Check key ratios
    print("\n\nKey Tuning Ratios:")
    print(f"  Q[d_Az] / Q[θ_Az] = {q_diag[8]/q_diag[0]:.0e} (should be >> 1e6)")
    print(f"  R[enc] = {r_diag[0]:.2e} (should be ~1e-12 for ideal sensors)")


def run_all_tests():
    """Run all EKF qualification tests."""
    print("\n" + "="*70)
    print("EKF QUALIFICATION TEST SUITE")
    print("="*70)
    
    # Test Q/R tuning
    test_q_r_tuning()
    
    # Static convergence
    result_static = test_ekf_static_convergence()
    
    # Sine wave at various frequencies
    test_cases = [
        (0.05, 5.0, 40.0),   # 20-second period (very slow)
        (0.1, 5.0, 20.0),    # 10-second period (user's problem case)
        (0.5, 5.0, 10.0),    # 2-second period
        (1.0, 5.0, 5.0),     # 1-second period
    ]
    
    results = []
    for freq, amp, dur in test_cases:
        result = test_ekf_sine_wave(freq, amp, dur)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = result_static['passed'] and all(r['passed'] for r in results)
    
    print(f"  Static Convergence: {'✓ PASSED' if result_static['passed'] else '✗ FAILED'}")
    for r in results:
        print(f"  Sine {r['freq_hz']:.2f} Hz: {'✓ PASSED' if r['passed'] else '✗ FAILED'} "
              f"(RMS: {r['rms_pos']*1e6:.1f} µrad)")
    
    print(f"\n  OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
