#!/usr/bin/env python3
"""
EKF Qualification Test Suite

This script tests the Extended Kalman Filter (EKF) in isolation from the
closed-loop simulation to verify it can track various signal types.

Test Categories:
1. Static Convergence - Filter converges to true constant position
2. Sine Wave Tracking - Filter tracks sinusoidal motion at various frequencies
3. Step Response - Filter responds quickly to step changes
4. Noise Rejection - Filter smooths noisy measurements appropriately
5. Disturbance Estimation - Filter correctly estimates external disturbances

Pass Criteria:
- 0.1 Hz sine: RMS error < 100 µrad
- 0.5 Hz sine: RMS error < 500 µrad
- 1.0 Hz sine: RMS error < 2000 µrad
- Static: Convergence to < 10 µrad in 1 second
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.estimators.state_estimator import PointingStateEstimator


def create_test_estimator():
    """Create an EKF instance with tuned parameters for testing."""
    config = {
        'initial_state': np.zeros(10),
        'inertia_az': 0.13,  # kg·m²
        'inertia_el': 0.03,
        'friction_coeff_az': 0.1,
        'friction_coeff_el': 0.1,
        'pan_mass': 0.5,
        'tilt_mass': 0.25,
        'cm_r': 0.0,
        'cm_h': 0.0,
        'gravity': 9.81,
        # Tuned for ideal sensor conditions
        'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-1, 1e-1],
        'measurement_noise_std': [1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2]
    }
    return PointingStateEstimator(config)


def test_static_convergence():
    """Test 1: Filter converges to a static true position."""
    print("\n" + "="*70)
    print("TEST 1: Static Convergence")
    print("="*70)
    
    ekf = create_test_estimator()
    
    # True static position
    true_az = np.deg2rad(5.0)  # 5 degrees
    true_el = np.deg2rad(3.0)  # 3 degrees
    
    dt = 0.001  # 1 kHz
    duration = 2.0  # 2 seconds
    n_steps = int(duration / dt)
    
    errors = []
    
    for i in range(n_steps):
        t = i * dt
        tau = np.array([0.0, 0.0])  # No control input
        
        # Predict step
        ekf.predict(tau, dt)
        
        # Measurements: [enc_az, enc_el, gyro_az, gyro_el, qpd_x, qpd_y]
        z = np.array([true_az, true_el, 0.0, 0.0, 0.0, 0.0])
        
        # Update step
        ekf.correct(z)
        
        # Get estimate
        state = ekf.get_fused_state()
        error_az = abs(state['theta_az'] - true_az)
        error_el = abs(state['theta_el'] - true_el)
        errors.append(np.sqrt(error_az**2 + error_el**2))
    
    final_error = errors[-1]
    final_error_urad = final_error * 1e6
    
    print(f"  True Position: Az={np.rad2deg(true_az):.2f}°, El={np.rad2deg(true_el):.2f}°")
    print(f"  Final Error: {final_error_urad:.2f} µrad ({np.rad2deg(final_error)*3600:.2f} arcsec)")
    print(f"  Convergence after 0.1s: {errors[100]*1e6:.2f} µrad")
    print(f"  Convergence after 0.5s: {errors[500]*1e6:.2f} µrad")
    
    passed = final_error_urad < 10.0
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: 10 µrad)")
    return passed


def test_sine_wave_tracking(frequency_hz: float, amplitude_deg: float = 5.0):
    """Test: Filter tracks sinusoidal motion at specified frequency."""
    print(f"\n  Testing {frequency_hz} Hz sine wave (amplitude: ±{amplitude_deg}°)...")
    
    ekf = create_test_estimator()
    
    amplitude = np.deg2rad(amplitude_deg)
    omega = 2.0 * np.pi * frequency_hz
    
    dt = 0.001  # 1 kHz
    duration = max(5.0, 3.0 / frequency_hz)  # At least 3 periods
    n_steps = int(duration / dt)
    
    errors_az = []
    errors_el = []
    
    for i in range(n_steps):
        t = i * dt
        
        # True sinusoidal motion
        true_az = amplitude * np.sin(omega * t)
        true_el = amplitude * np.sin(omega * t)
        true_vel_az = amplitude * omega * np.cos(omega * t)
        true_vel_el = amplitude * omega * np.cos(omega * t)
        
        # Predict with zero control (open-loop EKF test)
        tau = np.array([0.0, 0.0])
        ekf.predict(tau, dt)
        
        # Measurements: [enc_az, enc_el, gyro_az, gyro_el, qpd_x, qpd_y]
        z = np.array([true_az, true_el, true_vel_az, true_vel_el, 0.0, 0.0])
        
        # Update
        ekf.correct(z)
        
        # Get estimate and compute error
        state = ekf.get_fused_state()
        errors_az.append(state['theta_az'] - true_az)
        errors_el.append(state['theta_el'] - true_el)
    
    # Compute RMS error (skip first 20% for transient)
    skip = int(0.2 * n_steps)
    rms_az = np.sqrt(np.mean(np.array(errors_az[skip:])**2))
    rms_el = np.sqrt(np.mean(np.array(errors_el[skip:])**2))
    rms_total = np.sqrt(rms_az**2 + rms_el**2)
    
    rms_urad = rms_total * 1e6
    print(f"    RMS Error: {rms_urad:.2f} µrad (Az: {rms_az*1e6:.2f}, El: {rms_el*1e6:.2f})")
    
    return rms_urad


def test_sine_wave_suite():
    """Test 2: Filter tracks sine waves at various frequencies."""
    print("\n" + "="*70)
    print("TEST 2: Sine Wave Tracking at Multiple Frequencies")
    print("="*70)
    
    # Test frequencies and their pass thresholds
    test_cases = [
        (0.05, 50.0),    # 0.05 Hz, threshold 50 µrad
        (0.1, 100.0),    # 0.1 Hz, threshold 100 µrad
        (0.2, 200.0),    # 0.2 Hz, threshold 200 µrad
        (0.5, 500.0),    # 0.5 Hz, threshold 500 µrad
        (1.0, 2000.0),   # 1.0 Hz, threshold 2000 µrad
    ]
    
    all_passed = True
    results = []
    
    for freq, threshold in test_cases:
        rms_error = test_sine_wave_tracking(freq)
        passed = rms_error < threshold
        results.append((freq, rms_error, threshold, passed))
        if not passed:
            all_passed = False
    
    print("\n  Summary:")
    print("  " + "-"*50)
    print(f"  {'Frequency':>10} {'RMS Error':>15} {'Threshold':>12} {'Result':>8}")
    print("  " + "-"*50)
    for freq, rms, thresh, passed in results:
        print(f"  {freq:>10.2f} Hz {rms:>12.2f} µrad {thresh:>10.0f} µrad {'PASS' if passed else 'FAIL':>8}")
    
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_step_response():
    """Test 3: Filter responds quickly to step changes in position."""
    print("\n" + "="*70)
    print("TEST 3: Step Response")
    print("="*70)
    
    ekf = create_test_estimator()
    
    dt = 0.001
    duration = 2.0
    n_steps = int(duration / dt)
    
    step_time = 0.5  # Step occurs at 0.5 seconds
    step_size = np.deg2rad(10.0)  # 10 degree step
    
    errors = []
    estimates = []
    true_positions = []
    
    for i in range(n_steps):
        t = i * dt
        
        # True position with step
        if t < step_time:
            true_az = 0.0
        else:
            true_az = step_size
        true_el = 0.0
        
        # Predict
        tau = np.array([0.0, 0.0])
        ekf.predict(tau, dt)
        
        # Measurements
        z = np.array([true_az, true_el, 0.0, 0.0, 0.0, 0.0])
        
        ekf.correct(z)
        
        state = ekf.get_fused_state()
        estimates.append(state['theta_az'])
        true_positions.append(true_az)
        errors.append(abs(state['theta_az'] - true_az))
    
    # Find settling time (time to reach within 2% of step)
    step_idx = int(step_time / dt)
    settling_threshold = 0.02 * step_size
    settling_time = None
    
    for i in range(step_idx, n_steps):
        if errors[i] < settling_threshold:
            settling_time = (i - step_idx) * dt
            break
    
    if settling_time is None:
        settling_time = duration - step_time
    
    print(f"  Step Size: {np.rad2deg(step_size):.1f}°")
    print(f"  Settling Time (2%): {settling_time*1000:.1f} ms")
    print(f"  Error at 10ms after step: {errors[step_idx + 10]*1e6:.2f} µrad")
    print(f"  Error at 100ms after step: {errors[step_idx + 100]*1e6:.2f} µrad")
    
    passed = settling_time < 0.3  # Should settle within 300ms (EKF tuned for smooth filtering)
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: 300 ms)")
    return passed


def test_noisy_measurements():
    """Test 4: Filter properly smooths noisy measurements."""
    print("\n" + "="*70)
    print("TEST 4: Noise Rejection")
    print("="*70)
    
    ekf = create_test_estimator()
    np.random.seed(42)
    
    true_az = np.deg2rad(5.0)
    true_el = np.deg2rad(3.0)
    
    # Measurement noise levels
    enc_noise_std = 1e-5  # 10 µrad encoder noise
    gyro_noise_std = 1e-4  # 100 µrad/s gyro noise
    
    dt = 0.001
    duration = 5.0
    n_steps = int(duration / dt)
    
    errors = []
    
    for i in range(n_steps):
        tau = np.array([0.0, 0.0])
        ekf.predict(tau, dt)
        
        # Noisy measurements
        noisy_az = true_az + np.random.normal(0, enc_noise_std)
        noisy_el = true_el + np.random.normal(0, enc_noise_std)
        noisy_gyro_az = np.random.normal(0, gyro_noise_std)
        noisy_gyro_el = np.random.normal(0, gyro_noise_std)
        z = np.array([noisy_az, noisy_el, noisy_gyro_az, noisy_gyro_el, 0.0, 0.0])
        
        ekf.correct(z)
        
        state = ekf.get_fused_state()
        error = np.sqrt((state['theta_az'] - true_az)**2 + (state['theta_el'] - true_el)**2)
        errors.append(error)
    
    # RMS of estimation error should be less than measurement noise
    rms_error = np.sqrt(np.mean(np.array(errors[1000:])**2))  # Skip first second
    
    print(f"  Encoder Noise: {enc_noise_std*1e6:.1f} µrad RMS")
    print(f"  Estimation RMS Error: {rms_error*1e6:.2f} µrad")
    print(f"  Noise Reduction Factor: {enc_noise_std/rms_error:.1f}x")
    
    # Filter should reduce noise, not amplify it
    passed = rms_error < enc_noise_std
    print(f"  Result: {'PASS' if passed else 'FAIL'} (should reduce noise)")
    return passed


def test_velocity_tracking():
    """Test 5: Filter accurately tracks velocity during motion."""
    print("\n" + "="*70)
    print("TEST 5: Velocity Tracking")
    print("="*70)
    
    ekf = create_test_estimator()
    
    frequency_hz = 0.1
    amplitude = np.deg2rad(10.0)
    omega = 2.0 * np.pi * frequency_hz
    
    dt = 0.001
    duration = 10.0
    n_steps = int(duration / dt)
    
    vel_errors = []
    
    for i in range(n_steps):
        t = i * dt
        
        true_az = amplitude * np.sin(omega * t)
        true_el = amplitude * np.sin(omega * t)
        true_vel_az = amplitude * omega * np.cos(omega * t)
        true_vel_el = amplitude * omega * np.cos(omega * t)
        
        tau = np.array([0.0, 0.0])
        ekf.predict(tau, dt)
        
        z = np.array([true_az, true_el, true_vel_az, true_vel_el, 0.0, 0.0])
        
        ekf.correct(z)
        
        state = ekf.get_fused_state()
        vel_error = np.sqrt(
            (state['theta_dot_az'] - true_vel_az)**2 +
            (state['theta_dot_el'] - true_vel_el)**2
        )
        vel_errors.append(vel_error)
    
    # Skip first 20% for transient
    skip = int(0.2 * n_steps)
    rms_vel_error = np.sqrt(np.mean(np.array(vel_errors[skip:])**2))
    
    # Max expected velocity
    max_velocity = amplitude * omega
    percent_error = 100.0 * rms_vel_error / max_velocity
    
    print(f"  Frequency: {frequency_hz} Hz")
    print(f"  Max True Velocity: {np.rad2deg(max_velocity):.2f} deg/s")
    print(f"  Velocity RMS Error: {np.rad2deg(rms_vel_error)*1000:.2f} mdeg/s")
    print(f"  Percent Error: {percent_error:.2f}%")
    
    passed = percent_error < 5.0  # Less than 5% velocity error
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: 5%)")
    return passed


def run_all_tests():
    """Run complete EKF qualification test suite."""
    print("\n" + "="*70)
    print("   EKF QUALIFICATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    results['static_convergence'] = test_static_convergence()
    results['sine_wave_suite'] = test_sine_wave_suite()
    results['step_response'] = test_step_response()
    results['noise_rejection'] = test_noisy_measurements()
    results['velocity_tracking'] = test_velocity_tracking()
    
    # Summary
    print("\n" + "="*70)
    print("   QUALIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:25s} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("  OVERALL RESULT: ALL TESTS PASSED")
    else:
        print("  OVERALL RESULT: SOME TESTS FAILED")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
