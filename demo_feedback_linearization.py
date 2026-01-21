#!/usr/bin/env python3
"""
Demonstration script for Feedback Linearization Controller

This script showcases the complete signal flow architecture:
    Sensors → EKF Estimator → FL Controller → Actuators

The Feedback Linearization controller:
1. Receives filtered state from the EKF (position, velocity, disturbances)
2. Cancels nonlinear dynamics using M(q), C(q,dq), G(q) terms
3. Compensates for estimated disturbances
4. Achieves linear closed-loop behavior with high gains

Signal Flow Architecture:
------------------------
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                              │
│  Encoders (θ_az, θ_el) + Gyros (ω_az, ω_el) + QPD          │
└────────────────┬────────────────────────────────────────────┘
                 │ Raw noisy measurements
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   ESTIMATOR LAYER (EKF)                      │
│  Fuses sensor data → Filtered state + disturbance estimate  │
└────────────────┬────────────────────────────────────────────┘
                 │ state_estimate = {
                 │   'theta_az', 'theta_el',
                 │   'theta_dot_az', 'theta_dot_el',
                 │   'dist_az', 'dist_el'
                 │ }
                 ▼
┌─────────────────────────────────────────────────────────────┐
│          CONTROLLER LAYER (Feedback Linearization)           │
│  tau = M(q)*[ddq_ref + Kd*e_dot + Kp*e] + C*dq + G - d_hat │
└────────────────┬────────────────────────────────────────────┘
                 │ Torque commands [N·m]
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   ACTUATOR LAYER (Motors)                    │
└─────────────────────────────────────────────────────────────┘

Usage:
    python demo_feedback_linearization.py
"""

import numpy as np
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner
)


def run_comparison_study():
    """
    Run a comparison between PID and Feedback Linearization controllers.
    """
    print("\n" + "=" * 80)
    print("LASERCOM DIGITAL TWIN: CONTROLLER COMPARISON STUDY")
    print("=" * 80)
    print("\nThis demonstration compares two control strategies:")
    print("  1. Standard PID Controller (baseline)")
    print("  2. Feedback Linearization Controller (advanced)")
    print("\nBoth controllers use the same sensor and estimator architecture.")
    print("=" * 80 + "\n")
    
    # Common parameters
    target_az_deg = 5.0
    target_el_deg = 3.0
    duration = 2.5
    
    print(f"Test Conditions:")
    print(f"  - Target: Az={target_az_deg:.1f}°, El={target_el_deg:.1f}°")
    print(f"  - Duration: {duration:.1f} seconds")
    print(f"  - Initial position: [0°, 0°]")
    print(f"  - MAST VIBRATION: Enabled (Starts at t=1.0s)")
    print(f"  - Controller: Corrected PID gains (double-integrator design)")
    print(f"    • Pan:  Kp=3.257, Ki=10.232, Kd=0.147")
    print(f"    • Tilt: Kp=0.660, Ki=2.074,  Kd=0.030")
    print(f"  - Large slew maneuver to test tracking performance\n")
    
    # =========================================================================
    # Test 1: Standard PID Controller
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: STANDARD PID CONTROLLER")
    print("-" * 80)
    
    config_pid = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.010,
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=np.deg2rad(target_az_deg),
        target_el=np.deg2rad(target_el_deg),
        target_enabled=True,
        use_feedback_linearization=False,  # PID mode
        enable_visualization=False,
        real_time_factor=0.0,
        vibration_enabled=True,
        vibration_config={
            'start_time': 1.0,
            'frequency_hz': 40.0,
            'amplitude_rad': 100e-6,  # 100 µrad jitter
            'harmonics': [(1.0, 1.0), (2.1, 0.3)]
        },
        coarse_controller_config={
            # Corrected gains from double-integrator design (FIXED derivative calculation)
            # These gains are now correct after fixing the derivative term bug
            'kp': [3.257, 0.660],    # Per-axis: [Pan, Tilt]
            'ki': [10.232, 2.074],   # Designed for 5 Hz bandwidth
            'kd': [0.1046599, 0.021709],  # Corrected Kd values (40% higher than before)
            'anti_windup_gain': 1.0,
            'tau_rate_limit': 50.0,
            'enable_derivative': True  # Now works correctly with fixed implementation
        }
    )
    
    print("Initializing PID controller simulation...")
    runner_pid = DigitalTwinRunner(config_pid)
    print("Running simulation...\n")

    #results_pid = runner_pid.run_simulation(duration=duration)
    
    # =========================================================================
    # Test 2: Feedback Linearization Controller
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: FEEDBACK LINEARIZATION CONTROLLER")
    print("-" * 80)
    
    config_fl = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.010,
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=np.deg2rad(target_az_deg),
        target_el=np.deg2rad(target_el_deg),
        target_enabled=True,
        use_feedback_linearization=True,  # FL mode
        enable_visualization=False,
        real_time_factor=0.0,
        vibration_enabled=True,
        vibration_config={
            'start_time': 1.0,
            'frequency_hz': 40.0,
            'amplitude_rad': 100e-6,
            'harmonics': [(1.0, 1.0), (2.1, 0.3)]
        },
        feedback_linearization_config={
            # Gains tuned for tau_max=1.0 Nm and inertia~0.003 kg·m²
            # Natural frequency ωn = sqrt(Kp*M) ≈ sqrt(50/0.003) ≈ 130 rad/s ≈ 20 Hz
            # Damping ratio ζ = Kd/(2*sqrt(Kp*M)) ≈ 5/(2*sqrt(50*0.003)) ≈ 0.65
            'kp': [50.0, 50.0],    # Reduced from 150 to avoid saturation
            'kd': [5.0, 5.0],      # Reduced proportionally for critical damping
            'ki': [5.0, 5.0],      # Integral for steady-state error rejection
            'enable_integral': True,  # Enable robust tracking
            'tau_max': [1.0, 1.0],
            'tau_min': [-1.0, -1.0],
            # CRITICAL: Friction compensation (must match plant friction!)
            'friction_az': 0.1,    # N·m·s/rad - match plant default
            'friction_el': 0.1     # N·m·s/rad - match plant default
        },
        dynamics_config={
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.0,
            'cm_h': 0.0,
            'gravity': 9.81,
            'friction_az': 0.1,    # Explicitly set for clarity
            'friction_el': 0.1     # Explicitly set for clarity
        }
    )
    
    print("Initializing Feedback Linearization controller simulation...")
    runner_fl = DigitalTwinRunner(config_fl)
    print("Running simulation...\n")
    results_fl = runner_fl.run_simulation(duration=duration)
    
    # =========================================================================
    # Results Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Extract tracking performance metrics
    def compute_tracking_metrics(results, target_az_rad, target_el_rad):
        """Compute step response characteristics"""
        t = results['log_arrays']['time']
        q_az = results['log_arrays']['q_az']
        q_el = results['log_arrays']['q_el']
        
        # Azimuth tracking
        error_az = q_az - target_az_rad
        settling_criterion_az = 0.02 * abs(target_az_rad)  # 2% of final value
        settled_az = np.where(np.abs(error_az) < settling_criterion_az)[0]
        settling_time_az = t[settled_az[0]] if len(settled_az) > 0 else t[-1]
        overshoot_az = 100.0 * (np.max(q_az) - target_az_rad) / target_az_rad if target_az_rad != 0 else 0.0
        steady_state_error_az = np.mean(error_az[-100:])  # Last 100 samples
        
        # Elevation tracking
        error_el = q_el - target_el_rad
        settling_criterion_el = 0.02 * abs(target_el_rad)
        settled_el = np.where(np.abs(error_el) < settling_criterion_el)[0]
        settling_time_el = t[settled_el[0]] if len(settled_el) > 0 else t[-1]
        overshoot_el = 100.0 * (np.max(q_el) - target_el_rad) / target_el_rad if target_el_rad != 0 else 0.0
        steady_state_error_el = np.mean(error_el[-100:])
        
        return {
            'settling_time_az': settling_time_az,
            'settling_time_el': settling_time_el,
            'overshoot_az': overshoot_az,
            'overshoot_el': overshoot_el,
            'ss_error_az': steady_state_error_az,
            'ss_error_el': steady_state_error_el
        }
    
    target_az_rad = np.deg2rad(target_az_deg)
    target_el_rad = np.deg2rad(target_el_deg)
    
    metrics_pid = compute_tracking_metrics(results_pid, target_az_rad, target_el_rad)
    metrics_fl = compute_tracking_metrics(results_fl, target_az_rad, target_el_rad)
    
    print(f"\n{'Metric':<40} {'PID':<20} {'FL':<20} {'Improvement':<15}")
    print("-" * 95)
    
    # Settling Time
    print(f"{'Settling Time - Az (s)':<40} {metrics_pid['settling_time_az']:<20.3f} {metrics_fl['settling_time_az']:<20.3f} {(metrics_pid['settling_time_az']-metrics_fl['settling_time_az'])*1000:<15.0f} ms")
    print(f"{'Settling Time - El (s)':<40} {metrics_pid['settling_time_el']:<20.3f} {metrics_fl['settling_time_el']:<20.3f} {(metrics_pid['settling_time_el']-metrics_fl['settling_time_el'])*1000:<15.0f} ms")
    
    # Overshoot
    print(f"{'Overshoot - Az (%)':<40} {metrics_pid['overshoot_az']:<20.2f} {metrics_fl['overshoot_az']:<20.2f} {metrics_pid['overshoot_az']-metrics_fl['overshoot_az']:<15.2f}%")
    print(f"{'Overshoot - El (%)':<40} {metrics_pid['overshoot_el']:<20.2f} {metrics_fl['overshoot_el']:<20.2f} {metrics_pid['overshoot_el']-metrics_fl['overshoot_el']:<15.2f}%")
    
    # Steady-State Error
    print(f"{'Steady-State Error - Az (µrad)':<40} {np.rad2deg(metrics_pid['ss_error_az'])*3600:<20.2f} {np.rad2deg(metrics_fl['ss_error_az'])*3600:<20.2f}")
    print(f"{'Steady-State Error - El (µrad)':<40} {np.rad2deg(metrics_pid['ss_error_el'])*3600:<20.2f} {np.rad2deg(metrics_fl['ss_error_el'])*3600:<20.2f}")
    
    # LOS Error
    los_rms_pid = results_pid['los_error_rms'] * 1e6
    los_rms_fl = results_fl['los_error_rms'] * 1e6
    improvement_los = ((los_rms_pid - los_rms_fl) / los_rms_pid) * 100
    print(f"{'LOS Error RMS (µrad)':<40} {los_rms_pid:<20.2f} {los_rms_fl:<20.2f} {improvement_los:<15.1f}%")
    
    # Final LOS Error
    los_final_pid = results_pid['los_error_final'] * 1e6
    los_final_fl = results_fl['los_error_final'] * 1e6
    improvement_final = ((los_final_pid - los_final_fl) / los_final_pid) * 100 if los_final_pid > 0 else 0
    print(f"{'LOS Error Final (µrad)':<40} {los_final_pid:<20.2f} {los_final_fl:<20.2f} {improvement_final:<15.1f}%")
    
    # Torque effort
    torque_pid = np.sqrt(results_pid['torque_rms_az']**2 + results_pid['torque_rms_el']**2)
    torque_fl = np.sqrt(results_fl['torque_rms_az']**2 + results_fl['torque_rms_el']**2)
    torque_change = ((torque_fl - torque_pid) / torque_pid) * 100
    print(f"{'Total Torque RMS (N·m)':<40} {torque_pid:<20.3f} {torque_fl:<20.3f} {torque_change:+<15.1f}%")
    
    # FSM Saturation
    print(f"{'FSM Saturation (%)':<40} {results_pid['fsm_saturation_percent']:<20.1f} {results_fl['fsm_saturation_percent']:<20.1f}")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    print("\n1. FEEDBACK LINEARIZATION ADVANTAGES:")
    print("   ✓ Cancels nonlinear dynamics (M, C, G terms)")
    print("   ✓ Compensates for disturbances using EKF estimates")
    print("   ✓ Allows higher control gains without instability")
    print("   ✓ Better tracking performance during large maneuvers")
    
    print("\n2. SIGNAL FLOW ARCHITECTURE:")
    print("   Sensors → EKF (state fusion) → Controller → Actuators")
    print("   - Encoders: Absolute position measurement")
    print("   - Gyros: Angular velocity measurement")
    print("   - EKF: Fuses noisy measurements → clean state estimate")
    print("   - Controller: Uses filtered state for control law")
    
    print("\n3. IMPLEMENTATION DETAILS:")
    print("   - EKF provides: theta, theta_dot, dist_az, dist_el")
    print("   - FL Controller computes: M(q), C(q,dq), G(q) from dynamics model")
    print("   - Control law: tau = M*[ddq_ref + Kd*e_dot + Kp*e] + C*dq + G - d_hat")
    print("   - Result: Linear closed-loop behavior despite nonlinear plant")
    
    print("\n" + "=" * 80 + "\n")
    
    return results_pid, results_fl


if __name__ == "__main__":
    try:
        results_pid, results_fl = run_comparison_study()
        print("✓ Demonstration completed successfully!")
        print("\nTo visualize results, access:")
        print("  - results_pid['log_data'] for PID time-series")
        print("  - results_fl['log_data'] for FL time-series")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
