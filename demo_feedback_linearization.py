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
    target_az_deg = 15.0
    target_el_deg = 40.0
    duration = 5.0
    
    print(f"Test Conditions:")
    print(f"  - Target: Az={target_az_deg:.1f}°, El={target_el_deg:.1f}°")
    print(f"  - Duration: {duration:.1f} seconds")
    print(f"  - Initial position: [0°, 0°]")
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
        coarse_controller_config={
            'kp': 50.0,
            'ki': 5.0,
            'kd': 2.0,
            'anti_windup_gain': 1.0,
            'tau_rate_limit': 50.0
        }
    )
    
    print("Initializing PID controller simulation...")
    runner_pid = DigitalTwinRunner(config_pid)
    print("Running simulation...\n")
    results_pid = runner_pid.run_simulation(duration=duration)
    
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
        feedback_linearization_config={
            'kp': [150.0, 150.0],  # Higher gains stable due to linearization
            'kd': [30.0, 30.0],
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0]
        },
        dynamics_config={
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.02,
            'cm_h': 0.005,
            'gravity': 9.81
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
    
    print(f"\n{'Metric':<40} {'PID':<20} {'FL':<20} {'Improvement':<15}")
    print("-" * 95)
    
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
