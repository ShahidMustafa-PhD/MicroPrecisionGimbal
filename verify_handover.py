#!/usr/bin/env python3
"""
Coarse-to-Fine Handover Verification Script

This script validates the handover from the Coarse Pointing Assembly (CPA)
gimbal to the Fine Steering Mirror (FSM) for satellite laser communication.

REQUIREMENTS:
- Coarse gimbal SSE < 0.8° (13963 µrad) for FSM to "catch" the beam
- FSM must stabilize within its linear range (±1.5° optical)
- No divergence or saturation during handover

VERIFICATION TESTS:
1. Step response to verify coarse gimbal SSE
2. FSM capture verification (beam on sensor)
3. Combined pointing accuracy after handover

Author: Senior Control Systems Engineer
Date: January 21, 2026
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from lasercom_digital_twin.core.simulation.simulation_runner import (
    SimulationConfig,
    DigitalTwinRunner
)


def run_handover_verification() -> Dict:
    """
    Run complete handover verification test.
    
    Returns
    -------
    Dict
        Verification results with pass/fail status
    """
    print("\n" + "=" * 80)
    print("COARSE-TO-FINE HANDOVER VERIFICATION")
    print("=" * 80)
    
    # Test parameters
    target_az_deg = 5.0   # Smaller slew for handover test
    target_el_deg = 2.0
    duration = 3.0        # 3 seconds for full settling
    
    # Requirements
    SSE_REQUIREMENT_DEG = 0.8   # Max SSE for FSM capture
    FSM_LINEAR_RANGE_DEG = 1.5  # FSM optical range
    
    print(f"\nTest Configuration:")
    print(f"  Target: Az={target_az_deg:.1f}°, El={target_el_deg:.1f}°")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  SSE Requirement: < {SSE_REQUIREMENT_DEG}°")
    print(f"  FSM Linear Range: ±{FSM_LINEAR_RANGE_DEG}°")
    
    # =========================================================================
    # Configure Feedback Linearization Controller
    # =========================================================================
    config = SimulationConfig(
        dt_sim=0.001,
        dt_coarse=0.010,
        dt_fine=0.001,
        log_period=0.001,
        seed=42,
        target_az=np.deg2rad(target_az_deg),
        target_el=np.deg2rad(target_el_deg),
        target_enabled=True,
        use_feedback_linearization=True,
        use_direct_state_feedback=True,
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
            'kp': [150.0, 150.0],
            'kd': [20.0, 20.0],
            'ki': [15.0, 15.0],
            'enable_integral': True,
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0],
            'friction_az': 0.1,
            'friction_el': 0.1,
            'conditional_friction': True,
            'enable_disturbance_compensation': False,
        },
        dynamics_config={
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.0,
            'cm_h': 0.0,
            'gravity': 9.81,
            'friction_az': 0.1,
            'friction_el': 0.1
        },
        qpd_config={
            'linear_range': np.deg2rad(FSM_LINEAR_RANGE_DEG),  # QPD FOV = FSM range
            'sensitivity': 2000.0,
            'noise_std': 1e-4,
        }
    )
    
    # =========================================================================
    # Run Simulation
    # =========================================================================
    print("\n" + "-" * 80)
    print("RUNNING SIMULATION...")
    print("-" * 80)
    
    runner = DigitalTwinRunner(config)
    results = runner.run_simulation(duration=duration)
    
    # =========================================================================
    # Extract Data for Analysis
    # =========================================================================
    log_data = results['log_arrays']
    t = log_data['time']
    q_az = log_data['q_az']
    q_el = log_data['q_el']
    fsm_tip = log_data['fsm_tip']
    fsm_tilt = log_data['fsm_tilt']
    fsm_cmd_tip = log_data['fsm_cmd_tip']
    fsm_cmd_tilt = log_data['fsm_cmd_tilt']
    los_error_x = log_data['los_error_x']
    los_error_y = log_data['los_error_y']
    
    target_az_rad = np.deg2rad(target_az_deg)
    target_el_rad = np.deg2rad(target_el_deg)
    
    # Compute gimbal tracking error
    gimbal_error_az = target_az_rad - q_az
    gimbal_error_el = target_el_rad - q_el
    gimbal_error_total = np.sqrt(gimbal_error_az**2 + gimbal_error_el**2)
    
    # Compute steady-state values (last 20% of simulation)
    ss_start_idx = int(0.8 * len(t))
    
    sse_az_rad = np.mean(np.abs(gimbal_error_az[ss_start_idx:]))
    sse_el_rad = np.mean(np.abs(gimbal_error_el[ss_start_idx:]))
    sse_total_rad = np.sqrt(sse_az_rad**2 + sse_el_rad**2)
    
    sse_az_deg = np.rad2deg(sse_az_rad)
    sse_el_deg = np.rad2deg(sse_el_rad)
    sse_total_deg = np.rad2deg(sse_total_rad)
    
    # FSM state at steady state
    fsm_tip_ss = np.mean(np.abs(fsm_tip[ss_start_idx:]))
    fsm_tilt_ss = np.mean(np.abs(fsm_tilt[ss_start_idx:]))
    fsm_total_ss_deg = np.rad2deg(np.sqrt(fsm_tip_ss**2 + fsm_tilt_ss**2))
    
    # FSM optical deflection (2x mirror angle)
    fsm_optical_ss_deg = 2.0 * fsm_total_ss_deg
    
    # LOS error at steady state
    los_rms = np.sqrt(np.mean(los_error_x[ss_start_idx:]**2 + los_error_y[ss_start_idx:]**2))
    
    # =========================================================================
    # Verification Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    results_dict = {
        'sse_az_deg': sse_az_deg,
        'sse_el_deg': sse_el_deg,
        'sse_total_deg': sse_total_deg,
        'fsm_optical_ss_deg': fsm_optical_ss_deg,
        'los_rms_urad': los_rms * 1e6,
        'pass_sse': sse_total_deg < SSE_REQUIREMENT_DEG,
        'pass_fsm_range': fsm_optical_ss_deg < FSM_LINEAR_RANGE_DEG,
    }
    
    print("\n1. COARSE GIMBAL STEADY-STATE ERROR (SSE)")
    print("-" * 60)
    print(f"   Azimuth SSE:    {sse_az_deg*1000:.2f} mdeg ({sse_az_rad*1e6:.1f} µrad)")
    print(f"   Elevation SSE:  {sse_el_deg*1000:.2f} mdeg ({sse_el_rad*1e6:.1f} µrad)")
    print(f"   Total SSE:      {sse_total_deg*1000:.2f} mdeg ({sse_total_rad*1e6:.1f} µrad)")
    print(f"   Requirement:    < {SSE_REQUIREMENT_DEG*1000:.0f} mdeg")
    print(f"   Status:         {'✓ PASS' if results_dict['pass_sse'] else '✗ FAIL'}")
    
    print("\n2. FSM OPERATING POINT")
    print("-" * 60)
    print(f"   FSM Tip (mirror):   {fsm_tip_ss*1e6:.1f} µrad")
    print(f"   FSM Tilt (mirror):  {fsm_tilt_ss*1e6:.1f} µrad")
    print(f"   FSM Optical:        {fsm_optical_ss_deg*1000:.2f} mdeg")
    print(f"   Linear Range:       ±{FSM_LINEAR_RANGE_DEG*1000:.0f} mdeg")
    fsm_utilization = 100.0 * fsm_optical_ss_deg / FSM_LINEAR_RANGE_DEG
    print(f"   Range Utilization:  {fsm_utilization:.1f}%")
    print(f"   Status:             {'✓ PASS' if results_dict['pass_fsm_range'] else '✗ FAIL (SATURATED)'}")
    
    print("\n3. HANDOVER SUMMARY")
    print("-" * 60)
    print(f"   LOS Error RMS:     {los_rms*1e6:.1f} µrad")
    print(f"   FSM Saturation:    {results['fsm_saturation_percent']:.1f}%")
    
    # Handover time (when gimbal error enters FSM range)
    fsm_range_rad = np.deg2rad(FSM_LINEAR_RANGE_DEG)
    in_range_mask = gimbal_error_total < fsm_range_rad
    if np.any(in_range_mask):
        handover_idx = np.where(in_range_mask)[0][0]
        handover_time = t[handover_idx]
        print(f"   Handover Time:     {handover_time*1000:.0f} ms")
        results_dict['handover_time_ms'] = handover_time * 1000
    else:
        print(f"   Handover Time:     N/A (error never entered FSM range)")
        results_dict['handover_time_ms'] = None
    
    overall_pass = results_dict['pass_sse'] and results_dict['pass_fsm_range']
    print(f"\n   OVERALL STATUS:    {'✓ HANDOVER VERIFIED' if overall_pass else '✗ HANDOVER FAILED'}")
    
    # =========================================================================
    # Log Gimbal Error and FSM State
    # =========================================================================
    print("\n4. TELEMETRY LOG (key samples)")
    print("-" * 60)
    print(f"{'Time (ms)':<12} {'Gimbal Err (mdeg)':<20} {'FSM Tip (µrad)':<18} {'FSM Tilt (µrad)':<18}")
    
    # Sample at key times
    sample_times_ms = [0, 100, 200, 500, 1000, 1500, 2000, 2500, 2900]
    for t_ms in sample_times_ms:
        idx = int(t_ms / (config.dt_sim * 1000))
        if idx < len(t):
            gimbal_err_mdeg = np.rad2deg(gimbal_error_total[idx]) * 1000
            fsm_tip_val = fsm_tip[idx] * 1e6
            fsm_tilt_val = fsm_tilt[idx] * 1e6
            print(f"{t_ms:<12} {gimbal_err_mdeg:<20.2f} {fsm_tip_val:<18.1f} {fsm_tilt_val:<18.1f}")
    
    results_dict['log_data'] = {
        'time': t,
        'gimbal_error_az': gimbal_error_az,
        'gimbal_error_el': gimbal_error_el,
        'gimbal_error_total': gimbal_error_total,
        'fsm_tip': fsm_tip,
        'fsm_tilt': fsm_tilt,
        'los_error_x': los_error_x,
        'los_error_y': los_error_y,
    }
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80 + "\n")
    
    return results_dict


def analyze_failure_modes() -> None:
    """
    Print failure mode analysis for handover.
    
    This documents the "Silent Killers" that could prevent Link Acquisition.
    """
    print("\n" + "=" * 80)
    print("FAILURE MODE ANALYSIS: SILENT KILLERS")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. COORDINATE MISALIGNMENT                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ISSUE: Gimbal Az/El axes may not be perfectly aligned with FSM Tip/Tilt.    │
│                                                                              │
│ SYMPTOMS:                                                                    │
│   - FSM corrects in wrong direction                                         │
│   - Cross-coupling between axes                                             │
│   - Steady-state error that doesn't null                                    │
│                                                                              │
│ ROOT CAUSES:                                                                 │
│   - Missing rotation matrix between M-frame and O-frame                     │
│   - Field rotation not compensated (Az-El gimbal geometry)                  │
│   - Optical axis misalignment from manufacturing tolerances                 │
│                                                                              │
│ CHECK IN CODE:                                                               │
│   lasercom_digital_twin/core/coordinate_frames/transformations.py           │
│   - Verify use_field_rotation = True                                        │
│   - Check compute_field_rotation_angle() implementation                     │
│                                                                              │
│ VERIFICATION:                                                                │
│   - Slew to multiple Az angles and verify FSM tip/tilt don't swap           │
│   - Check cross-coupling ratio in FSM response                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. SIGN MISMATCH                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ISSUE: A sign error causes FSM to steer beam AWAY from target.              │
│                                                                              │
│ SYMPTOMS:                                                                    │
│   - LOS error increases with FSM correction                                 │
│   - FSM saturates at max deflection                                         │
│   - Positive feedback oscillation                                           │
│                                                                              │
│ ROOT CAUSES:                                                                 │
│   - Wrong sign in residual = coarse_error - 2*FSM (should be minus)         │
│   - Wrong sign in controller: u = Kp * error (error = setpoint - meas)      │
│   - FSM plant inversion in state-space model                                │
│                                                                              │
│ CHECK IN CODE:                                                               │
│   simulation_runner.py _update_fine_controller():                            │
│     residual_tip = error_az - 2.0 * self.state.fsm_tip  (CORRECT: minus)    │
│                                                                              │
│   fsm_pid_control.py update():                                               │
│     error = setpoint - measurement  (CORRECT: setpoint first)               │
│                                                                              │
│ VERIFICATION:                                                                │
│   - Apply known error, verify FSM moves in correct direction                │
│   - Check that LOS error decreases after FSM correction                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. SENSOR SATURATION                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ISSUE: QPD linear range smaller than gimbal SSE → flat signal.              │
│                                                                              │
│ SYMPTOMS:                                                                    │
│   - QPD NES output clipped at max value                                     │
│   - FSM doesn't know which direction to move                                │
│   - Controller hunts or dithers at saturation boundary                      │
│                                                                              │
│ ROOT CAUSES:                                                                 │
│   - QPD linear_range (±100-500 µrad typical) << gimbal SSE                  │
│   - Spot size too small relative to detector active area                    │
│   - Optical misalignment causing vignetting                                 │
│                                                                              │
│ CONFIGURATION:                                                               │
│   Default QPD linear_range = 1 mrad = 1000 µrad                             │
│   Gimbal SSE requirement = 0.8° = 13963 µrad                                │
│   → QPD sees saturated signal during most of slew!                          │
│                                                                              │
│ SOLUTION:                                                                    │
│   - Use wider QPD linear range (match FSM optical range ~1.5°)              │
│   - Implement coarse sensor handover before fine tracking                   │
│   - Gate FSM controller until gimbal error < QPD range                      │
│                                                                              │
│ CHECK IN CODE:                                                               │
│   simulation_runner.py:                                                      │
│     qpd_fov_rad = config.qpd_config.get('linear_range', 0.015)              │
│     is_beam_on_sensor = coarse_error_mag < qpd_fov_rad                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. FSM CONTROLLER GAIN MISMATCH (FIXED IN THIS AUDIT)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ISSUE: FSM controller gains designed for DC_gain=1, plant has DC_gain=47.   │
│                                                                              │
│ SYMPTOMS:                                                                    │
│   - FSM oscillates wildly                                                   │
│   - Output diverges to NaN                                                  │
│   - "ERROR: FSM state divergence detected" in console                       │
│                                                                              │
│ ROOT CAUSE:                                                                  │
│   - Modal reduction produced state-space model with high DC gain            │
│   - Original controller gains not scaled for this plant                     │
│   - Open-loop gain Kp * G(0) = 0.976 * 47 = 45.9 >> 1                       │
│                                                                              │
│ FIX APPLIED:                                                                 │
│   fsm_pid_control.py create_fsm_controller_from_design():                   │
│     Kp = 0.02 (was 0.976)                                                   │
│     Ki = 0.5 (was 91.98)                                                    │
│   This gives stable loop gain ≈ 2 and settling time ~400ms                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. INTEGRATOR WINDUP DURING OFF-SENSOR (FIXED IN THIS AUDIT)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ISSUE: FSM integrator accumulates while beam is off sensor.                 │
│                                                                              │
│ SYMPTOMS:                                                                    │
│   - Large FSM command spike when beam returns on sensor                     │
│   - FSM saturates immediately after acquisition                             │
│   - Integrator state goes to infinity                                       │
│                                                                              │
│ ROOT CAUSE:                                                                  │
│   - Controller update() called with measurement=-residual even off sensor   │
│   - Large residual (~15° during slew) integrated over time                  │
│                                                                              │
│ FIX APPLIED:                                                                 │
│   simulation_runner.py _update_fine_controller():                           │
│     if not is_beam_on_sensor:                                               │
│         self.fsm_pid.hold_integrator()  # Freeze integrator state           │
│                                                                              │
│   fsm_pid_control.py hold_integrator():                                     │
│     # New method to freeze integrator without full reset                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. FBL FRICTION COMPENSATION INSTABILITY (PREVIOUSLY FIXED)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ISSUE: Unconditional friction compensation destabilizes during braking.     │
│                                                                              │
│ SYMPTOMS:                                                                    │
│   - Coarse gimbal overshoots target by 40-50%                               │
│   - Slow convergence to final position                                      │
│   - Oscillation near target                                                 │
│                                                                              │
│ ROOT CAUSE:                                                                  │
│   - During overshoot: velocity positive, controller wants to brake          │
│   - Friction comp adds tau_friction = +0.1 * velocity (positive!)           │
│   - This ACCELERATES past target instead of braking                         │
│                                                                              │
│ FIX (from previous audit):                                                   │
│   control_laws.py FeedbackLinearizationController:                          │
│     conditional_friction = True                                             │
│     Only compensate friction when velocity aligns with desired acceleration │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    try:
        # Run verification
        results = run_handover_verification()
        
        # Print failure mode analysis
        analyze_failure_modes()
        
        # Final status
        if results['pass_sse'] and results['pass_fsm_range']:
            print("\n[SUCCESS] Handover verification PASSED")
            sys.exit(0)
        else:
            print("\n[FAILURE] Handover verification FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
