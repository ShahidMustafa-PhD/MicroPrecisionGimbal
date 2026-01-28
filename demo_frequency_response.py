#!/usr/bin/env python3
"""
Frequency Response Analysis Demo for Gimbal Control Systems

This script demonstrates the comprehensive frequency response analysis suite
for comparing PID, FBL, and FBL+NDOB controllers on a nonlinear gimbal system.

The analysis uses empirical sinusoidal sweep methodology (SIDF) which is the
gold standard for characterizing nonlinear systems in the frequency domain.

Generated Plots (Publication Quality - 300 DPI):
------------------------------------------------
1. Bode Plot: Magnitude & Phase vs Frequency (all controllers overlaid)
2. Sensitivity Function: S(jω) showing disturbance rejection bands
3. Disturbance Rejection Bands: Annotated frequency regions
4. Coherence Quality: Measurement validity verification
5. Performance Summary: Bar charts of key metrics

Key Metrics Computed:
--------------------
- Closed-Loop Bandwidth [Hz]: -3dB crossover frequency
- Peak Sensitivity Ms: Maximum |S(jω)|, robustness indicator
- Gain Margin [dB]: Stability margin at phase = -180°
- Phase Margin [deg]: Stability margin at |G| = 0dB

Disturbance Rejection Analysis:
-------------------------------
The sensitivity function S(jω) reveals which frequency bands have:
- |S| < 0 dB: Disturbances ATTENUATED (controller rejects them)
- |S| > 0 dB: Disturbances AMPLIFIED (controller makes them worse)

Typical disturbance sources:
- 0.01-0.1 Hz: Thermal drift
- 0.1-1 Hz: Platform/base motion
- 1-10 Hz: Low-frequency vibration
- 10-50 Hz: Motor cogging, structural resonance
- 50-200 Hz: Structural modes
- >200 Hz: Acoustic/electrical noise

Usage:
------
    python demo_frequency_response.py

    # Or with custom parameters:
    python demo_frequency_response.py --f_min 0.05 --f_max 100 --n_points 50

Author: Dr. S. Shahid Mustafa
Date: January 28, 2026
"""

import numpy as np
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# Configure matplotlib for publication-quality output
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['lines.linewidth'] = 2.0

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import frequency response suite
from lasercom_digital_twin.core.frequency_response import (
    FrequencySweepEngine,
    FrequencySweepConfig,
    FrequencyResponseData,
    ControllerType,
    FrequencyResponsePlotter,
    PlotConfig,
    PlotStyle,
    FrequencyResponseLogger,
    LoggerConfig,
    SweepType
)

# Import dynamics and controllers
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics
from lasercom_digital_twin.core.controllers.control_laws import (
    CoarseGimbalController,
    FeedbackLinearizationController
)
from lasercom_digital_twin.core.n_dist_observer import (
    NonlinearDisturbanceObserver,
    NDOBConfig
)
from typing import Dict, Tuple


class FrequencyResponseSimulator:
    """
    Simulation wrapper for frequency response analysis.
    
    Provides simulation callbacks for the sweep engine to extract empirical
    frequency response from closed-loop nonlinear systems.
    """
    
    def __init__(
        self,
        controller_type: ControllerType,
        dt: float = 0.001,
        operating_point_az: float = 0.0,
        operating_point_el: float = 0.0
    ):
        self.controller_type = controller_type
        self.dt = dt
        self.op_az = operating_point_az
        self.op_el = operating_point_el
        
        # Initialize plant dynamics
        self.dynamics = GimbalDynamics(
            pan_mass=1.0,
            tilt_mass=0.5,
            cm_r=0.0,
            cm_h=0.0,
            gravity=9.81
        )
        
        # Initialize controller
        self._init_controller()
        
        # Plant friction
        self.friction_coef = 0.1
    
    def _init_controller(self) -> None:
        """Initialize controller for the specified type."""
        if self.controller_type == ControllerType.OPEN_LOOP:
            self.controller = None
        
        elif self.controller_type == ControllerType.PID:
            self.controller = CoarseGimbalController({
                'kp': [3.514, 1.320],
                'ki': [15.464, 4.148],
                'kd': [0.293, 0.059418],
                'tau_max': [10.0, 10.0],
                'tau_min': [-10.0, -10.0],
                'enable_derivative': True
            })
        
        elif self.controller_type == ControllerType.FBL:
            self.controller = FeedbackLinearizationController(
                config={
                    'kp': [400.0, 400.0],
                    'kd': [40.0, 40.0],
                    'ki': [0.0, 0.0],
                    'enable_integral': False,
                    'tau_max': [10.0, 10.0],
                    'tau_min': [-10.0, -10.0],
                    'friction_az': 0.1,
                    'friction_el': 0.1
                },
                dynamics_model=self.dynamics,
                ndob=None
            )
        
        elif self.controller_type == ControllerType.FBL_NDOB:
            ndob_config = NDOBConfig(
                lambda_az=50.0,
                lambda_el=50.0,
                d_max=0.5,
                enable=True
            )
            ndob = NonlinearDisturbanceObserver(self.dynamics, ndob_config)
            
            self.controller = FeedbackLinearizationController(
                config={
                    'kp': [400.0, 400.0],
                    'kd': [40.0, 40.0],
                    'ki': [0.0, 0.0],
                    'enable_integral': False,
                    'tau_max': [10.0, 10.0],
                    'tau_min': [-10.0, -10.0],
                    'friction_az': 0.1,
                    'friction_el': 0.1
                },
                dynamics_model=self.dynamics,
                ndob=ndob
            )
    
    def simulate_sweep(
        self,
        omega: float,
        duration: float,
        amplitude: float,
        sweep_type: SweepType,
        axis: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run closed-loop simulation with sinusoidal excitation."""
        n_steps = int(duration / self.dt)
        t = np.arange(n_steps) * self.dt
        
        q = np.array([self.op_az, self.op_el])
        dq = np.zeros(2)
        
        u_signal = np.zeros(n_steps)
        y_signal = np.zeros(n_steps)
        
        if self.controller is not None:
            self.controller.reset()
        
        axis_idx = 0 if axis == 'az' else 1
        
        for k in range(n_steps):
            sin_value = amplitude * np.sin(omega * t[k])
            
            if sweep_type == SweepType.REFERENCE_TRACKING:
                if axis == 'az':
                    ref = np.array([self.op_az + sin_value, self.op_el])
                else:
                    ref = np.array([self.op_az, self.op_el + sin_value])
                disturbance = np.zeros(2)
                u_signal[k] = sin_value
            else:
                ref = np.array([self.op_az, self.op_el])
                disturbance = np.zeros(2)
                disturbance[axis_idx] = sin_value * 0.1
                u_signal[k] = sin_value * 0.1
            
            if self.controller_type == ControllerType.OPEN_LOOP:
                tau = np.zeros(2)
            elif self.controller_type == ControllerType.PID:
                tau, _ = self.controller.compute_control(
                    reference=ref,
                    measurement=q,
                    dt=self.dt,
                    velocity_estimate=dq
                )
            else:
                state_estimate = {
                    'theta_az': q[0],
                    'theta_el': q[1],
                    'theta_dot_az': dq[0],
                    'theta_dot_el': dq[1],
                    'dist_az': 0.0,
                    'dist_el': 0.0
                }
                tau, _ = self.controller.compute_control(
                    q_ref=ref,
                    dq_ref=np.zeros(2),
                    state_estimate=state_estimate,
                    dt=self.dt
                )
            
            tau_applied = tau + disturbance
            tau_net = tau_applied - self.friction_coef * dq
            ddq = self.dynamics.compute_forward_dynamics(q, dq, tau_net)
            dq = dq + ddq * self.dt
            q = q + dq * self.dt
            
            if sweep_type == SweepType.REFERENCE_TRACKING:
                y_signal[k] = q[axis_idx] - (self.op_az if axis == 'az' else self.op_el)
            else:
                y_signal[k] = ref[axis_idx] - q[axis_idx]
        
        return t, u_signal, y_signal


def run_analysis(
    f_min: float = 0.1,
    f_max: float = 50.0,
    n_points: int = 30,
    amplitude_deg: float = 1.0,
    axis: str = 'az'
) -> Dict[ControllerType, FrequencyResponseData]:
    """
    Execute comprehensive frequency response analysis.
    
    Parameters
    ----------
    f_min : float
        Minimum frequency [Hz]
    f_max : float
        Maximum frequency [Hz]
    n_points : int
        Number of logarithmically-spaced frequency points
    amplitude_deg : float
        Sinusoidal excitation amplitude [degrees]
    axis : str
        Axis to analyze ('az' or 'el')
        
    Returns
    -------
    Dict[ControllerType, FrequencyResponseData]
        Frequency response data for each controller
    """
    print("\n" + "=" * 80)
    print("FREQUENCY RESPONSE ANALYSIS SUITE")
    print("Empirical Sinusoidal Sweep for Nonlinear Gimbal Control Systems")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Frequency Range: {f_min:.2f} Hz - {f_max:.1f} Hz")
    print(f"  Frequency Points: {n_points}")
    print(f"  Excitation Amplitude: {amplitude_deg}°")
    print(f"  Analysis Axis: {axis.upper()}")
    print("=" * 80)
    
    sweep_config = FrequencySweepConfig(
        f_min=f_min,
        f_max=f_max,
        n_points=n_points,
        amplitude=np.deg2rad(amplitude_deg),
        settling_cycles=6,
        measurement_cycles=12,
        max_settling_time=15.0,
        min_measurement_time=0.5,
        use_hanning_window=True,
        dt=0.001
    )
    
    controller_types = [
        ControllerType.OPEN_LOOP,
        ControllerType.PID,
        ControllerType.FBL,
        ControllerType.FBL_NDOB
    ]
    
    all_results = {}
    
    for ctrl_type in controller_types:
        print(f"\n{'='*60}")
        print(f"Analyzing: {ctrl_type.name}")
        print(f"{'='*60}")
        
        simulator = FrequencyResponseSimulator(ctrl_type, dt=sweep_config.dt)
        engine = FrequencySweepEngine(sweep_config, verbose=True)
        
        print("\n[Phase 1] Closed-Loop Tracking Response T(jω)")
        tracking_results = engine.run_sweep(
            simulator.simulate_sweep,
            SweepType.REFERENCE_TRACKING,
            axis
        )
        
        print("\n[Phase 2] Sensitivity Function S(jω)")
        engine2 = FrequencySweepEngine(sweep_config, verbose=True)
        sensitivity_results = engine2.run_sweep(
            simulator.simulate_sweep,
            SweepType.DISTURBANCE_INJECTION,
            axis
        )
        
        n = len(tracking_results)
        
        # Compute bandwidth
        cl_gain = np.array([r.gain_db for r in tracking_results])
        valid = ~np.isnan(cl_gain)
        bandwidth = 0.0
        if np.any(valid):
            dc_gain = cl_gain[valid][0]
            threshold = dc_gain - 3.0
            crossings = np.where((cl_gain < threshold) & valid)[0]
            if len(crossings) > 0:
                freqs = np.array([r.frequency_hz for r in tracking_results])
                bandwidth = freqs[crossings[0]]
            else:
                bandwidth = tracking_results[-1].frequency_hz
        
        # Compute peak sensitivity
        sens_gain = np.array([r.gain_db for r in sensitivity_results])
        valid_sens = ~np.isnan(sens_gain)
        peak_sens = 1.0
        if np.any(valid_sens):
            peak_db = np.max(sens_gain[valid_sens])
            peak_sens = 10 ** (peak_db / 20.0)
        
        freq_data = FrequencyResponseData(
            controller_type=ctrl_type,
            axis=axis,
            frequencies_hz=np.array([r.frequency_hz for r in tracking_results]),
            frequencies_rad=np.array([r.frequency_rad for r in tracking_results]),
            closed_loop_gain_db=cl_gain,
            closed_loop_phase_deg=np.array([r.phase_deg for r in tracking_results]),
            sensitivity_gain_db=sens_gain,
            sensitivity_phase_deg=np.array([r.phase_deg for r in sensitivity_results]),
            control_effort_gain_db=np.zeros(n),
            coherence=np.array([r.coherence for r in tracking_results]),
            bandwidth_hz=bandwidth,
            peak_sensitivity=peak_sens,
            metadata={'axis': axis, 'n_points': n}
        )
        
        all_results[ctrl_type] = freq_data
        
        print(f"\n[OK] {ctrl_type.name} Complete:")
        print(f"     Bandwidth: {bandwidth:.2f} Hz")
        print(f"     Peak Sensitivity Ms: {peak_sens:.2f}")
    
    return all_results


def generate_plots(
    results: Dict[ControllerType, FrequencyResponseData],
    save: bool = True
) -> None:
    """Generate all frequency response comparison plots."""
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 70)
    
    config = PlotConfig(
        style=PlotStyle.PUBLICATION,
        output_dir=Path('figures_bode'),
        dpi=300,
        save_format='png'
    )
    plotter = FrequencyResponsePlotter(config)
    
    for ctrl_type, data in results.items():
        plotter.add_response(data)
    
    print("\n[1/5] Bode Plot...")
    plotter.plot_bode_comparison(save=save)
    
    print("[2/5] Sensitivity Function...")
    plotter.plot_sensitivity_comparison(save=save)
    
    print("[3/5] Disturbance Rejection Bands...")
    plotter.plot_disturbance_rejection_bands(save=save)
    
    print("[4/5] Coherence Quality...")
    plotter.plot_coherence_overlay(save=save)
    
    print("[5/5] Performance Summary...")
    plotter.plot_performance_summary(save=save)
    
    # Save data
    print("\n[DATA] Saving to JSON/CSV...")
    logger = FrequencyResponseLogger(LoggerConfig(
        output_dir=Path('frequency_response_data'),
        save_json=True,
        save_csv=True
    ))
    logger.add_results_dict(results)
    logger.save()
    
    # Print summary
    print("\n" + "=" * 90)
    print("FREQUENCY-DOMAIN PERFORMANCE SUMMARY")
    print("=" * 90)
    print(f"{'Controller':<15} {'Bandwidth [Hz]':>15} {'Peak Ms':>12} {'Ms [dB]':>10} {'DC Gain [dB]':>14}")
    print("-" * 90)
    
    for ctrl_type, data in results.items():
        dc_gain = data.closed_loop_gain_db[0] if len(data.closed_loop_gain_db) > 0 else np.nan
        ms_db = 20 * np.log10(data.peak_sensitivity + 1e-12)
        print(f"{ctrl_type.name:<15} {data.bandwidth_hz:>15.2f} {data.peak_sensitivity:>12.2f} "
              f"{ms_db:>10.1f} {dc_gain:>14.1f}")
    
    print("=" * 90)
    
    if save:
        print(f"\n[OK] Figures saved to figures_bode/")
        print("[OK] Data saved to frequency_response_data/")
    
    plt.show()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Frequency Response Analysis for Gimbal Control Systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_frequency_response.py
  python demo_frequency_response.py --f_min 0.05 --f_max 100 --n_points 50
  python demo_frequency_response.py --axis el --amplitude 0.5
        """
    )
    
    parser.add_argument('--f_min', type=float, default=0.1,
                       help='Minimum frequency [Hz] (default: 0.1)')
    parser.add_argument('--f_max', type=float, default=50.0,
                       help='Maximum frequency [Hz] (default: 50)')
    parser.add_argument('--n_points', type=int, default=30,
                       help='Number of frequency points (default: 30)')
    parser.add_argument('--amplitude', type=float, default=1.0,
                       help='Excitation amplitude [degrees] (default: 1.0)')
    parser.add_argument('--axis', type=str, default='az', choices=['az', 'el'],
                       help='Axis to analyze (default: az)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save figures to disk')
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_analysis(
        f_min=args.f_min,
        f_max=args.f_max,
        n_points=args.n_points,
        amplitude_deg=args.amplitude,
        axis=args.axis
    )
    
    # Generate plots
    generate_plots(results, save=not args.no_save)


if __name__ == '__main__':
    main()
