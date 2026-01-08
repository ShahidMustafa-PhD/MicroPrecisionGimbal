"""
Monte Carlo Analysis Demonstration

This script demonstrates industrial-grade uncertainty quantification using
the Monte Carlo engine with performance analysis. It shows how to:

1. Define parameter uncertainties across subsystems
2. Configure and execute Monte Carlo batch runs
3. Analyze performance metrics statistically
4. Generate reports and visualizations
5. Assess compliance with requirements under uncertainty

Typical use case: Evaluate whether the system meets pointing accuracy
requirements when component parameters vary within manufacturing tolerances.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from lasercom_digital_twin.core.simulation.monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloConfig,
    ParameterUncertainty,
    DistributionType,
    create_default_uncertainties
)
from lasercom_digital_twin.core.simulation.performance_analyzer import (
    PerformanceAnalyzer
)


def create_mock_simulation_factory():
    """
    Create a mock simulation factory for demonstration.
    
    In a real application, this would instantiate DigitalTwinRunner
    with the randomized configuration and return actual telemetry.
    
    For demo purposes, we generate synthetic telemetry that responds
    to parameter variations (e.g., higher noise → worse RMS error).
    """
    def factory(config, seed):
        """
        Simulation factory: (config_dict, seed) → (runner, telemetry)
        """
        np.random.seed(seed)
        
        # Extract simulation duration
        duration = config.get('simulation_duration', 10.0)
        t = np.linspace(0, duration, int(duration * 100))  # 100 Hz
        dt = t[1] - t[0]
        
        # Extract randomized parameters (if any)
        motor_Kt = config.get('motor_az', {}).get('K_t', 0.1)
        gyro_noise = config.get('gyro_az', {}).get('noise_std', 1e-5)
        backlash = config.get('gimbal_az', {}).get('backlash', 1e-5)
        friction = config.get('gimbal_az', {}).get('friction', 0.1)
        
        # Simulate LOS error with parameter-dependent performance
        # Base error increases with gyro noise, backlash, and friction
        # Base error decreases with higher motor torque constant
        
        base_rms = 5.0  # µrad baseline
        
        # Scale error based on parameters
        noise_factor = gyro_noise / 1e-5  # Normalized to nominal
        backlash_factor = backlash / 1e-5
        friction_factor = friction / 0.1
        motor_factor = 0.1 / motor_Kt
        
        error_scale = base_rms * (
            0.5 * noise_factor +
            0.3 * backlash_factor +
            0.2 * friction_factor +
            0.2 * motor_factor
        ) / 1.2  # Normalize
        
        # Generate pointing error: sinusoid + noise
        los_error_az = (
            error_scale * np.sin(2 * np.pi * 1.0 * t) +
            np.random.randn(len(t)) * error_scale * 0.3
        )
        
        los_error_el = (
            error_scale * 0.7 * np.cos(2 * np.pi * 0.5 * t) +
            np.random.randn(len(t)) * error_scale * 0.3
        )
        
        # FSM commands (with occasional saturation)
        fsm_cmd_az = 200.0 * np.sin(2 * np.pi * 5.0 * t)
        fsm_cmd_el = 150.0 * np.cos(2 * np.pi * 3.0 * t)
        
        # Randomly increase FSM amplitude for some runs (simulates disturbance)
        if np.random.rand() > 0.7:
            fsm_cmd_az *= 1.5
            fsm_cmd_el *= 1.5
        
        # Apply saturation
        fsm_limit = 400.0  # µrad
        fsm_cmd_az = np.clip(fsm_cmd_az, -fsm_limit, fsm_limit)
        fsm_cmd_el = np.clip(fsm_cmd_el, -fsm_limit, fsm_limit)
        
        # Gimbal angles (tracking profile)
        gimbal_az = np.deg2rad(10.0) * (1 + 0.1 * np.sin(2 * np.pi * 0.1 * t))
        gimbal_el = np.deg2rad(5.0) * (1 + 0.1 * np.cos(2 * np.pi * 0.15 * t))
        
        # Control torques
        torque_az = 0.05 * np.sin(2 * np.pi * 2.0 * t) * motor_factor
        torque_el = 0.03 * np.cos(2 * np.pi * 1.5 * t) * motor_factor
        
        # Estimator convergence (starts False, converges after 1 s)
        estimator_converged = (t > 1.0).tolist()
        
        telemetry = {
            'time': t.tolist(),
            'los_error_az': los_error_az.tolist(),
            'los_error_el': los_error_el.tolist(),
            'fsm_cmd_az': fsm_cmd_az.tolist(),
            'fsm_cmd_el': fsm_cmd_el.tolist(),
            'gimbal_az': gimbal_az.tolist(),
            'gimbal_el': gimbal_el.tolist(),
            'torque_az': torque_az.tolist(),
            'torque_el': torque_el.tolist(),
            'estimator_converged': estimator_converged
        }
        
        # Return (None, telemetry) - runner not needed for demo
        return None, telemetry
    
    return factory


def run_monte_carlo_demo():
    """
    Execute Monte Carlo demonstration with visualization.
    """
    print("=" * 70)
    print("MONTE CARLO UNCERTAINTY ANALYSIS DEMONSTRATION")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 1. Define parameter uncertainties
    # -------------------------------------------------------------------------
    print("Step 1: Defining parameter uncertainties...")
    print()
    
    uncertainties = [
        # Motor parameters
        ParameterUncertainty(
            name='motor_az.K_t',
            nominal=0.1,  # N·m/A
            distribution=DistributionType.NORMAL,
            uncertainty=5.0,  # ±5%
            bounds=(0.08, 0.12),
            metadata={'description': 'Motor torque constant', 'unit': 'N·m/A'}
        ),
        ParameterUncertainty(
            name='motor_az.R',
            nominal=1.0,  # Ω
            distribution=DistributionType.NORMAL,
            uncertainty=3.0,  # ±3%
            bounds=(0.85, 1.15),
            metadata={'description': 'Motor resistance', 'unit': 'Ω'}
        ),
        
        # Sensor parameters
        ParameterUncertainty(
            name='gyro_az.noise_std',
            nominal=1e-5,  # rad/s
            distribution=DistributionType.LOGNORMAL,
            uncertainty=30.0,  # Relative std
            bounds=(5e-6, 3e-5),
            metadata={'description': 'Gyro noise density', 'unit': 'rad/s'}
        ),
        ParameterUncertainty(
            name='gyro_az.bias',
            nominal=0.0,  # rad/s
            distribution=DistributionType.UNIFORM,
            uncertainty=100.0,
            bounds=(-1e-4, 1e-4),
            metadata={'description': 'Gyro bias offset', 'unit': 'rad/s'}
        ),
        
        # Structural parameters
        ParameterUncertainty(
            name='gimbal_az.backlash',
            nominal=1e-5,  # rad
            distribution=DistributionType.LOGNORMAL,
            uncertainty=50.0,  # Highly variable
            bounds=(5e-6, 5e-5),
            metadata={'description': 'Gimbal backlash', 'unit': 'rad'}
        ),
        ParameterUncertainty(
            name='gimbal_az.friction',
            nominal=0.1,  # N·m·s/rad
            distribution=DistributionType.NORMAL,
            uncertainty=20.0,  # ±20%
            bounds=(0.05, 0.3),
            metadata={'description': 'Viscous friction', 'unit': 'N·m·s/rad'}
        ),
    ]
    
    print(f"  Defined {len(uncertainties)} uncertain parameters:")
    for unc in uncertainties:
        print(f"    - {unc.name:30s}: {unc.nominal:10.3g} ± {unc.uncertainty:5.1f}% "
              f"[{unc.distribution.value}]")
    print()
    
    # -------------------------------------------------------------------------
    # 2. Configure Monte Carlo analysis
    # -------------------------------------------------------------------------
    print("Step 2: Configuring Monte Carlo analysis...")
    print()
    
    mc_config = MonteCarloConfig(
        n_runs=100,  # 100 Monte Carlo runs
        base_seed=42,  # For reproducibility
        parameter_uncertainties=uncertainties,
        simulation_duration=10.0,  # 10 seconds per run
        save_telemetry=False,  # Don't save full telemetry (saves space)
        output_dir=None  # Don't save to disk for demo
    )
    
    print(f"  Number of runs:        {mc_config.n_runs}")
    print(f"  Simulation duration:   {mc_config.simulation_duration} s")
    print(f"  Base seed:             {mc_config.base_seed}")
    print()
    
    # -------------------------------------------------------------------------
    # 3. Create performance analyzer with requirements
    # -------------------------------------------------------------------------
    print("Step 3: Setting performance requirements...")
    print()
    
    analyzer = PerformanceAnalyzer(
        rms_requirement=10.0,  # 10 µrad RMS (CRITICAL)
        peak_requirement=50.0,  # 50 µrad peak
        saturation_requirement=30.0  # 30% FSM saturation limit
    )
    
    print(f"  RMS pointing error:    ≤ {analyzer.rms_requirement:.1f} µrad")
    print(f"  Peak pointing error:   ≤ {analyzer.peak_requirement:.1f} µrad")
    print(f"  FSM saturation:        ≤ {analyzer.saturation_requirement:.1f} %")
    print()
    
    # -------------------------------------------------------------------------
    # 4. Execute Monte Carlo batch
    # -------------------------------------------------------------------------
    print("Step 4: Executing Monte Carlo batch...")
    print()
    
    engine = MonteCarloEngine(mc_config, analyzer, verbose=True)
    
    simulation_factory = create_mock_simulation_factory()
    
    results = engine.run_batch(simulation_factory)
    
    print()
    
    # -------------------------------------------------------------------------
    # 5. Generate and display report
    # -------------------------------------------------------------------------
    print("Step 5: Analyzing results...")
    print()
    
    report = engine.generate_report(results)
    print(report)
    print()
    
    # -------------------------------------------------------------------------
    # 6. Create visualizations
    # -------------------------------------------------------------------------
    print("Step 6: Generating visualizations...")
    print()
    
    # Extract metrics for plotting
    successful_runs = [r for r in results.runs if r.success]
    
    rms_errors = [r.metrics.rms_pointing_error for r in successful_runs]
    peak_errors = [r.metrics.peak_pointing_error for r in successful_runs]
    sat_pcts = [r.metrics.fsm_saturation_pct for r in successful_runs]
    
    # Extract parameter values
    motor_Kt_values = [r.parameters.get('motor_az.K_t', 0.1) for r in successful_runs]
    gyro_noise_values = [r.parameters.get('gyro_az.noise_std', 1e-5) for r in successful_runs]
    backlash_values = [r.parameters.get('gimbal_az.backlash', 1e-5) for r in successful_runs]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Subplot 1: RMS error histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(rms_errors, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(analyzer.rms_requirement, color='red', linestyle='--',
                linewidth=2, label=f'Requirement: {analyzer.rms_requirement} µrad')
    ax1.axvline(np.mean(rms_errors), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(rms_errors):.2f} µrad')
    ax1.set_xlabel('RMS Pointing Error (µrad)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('RMS Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Peak error histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(peak_errors, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(analyzer.peak_requirement, color='red', linestyle='--',
                linewidth=2, label=f'Requirement: {analyzer.peak_requirement} µrad')
    ax2.axvline(np.mean(peak_errors), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(peak_errors):.2f} µrad')
    ax2.set_xlabel('Peak Pointing Error (µrad)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Peak Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: FSM saturation histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(sat_pcts, bins=20, alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax3.axvline(analyzer.saturation_requirement, color='red', linestyle='--',
                linewidth=2, label=f'Requirement: {analyzer.saturation_requirement} %')
    ax3.axvline(np.mean(sat_pcts), color='green', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(sat_pcts):.2f} %')
    ax3.set_xlabel('FSM Saturation (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('FSM Saturation Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: RMS vs Motor K_t
    ax4 = plt.subplot(2, 3, 4)
    sc = ax4.scatter(motor_Kt_values, rms_errors, c=gyro_noise_values,
                     cmap='viridis', alpha=0.6, s=50, edgecolor='black')
    ax4.axhline(analyzer.rms_requirement, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Motor K_t (N·m/A)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('RMS Error (µrad)', fontsize=11, fontweight='bold')
    ax4.set_title('RMS vs Motor Torque Constant', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label('Gyro Noise (rad/s)', fontsize=10, fontweight='bold')
    
    # Subplot 5: RMS vs Gyro Noise
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(np.array(gyro_noise_values) * 1e6, rms_errors,
                c=backlash_values, cmap='plasma', alpha=0.6, s=50, edgecolor='black')
    ax5.axhline(analyzer.rms_requirement, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Gyro Noise (µrad/s)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('RMS Error (µrad)', fontsize=11, fontweight='bold')
    ax5.set_title('RMS vs Gyro Noise', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: CDF of RMS error
    ax6 = plt.subplot(2, 3, 6)
    sorted_rms = np.sort(rms_errors)
    cdf = np.arange(1, len(sorted_rms) + 1) / len(sorted_rms) * 100
    ax6.plot(sorted_rms, cdf, linewidth=2, color='navy')
    ax6.axvline(analyzer.rms_requirement, color='red', linestyle='--',
                linewidth=2, label='Requirement')
    ax6.axhline(95, color='orange', linestyle=':', linewidth=2, label='95th percentile')
    ax6.set_xlabel('RMS Pointing Error (µrad)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Cumulative Probability (%)', fontsize=11, fontweight='bold')
    ax6.set_title('RMS Error CDF', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved: monte_carlo_analysis.png")
    
    plt.show()
    
    # -------------------------------------------------------------------------
    # 7. Statistical summary table
    # -------------------------------------------------------------------------
    print()
    print("Step 7: Statistical Summary Table")
    print()
    
    summary_data = {
        'Metric': ['RMS Error (µrad)', 'Peak Error (µrad)', 'FSM Saturation (%)'],
        'Mean': [
            f"{np.mean(rms_errors):.2f}",
            f"{np.mean(peak_errors):.2f}",
            f"{np.mean(sat_pcts):.2f}"
        ],
        'Std': [
            f"{np.std(rms_errors):.2f}",
            f"{np.std(peak_errors):.2f}",
            f"{np.std(sat_pcts):.2f}"
        ],
        'Min': [
            f"{np.min(rms_errors):.2f}",
            f"{np.min(peak_errors):.2f}",
            f"{np.min(sat_pcts):.2f}"
        ],
        'Max': [
            f"{np.max(rms_errors):.2f}",
            f"{np.max(peak_errors):.2f}",
            f"{np.max(sat_pcts):.2f}"
        ],
        '95th %ile': [
            f"{np.percentile(rms_errors, 95):.2f}",
            f"{np.percentile(peak_errors, 95):.2f}",
            f"{np.percentile(sat_pcts, 95):.2f}"
        ],
        'Requirement': [
            f"{analyzer.rms_requirement:.1f}",
            f"{analyzer.peak_requirement:.1f}",
            f"{analyzer.saturation_requirement:.1f}"
        ],
        'Pass Rate': [
            f"{sum(1 for e in rms_errors if e <= analyzer.rms_requirement) / len(rms_errors) * 100:.1f}%",
            f"{sum(1 for e in peak_errors if e <= analyzer.peak_requirement) / len(peak_errors) * 100:.1f}%",
            f"{sum(1 for s in sat_pcts if s <= analyzer.saturation_requirement) / len(sat_pcts) * 100:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()
    
    # -------------------------------------------------------------------------
    # 8. Performance margin assessment
    # -------------------------------------------------------------------------
    print()
    print("Step 8: Performance Margin Assessment")
    print()
    
    rms_p95 = np.percentile(rms_errors, 95)
    rms_margin = (analyzer.rms_requirement - rms_p95) / analyzer.rms_requirement * 100
    
    print(f"  RMS Requirement:         {analyzer.rms_requirement:.2f} µrad")
    print(f"  Mean RMS:                {np.mean(rms_errors):.2f} µrad")
    print(f"  95th percentile RMS:     {rms_p95:.2f} µrad")
    print(f"  Worst-case RMS:          {np.max(rms_errors):.2f} µrad")
    print(f"  Performance Margin:      {rms_margin:+.1f}%")
    print()
    
    if rms_margin > 0:
        print(f"  ✓ PASS: System meets requirements with {rms_margin:.1f}% margin")
    else:
        print(f"  ✗ FAIL: System exceeds requirements by {-rms_margin:.1f}%")
    print()
    
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_monte_carlo_demo()
