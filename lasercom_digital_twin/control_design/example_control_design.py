#!/usr/bin/env python3
"""
Example: Control Design for Lasercom Gimbal System

This script demonstrates the complete control design workflow:
1. Plant model creation
2. Controller synthesis
3. Performance analysis
4. Requirements validation
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Import control design modules
from lasercom_digital_twin.control_design import (
    ControllerDesigner, ControlAnalyzer, SystemModeler, DesignRequirements
)


def main():
    """Main control design example."""

    print("Lasercom Gimbal Control Design Example")
    print("=" * 40)

    # 1. Create system model
    print("\n1. Creating plant model...")
    modeler = SystemModeler()

    # Gimbal plant parameters (typical values)
    plant = modeler.create_gimbal_plant_model(
        inertia_az=2.0,      # kg·m²
        inertia_el=1.5,      # kg·m²
        friction_az=0.05,    # N·m·s/rad
        friction_el=0.05,    # N·m·s/rad
        motor_kt=0.5,        # N·m/A
        motor_r=2.0,         # Ω
        motor_l=0.05         # H
    )

    print(f"   Created {plant.A.shape[0]}-state linear model")
    print(f"   States: {plant.state_names}")
    print(f"   Inputs: {plant.input_names}")

    # 2. Design controller
    print("\n2. Designing PID controller...")
    designer = ControllerDesigner()

    # Get requirements for gimbal coarse control
    requirements = DesignRequirements()
    specs = requirements.get_requirements_for_level(requirements.control_levels[0])

    # Design PID controller
    controller = designer.design_pid_controller(plant.to_control(), specs[0])
    print(f"   Designed PID controller with gains:")
    print(f"   Kp = {controller.gains.kp:.2f}")
    print(f"   Ki = {controller.gains.ki:.2f}")
    print(f"   Kd = {controller.gains.kd:.2f}")

    # 3. Analyze performance
    print("\n3. Analyzing closed-loop performance...")
    analyzer = ControlAnalyzer()
    results = analyzer.analyze_system(plant.to_control(), controller.get_transfer_function())

    print(f"   Stability: {'Stable' if results.stable else 'Unstable'}")
    print(f"   Bandwidth: {results.bandwidth:.2f} Hz")
    print(f"   Phase Margin: {results.phase_margin:.1f} deg")
    print(f"   Gain Margin: {results.gain_margin:.1f} dB")
    print(f"   Settling Time: {results.settling_time:.4f} s")
    print(f"   Overshoot: {results.overshoot:.1f}%")
    print(f"   Steady-State Error: {results.steady_state_error:.2e}")

    # 4. Validate against requirements
    print("\n4. Validating against requirements...")
    design_metrics = {
        requirements.requirements[requirements.control_levels[0]][0].metric: results.bandwidth,
        requirements.requirements[requirements.control_levels[0]][1].metric: results.settling_time,
        requirements.requirements[requirements.control_levels[0]][2].metric: results.overshoot,
        requirements.requirements[requirements.control_levels[0]][3].metric: results.phase_margin,
    }

    validation = requirements.validate_design(design_metrics, requirements.control_levels[0])

    print("   Requirements validation:")
    for req_name, passed in validation.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {req_name}: {status}")

    # 5. Generate plots (if matplotlib available)
    try:
        print("\n5. Generating analysis plots...")

        # Bode plot
        analyzer.plot_bode(
            controller.get_transfer_function() * plant.to_control(),
            title="Open-Loop Bode Plot",
            save_path="bode_plot.png"
        )

        # Step response - handle MIMO systems
        try:
            # For MIMO systems, create closed-loop using state-space
            plant_ss = plant.to_control()
            controller_tf = controller.get_transfer_function()

            if plant_ss.noutputs > 1 or plant_ss.ninputs > 1:
                # MIMO case - skip step response for now due to library limitations
                print("   Step response plot skipped (MIMO system - requires Slycot)")
                print("   Bode plot saved as 'bode_plot.png'")
            else:
                # SISO case
                closed_loop = ctrl.feedback(controller_tf * plant_ss)
                analyzer.plot_step_response(
                    closed_loop,
                    title="Closed-Loop Step Response",
                    save_path="step_response.png"
                )
                print("   Plots saved as 'bode_plot.png' and 'step_response.png'")
        except Exception as e:
            print(f"   Step response plot failed: {e}")
            print("   Bode plot saved as 'bode_plot.png'")

    except ImportError:
        print("   Matplotlib not available - skipping plots")

    # 6. Generate report
    print("\n6. Generating analysis report...")
    analyzer.generate_analysis_report(results, "control_analysis_report.txt")
    requirements.generate_requirements_report("design_requirements_report.txt")
    print("   Reports saved as 'control_analysis_report.txt' and 'design_requirements_report.txt'")

    print("\nControl design example completed successfully!")


if __name__ == "__main__":
    main()