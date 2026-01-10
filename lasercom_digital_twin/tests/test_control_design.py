#!/usr/bin/env python3
"""
Tests for the control design module.
"""

import unittest
import numpy as np
import control as ctrl

from lasercom_digital_twin.control_design import (
    ControllerDesigner, ControlAnalyzer, SystemModeler, DesignRequirements, LinearModel, ControllerSpecs
)


class TestControlDesign(unittest.TestCase):
    """Test cases for control design functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.modeler = SystemModeler()
        self.designer = ControllerDesigner()
        self.analyzer = ControlAnalyzer()
        self.requirements = DesignRequirements()

    def test_system_modeler(self):
        """Test system model creation."""
        # Create a gimbal plant model with typical parameters
        plant = self.modeler.create_gimbal_plant_model(
            inertia_az=0.01, inertia_el=0.01,
            friction_az=0.001, friction_el=0.001,
            motor_kt=0.1, motor_r=1.0, motor_l=0.001
        )
        self.assertIsInstance(plant, LinearModel)

        # Check dimensions
        self.assertEqual(plant.A.shape, (6, 6))  # 6 states
        self.assertEqual(plant.B.shape, (6, 2))  # 2 inputs
        self.assertEqual(plant.C.shape, (6, 6))  # 6 outputs

        # Convert to control system
        plant_ctrl = plant.to_control()
        self.assertIsInstance(plant_ctrl, ctrl.StateSpace)

    def test_pid_controller_design(self):
        """Test PID controller design."""
        # Create plant
        plant = self.modeler.create_gimbal_plant_model(
            inertia_az=0.01, inertia_el=0.01,
            friction_az=0.001, friction_el=0.001,
            motor_kt=0.1, motor_r=1.0, motor_l=0.001
        )

        # Create controller specs
        specs = ControllerSpecs()

        # Design PID controller
        controller = self.designer.design_pid_controller(plant.to_control(), specs)

        # Check that controller is created and tuned
        self.assertIsNotNone(controller)
        self.assertTrue(controller.is_tuned)

        # Check transfer function
        tf = controller.get_transfer_function()
        self.assertIsInstance(tf, ctrl.TransferFunction)

    def test_stability_analysis(self):
        """Test stability analysis."""
        # Create a stable system
        plant = self.modeler.create_gimbal_plant_model(
            inertia_az=0.01, inertia_el=0.01,
            friction_az=0.001, friction_el=0.001,
            motor_kt=0.1, motor_r=1.0, motor_l=0.001
        )

        # Analyze stability
        stable, poles = self.analyzer.check_stability(plant.to_control())
        self.assertIsInstance(stable, bool)
        self.assertIsInstance(poles, np.ndarray)

    def test_requirements_validation(self):
        """Test requirements validation."""
        # Create dummy analysis results
        class DummyResults:
            def __init__(self):
                self.bandwidth = 5.0
                self.settling_time = 0.05
                self.overshoot = 3.0
                self.phase_margin = 50.0

        results = DummyResults()

        # Validate requirements
        validation = self.requirements.validate_design(results, "gimbal_coarse")
        self.assertIsInstance(validation, dict)

    def test_analysis_results(self):
        """Test analysis results structure."""
        plant = self.modeler.create_gimbal_plant_model(
            inertia_az=0.01, inertia_el=0.01,
            friction_az=0.001, friction_el=0.001,
            motor_kt=0.1, motor_r=1.0, motor_l=0.001
        )

        results = self.analyzer.analyze_system(plant.to_control())

        # Check that all expected attributes exist
        expected_attrs = [
            'stable', 'poles', 'bandwidth', 'phase_margin',
            'gain_margin', 'settling_time', 'overshoot', 'steady_state_error'
        ]

        for attr in expected_attrs:
            self.assertTrue(hasattr(results, attr))


if __name__ == '__main__':
    unittest.main()