"""
Control System Design Requirements

This module defines the performance requirements and specifications for the
lasercom pointing control system. It provides a centralized location for
design constraints, performance metrics, and validation criteria.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ControlLevel(Enum):
    """Control hierarchy levels."""
    GIMBAL_COARSE = "gimbal_coarse"
    FSM_FINE = "fsm_fine"
    INTEGRATED = "integrated"


class PerformanceMetric(Enum):
    """Performance metric types."""
    BANDWIDTH = "bandwidth"
    SETTLING_TIME = "settling_time"
    OVERSHOOT = "overshoot"
    STEADY_STATE_ERROR = "steady_state_error"
    PHASE_MARGIN = "phase_margin"
    GAIN_MARGIN = "gain_margin"


@dataclass
class ControlRequirement:
    """Individual control requirement specification."""
    metric: PerformanceMetric
    value: float
    unit: str
    tolerance: float = 0.0
    description: str = ""
    level: ControlLevel = ControlLevel.INTEGRATED


@dataclass
class DisturbanceSpec:
    """Disturbance specification."""
    name: str
    amplitude: float
    frequency: float
    unit: str
    description: str = ""


@dataclass
class DesignRequirements:
    """Complete set of control design requirements."""

    # System specifications
    control_levels: List[ControlLevel] = field(default_factory=lambda: [
        ControlLevel.GIMBAL_COARSE,
        ControlLevel.FSM_FINE,
        ControlLevel.INTEGRATED
    ])

    # Performance requirements
    requirements: Dict[ControlLevel, List[ControlRequirement]] = field(default_factory=dict)

    # Disturbance specifications
    disturbances: List[DisturbanceSpec] = field(default_factory=list)

    # Operating conditions
    operating_conditions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default requirements."""
        self._set_default_requirements()
        self._set_default_disturbances()
        self._set_default_operating_conditions()

    def _set_default_requirements(self):
        """Set default performance requirements based on lasercom standards."""

        # Gimbal coarse pointing requirements
        gimbal_reqs = [
            ControlRequirement(
                metric=PerformanceMetric.BANDWIDTH,
                value=5.0,
                unit="Hz",
                tolerance=0.5,
                description="Coarse pointing bandwidth for large angle maneuvers",
                level=ControlLevel.GIMBAL_COARSE
            ),
            ControlRequirement(
                metric=PerformanceMetric.SETTLING_TIME,
                value=0.2,
                unit="s",
                tolerance=0.05,
                description="Settling time for 1° step input",
                level=ControlLevel.GIMBAL_COARSE
            ),
            ControlRequirement(
                metric=PerformanceMetric.OVERSHOOT,
                value=10.0,
                unit="%",
                tolerance=5.0,
                description="Maximum overshoot for step response",
                level=ControlLevel.GIMBAL_COARSE
            ),
            ControlRequirement(
                metric=PerformanceMetric.PHASE_MARGIN,
                value=45.0,
                unit="deg",
                tolerance=10.0,
                description="Phase margin for stability",
                level=ControlLevel.GIMBAL_COARSE
            )
        ]

        # FSM fine pointing requirements
        fsm_reqs = [
            ControlRequirement(
                metric=PerformanceMetric.BANDWIDTH,
                value=200.0,
                unit="Hz",
                tolerance=20.0,
                description="Fine pointing bandwidth for residual error correction",
                level=ControlLevel.FSM_FINE
            ),
            ControlRequirement(
                metric=PerformanceMetric.STEADY_STATE_ERROR,
                value=1e-6,
                unit="rad",
                tolerance=1e-7,
                description="Steady-state pointing accuracy",
                level=ControlLevel.FSM_FINE
            ),
            ControlRequirement(
                metric=PerformanceMetric.SETTLING_TIME,
                value=0.01,
                unit="s",
                tolerance=0.005,
                description="Settling time for fine corrections",
                level=ControlLevel.FSM_FINE
            )
        ]

        # Integrated system requirements
        integrated_reqs = [
            ControlRequirement(
                metric=PerformanceMetric.STEADY_STATE_ERROR,
                value=1e-6,
                unit="rad",
                tolerance=1e-7,
                description="Overall pointing accuracy (LOS error)",
                level=ControlLevel.INTEGRATED
            ),
            ControlRequirement(
                metric=PerformanceMetric.BANDWIDTH,
                value=50.0,
                unit="Hz",
                tolerance=10.0,
                description="Integrated system bandwidth",
                level=ControlLevel.INTEGRATED
            )
        ]

        self.requirements = {
            ControlLevel.GIMBAL_COARSE: gimbal_reqs,
            ControlLevel.FSM_FINE: fsm_reqs,
            ControlLevel.INTEGRATED: integrated_reqs
        }

    def _set_default_disturbances(self):
        """Set default disturbance specifications."""

        self.disturbances = [
            DisturbanceSpec(
                name="vibration_base",
                amplitude=0.001,
                frequency=100.0,
                unit="rad",
                description="Base vibration disturbance at 100 Hz"
            ),
            DisturbanceSpec(
                name="thermal_drift",
                amplitude=1e-6,
                frequency=0.01,
                unit="rad/s",
                description="Thermal drift rate"
            ),
            DisturbanceSpec(
                name="cable_torque",
                amplitude=0.01,
                frequency=0.0,
                unit="N·m",
                description="Constant cable torque disturbance"
            ),
            DisturbanceSpec(
                name="wind_gust",
                amplitude=0.0001,
                frequency=1.0,
                unit="rad/s²",
                description="Wind gust acceleration disturbance"
            )
        ]

    def _set_default_operating_conditions(self):
        """Set default operating conditions."""

        self.operating_conditions = {
            "nominal": {
                "temperature": 20.0,  # °C
                "pressure": 101325.0,  # Pa
                "humidity": 50.0,  # %
                "vibration_level": 0.001  # g-rms
            },
            "extreme": {
                "temperature": -40.0,  # °C
                "pressure": 50000.0,  # Pa
                "humidity": 95.0,  # %
                "vibration_level": 0.01  # g-rms
            }
        }

    def get_requirements_for_level(self, level: ControlLevel) -> List[ControlRequirement]:
        """Get requirements for a specific control level."""
        return self.requirements.get(level, [])

    def get_requirement_by_metric(self, metric: PerformanceMetric,
                                level: ControlLevel) -> Optional[ControlRequirement]:
        """Get specific requirement by metric and level."""
        level_reqs = self.requirements.get(level, [])
        for req in level_reqs:
            if req.metric == metric:
                return req
        return None

    def validate_design(self, design_metrics: Dict[PerformanceMetric, float],
                       level: ControlLevel) -> Dict[str, bool]:
        """
        Validate design metrics against requirements.

        Parameters
        ----------
        design_metrics : Dict[PerformanceMetric, float]
            Design performance metrics
        level : ControlLevel
            Control level to validate

        Returns
        -------
        Dict[str, bool]
            Validation results {requirement_name: passed}
        """
        results = {}
        requirements = self.get_requirements_for_level(level)

        for req in requirements:
            if req.metric in design_metrics:
                actual_value = design_metrics[req.metric]
                tolerance = req.tolerance

                # Check if within tolerance
                if req.metric in [PerformanceMetric.STEADY_STATE_ERROR]:
                    # For error metrics, smaller is better
                    passed = actual_value <= (req.value + tolerance)
                else:
                    # For other metrics, check both bounds
                    passed = (req.value - tolerance) <= actual_value <= (req.value + tolerance)

                results[f"{req.metric.value}_{req.level.value}"] = passed
            else:
                results[f"{req.metric.value}_{req.level.value}"] = False

        return results

    def generate_requirements_report(self, filename: str = "design_requirements.txt") -> None:
        """
        Generate a text report of all design requirements.

        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            f.write("LASERCOM POINTING CONTROL DESIGN REQUIREMENTS\n")
            f.write("=" * 55 + "\n\n")

            for level in self.control_levels:
                f.write(f"{level.value.upper().replace('_', ' ')} CONTROL\n")
                f.write("-" * 40 + "\n")

                requirements = self.get_requirements_for_level(level)
                for req in requirements:
                    f.write(f"  {req.metric.value}: {req.value} {req.unit}")
                    if req.tolerance > 0:
                        f.write(f" ± {req.tolerance}")
                    f.write(f"\n    {req.description}\n")

                f.write("\n")

            f.write("DISTURBANCE SPECIFICATIONS\n")
            f.write("-" * 40 + "\n")
            for dist in self.disturbances:
                f.write(f"  {dist.name}: {dist.amplitude} {dist.unit}")
                if dist.frequency > 0:
                    f.write(f" @ {dist.frequency} Hz")
                f.write(f"\n    {dist.description}\n")

            f.write("\nOPERATING CONDITIONS\n")
            f.write("-" * 40 + "\n")
            for condition, params in self.operating_conditions.items():
                f.write(f"  {condition.title()}:\n")
                for param, value in params.items():
                    f.write(f"    {param}: {value}\n")
                f.write("\n")