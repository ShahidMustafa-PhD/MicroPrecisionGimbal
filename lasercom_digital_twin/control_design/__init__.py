"""
Control Design Module for Lasercom Digital Twin

This module provides tools for controller synthesis, analysis, and validation
for the laser communication terminal pointing system. It implements a
systematic approach to control design following aerospace standards.

Key Features:
- Linear system modeling and linearization
- Controller synthesis (PID, LQG, H-infinity)
- Stability and performance analysis
- Monte Carlo robustness evaluation
- Hardware-in-the-loop validation interfaces

Design Philosophy:
- Separation of concerns: Design tools separate from simulation execution
- Reproducible results: Deterministic algorithms with seeded random processes
- Modular architecture: Easy integration with different control strategies
- Standards compliance: Follows aerospace control design best practices
"""

from .controller_design import ControllerDesigner, ControllerSpecs
from .analysis_tools import ControlAnalyzer
from .system_models import SystemModeler, LinearModel
from .design_requirements import DesignRequirements

__version__ = "1.0.0"
__all__ = [
    "ControllerDesigner",
    "ControlAnalyzer", 
    "SystemModeler",
    "DesignRequirements",
    "LinearModel",
    "ControllerSpecs"
]