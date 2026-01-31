"""
Twin - High-Fidelity Digital Twin Module
=========================================
PyVista-based 3D visualization assets for the laser communication gimbal.

Modules:
--------
- gimbal_model: Hierarchical CAD geometry and kinematics
- simulation_demo: Trajectory playback and visualization orchestration
"""

from .gimbal_model import GimbalModel

__all__ = ['GimbalModel']
__version__ = '1.0.0'
