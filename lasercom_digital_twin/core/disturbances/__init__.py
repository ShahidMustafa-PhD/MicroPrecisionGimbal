"""
Disturbance modeling package for laser communication terminal digital twin.

This package provides high-fidelity environmental disturbance models including:
- Dryden wind turbulence (MIL-F-8785C compliant)
- PSD-based structural vibration with modal superposition
- High-frequency structural noise

All disturbances are designed for injection as additive torques τ_d into the
plant dynamics: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ_control + τ_disturbance
"""

from .disturbance_models import (
    # Main classes
    EnvironmentalDisturbances,
    SimpleDisturbanceModel,
    DisturbanceState,
    # Filter classes (internal, but exported for testing/advanced use)
    DrydenTurbulenceFilter,
    ModalVibrationFilter,
    # Configuration dataclasses
    EnvironmentalDisturbanceConfig,
    DrydenWindConfig,
    StructuralVibrationConfig,
    # Factory functions
    create_default_disturbances,
    create_wind_disturbance,
    create_vibration_disturbance,
    create_environmental_disturbances,
)

__all__ = [
    'EnvironmentalDisturbances',
    'SimpleDisturbanceModel',
    'DisturbanceState',
    'DrydenTurbulenceFilter',
    'ModalVibrationFilter',
    'EnvironmentalDisturbanceConfig',
    'DrydenWindConfig',
    'StructuralVibrationConfig',
    'create_default_disturbances',
    'create_wind_disturbance',
    'create_vibration_disturbance',
    'create_environmental_disturbances',
]
