"""
Performance Metrics Package — FSO Dual-Stage Gimbal Digital Twin
================================================================

This package provides industrial-grade, physics-based performance metrics
for quantifying the reliability and efficiency of the spectral handover
between the coarse gimbal and the Fast Steering Mirror (FSM).

Modules
-------
stroke_metrics
    Stroke Consumption Ratio (SCR), Bias Consumption (S_bias), and
    Dynamic Stroke Margin (DSM) — the three canonical handover bottleneck
    benchmark metrics for SWaP-constrained FSO terminals.

Usage
-----
    from lasercom_digital_twin.core.performance_metrics import StrokeMetrics

    calculator = StrokeMetrics(theta_max=0.010)  # 10 mrad stroke limit
    result = calculator.compute(
        time=log_arrays['time'],
        fsm_tip=log_arrays['fsm_tip'],
        fsm_tilt=log_arrays['fsm_tilt'],
        dt=1e-4
    )
    print(f"SCR Tip: {result.scr_tip:.1f}%")
    print(f"DSM Tip: {result.dsm_tip*1e3:.2f} mrad")
"""

from lasercom_digital_twin.core.performance_metrics.stroke_metrics import (
    StrokeMetrics,
    StrokeMetricsResult,
)

__all__ = [
    "StrokeMetrics",
    "StrokeMetricsResult",
]
