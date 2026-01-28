"""
Frequency Response Analysis Suite for Nonlinear Gimbal Control Systems

This module provides industrial-grade frequency response analysis for nonlinear
mechanical systems using empirical sinusoidal sweep methodology. It is designed
for characterizing closed-loop performance of:

- PID Controllers (Baseline)
- Feedback Linearization (FBL)  
- Feedback Linearization + Nonlinear Disturbance Observer (FBL+NDOB)

Mathematical Foundation
-----------------------
For nonlinear systems, traditional Bode analysis (based on Laplace transforms)
does not directly apply. This module implements **Empirical Frequency Response**
using the **Sinusoidal Input Describing Function (SIDF)** approach:

1. Apply sinusoidal reference/disturbance at frequency ω
2. Wait for transients to decay (settle to periodic steady-state)
3. Extract fundamental component via DFT correlation
4. Compute gain |G(jω)| and phase ∠G(jω)
5. Repeat for logarithmically-spaced frequencies

Key Metrics Computed
--------------------
- **Closed-Loop Frequency Response T(jω)**: Reference → Output
- **Sensitivity Function S(jω)**: Disturbance → Error  
- **Complementary Sensitivity T(jω)**: Reference → Output (= 1 - S)
- **Input Sensitivity**: Disturbance → Control Effort

For proper disturbance rejection analysis:
- |S(jω)| < 1 in a frequency band → disturbances attenuated
- |S(jω)| > 1 → disturbances amplified (typically at crossover)
- Peak |S(jω)| indicates robustness (Ms < 2 desired)

References
----------
[1] Slotine, J.J.E., Li, W., "Applied Nonlinear Control", Prentice Hall, 1991.
[2] Khalil, H.K., "Nonlinear Systems", 3rd Ed., Prentice Hall, 2002.
[3] Ljung, L., "System Identification: Theory for the User", Prentice Hall, 1999.
[4] Pintelon, R., Schoukens, J., "System Identification: A Frequency Domain 
    Approach", Wiley-IEEE Press, 2012.

Author: Dr. S. Shahid Mustafa
Date: January 28, 2026
"""

from .frequency_sweep_engine import (
    FrequencySweepEngine,
    FrequencySweepConfig,
    FrequencyPoint,
    SweepType
)

from .frequency_response_analyzer import (
    FrequencyResponseAnalyzer,
    AnalyzerConfig,
    ControllerType,
    FrequencyResponseData
)

from .frequency_response_plotter import (
    FrequencyResponsePlotter,
    PlotConfig,
    PlotStyle
)

from .data_logger import (
    FrequencyResponseLogger,
    LoggerConfig
)

__all__ = [
    # Engine
    'FrequencySweepEngine',
    'FrequencySweepConfig', 
    'FrequencyPoint',
    'SweepType',
    # Analyzer
    'FrequencyResponseAnalyzer',
    'AnalyzerConfig',
    'ControllerType',
    'FrequencyResponseData',
    # Plotter
    'FrequencyResponsePlotter',
    'PlotConfig',
    'PlotStyle',
    # Logger
    'FrequencyResponseLogger',
    'LoggerConfig',
]
