"""
Plotting Module for Lasercom Digital Twin.

This module provides publication-quality visualization capabilities for
gimbal control system analysis and comparison studies.

Architecture
------------
- `style_config`: Matplotlib styling, color schemes, publication presets
- `metrics_utils`: Performance metric computation for plotting
- `research_comparison_plotter`: Three-way controller comparison plots (13 figures)

Usage Example
-------------
```python
from lasercom_digital_twin.core.plots import (
    ResearchComparisonPlotter,
    PlotStyleConfig,
    compute_tracking_metrics
)

# Create plotter with style configuration
style = PlotStyleConfig()
plotter = ResearchComparisonPlotter(style)

# Generate plots from simulation results
plotter.plot_all(results_pid, results_fbl, results_ndob, target_az_deg, target_el_deg)
```

Design Principles
-----------------
1. Separation of Concerns: Style config separate from plotting logic
2. Composability: Individual plot methods can be called separately
3. Publication-Ready: 300 DPI, LaTeX typography, IEEE/AIAA compliant
4. Reproducibility: Deterministic output, version-controlled styles

Author: Dr. S. Shahid Mustafa
Version: 2.0.0
"""

from lasercom_digital_twin.core.plots.style_config import (
    PlotStyleConfig,
    ControllerColors,
    AxisColors,
    configure_matplotlib_defaults
)

from lasercom_digital_twin.core.plots.metrics_utils import (
    compute_tracking_metrics,
    TrackingMetrics
)

from lasercom_digital_twin.core.plots.research_comparison_plotter import (
    ResearchComparisonPlotter
)

from lasercom_digital_twin.core.plots.interactive_plotter import (
    InteractiveFigureManager,
    InteractiveStyleConfig,
    InteractionMode,
    make_interactive
)

__all__ = [
    # Style Configuration
    'PlotStyleConfig',
    'ControllerColors',
    'AxisColors',
    'configure_matplotlib_defaults',
    
    # Metrics Utilities
    'compute_tracking_metrics',
    'TrackingMetrics',
    
    # Plotters
    'ResearchComparisonPlotter',
    
    # Interactive Plotting
    'InteractiveFigureManager',
    'InteractiveStyleConfig',
    'InteractionMode',
    'make_interactive',
]

__version__ = '2.0.0'
