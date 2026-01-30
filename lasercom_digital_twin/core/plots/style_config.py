"""
Matplotlib Style Configuration for Publication-Quality Figures.

This module centralizes all visual styling constants and matplotlib
configuration for consistent, publication-ready output across the
entire lasercom_digital_twin project.

Style Standards
---------------
- IEEE Transactions: 3.5" single column, 7" double column
- AIAA: Similar to IEEE with specific font requirements
- LaTeX Typography: Computer Modern via STIXGeneral

Usage
-----
```python
from lasercom_digital_twin.core.plots.style_config import (
    configure_matplotlib_defaults,
    PlotStyleConfig,
    ControllerColors
)

# Apply global matplotlib settings
configure_matplotlib_defaults()

# Access color schemes
color = ControllerColors.PID  # '#1f77b4'
```

Author: Dr. S. Shahid Mustafa
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
from pathlib import Path
import matplotlib


class ControllerColors:
    """
    Color scheme for controller comparison traces.
    
    These colors are optimized for:
    - Colorblind accessibility (partial)
    - Black & white printing (distinct luminance)
    - Professional appearance
    """
    # Primary controller traces
    PID: str = '#1f77b4'       # Blue - Standard/Baseline
    FBL: str = '#ff7f0e'       # Orange - Advanced
    FBL_NDOB: str = '#2ca02c'  # Green - Optimal
    
    # Reference signals
    TARGET: str = '#000000'    # Black - Command/Reference
    COMMAND: str = '#2ca02c'   # Green - Same as optimal for consistency
    
    # Limits and thresholds
    THRESHOLD: str = '#d62728' # Red - Safety/Performance limits
    HANDOVER: str = '#ff7f00'  # Orange - FSM handover threshold
    SATURATION: str = '#d62728'# Red - Motor saturation limits
    
    # Disturbance analysis
    GROUND_TRUTH: str = '#9467bd'  # Purple - True values
    ESTIMATE: str = '#2ca02c'      # Green - Observer estimates
    
    # Classification by type
    @classmethod
    def by_controller(cls, name: str) -> str:
        """Get color by controller name string."""
        mapping = {
            'pid': cls.PID,
            'fbl': cls.FBL,
            'fbl_ndob': cls.FBL_NDOB,
            'fbl+ndob': cls.FBL_NDOB,
            'ndob': cls.FBL_NDOB
        }
        return mapping.get(name.lower(), cls.PID)


class AxisColors:
    """
    Color scheme for axis-specific traces (Az, El, X, Y).
    
    Consistent axis coloring across all figures ensures
    readers can instantly identify which data belongs to which axis.
    """
    # Gimbal axes
    AZIMUTH: str = '#1f77b4'   # Blue
    ELEVATION: str = '#d62728' # Red
    
    # FSM axes (optical frame)
    TIP: str = '#ff7f0e'       # Orange - X axis
    TILT: str = '#9467bd'      # Purple - Y axis
    
    # Aliases for convenience
    AZ: str = AZIMUTH
    EL: str = ELEVATION
    X: str = TIP
    Y: str = TILT


class DisturbanceColors:
    """Color scheme for environmental disturbance visualization."""
    TOTAL: str = '#2c3e50'     # Dark blue-gray
    WIND: str = '#e74c3c'      # Red
    VIBRATION: str = '#3498db' # Blue
    STRUCTURAL: str = '#9b59b6'# Purple
    NOISE: str = '#7f8c8d'     # Gray


@dataclass
class PlotStyleConfig:
    """
    Centralized configuration for all plotting parameters.
    
    This dataclass consolidates style settings that can be customized
    for different output targets (journal, presentation, web, etc.).
    
    Attributes
    ----------
    figure_sizes : Dict[str, Tuple[float, float]]
        Named figure sizes for different layouts
    dpi : int
        Resolution for saved figures
    linewidth_primary : float
        Line width for main data traces
    linewidth_secondary : float
        Line width for secondary/reference traces
    grid_alpha : float
        Transparency of grid lines
    grid_linestyle : str
        Grid line style
    legend_fontsize : int
        Font size for legend entries
    axis_label_fontsize : int
        Font size for axis labels
    title_fontsize : int
        Font size for subplot titles
    suptitle_fontsize : int
        Font size for figure super titles
    tick_fontsize : int
        Font size for axis tick labels
    """
    
    # Figure sizes (width, height) in inches
    figure_sizes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        '2x1': (10, 7),      # Standard 2-row vertical
        '2x2': (10, 7),      # 2x2 grid
        '1x2': (10, 7),      # Side-by-side
        '3x1': (10, 7),      # 3-row vertical
        '3x1_tall': (10, 9), # Extended 3-row
        'double_column': (14, 10),  # IEEE double column
    })
    
    # Resolution
    dpi: int = 300
    save_format: str = 'png'
    
    # Line properties
    linewidth_primary: float = 2.0
    linewidth_secondary: float = 1.5
    linewidth_threshold: float = 2.0
    alpha_primary: float = 0.9
    alpha_secondary: float = 0.7
    alpha_threshold: float = 0.6
    
    # Grid properties
    grid_alpha: float = 0.3
    grid_linestyle: str = ':'
    
    # Font sizes
    legend_fontsize: int = 14
    axis_label_fontsize: int = 14
    title_fontsize: int = 14
    suptitle_fontsize: int = 14
    tick_fontsize: int = 10
    
    # Legend properties
    legend_framealpha: float = 0.9
    legend_loc: str = 'best'
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: Path('figures_comparative'))
    
    def get_figure_size(self, layout: str) -> Tuple[float, float]:
        """Get figure size for a named layout."""
        return self.figure_sizes.get(layout, (10, 7))


def configure_matplotlib_defaults() -> None:
    """
    Apply publication-quality matplotlib defaults globally.
    
    This function configures matplotlib's rcParams for:
    - LaTeX-compatible typography (STIX fonts)
    - Appropriate font sizes for publications
    - High-quality figure rendering
    
    Call this once at module import or script start.
    """
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['figure.titlesize'] = 16
    
    # Additional publication settings
    matplotlib.rcParams['figure.dpi'] = 100  # Screen display
    matplotlib.rcParams['savefig.dpi'] = 300  # Saved figures
    matplotlib.rcParams['figure.constrained_layout.use'] = True
    matplotlib.rcParams['axes.grid'] = False  # Manual grid control
    matplotlib.rcParams['axes.axisbelow'] = True  # Grid behind data


# Apply defaults on import
configure_matplotlib_defaults()
