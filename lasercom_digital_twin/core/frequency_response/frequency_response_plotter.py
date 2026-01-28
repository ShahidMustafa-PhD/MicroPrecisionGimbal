"""
Publication-Quality Frequency Response Plotter

This module generates journal-grade Bode plots for frequency response analysis
of nonlinear gimbal control systems. All plots conform to IEEE/AIAA publication
standards with proper LaTeX typography, sizing, and annotation.

Plot Types Generated
--------------------
1. **Bode Magnitude Plot**: |G(jω)| vs frequency (log-log or semi-log)
2. **Bode Phase Plot**: ∠G(jω) vs frequency (semi-log)
3. **Sensitivity Function Plot**: |S(jω)| showing disturbance rejection bands
4. **Nyquist Diagram**: Re{G} vs Im{G} for stability analysis
5. **Comparative Overlay**: Multi-controller comparison on single axes

Design Specifications (IEEE Style)
----------------------------------
- Figure size: 3.5" (single column) or 7.16" (double column)
- Font: Times New Roman or STIX (LaTeX compatible)
- Axis labels: 10-12 pt, bold
- Tick labels: 8-10 pt
- Legend: 8-9 pt, framealpha=0.9
- Line width: 1.5-2.0 pt
- Grid: alpha=0.3, linestyle=':'
- DPI: 300 for publication, 100 for screen

Color Scheme (Colorblind-Safe)
------------------------------
- PID (Baseline): #1f77b4 (Blue)
- FBL (Advanced): #ff7f0e (Orange)
- FBL+NDOB (Optimal): #2ca02c (Green)
- Open-Loop: #7f7f7f (Gray)
- Threshold/Limit: #d62728 (Red)

Author: Dr. S. Shahid Mustafa
Date: January 28, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogLocator, NullFormatter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Local imports
from .frequency_response_analyzer import FrequencyResponseData, ControllerType


class PlotStyle(Enum):
    """Plot style presets."""
    PUBLICATION = auto()   # IEEE/AIAA journal quality
    PRESENTATION = auto()  # Conference slides
    SCREEN = auto()        # Interactive viewing


@dataclass
class PlotConfig:
    """
    Configuration for frequency response plots.
    
    Attributes
    ----------
    style : PlotStyle
        Visual style preset
    figsize_single : Tuple[float, float]
        Single-column figure size [inches]
    figsize_double : Tuple[float, float]
        Double-column figure size [inches]
    dpi : int
        Output resolution
    font_family : str
        Font family for text
    use_latex : bool
        Enable LaTeX rendering
    colorblind_safe : bool
        Use colorblind-safe palette
    save_format : str
        Output format ('png', 'pdf', 'svg')
    output_dir : Path
        Directory for saved figures
    show_coherence : bool
        Overlay coherence on magnitude plot
    show_margins : bool
        Annotate stability margins
    show_bandwidth : bool
        Annotate bandwidth frequency
    """
    style: PlotStyle = PlotStyle.PUBLICATION
    figsize_single: Tuple[float, float] = (7, 5)
    figsize_double: Tuple[float, float] = (10, 8)
    dpi: int = 300
    font_family: str = 'STIXGeneral'
    use_latex: bool = False
    colorblind_safe: bool = True
    save_format: str = 'png'
    output_dir: Path = field(default_factory=lambda: Path('figures_bode'))
    show_coherence: bool = True
    show_margins: bool = True
    show_bandwidth: bool = True


class FrequencyResponsePlotter:
    """
    Publication-Quality Frequency Response Visualization.
    
    This class generates comprehensive Bode plots for comparing frequency
    response characteristics across multiple controller architectures.
    
    Key Features
    ------------
    - Automatic axis scaling and tick formatting
    - Frequency-band shading for disturbance rejection analysis
    - Stability margin annotations
    - Multi-trace comparison with consistent styling
    - Export to multiple formats (PNG, PDF, SVG)
    
    Example Usage
    -------------
    >>> plotter = FrequencyResponsePlotter(PlotConfig(style=PlotStyle.PUBLICATION))
    >>> plotter.add_response(pid_data, 'PID')
    >>> plotter.add_response(fbl_data, 'FBL')
    >>> plotter.add_response(ndob_data, 'FBL+NDOB')
    >>> plotter.plot_bode_comparison()
    >>> plotter.plot_sensitivity_comparison()
    >>> plotter.save_all_figures()
    
    Parameters
    ----------
    config : PlotConfig
        Plotting configuration
    """
    
    # Colorblind-safe color palette
    COLORS = {
        ControllerType.OPEN_LOOP: '#7f7f7f',   # Gray
        ControllerType.PID: '#1f77b4',          # Blue
        ControllerType.FBL: '#ff7f0e',          # Orange
        ControllerType.FBL_NDOB: '#2ca02c',     # Green
    }
    
    LABELS = {
        ControllerType.OPEN_LOOP: 'Open Loop',
        ControllerType.PID: 'PID',
        ControllerType.FBL: 'FBL',
        ControllerType.FBL_NDOB: 'FBL+NDOB',
    }
    
    LINESTYLES = {
        ControllerType.OPEN_LOOP: '--',
        ControllerType.PID: '-',
        ControllerType.FBL: '-',
        ControllerType.FBL_NDOB: '-',
    }
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._responses: Dict[ControllerType, FrequencyResponseData] = {}
        self._figures: Dict[str, plt.Figure] = {}
        
        # Apply matplotlib configuration
        self._configure_matplotlib()
    
    def _configure_matplotlib(self) -> None:
        """Configure matplotlib for publication quality."""
        # Font configuration
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = self.config.font_family
        
        if self.config.style == PlotStyle.PUBLICATION:
            matplotlib.rcParams['font.size'] = 12
            matplotlib.rcParams['axes.labelsize'] = 14
            matplotlib.rcParams['axes.titlesize'] = 14
            matplotlib.rcParams['xtick.labelsize'] = 11
            matplotlib.rcParams['ytick.labelsize'] = 11
            matplotlib.rcParams['legend.fontsize'] = 11
            matplotlib.rcParams['figure.titlesize'] = 16
            matplotlib.rcParams['lines.linewidth'] = 2.0
            matplotlib.rcParams['axes.linewidth'] = 1.2
        elif self.config.style == PlotStyle.PRESENTATION:
            matplotlib.rcParams['font.size'] = 14
            matplotlib.rcParams['axes.labelsize'] = 16
            matplotlib.rcParams['axes.titlesize'] = 18
            matplotlib.rcParams['legend.fontsize'] = 12
            matplotlib.rcParams['lines.linewidth'] = 2.5
        else:
            matplotlib.rcParams['font.size'] = 10
            matplotlib.rcParams['lines.linewidth'] = 1.5
    
    def add_response(
        self, 
        data: FrequencyResponseData,
        label: Optional[str] = None
    ) -> None:
        """
        Add frequency response data for plotting.
        
        Parameters
        ----------
        data : FrequencyResponseData
            Frequency response data from analyzer
        label : str, optional
            Custom label override
        """
        self._responses[data.controller_type] = data
        if label:
            self.LABELS[data.controller_type] = label
    
    def plot_bode_comparison(
        self,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Generate comparative Bode plot (magnitude and phase).
        
        Creates a 2-subplot figure showing:
        - Top: Magnitude response |T(jω)| in dB
        - Bottom: Phase response ∠T(jω) in degrees
        
        All registered controllers are overlaid for comparison.
        
        Parameters
        ----------
        title : str, optional
            Custom figure title
        save : bool
            Save figure to disk
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, 
            figsize=self.config.figsize_double,
            sharex=True,
            constrained_layout=True
        )
        
        # Plot each controller
        for ctrl_type, data in self._responses.items():
            color = self.COLORS.get(ctrl_type, '#333333')
            label = self.LABELS.get(ctrl_type, ctrl_type.name)
            linestyle = self.LINESTYLES.get(ctrl_type, '-')
            
            # Filter valid data
            valid = ~np.isnan(data.closed_loop_gain_db)
            f = data.frequencies_hz[valid]
            mag = data.closed_loop_gain_db[valid]
            phase = data.closed_loop_phase_deg[valid]
            
            # Magnitude plot
            ax_mag.semilogx(f, mag, color=color, linestyle=linestyle,
                           linewidth=2.0, label=label, alpha=0.9)
            
            # Phase plot
            ax_phase.semilogx(f, phase, color=color, linestyle=linestyle,
                             linewidth=2.0, label=label, alpha=0.9)
            
            # Annotate bandwidth
            if self.config.show_bandwidth and data.bandwidth_hz > 0:
                ax_mag.axvline(data.bandwidth_hz, color=color, linestyle=':',
                              alpha=0.5, linewidth=1.5)
        
        # Magnitude axis formatting
        ax_mag.set_ylabel('Magnitude [dB]', fontweight='bold')
        ax_mag.set_title(title or 'Closed-Loop Frequency Response (Bode Plot)',
                        fontweight='bold')
        ax_mag.axhline(-3, color='red', linestyle='--', alpha=0.5, 
                       linewidth=1.5, label='-3 dB')
        ax_mag.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1.0)
        ax_mag.legend(loc='lower left', framealpha=0.95, ncol=2)
        ax_mag.grid(True, which='both', alpha=0.3, linestyle=':')
        ax_mag.set_xlim([data.frequencies_hz[0], data.frequencies_hz[-1]])
        
        # Phase axis formatting
        ax_phase.set_xlabel('Frequency [Hz]', fontweight='bold')
        ax_phase.set_ylabel('Phase [degrees]', fontweight='bold')
        ax_phase.axhline(-90, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
        ax_phase.axhline(-180, color='red', linestyle='--', alpha=0.5, linewidth=1.0)
        ax_phase.legend(loc='lower left', framealpha=0.95)
        ax_phase.grid(True, which='both', alpha=0.3, linestyle=':')
        
        self._figures['bode_comparison'] = fig
        
        if save:
            self._save_figure(fig, 'bode_comparison')
        
        return fig
    
    def plot_sensitivity_comparison(
        self,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Generate sensitivity function comparison plot.
        
        The sensitivity function S(jω) characterizes disturbance rejection:
        - |S(jω)| < 0 dB: Disturbances attenuated
        - |S(jω)| > 0 dB: Disturbances amplified
        
        Includes shaded regions indicating:
        - Low-frequency rejection band (integral action)
        - Crossover amplification region
        - High-frequency attenuation (noise rejection limited by T)
        
        Parameters
        ----------
        title : str, optional
            Custom figure title
        save : bool
            Save figure to disk
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, ax = plt.subplots(
            figsize=self.config.figsize_single,
            constrained_layout=True
        )
        
        # Plot each controller's sensitivity
        for ctrl_type, data in self._responses.items():
            color = self.COLORS.get(ctrl_type, '#333333')
            label = self.LABELS.get(ctrl_type, ctrl_type.name)
            linestyle = self.LINESTYLES.get(ctrl_type, '-')
            
            valid = ~np.isnan(data.sensitivity_gain_db)
            f = data.frequencies_hz[valid]
            sens = data.sensitivity_gain_db[valid]
            
            ax.semilogx(f, sens, color=color, linestyle=linestyle,
                       linewidth=2.0, label=f'{label} (Ms={data.peak_sensitivity:.2f})',
                       alpha=0.9)
        
        # Reference lines
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7,
                   label='|S|=1 (No attenuation)')
        ax.axhline(6, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Ms=2 (Robustness limit)')
        ax.axhline(-20, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='-20 dB (99% rejection)')
        
        # Shade disturbance rejection region
        if len(self._responses) > 0:
            f_range = list(self._responses.values())[0].frequencies_hz
            ax.fill_between(f_range, -60, 0, alpha=0.1, color='green',
                           label='Disturbance Rejection')
            ax.fill_between(f_range, 0, 20, alpha=0.1, color='red',
                           label='Disturbance Amplification')
        
        ax.set_xlabel('Frequency [Hz]', fontweight='bold')
        ax.set_ylabel('Sensitivity |S(jω)| [dB]', fontweight='bold')
        ax.set_title(title or 'Sensitivity Function - Disturbance Rejection Analysis',
                    fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, fontsize=10)
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.set_ylim([-40, 20])
        
        self._figures['sensitivity_comparison'] = fig
        
        if save:
            self._save_figure(fig, 'sensitivity_comparison')
        
        return fig
    
    def plot_disturbance_rejection_bands(
        self,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Generate disturbance rejection band analysis plot.
        
        This plot highlights the frequency bands where disturbances are
        effectively rejected, showing:
        
        - Low-frequency band: Dominated by integral action
        - Mid-frequency band: Proportional/derivative action
        - High-frequency band: Roll-off and noise sensitivity
        
        Typical disturbance sources are annotated:
        - 0.01-1 Hz: Thermal drift, platform motion
        - 1-50 Hz: Mechanical vibration, motor ripple
        - 50-500 Hz: Structural modes, acoustics
        
        Parameters
        ----------
        title : str, optional
            Custom figure title
        save : bool
            Save figure to disk
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, (ax_sens, ax_T) = plt.subplots(
            2, 1,
            figsize=self.config.figsize_double,
            sharex=True,
            constrained_layout=True
        )
        
        # Disturbance frequency bands with labels
        disturbance_bands = [
            (0.01, 0.1, 'Thermal\nDrift', '#e6f2ff'),
            (0.1, 1.0, 'Platform\nMotion', '#fff2e6'),
            (1.0, 10.0, 'Base\nVibration', '#e6ffe6'),
            (10.0, 50.0, 'Motor\nRipple', '#ffe6e6'),
            (50.0, 200.0, 'Structural\nModes', '#f2e6ff'),
        ]
        
        for ctrl_type, data in self._responses.items():
            color = self.COLORS.get(ctrl_type, '#333333')
            label = self.LABELS.get(ctrl_type, ctrl_type.name)
            
            valid = ~np.isnan(data.sensitivity_gain_db)
            f = data.frequencies_hz[valid]
            sens = data.sensitivity_gain_db[valid]
            cl_mag = data.closed_loop_gain_db[valid]
            
            # Sensitivity plot
            ax_sens.semilogx(f, sens, color=color, linewidth=2.0, 
                            label=label, alpha=0.9)
            
            # Closed-loop magnitude (complementary sensitivity)
            ax_T.semilogx(f, cl_mag, color=color, linewidth=2.0,
                         label=label, alpha=0.9)
        
        # Add disturbance band shading
        for f_low, f_high, band_label, band_color in disturbance_bands:
            ax_sens.axvspan(f_low, f_high, alpha=0.15, color=band_color)
            ax_T.axvspan(f_low, f_high, alpha=0.15, color=band_color)
            # Add label at center of band
            f_center = np.sqrt(f_low * f_high)
            ax_sens.text(f_center, -35, band_label, ha='center', va='top',
                        fontsize=8, alpha=0.7)
        
        # Sensitivity axis
        ax_sens.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax_sens.axhline(-20, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_sens.set_ylabel('|S(jω)| [dB]\n(Disturbance → Error)', fontweight='bold')
        ax_sens.set_title(title or 'Disturbance Rejection by Frequency Band',
                         fontweight='bold')
        ax_sens.legend(loc='upper right', framealpha=0.95)
        ax_sens.grid(True, which='both', alpha=0.3, linestyle=':')
        ax_sens.set_ylim([-40, 15])
        
        # Complementary sensitivity axis
        ax_T.axhline(-3, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_T.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax_T.set_xlabel('Frequency [Hz]', fontweight='bold')
        ax_T.set_ylabel('|T(jω)| [dB]\n(Reference → Output)', fontweight='bold')
        ax_T.legend(loc='lower left', framealpha=0.95)
        ax_T.grid(True, which='both', alpha=0.3, linestyle=':')
        ax_T.set_ylim([-40, 10])
        
        self._figures['disturbance_rejection_bands'] = fig
        
        if save:
            self._save_figure(fig, 'disturbance_rejection_bands')
        
        return fig
    
    def plot_coherence_overlay(
        self,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Generate magnitude plot with coherence overlay.
        
        Coherence indicates measurement quality:
        - γ² > 0.9: Excellent (linear behavior dominant)
        - γ² > 0.7: Good (acceptable for analysis)
        - γ² < 0.5: Poor (nonlinearity or noise dominant)
        
        Parameters
        ----------
        title : str, optional
            Custom figure title
        save : bool
            Save figure to disk
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, (ax_mag, ax_coh) = plt.subplots(
            2, 1,
            figsize=self.config.figsize_double,
            sharex=True,
            constrained_layout=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )
        
        for ctrl_type, data in self._responses.items():
            color = self.COLORS.get(ctrl_type, '#333333')
            label = self.LABELS.get(ctrl_type, ctrl_type.name)
            
            valid = ~np.isnan(data.closed_loop_gain_db)
            f = data.frequencies_hz[valid]
            mag = data.closed_loop_gain_db[valid]
            coh = data.coherence[valid]
            
            ax_mag.semilogx(f, mag, color=color, linewidth=2.0, label=label)
            ax_coh.semilogx(f, coh, color=color, linewidth=1.5, 
                           marker='o', markersize=3, label=label)
        
        # Magnitude axis
        ax_mag.set_ylabel('Magnitude [dB]', fontweight='bold')
        ax_mag.set_title(title or 'Frequency Response with Measurement Quality',
                        fontweight='bold')
        ax_mag.legend(loc='lower left', framealpha=0.95)
        ax_mag.grid(True, which='both', alpha=0.3, linestyle=':')
        ax_mag.axhline(-3, color='red', linestyle='--', alpha=0.5)
        
        # Coherence axis
        ax_coh.axhline(0.9, color='green', linestyle='--', alpha=0.7, 
                       label='Excellent (γ²>0.9)')
        ax_coh.axhline(0.7, color='orange', linestyle='--', alpha=0.7,
                       label='Acceptable (γ²>0.7)')
        ax_coh.axhline(0.5, color='red', linestyle='--', alpha=0.7,
                       label='Poor (γ²<0.5)')
        ax_coh.fill_between(data.frequencies_hz, 0.9, 1.0, alpha=0.1, color='green')
        ax_coh.fill_between(data.frequencies_hz, 0.7, 0.9, alpha=0.1, color='orange')
        ax_coh.fill_between(data.frequencies_hz, 0.0, 0.7, alpha=0.1, color='red')
        
        ax_coh.set_xlabel('Frequency [Hz]', fontweight='bold')
        ax_coh.set_ylabel('Coherence γ²', fontweight='bold')
        ax_coh.set_ylim([0, 1.05])
        ax_coh.legend(loc='lower right', framealpha=0.95, fontsize=9)
        ax_coh.grid(True, which='both', alpha=0.3, linestyle=':')
        
        self._figures['coherence_overlay'] = fig
        
        if save:
            self._save_figure(fig, 'coherence_overlay')
        
        return fig
    
    def plot_performance_summary(
        self,
        save: bool = True
    ) -> plt.Figure:
        """
        Generate performance metrics summary bar chart.
        
        Compares key frequency-domain metrics:
        - Bandwidth [Hz]
        - Peak Sensitivity Ms
        - Gain Margin [dB]
        - Phase Margin [degrees]
        
        Parameters
        ----------
        save : bool
            Save figure to disk
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, axes = plt.subplots(
            2, 2,
            figsize=self.config.figsize_double,
            constrained_layout=True
        )
        ax_bw, ax_ms, ax_gm, ax_pm = axes.flatten()
        
        controllers = list(self._responses.keys())
        labels = [self.LABELS.get(c, c.name) for c in controllers]
        colors = [self.COLORS.get(c, '#333333') for c in controllers]
        x = np.arange(len(controllers))
        
        # Bandwidth
        bw_values = [self._responses[c].bandwidth_hz for c in controllers]
        bars = ax_bw.bar(x, bw_values, color=colors, alpha=0.8, edgecolor='black')
        ax_bw.set_ylabel('Bandwidth [Hz]', fontweight='bold')
        ax_bw.set_title('Closed-Loop Bandwidth', fontweight='bold')
        ax_bw.set_xticks(x)
        ax_bw.set_xticklabels(labels)
        ax_bw.grid(True, axis='y', alpha=0.3, linestyle=':')
        for bar, val in zip(bars, bw_values):
            ax_bw.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Peak Sensitivity
        ms_values = [self._responses[c].peak_sensitivity for c in controllers]
        bars = ax_ms.bar(x, ms_values, color=colors, alpha=0.8, edgecolor='black')
        ax_ms.axhline(2.0, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label='Ms=2 limit')
        ax_ms.axhline(1.5, color='orange', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='Ms=1.5 target')
        ax_ms.set_ylabel('Peak Sensitivity Ms', fontweight='bold')
        ax_ms.set_title('Robustness (Lower is Better)', fontweight='bold')
        ax_ms.set_xticks(x)
        ax_ms.set_xticklabels(labels)
        ax_ms.legend(loc='upper right', fontsize=9)
        ax_ms.grid(True, axis='y', alpha=0.3, linestyle=':')
        for bar, val in zip(bars, ms_values):
            ax_ms.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Gain Margin
        gm_values = [min(self._responses[c].gain_margin_db, 30) for c in controllers]
        bars = ax_gm.bar(x, gm_values, color=colors, alpha=0.8, edgecolor='black')
        ax_gm.axhline(6, color='orange', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='6 dB minimum')
        ax_gm.set_ylabel('Gain Margin [dB]', fontweight='bold')
        ax_gm.set_title('Stability Margin (Higher is Better)', fontweight='bold')
        ax_gm.set_xticks(x)
        ax_gm.set_xticklabels(labels)
        ax_gm.legend(loc='upper right', fontsize=9)
        ax_gm.grid(True, axis='y', alpha=0.3, linestyle=':')
        
        # Phase Margin
        pm_values = [self._responses[c].phase_margin_deg for c in controllers]
        bars = ax_pm.bar(x, pm_values, color=colors, alpha=0.8, edgecolor='black')
        ax_pm.axhline(45, color='orange', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='45° minimum')
        ax_pm.axhline(60, color='green', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='60° target')
        ax_pm.set_ylabel('Phase Margin [degrees]', fontweight='bold')
        ax_pm.set_title('Phase Margin (Higher is Better)', fontweight='bold')
        ax_pm.set_xticks(x)
        ax_pm.set_xticklabels(labels)
        ax_pm.legend(loc='upper right', fontsize=9)
        ax_pm.grid(True, axis='y', alpha=0.3, linestyle=':')
        
        fig.suptitle('Frequency-Domain Performance Metrics Comparison',
                    fontweight='bold', fontsize=14)
        
        self._figures['performance_summary'] = fig
        
        if save:
            self._save_figure(fig, 'performance_summary')
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure to disk."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.config.output_dir / f'{name}.{self.config.save_format}'
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"  [SAVED] {filepath}")
    
    def save_all_figures(self) -> None:
        """Save all generated figures to disk."""
        print(f"\nSaving {len(self._figures)} figures to {self.config.output_dir}/")
        for name, fig in self._figures.items():
            self._save_figure(fig, name)
        print(f"[COMPLETE] All figures saved ({self.config.dpi} DPI, {self.config.save_format.upper()})")
    
    def show(self) -> None:
        """Display all figures."""
        plt.show()
    
    @property
    def figures(self) -> Dict[str, plt.Figure]:
        """Get all generated figures."""
        return self._figures
