"""
Optical Beam Spot Analysis for Quadrant Photo Detector (QPD)

This module provides micron-level visualization of the laser beam spot
position on the QPD focal plane, essential for validating pointing accuracy
and understanding detector margin.

Key Features:
-------------
- 2D trajectory plots showing beam spot motion over time
- QPD boundary visualization with track-center reference
- RMS radius circle indicating typical pointing error
- Peak error marker showing worst-case excursion
- Time-colored trajectory for temporal analysis

Design Philosophy:
------------------
Focuses on immediate engineering insight: Can we see the spot position relative
to detector limits? What is the typical scatter (RMS)? Where are the worst
transients (peak)?
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from typing import Optional, Dict, List, Tuple, Union
import pandas as pd


class SpotPlotter:
    """
    Beam spot trajectory plotter for QPD focal plane analysis.
    
    This class generates publication-quality plots showing the laser beam
    spot's motion on the detector, with clear indicators of RMS performance,
    peak errors, and detector boundaries.
    
    Usage:
    ------
    >>> plotter = SpotPlotter(qpd_size=1000.0)  # 1000 µm detector
    >>> 
    >>> # Plot spot trajectory from telemetry
    >>> fig, ax = plotter.plot_spot_trajectory(
    ...     telemetry,
    ...     time_window=(0, 10),
    ...     show_rms=True,
    ...     show_peak=True
    ... )
    >>> plt.savefig('spot_analysis.png', dpi=300)
    >>> plt.show()
    """
    
    def __init__(
        self,
        qpd_size: float = 1000.0,  # µm
        qpd_shape: str = 'circular',  # 'circular' or 'square'
        target_rms: Optional[float] = None,  # µm
        figure_size: Tuple[int, int] = (10, 10)
    ):
        """
        Initialize spot plotter with detector specifications.
        
        Parameters
        ----------
        qpd_size : float
            QPD detector size [µm]
            For circular: diameter
            For square: side length
        qpd_shape : str
            Detector shape: 'circular' or 'square'
        target_rms : float, optional
            Target RMS pointing error [µm] for reference circle
        figure_size : tuple
            Figure size in inches (width, height)
        """
        self.qpd_size = qpd_size
        self.qpd_shape = qpd_shape
        self.target_rms = target_rms
        self.figure_size = figure_size
    
    def plot_spot_trajectory(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        show_rms: bool = True,
        show_peak: bool = True,
        show_percentiles: bool = False,
        percentiles: List[float] = [50, 90, 95],
        colormap: str = 'viridis',
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot beam spot trajectory on QPD focal plane.
        
        This is the primary method for visualizing pointing performance
        at the micron level. The plot shows:
        - Spot trajectory colored by time
        - QPD boundaries (detector limits)
        - Track center (origin)
        - RMS circle (typical pointing scatter)
        - Peak error marker (worst-case excursion)
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry data containing:
            - 'time': Time vector [s]
            - 'los_error_x' or 'spot_x': X position [µrad or µm]
            - 'los_error_y' or 'spot_y': Y position [µrad or µm]
        time_window : tuple, optional
            (start, end) time range to plot [s]
            If None, plots entire trajectory
        show_rms : bool
            Draw RMS radius circle
        show_peak : bool
            Mark peak error position
        show_percentiles : bool
            Draw percentile circles (50%, 90%, 95%)
        percentiles : list
            Percentile values to plot if show_percentiles=True
        colormap : str
            Matplotlib colormap name for trajectory coloring
        title : str, optional
            Plot title (auto-generated if None)
        ax : plt.Axes, optional
            Existing axes to plot on (creates new if None)
            
        Returns
        -------
        fig : plt.Figure
            Figure handle
        ax : plt.Axes
            Axes handle
            
        Example
        -------
        >>> plotter = SpotPlotter(qpd_size=1000.0)
        >>> fig, ax = plotter.plot_spot_trajectory(
        ...     telemetry,
        ...     time_window=(5, 15),
        ...     show_rms=True,
        ...     show_peak=True
        ... )
        >>> ax.set_title('Pointing Performance: Steady-State Tracking')
        >>> plt.savefig('pointing_analysis.png', dpi=300, bbox_inches='tight')
        """
        # Convert telemetry to arrays
        time, spot_x, spot_y = self._extract_spot_data(telemetry, time_window)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        else:
            fig = ax.figure
        
        # Set equal aspect ratio (critical for spatial accuracy)
        ax.set_aspect('equal')
        
        # Draw QPD boundaries
        self._draw_qpd_boundary(ax)
        
        # Draw track center (origin)
        ax.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, 
                label='Track Center', zorder=5)
        
        # Plot spot trajectory with time coloring
        self._plot_colored_trajectory(ax, spot_x, spot_y, time, colormap)
        
        # Compute and display performance metrics
        rms_radius = np.sqrt(np.mean(spot_x**2 + spot_y**2))
        peak_radius = np.max(np.sqrt(spot_x**2 + spot_y**2))
        peak_idx = np.argmax(np.sqrt(spot_x**2 + spot_y**2))
        
        # Draw RMS circle
        if show_rms:
            rms_circle = Circle(
                (0, 0), rms_radius,
                fill=False, edgecolor='red', linewidth=2,
                linestyle='--', label=f'RMS: {rms_radius:.2f} µm',
                zorder=3
            )
            ax.add_patch(rms_circle)
        
        # Draw target RMS circle (if specified)
        if self.target_rms is not None:
            target_circle = Circle(
                (0, 0), self.target_rms,
                fill=False, edgecolor='green', linewidth=2,
                linestyle=':', label=f'Target RMS: {self.target_rms:.2f} µm',
                alpha=0.7, zorder=2
            )
            ax.add_patch(target_circle)
        
        # Draw percentile circles
        if show_percentiles:
            for pct in percentiles:
                pct_radius = np.percentile(np.sqrt(spot_x**2 + spot_y**2), pct)
                pct_circle = Circle(
                    (0, 0), pct_radius,
                    fill=False, edgecolor='gray', linewidth=1,
                    linestyle='-.', alpha=0.5,
                    label=f'{pct}%ile: {pct_radius:.2f} µm',
                    zorder=1
                )
                ax.add_patch(pct_circle)
        
        # Mark peak error position
        if show_peak:
            ax.plot(
                spot_x[peak_idx], spot_y[peak_idx],
                'r*', markersize=20, markeredgewidth=1,
                markeredgecolor='darkred',
                label=f'Peak: {peak_radius:.2f} µm @ t={time[peak_idx]:.2f}s',
                zorder=6
            )
        
        # Labels and formatting
        ax.set_xlabel('X Position (µm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (µm)', fontsize=12, fontweight='bold')
        
        if title is None:
            if time_window is not None:
                title = f'Beam Spot Trajectory [{time_window[0]:.1f}-{time_window[1]:.1f}s]'
            else:
                title = 'Beam Spot Trajectory (Full Duration)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add performance summary text box
        summary_text = (
            f"RMS Radius: {rms_radius:.2f} µm\n"
            f"Peak Error: {peak_radius:.2f} µm\n"
            f"Samples: {len(time)}\n"
            f"Duration: {time[-1] - time[0]:.2f} s"
        )
        
        ax.text(
            0.02, 0.98, summary_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, family='monospace'
        )
        
        return fig, ax
    
    def plot_spot_heatmap(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        bins: int = 50,
        colormap: str = 'hot',
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 2D histogram (heatmap) of beam spot density.
        
        Shows where the beam spends most time on the detector,
        useful for identifying bias or systematic errors.
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry data
        time_window : tuple, optional
            Time range to analyze
        bins : int
            Number of bins for 2D histogram
        colormap : str
            Colormap for heatmap
        title : str, optional
            Plot title
        ax : plt.Axes, optional
            Existing axes
            
        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes
        """
        # Extract data
        time, spot_x, spot_y = self._extract_spot_data(telemetry, time_window)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        else:
            fig = ax.figure
        
        ax.set_aspect('equal')
        
        # Draw QPD boundary
        self._draw_qpd_boundary(ax)
        
        # Compute 2D histogram
        detector_range = self.qpd_size / 2
        hist_range = [[-detector_range, detector_range], 
                      [-detector_range, detector_range]]
        
        H, xedges, yedges = np.histogram2d(
            spot_x, spot_y, bins=bins, range=hist_range
        )
        
        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            H.T, extent=extent, origin='lower',
            cmap=colormap, aspect='auto', alpha=0.8
        )
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, label='Dwell Time (samples)')
        
        # Track center
        ax.plot(0, 0, 'w+', markersize=15, markeredgewidth=2, label='Track Center')
        
        # Labels
        ax.set_xlabel('X Position (µm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (µm)', fontsize=12, fontweight='bold')
        
        if title is None:
            title = 'Beam Spot Density Heatmap'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right')
        
        return fig, ax
    
    def plot_radial_error_histogram(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        bins: int = 50,
        show_stats: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot histogram of radial error distribution.
        
        Shows the statistical distribution of pointing error magnitude,
        useful for comparing against Gaussian assumptions and identifying
        outliers.
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry data
        time_window : tuple, optional
            Time range
        bins : int
            Number of histogram bins
        show_stats : bool
            Display statistics on plot
        ax : plt.Axes, optional
            Existing axes
            
        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes
        """
        # Extract data
        time, spot_x, spot_y = self._extract_spot_data(telemetry, time_window)
        
        # Compute radial error
        radial_error = np.sqrt(spot_x**2 + spot_y**2)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot histogram
        n, bins_edge, patches = ax.hist(
            radial_error, bins=bins,
            density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Empirical PDF'
        )
        
        # Fit Rayleigh distribution (theoretical for 2D Gaussian)
        from scipy.stats import rayleigh
        rms = np.sqrt(np.mean(radial_error**2))
        sigma_rayleigh = rms / np.sqrt(2)  # Convert RMS to Rayleigh scale
        
        x_fit = np.linspace(0, radial_error.max(), 200)
        pdf_fit = rayleigh.pdf(x_fit, scale=sigma_rayleigh)
        
        ax.plot(
            x_fit, pdf_fit,
            'r-', linewidth=2,
            label=f'Rayleigh Fit (σ={sigma_rayleigh:.2f} µm)'
        )
        
        # Mark RMS and percentiles
        ax.axvline(rms, color='red', linestyle='--', linewidth=2, 
                   label=f'RMS: {rms:.2f} µm')
        
        p50 = np.percentile(radial_error, 50)
        p95 = np.percentile(radial_error, 95)
        
        ax.axvline(p50, color='orange', linestyle=':', linewidth=2,
                   label=f'Median: {p50:.2f} µm')
        ax.axvline(p95, color='purple', linestyle='-.', linewidth=2,
                   label=f'95%: {p95:.2f} µm')
        
        # Labels
        ax.set_xlabel('Radial Error (µm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title('Radial Error Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Statistics box
        if show_stats:
            stats_text = (
                f"Mean: {radial_error.mean():.2f} µm\n"
                f"Std: {radial_error.std():.2f} µm\n"
                f"RMS: {rms:.2f} µm\n"
                f"Max: {radial_error.max():.2f} µm\n"
                f"Samples: {len(radial_error)}"
            )
            ax.text(
                0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace'
            )
        
        return fig, ax
    
    def _extract_spot_data(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and process spot position data from telemetry.
        
        Handles both dict and DataFrame inputs, and performs unit conversion
        if needed (µrad → µm assumes focal length for conversion).
        """
        # Convert to DataFrame if dict
        if isinstance(telemetry, dict):
            df = pd.DataFrame(telemetry)
        else:
            df = telemetry.copy()
        
        # Extract time
        time = df['time'].values
        
        # Extract spot positions (try multiple key names)
        if 'spot_x' in df.columns:
            spot_x = df['spot_x'].values
            spot_y = df['spot_y'].values
        elif 'los_error_x' in df.columns:
            # Convert µrad to µm (assume focal length = 100 mm = 100000 µm)
            # 1 µrad × 100000 µm = 0.1 µm
            # For visualization, scale to reasonable detector size
            focal_length_um = 100000.0  # 100 mm focal length
            spot_x = df['los_error_x'].values * focal_length_um / 1e6  # µrad to µm
            spot_y = df['los_error_y'].values * focal_length_um / 1e6
        else:
            raise ValueError("Telemetry must contain 'spot_x/spot_y' or 'los_error_x/los_error_y'")
        
        # Apply time window
        if time_window is not None:
            mask = (time >= time_window[0]) & (time <= time_window[1])
            time = time[mask]
            spot_x = spot_x[mask]
            spot_y = spot_y[mask]
        
        return time, spot_x, spot_y
    
    def _draw_qpd_boundary(self, ax: plt.Axes):
        """Draw QPD detector boundary on axes."""
        if self.qpd_shape == 'circular':
            # Circular detector
            radius = self.qpd_size / 2
            boundary = Circle(
                (0, 0), radius,
                fill=False, edgecolor='black', linewidth=2,
                label=f'QPD Boundary (Ø{self.qpd_size:.0f} µm)',
                zorder=1
            )
            ax.add_patch(boundary)
            
            # Set axis limits with margin
            margin = radius * 0.2
            ax.set_xlim(-radius - margin, radius + margin)
            ax.set_ylim(-radius - margin, radius + margin)
            
        elif self.qpd_shape == 'square':
            # Square detector
            half_size = self.qpd_size / 2
            boundary = Rectangle(
                (-half_size, -half_size), self.qpd_size, self.qpd_size,
                fill=False, edgecolor='black', linewidth=2,
                label=f'QPD Boundary ({self.qpd_size:.0f}×{self.qpd_size:.0f} µm)',
                zorder=1
            )
            ax.add_patch(boundary)
            
            # Set axis limits
            margin = half_size * 0.2
            ax.set_xlim(-half_size - margin, half_size + margin)
            ax.set_ylim(-half_size - margin, half_size + margin)
        
        # Draw quadrant lines
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    def _plot_colored_trajectory(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        time: np.ndarray,
        colormap: str
    ):
        """
        Plot trajectory with time-based color gradient.
        
        Uses LineCollection for efficient rendering of colored segments.
        """
        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create line collection with time-based colors
        lc = LineCollection(
            segments,
            cmap=colormap,
            linewidth=1.5,
            alpha=0.8,
            zorder=4
        )
        
        # Set color based on time
        lc.set_array(time)
        
        # Add to axes
        line = ax.add_collection(lc)
        
        # Add colorbar
        cbar = ax.figure.colorbar(line, ax=ax, label='Time (s)', pad=0.02)
        
        # Mark start and end points
        ax.plot(x[0], y[0], 'go', markersize=10, markeredgewidth=2,
                markeredgecolor='darkgreen', label='Start', zorder=7)
        ax.plot(x[-1], y[-1], 'rs', markersize=10, markeredgewidth=2,
                markeredgecolor='darkred', label='End', zorder=7)
