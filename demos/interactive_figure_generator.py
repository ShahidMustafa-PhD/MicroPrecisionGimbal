#!/usr/bin/env python3
"""
Interactive Figure Generator for Control Engineering Publications

This script creates publication-quality figures with interactive annotation
capabilities for 2-DOF gimbal control system visualization.

Features:
---------
- Interactive zoomed inset with connector lines
- Keyboard-triggered annotations:
  - 'v': Place vertical dashed line at click location
  - 'c': Place highlight circle at click location
  - 't': Add text label at click location
  - 'z': Define zoom region (click two corners)
  - 's': Save figure to file
  - 'u': Undo last annotation
  - 'h': Show help overlay
  - 'q': Quit and save
- LaTeX rendering for all labels
- Publication-standard sizing (7x5 inches, 300 DPI)
- Professional Seaborn-inspired color palette

Usage:
------
    python interactive_figure_generator.py

Author: Dr. S. Shahid Mustafa
Date: January 30, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive features
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.widgets import TextBox, Button
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto
import json
from pathlib import Path


# =============================================================================
# Configuration and Style Settings
# =============================================================================

class InteractionMode(Enum):
    """Available interaction modes."""
    NONE = auto()
    VERTICAL_LINE = auto()
    CIRCLE = auto()
    TEXT = auto()
    ZOOM_START = auto()
    ZOOM_END = auto()


@dataclass
class ColorPalette:
    """Professional color palette inspired by Seaborn 'deep'."""
    reference: str = '#1f77b4'      # Blue - reference signal
    measured: str = '#ff7f0e'       # Orange - measured signal
    ndob: str = '#2ca02c'           # Green - NDOB corrected
    error: str = '#d62728'          # Red - error/highlight
    disturbance: str = '#9467bd'    # Purple - disturbance
    annotation: str = '#8c564b'     # Brown - annotations
    grid: str = '#cccccc'           # Light gray - grid
    background: str = '#fafafa'     # Off-white background
    
    # Secondary palette for additional traces
    secondary: Tuple[str, ...] = (
        '#17becf',  # Cyan
        '#bcbd22',  # Yellow-green
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    )


@dataclass
class FigureConfig:
    """Publication-standard figure configuration."""
    # Size for double-column journal (inches)
    width: float = 7.0
    height: float = 5.0
    dpi: int = 300
    
    # Font sizes (points)
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    annotation_size: int = 9
    
    # Line properties
    line_width: float = 1.5
    reference_style: str = '--'
    measured_style: str = '-'
    
    # Inset properties
    inset_width: str = "40%"
    inset_height: str = "35%"
    inset_loc: str = 'upper right'
    
    # Output formats
    output_formats: Tuple[str, ...] = ('png', 'pdf', 'svg')


@dataclass
class Annotation:
    """Represents a user-placed annotation."""
    type: str  # 'vline', 'circle', 'text'
    x: float
    y: float
    artist: Any = None  # matplotlib artist object
    text: str = ""
    properties: Dict = field(default_factory=dict)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

class GimbalSimulator:
    """
    Generate synthetic 2-DOF gimbal tracking data with disturbances.
    
    Simulates a step response with:
    - Reference command (step)
    - Measured position (with dynamics)
    - NDOB-corrected response
    - Wind gust disturbances
    """
    
    def __init__(
        self,
        duration: float = 10.0,
        dt: float = 0.001,
        seed: int = 42
    ):
        self.duration = duration
        self.dt = dt
        self.t = np.arange(0, duration, dt)
        self.n_samples = len(self.t)
        np.random.seed(seed)
        
        # System parameters
        self.natural_freq = 15.0  # rad/s
        self.damping = 0.7
        self.ndob_bandwidth = 50.0  # rad/s
        
    def generate_reference(
        self,
        step_time: float = 1.0,
        step_amplitude: float = 0.1,  # rad (~5.7 deg)
        ramp_rate: float = 0.0
    ) -> np.ndarray:
        """Generate step reference command."""
        ref = np.zeros(self.n_samples)
        step_idx = int(step_time / self.dt)
        ref[step_idx:] = step_amplitude
        
        # Add optional ramp
        if ramp_rate > 0:
            ramp = ramp_rate * (self.t - step_time)
            ramp[ramp < 0] = 0
            ref += ramp
            
        return ref
    
    def generate_disturbance(
        self,
        gust_times: List[float] = [3.0, 6.0],
        gust_amplitudes: List[float] = [0.02, -0.015],
        gust_durations: List[float] = [0.5, 0.3],
        noise_std: float = 0.001
    ) -> np.ndarray:
        """Generate wind gust disturbances."""
        disturbance = np.zeros(self.n_samples)
        
        # Add discrete gusts
        for t_gust, amp, dur in zip(gust_times, gust_amplitudes, gust_durations):
            idx_start = int(t_gust / self.dt)
            idx_end = int((t_gust + dur) / self.dt)
            
            # Smooth gust profile (raised cosine)
            n_gust = idx_end - idx_start
            if n_gust > 0:
                gust_profile = amp * 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_gust) / n_gust))
                disturbance[idx_start:idx_end] += gust_profile
        
        # Add colored noise (low-pass filtered white noise)
        white_noise = np.random.randn(self.n_samples) * noise_std
        disturbance += gaussian_filter1d(white_noise, sigma=50)
        
        return disturbance
    
    def simulate_response(
        self,
        reference: np.ndarray,
        disturbance: np.ndarray,
        with_ndob: bool = False
    ) -> np.ndarray:
        """
        Simulate closed-loop response.
        
        Uses a second-order transfer function model.
        """
        # Create second-order system
        wn = self.natural_freq
        zeta = self.damping
        
        # Closed-loop transfer function: wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        num = [wn**2]
        den = [1, 2*zeta*wn, wn**2]
        system = signal.TransferFunction(num, den)
        system_d = system.to_discrete(self.dt, method='tustin')
        
        # Simulate reference tracking
        tout, response = signal.dlsim(system_d, reference)
        response = response.flatten()
        
        # Add disturbance effect (disturbance enters after plant)
        if with_ndob:
            # NDOB reduces disturbance effect significantly
            # Model as high-pass filtered disturbance rejection
            ndob_cutoff = self.ndob_bandwidth / (2 * np.pi)
            b_hp, a_hp = signal.butter(2, ndob_cutoff * self.dt * 2, btype='high')
            disturbance_effect = signal.filtfilt(b_hp, a_hp, disturbance) * 0.2
        else:
            # Without NDOB, disturbance has larger effect
            disturbance_effect = disturbance * 0.8
            
        response += disturbance_effect
        
        # Add measurement noise
        response += np.random.randn(self.n_samples) * 0.0005
        
        return response
    
    def generate_dataset(self) -> Dict[str, np.ndarray]:
        """Generate complete dataset for visualization."""
        reference = self.generate_reference(step_time=1.0, step_amplitude=0.1)
        disturbance = self.generate_disturbance(
            gust_times=[3.0, 6.5],
            gust_amplitudes=[0.025, -0.018],
            gust_durations=[0.6, 0.4]
        )
        
        measured_no_ndob = self.simulate_response(reference, disturbance, with_ndob=False)
        measured_with_ndob = self.simulate_response(reference, disturbance, with_ndob=True)
        
        return {
            'time': self.t,
            'reference': reference,
            'measured': measured_no_ndob,
            'ndob': measured_with_ndob,
            'disturbance': disturbance
        }


# =============================================================================
# Interactive Figure Class
# =============================================================================

class InteractiveFigureGenerator:
    """
    Professional-grade interactive figure generator with annotation tools.
    
    Keyboard Controls:
    ------------------
    v : Vertical line mode - click to place dashed vertical line
    c : Circle mode - click to place highlight circle
    t : Text mode - click to add text label
    z : Zoom mode - click two corners to define inset region
    s : Save figure to all configured formats
    u : Undo last annotation
    h : Toggle help overlay
    q : Quit (prompts to save)
    Esc : Cancel current mode
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        config: Optional[FigureConfig] = None,
        colors: Optional[ColorPalette] = None
    ):
        self.data = data
        self.config = config or FigureConfig()
        self.colors = colors or ColorPalette()
        
        # State
        self.mode = InteractionMode.NONE
        self.annotations: List[Annotation] = []
        self.zoom_start: Optional[Tuple[float, float]] = None
        self.inset_ax: Optional[plt.Axes] = None
        self.help_visible = False
        self.help_text = None
        
        # Setup matplotlib
        self._setup_matplotlib()
        self._create_figure()
        self._plot_data()
        self._setup_event_handlers()
        self._create_status_bar()
        
    def _setup_matplotlib(self) -> None:
        """Configure matplotlib for publication quality."""
        plt.rcParams.update({
            'text.usetex': False,  # Use mathtext instead for portability
            'mathtext.fontset': 'stix',
            'font.family': 'STIXGeneral',
            'font.size': self.config.label_size,
            'axes.labelsize': self.config.label_size,
            'axes.titlesize': self.config.title_size,
            'xtick.labelsize': self.config.tick_size,
            'ytick.labelsize': self.config.tick_size,
            'legend.fontsize': self.config.legend_size,
            'figure.dpi': 100,  # Screen DPI
            'savefig.dpi': self.config.dpi,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': self.colors.background,
            'figure.facecolor': 'white',
            'axes.linewidth': 0.8,
            'lines.linewidth': self.config.line_width,
        })
        
    def _create_figure(self) -> None:
        """Create the main figure and axes."""
        self.fig, self.ax = plt.subplots(
            figsize=(self.config.width, self.config.height),
            constrained_layout=True
        )
        
        self.fig.canvas.manager.set_window_title(
            'Interactive Figure Generator - Press H for Help'
        )
        
    def _plot_data(self) -> None:
        """Plot the gimbal tracking data."""
        t = self.data['time']
        
        # Plot reference
        self.ax.plot(
            t, self.data['reference'] * 1000,  # Convert to mrad
            linestyle=self.config.reference_style,
            color=self.colors.reference,
            label=r'Reference $\psi_{ref}$',
            linewidth=self.config.line_width,
            zorder=3
        )
        
        # Plot measured (no NDOB)
        self.ax.plot(
            t, self.data['measured'] * 1000,
            linestyle=self.config.measured_style,
            color=self.colors.measured,
            label=r'Measured $\psi$ (PID)',
            linewidth=self.config.line_width * 0.8,
            alpha=0.8,
            zorder=2
        )
        
        # Plot NDOB-corrected
        self.ax.plot(
            t, self.data['ndob'] * 1000,
            linestyle=self.config.measured_style,
            color=self.colors.ndob,
            label=r'Measured $\psi$ (FBL+NDOB)',
            linewidth=self.config.line_width,
            zorder=4
        )
        
        # Configure axes
        self.ax.set_xlabel(r'Time $t$ [s]')
        self.ax.set_ylabel(r'Position $\psi$ [mrad]')
        self.ax.set_title(
            r'2-DOF Gimbal Step Response with Wind Disturbance Rejection',
            fontsize=self.config.title_size,
            fontweight='bold'
        )
        
        # Set axis limits with some padding
        t_max = t[-1]
        y_min = min(self.data['measured'].min(), self.data['ndob'].min()) * 1000
        y_max = max(self.data['reference'].max(), self.data['ndob'].max()) * 1000
        y_range = y_max - y_min
        self.ax.set_xlim(0, t_max)
        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)
        
        # Legend
        self.ax.legend(
            loc='lower right',
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True
        )
        
        # Grid
        self.ax.grid(True, linestyle='-', alpha=0.3, color=self.colors.grid)
        self.ax.set_axisbelow(True)
        
    def _setup_event_handlers(self) -> None:
        """Connect event handlers for interactivity."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        
    def _create_status_bar(self) -> None:
        """Create status bar text at bottom of figure."""
        self.status_text = self.fig.text(
            0.02, 0.01,
            'Mode: Navigate | Press H for help',
            fontsize=9,
            family='monospace',
            color='gray',
            transform=self.fig.transFigure
        )
        
    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        mode_names = {
            InteractionMode.NONE: 'Navigate',
            InteractionMode.VERTICAL_LINE: 'Vertical Line (click to place)',
            InteractionMode.CIRCLE: 'Circle Highlight (click to place)',
            InteractionMode.TEXT: 'Text Label (click to place)',
            InteractionMode.ZOOM_START: 'Zoom Region (click first corner)',
            InteractionMode.ZOOM_END: 'Zoom Region (click second corner)',
        }
        mode_str = mode_names.get(self.mode, 'Unknown')
        self.status_text.set_text(f'Mode: {mode_str} | {message}')
        self.fig.canvas.draw_idle()
        
    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        if event.key == 'v':
            self.mode = InteractionMode.VERTICAL_LINE
            self._update_status('Click to place vertical line')
            
        elif event.key == 'c':
            self.mode = InteractionMode.CIRCLE
            self._update_status('Click to place highlight circle')
            
        elif event.key == 't':
            self.mode = InteractionMode.TEXT
            self._update_status('Click to place text label')
            
        elif event.key == 'z':
            self.mode = InteractionMode.ZOOM_START
            self.zoom_start = None
            self._update_status('Click first corner of zoom region')
            
        elif event.key == 's':
            self._save_figure()
            
        elif event.key == 'u':
            self._undo_last_annotation()
            
        elif event.key == 'h':
            self._toggle_help()
            
        elif event.key == 'escape':
            self.mode = InteractionMode.NONE
            self.zoom_start = None
            self._update_status('Press H for help')
            
        elif event.key == 'q':
            self._quit()
            
    def _on_click(self, event) -> None:
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        
        if self.mode == InteractionMode.VERTICAL_LINE:
            self._add_vertical_line(x)
            self.mode = InteractionMode.NONE
            self._update_status('Vertical line placed')
            
        elif self.mode == InteractionMode.CIRCLE:
            self._add_circle(x, y)
            self.mode = InteractionMode.NONE
            self._update_status('Circle placed')
            
        elif self.mode == InteractionMode.TEXT:
            self._add_text_label(x, y)
            self.mode = InteractionMode.NONE
            
        elif self.mode == InteractionMode.ZOOM_START:
            self.zoom_start = (x, y)
            self.mode = InteractionMode.ZOOM_END
            self._update_status(f'Corner 1: ({x:.2f}, {y:.2f}) - Click second corner')
            
        elif self.mode == InteractionMode.ZOOM_END:
            if self.zoom_start is not None:
                self._create_zoom_inset(self.zoom_start, (x, y))
                self.mode = InteractionMode.NONE
                self._update_status('Zoom inset created')
                
    def _on_motion(self, event) -> None:
        """Handle mouse motion for preview effects."""
        # Could add preview effects here (e.g., crosshairs)
        pass
        
    def _add_vertical_line(self, x: float) -> None:
        """Add a dashed vertical line annotation."""
        line = self.ax.axvline(
            x=x,
            color=self.colors.annotation,
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
            zorder=10
        )
        
        # Add time label
        label = self.ax.text(
            x, self.ax.get_ylim()[1] * 0.98,
            f't = {x:.2f}s',
            fontsize=self.config.annotation_size,
            ha='center',
            va='top',
            color=self.colors.annotation,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )
        
        annotation = Annotation(
            type='vline',
            x=x,
            y=0,
            artist=[line, label],
            properties={'time': x}
        )
        self.annotations.append(annotation)
        self.fig.canvas.draw_idle()
        
    def _add_circle(self, x: float, y: float, radius: float = None) -> None:
        """Add a highlight circle annotation."""
        # Auto-calculate radius based on axis scale
        if radius is None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            # Make circle appear circular in display coordinates
            radius = min(x_range, y_range) * 0.03
            
        circle = Circle(
            (x, y),
            radius=radius,
            fill=False,
            edgecolor=self.colors.error,
            linewidth=2.5,
            linestyle='-',
            alpha=0.9,
            zorder=15
        )
        self.ax.add_patch(circle)
        
        # Add semi-transparent fill
        fill = Circle(
            (x, y),
            radius=radius,
            fill=True,
            facecolor=self.colors.error,
            alpha=0.15,
            zorder=14
        )
        self.ax.add_patch(fill)
        
        annotation = Annotation(
            type='circle',
            x=x,
            y=y,
            artist=[circle, fill],
            properties={'radius': radius}
        )
        self.annotations.append(annotation)
        self.fig.canvas.draw_idle()
        
    def _add_text_label(self, x: float, y: float) -> None:
        """Add a text label annotation with input dialog."""
        # Create a simple input using matplotlib
        label_text = self._get_text_input()
        
        if label_text:
            text = self.ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(x + 0.3, y + 5),  # Offset
                fontsize=self.config.annotation_size,
                color=self.colors.annotation,
                arrowprops=dict(
                    arrowstyle='->',
                    color=self.colors.annotation,
                    connectionstyle='arc3,rad=0.2'
                ),
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor=self.colors.annotation,
                    alpha=0.9
                ),
                zorder=20
            )
            
            annotation = Annotation(
                type='text',
                x=x,
                y=y,
                artist=text,
                text=label_text
            )
            self.annotations.append(annotation)
            self.fig.canvas.draw_idle()
            self._update_status(f'Text label added: "{label_text}"')
        else:
            self._update_status('Text label cancelled')
            
    def _get_text_input(self) -> str:
        """Get text input from user via simple dialog."""
        import tkinter as tk
        from tkinter import simpledialog
        
        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        
        # Show input dialog
        text = simpledialog.askstring(
            "Add Label",
            "Enter annotation text:",
            parent=root
        )
        
        root.destroy()
        return text if text else ""
        
    def _create_zoom_inset(
        self,
        corner1: Tuple[float, float],
        corner2: Tuple[float, float]
    ) -> None:
        """Create a zoomed inset with connector lines."""
        # Remove existing inset if present
        if self.inset_ax is not None:
            self.inset_ax.remove()
            
        x1, y1 = corner1
        x2, y2 = corner2
        
        # Ensure proper ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Create inset axes
        self.inset_ax = inset_axes(
            self.ax,
            width=self.config.inset_width,
            height=self.config.inset_height,
            loc='upper left',
            borderpad=2
        )
        
        # Plot same data in inset
        t = self.data['time']
        
        self.inset_ax.plot(
            t, self.data['reference'] * 1000,
            linestyle=self.config.reference_style,
            color=self.colors.reference,
            linewidth=self.config.line_width * 0.8
        )
        
        self.inset_ax.plot(
            t, self.data['measured'] * 1000,
            linestyle=self.config.measured_style,
            color=self.colors.measured,
            linewidth=self.config.line_width * 0.6,
            alpha=0.8
        )
        
        self.inset_ax.plot(
            t, self.data['ndob'] * 1000,
            linestyle=self.config.measured_style,
            color=self.colors.ndob,
            linewidth=self.config.line_width * 0.8
        )
        
        # Set zoom limits
        self.inset_ax.set_xlim(x_min, x_max)
        self.inset_ax.set_ylim(y_min, y_max)
        
        # Style inset
        self.inset_ax.tick_params(labelsize=8)
        self.inset_ax.grid(True, alpha=0.3)
        self.inset_ax.set_facecolor(self.colors.background)
        
        # Add connector lines
        mark_inset(
            self.ax, self.inset_ax,
            loc1=2, loc2=4,  # Corners to connect
            fc="none",
            ec=self.colors.annotation,
            linewidth=1,
            linestyle='--',
            alpha=0.7
        )
        
        # Draw rectangle on main plot showing zoom region
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=self.colors.annotation,
            linewidth=1.5,
            linestyle='-',
            zorder=5
        )
        self.ax.add_patch(rect)
        
        self.fig.canvas.draw_idle()
        
    def _undo_last_annotation(self) -> None:
        """Remove the last added annotation."""
        if not self.annotations:
            self._update_status('Nothing to undo')
            return
            
        annotation = self.annotations.pop()
        
        # Remove artist(s)
        if isinstance(annotation.artist, list):
            for artist in annotation.artist:
                artist.remove()
        elif annotation.artist is not None:
            annotation.artist.remove()
            
        self.fig.canvas.draw_idle()
        self._update_status(f'Undid {annotation.type} annotation')
        
    def _toggle_help(self) -> None:
        """Toggle help overlay visibility."""
        if self.help_visible:
            if self.help_text:
                self.help_text.remove()
                self.help_text = None
            self.help_visible = False
        else:
            help_message = """
╔════════════════════════════════════════╗
║     INTERACTIVE FIGURE CONTROLS        ║
╠════════════════════════════════════════╣
║  V : Place vertical line               ║
║  C : Place highlight circle            ║
║  T : Add text label                    ║
║  Z : Define zoom inset region          ║
║  S : Save figure (PNG, PDF, SVG)       ║
║  U : Undo last annotation              ║
║  H : Toggle this help                  ║
║  Q : Quit                              ║
║  Esc : Cancel current mode             ║
╚════════════════════════════════════════╝
"""
            self.help_text = self.fig.text(
                0.5, 0.5,
                help_message,
                fontsize=11,
                family='monospace',
                ha='center',
                va='center',
                transform=self.fig.transFigure,
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.95
                ),
                zorder=100
            )
            self.help_visible = True
            
        self.fig.canvas.draw_idle()
        
    def _save_figure(self) -> None:
        """Save figure to multiple formats."""
        base_path = Path('figures_comparative')
        base_path.mkdir(exist_ok=True)
        
        base_name = 'fig2_gimbal_tracking_interactive'
        
        for fmt in self.config.output_formats:
            filepath = base_path / f'{base_name}.{fmt}'
            self.fig.savefig(
                filepath,
                format=fmt,
                dpi=self.config.dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            print(f'Saved: {filepath}')
            
        # Also save annotations to JSON
        annotations_path = base_path / f'{base_name}_annotations.json'
        self._save_annotations(annotations_path)
        
        self._update_status(f'Saved to {len(self.config.output_formats)} formats')
        
    def _save_annotations(self, filepath: Path) -> None:
        """Save annotation data to JSON for reproducibility."""
        annotation_data = []
        for ann in self.annotations:
            annotation_data.append({
                'type': ann.type,
                'x': ann.x,
                'y': ann.y,
                'text': ann.text,
                'properties': ann.properties
            })
            
        with open(filepath, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        print(f'Saved annotations: {filepath}')
        
    def _quit(self) -> None:
        """Clean exit with save prompt."""
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()
        
        if messagebox.askyesno("Save Before Quit", "Save figure before quitting?"):
            self._save_figure()
            
        root.destroy()
        plt.close(self.fig)
        
    def show(self) -> None:
        """Display the interactive figure."""
        plt.show()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Generate and display interactive figure."""
    print("=" * 60)
    print("  Interactive Figure Generator for Control Engineering")
    print("  Publication-Quality 2-DOF Gimbal Visualization")
    print("=" * 60)
    print()
    
    # Generate synthetic data
    print("Generating synthetic gimbal tracking data...")
    simulator = GimbalSimulator(duration=10.0, dt=0.001)
    data = simulator.generate_dataset()
    print(f"  - Duration: {data['time'][-1]:.1f} s")
    print(f"  - Samples: {len(data['time'])}")
    print(f"  - Step amplitude: 100 mrad")
    print()
    
    # Create interactive figure
    print("Launching interactive figure...")
    print("  Press H for help on keyboard controls")
    print()
    
    config = FigureConfig(
        width=7.0,
        height=5.0,
        dpi=300
    )
    
    generator = InteractiveFigureGenerator(
        data=data,
        config=config
    )
    
    # Add default annotations to demonstrate features
    print("Adding example annotations...")
    
    # Add vertical line at first wind gust
    generator._add_vertical_line(3.0)
    
    # Add circle at disturbance response
    generator._add_circle(3.3, 105)
    
    # Show the figure
    generator.show()
    
    print()
    print("Figure closed.")


if __name__ == '__main__':
    main()
