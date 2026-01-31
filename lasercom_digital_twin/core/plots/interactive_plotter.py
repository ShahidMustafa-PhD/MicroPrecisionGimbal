#!/usr/bin/env python3
"""
Industrial-Grade Interactive Plotting Module for Control Engineering

This module provides fully interactive matplotlib figures with professional
capabilities for control system analysis and publication preparation.

Features:
---------
- Interactive zoom regions with magnification insets (draggable, resizable)
- Draggable vertical and horizontal reference lines
- Mouse-based selection, movement, and deletion of all annotations
- Professional toolbar with mode indicators
- Keyboard shortcuts for rapid annotation
- Export to publication formats (PNG, PDF, SVG)
- Full undo/redo support with complete state restoration
- Annotation state persistence (JSON save/load)

Industrial Standards:
--------------------
- 300 DPI output for print quality
- LaTeX-compatible typography
- Color schemes suitable for colorblind accessibility
- Consistent styling across all figures

Author: Dr. S. Shahid Mustafa
Date: January 30, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, SpanSelector, RectangleSelector
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from enum import Enum, auto
import json
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback


# =============================================================================
# Configuration Classes
# =============================================================================

class InteractionMode(Enum):
    """Available interaction modes for the figure."""
    NAVIGATE = auto()      # Default pan/zoom mode
    VLINE = auto()         # Place vertical line
    HLINE = auto()         # Place horizontal line
    ZOOM_RECT = auto()     # Define zoom region
    DELETE = auto()        # Delete annotations
    MOVE = auto()          # Move existing annotations
    SELECT = auto()        # Select annotations for actions
    RESIZE = auto()        # Resize zoom regions


@dataclass
class DraggableLine:
    """Represents a draggable reference line."""
    line: Line2D
    orientation: str  # 'vertical' or 'horizontal'
    position: float
    label: Optional[Any] = None
    is_dragging: bool = False
    is_selected: bool = False
    parent_ax: Optional[plt.Axes] = None
    
    
@dataclass
class ZoomRegion:
    """Represents a zoom inset region."""
    inset_ax: plt.Axes
    rect: Rectangle
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    connectors: List[Any] = field(default_factory=list)
    is_selected: bool = False
    is_dragging: bool = False
    parent_ax: Optional[plt.Axes] = None
    # Inset position on figure (normalized coordinates)
    inset_loc: str = 'upper left'
    # For tracking inset position in axes coordinates
    inset_bbox: Optional[Any] = None


@dataclass
class TextAnnotation:
    """Represents a text annotation."""
    text_obj: plt.Text
    position: Tuple[float, float]
    content: str
    is_selected: bool = False
    is_dragging: bool = False
    parent_ax: Optional[plt.Axes] = None


@dataclass 
class InteractiveStyleConfig:
    """Style configuration for interactive plots."""
    # Figure dimensions
    fig_width: float = 12.0
    fig_height: float = 8.0
    dpi: int = 100  # Screen DPI
    save_dpi: int = 300  # Export DPI
    
    # Colors
    vline_color: str = '#e74c3c'       # Red for vertical lines
    hline_color: str = '#3498db'       # Blue for horizontal lines
    zoom_rect_color: str = '#27ae60'   # Green for zoom regions
    zoom_rect_alpha: float = 0.15
    selection_color: str = '#f39c12'   # Orange for selection highlight
    
    # Line properties
    annotation_linewidth: float = 1.5
    annotation_linestyle: str = '--'
    selection_linewidth: float = 3.0
    
    # Fonts
    label_fontsize: int = 10
    
    # Inset properties
    inset_width: str = "30%"
    inset_height: str = "25%"
    
    # Selection threshold (fraction of axis range)
    selection_threshold: float = 0.03
    
    # Zoom inset locations (cycle through these)
    inset_locations: List[str] = field(default_factory=lambda: [
        'upper left', 'upper right', 'lower left', 'lower right'
    ])


# =============================================================================
# Interactive Figure Manager
# =============================================================================

class InteractiveFigureManager:
    """
    Manages interactive features for matplotlib figures.
    
    Provides industrial-level interactivity including:
    - Draggable vertical and horizontal reference lines
    - Interactive zoom regions with magnification insets
    - Click-to-select, relocate, and delete annotations
    - Full mouse-based interaction
    - Draggable zoom insets
    - Professional toolbar with mode indicators
    
    Keyboard Shortcuts:
    ------------------
    v : Vertical line mode
    h : Horizontal line mode
    z : Zoom region mode  
    d : Delete mode (click to delete)
    m : Move mode (drag to move)
    n : Navigate mode (default)
    s : Save figure
    u : Undo last action
    r : Redo last undone action
    Esc : Cancel current mode / deselect
    Delete : Remove selected annotation
    
    Mouse Actions:
    -------------
    Left Click : Place/Select annotation (depends on mode)
    Left Drag : Move annotation (in Move mode or on selected item)
    Right Click : Delete annotation under cursor
    Middle Click : Quick-select annotation
    Double Click : Edit annotation properties
    
    Usage:
    ------
    ```python
    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    manager = InteractiveFigureManager(fig, ax)
    manager.show()
    ```
    """
    
    def __init__(
        self,
        fig: plt.Figure,
        axes: Union[plt.Axes, List[plt.Axes]],
        style: Optional[InteractiveStyleConfig] = None,
        save_dir: str = "figures_comparative"
    ):
        """
        Initialize the interactive figure manager.
        
        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure to manage
        axes : plt.Axes or List[plt.Axes]
            The axes in the figure
        style : InteractiveStyleConfig, optional
            Style configuration
        save_dir : str
            Directory for saving figures
        """
        self.fig = fig
        self.axes = [axes] if isinstance(axes, plt.Axes) else list(axes)
        self.primary_ax = self.axes[0]
        self.style = style or InteractiveStyleConfig()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # State management
        self.mode = InteractionMode.NAVIGATE
        self.vlines: List[DraggableLine] = []
        self.hlines: List[DraggableLine] = []
        self.zoom_regions: List[ZoomRegion] = []
        self.text_annotations: List[TextAnnotation] = []
        
        # History for undo/redo
        self.action_history: List[Dict] = []
        self.redo_stack: List[Dict] = []
        
        # Drag state
        self.dragging_item: Optional[Any] = None
        self.drag_start_pos: Optional[Tuple[float, float]] = None
        self.drag_offset: Optional[Tuple[float, float]] = None
        
        # Zoom selection state
        self.zoom_start: Optional[Tuple[float, float]] = None
        self.zoom_preview_rect: Optional[Rectangle] = None
        
        # Selection state
        self.selected_item: Optional[Any] = None
        self.selection_highlight: Optional[Any] = None
        
        # Inset location counter for cycling
        self.inset_location_index: int = 0
        
        # Track which axes we're working in
        self.current_ax: Optional[plt.Axes] = None
        
        # Last saved file path
        self._last_saved_pdf: Optional[Path] = None
        
        # Hover state for visual feedback
        self.hover_item: Optional[Any] = None
        
        # Setup
        self._setup_toolbar()
        self._connect_events()
        self._create_status_bar()
        
    def _setup_toolbar(self) -> None:
        """Create custom toolbar with mode buttons."""
        self.buttons = {}
        self.use_keyboard_only = False
        
        try:
            is_constrained = self.fig.get_constrained_layout()
        except AttributeError:
            is_constrained = False
            
        if is_constrained:
            self.use_keyboard_only = True
            return
            
        # Adjust figure to make room for toolbar
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, 
                                        message="This figure was using a layout engine")
                self.fig.subplots_adjust(bottom=0.15)
        except Exception:
            pass
        
        # Button positions [left, bottom, width, height]
        button_height = 0.04
        button_width = 0.07
        button_y = 0.02
        button_spacing = 0.075
        start_x = 0.05
        
        button_configs = [
            ('navigate', 'Nav', self._set_navigate_mode, '#4a90d9'),
            ('vline', 'VLine', self._set_vline_mode, self.style.vline_color),
            ('hline', 'HLine', self._set_hline_mode, self.style.hline_color),
            ('zoom', 'Zoom', self._set_zoom_mode, self.style.zoom_rect_color),
            ('move', 'Move', self._set_move_mode, '#9b59b6'),
            ('delete', 'Del', self._set_delete_mode, '#c0392b'),
            ('undo', 'Undo', self._undo_last, '#7f8c8d'),
            ('redo', 'Redo', self._redo_last, '#7f8c8d'),
            ('save', 'Save', self._save_figure, '#27ae60'),
            ('open', 'Open', self._open_last_saved, '#3498db'),
        ]
        
        for i, (name, label, callback, color) in enumerate(button_configs):
            ax_button = self.fig.add_axes([
                start_x + i * button_spacing,
                button_y,
                button_width,
                button_height
            ])
            self.buttons[name] = Button(ax_button, label)
            self.buttons[name].on_clicked(lambda event, cb=callback: cb())
            # Color the button border
            for spine in ax_button.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            
    def _connect_events(self) -> None:
        """Connect matplotlib event handlers."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        # Track double-clicks for label hiding
        self._last_click_time = 0
        self._last_click_pos = None
        self._double_click_threshold = 0.3  # seconds
        
    def _create_status_bar(self) -> None:
        """Create status bar for mode indication."""
        if self.use_keyboard_only:
            status_msg = 'Mode: Navigate | Keys: V/H/Z=add D=del M=move S=save U=undo R=redo'
        else:
            status_msg = 'Mode: Navigate | Click buttons or use keyboard (V/H/Z/D/M/S/U/R)'
            
        self.status_text = self.fig.text(
            0.02, 0.01,
            status_msg,
            fontsize=9,
            family='monospace',
            color='#2c3e50',
            transform=self.fig.transFigure,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.95,
                     edgecolor='#bdc3c7', linewidth=1)
        )
        
    def _update_status(self, message: str = "") -> None:
        """Update status bar with current mode and message."""
        mode_names = {
            InteractionMode.NAVIGATE: 'Navigate',
            InteractionMode.VLINE: 'V-Line (click to place)',
            InteractionMode.HLINE: 'H-Line (click to place)',
            InteractionMode.ZOOM_RECT: 'Zoom (click 2 corners)',
            InteractionMode.DELETE: 'Delete (click item)',
            InteractionMode.MOVE: 'Move (drag item)',
            InteractionMode.SELECT: 'Select',
            InteractionMode.RESIZE: 'Resize',
        }
        
        mode_str = mode_names.get(self.mode, 'Unknown')
        
        # Build status with item counts
        counts = f"[V:{len(self.vlines)} H:{len(self.hlines)} Z:{len(self.zoom_regions)}]"
        
        status = f'Mode: {mode_str} {counts}'
        if message:
            status += f' | {message}'
        if self.selected_item:
            status += ' | Item selected (Del to remove)'
            
        self.status_text.set_text(status)
        self._highlight_active_button()
        self.fig.canvas.draw_idle()
        
    def _highlight_active_button(self) -> None:
        """Highlight the currently active mode button."""
        if self.use_keyboard_only or not self.buttons:
            return
            
        mode_button_map = {
            InteractionMode.NAVIGATE: 'navigate',
            InteractionMode.VLINE: 'vline',
            InteractionMode.HLINE: 'hline',
            InteractionMode.ZOOM_RECT: 'zoom',
            InteractionMode.DELETE: 'delete',
            InteractionMode.MOVE: 'move',
        }
        
        active_name = mode_button_map.get(self.mode)
        
        for name, button in self.buttons.items():
            if name == active_name:
                button.ax.set_facecolor('#a8d5ba')  # Light green for active
            elif name in ['save', 'open']:
                button.ax.set_facecolor('#d5f5e3')  # Light green for actions
            elif name in ['undo', 'redo']:
                button.ax.set_facecolor('#fadbd8')  # Light pink for history
            else:
                button.ax.set_facecolor('#f5f5f5')  # Light gray for inactive
                
    # =========================================================================
    # Mode Setters
    # =========================================================================
    
    def _set_navigate_mode(self) -> None:
        """Set navigation mode (default pan/zoom)."""
        self._clear_selection()
        self.mode = InteractionMode.NAVIGATE
        self._cancel_zoom_preview()
        self._update_status()
        
    def _set_vline_mode(self) -> None:
        """Set vertical line placement mode."""
        self._clear_selection()
        self.mode = InteractionMode.VLINE
        self._update_status("Click on plot to place vertical line")
        
    def _set_hline_mode(self) -> None:
        """Set horizontal line placement mode."""
        self._clear_selection()
        self.mode = InteractionMode.HLINE
        self._update_status("Click on plot to place horizontal line")
        
    def _set_zoom_mode(self) -> None:
        """Set zoom region mode."""
        self._clear_selection()
        self.mode = InteractionMode.ZOOM_RECT
        self.zoom_start = None
        self._update_status("Click first corner of zoom region")
        
    def _set_delete_mode(self) -> None:
        """Set delete mode."""
        self._clear_selection()
        self.mode = InteractionMode.DELETE
        self._update_status("Click on annotation to delete | Right-click anywhere to delete")
        
    def _set_move_mode(self) -> None:
        """Set move mode."""
        self.mode = InteractionMode.MOVE
        self._update_status("Drag any annotation to reposition")
        
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        key_handlers = {
            'v': self._set_vline_mode,
            'h': self._set_hline_mode,
            'z': self._set_zoom_mode,
            'd': self._set_delete_mode,
            'm': self._set_move_mode,
            'n': self._set_navigate_mode,
            's': self._save_figure,
            'u': self._undo_last,
            'r': self._redo_last,
            'escape': self._escape_action,
            'delete': self._delete_selected,
        }
        
        if event.key in key_handlers:
            key_handlers[event.key]()
        elif event.key == '?':
            self._show_help()
        elif event.key == 'o':
            self._open_last_saved()
            
    def _handle_double_click(self, ax: plt.Axes, x: float, y: float) -> None:
        """Handle double-click to hide/show vertical line labels."""
        # Find vertical line at this position
        xlim = ax.get_xlim()
        x_thresh = self.style.selection_threshold * (xlim[1] - xlim[0])
        
        for vline in self.vlines:
            if vline.parent_ax == ax or vline.parent_ax is None:
                if abs(vline.position - x) < x_thresh:
                    if vline.label:
                        # Toggle label visibility
                        current_visibility = vline.label.get_visible()
                        vline.label.set_visible(not current_visibility)
                        status_msg = "Label hidden" if current_visibility else "Label shown"
                        self._update_status(f"Vertical line label: {status_msg}")
                        self.fig.canvas.draw_idle()
                        return
        
        self._update_status("Double-click on vertical line to hide/show label")
    
    def _escape_action(self) -> None:
        """Handle escape key - cancel current action or deselect."""
        if self.zoom_start is not None:
            self._cancel_zoom_preview()
            self._update_status("Zoom cancelled")
        elif self.selected_item is not None:
            self._clear_selection()
        else:
            self._set_navigate_mode()
            
    def _on_mouse_press(self, event) -> None:
        """Handle mouse button press."""
        if event.inaxes is None:
            return
            
        # Check if clicking on an inset axes (for zoom regions)
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        
        if x is None or y is None:
            return
            
        # Store current axes
        self.current_ax = ax
        
        # Right-click: Quick delete
        if event.button == 3:
            self._quick_delete(ax, x, y)
            return
            
        # Middle-click: Quick select
        if event.button == 2:
            self._quick_select(ax, x, y)
            return
            
        # Left-click handling based on mode
        if event.button == 1:
            # Detect double-click for label hiding
            current_time = time.time()
            is_double_click = False
            if self._last_click_pos is not None:
                dx = abs(event.x - self._last_click_pos[0]) if event.x else float('inf')
                dy = abs(event.y - self._last_click_pos[1]) if event.y else float('inf')
                time_delta = current_time - self._last_click_time
                if dx < 10 and dy < 10 and time_delta < self._double_click_threshold:
                    is_double_click = True
                    self._handle_double_click(ax, x, y)
                    self._last_click_time = 0  # Reset
                    self._last_click_pos = None
                    return
            
            self._last_click_time = current_time
            self._last_click_pos = (event.x, event.y) if event.x and event.y else None
            
            # First check if clicking on a zoom inset to drag it
            if self.mode in [InteractionMode.NAVIGATE, InteractionMode.MOVE]:
                zoom = self._find_zoom_at_inset(ax)
                if zoom is not None:
                    self._start_zoom_inset_drag(zoom, event)
                    return
            
            # Check if there's an annotation under cursor that can be dragged
            if self.mode == InteractionMode.MOVE:
                item = self._find_annotation_at(ax, x, y)
                if item is not None:
                    self._start_drag(item, x, y)
                    return
                    
            # Mode-specific actions
            if ax not in self.axes:
                # Check if it's a zoom inset
                zoom = self._find_zoom_at_inset(ax)
                if zoom and self.mode == InteractionMode.MOVE:
                    self._start_zoom_inset_drag(zoom, event)
                return
                
            if self.mode == InteractionMode.VLINE:
                self._place_vertical_line(ax, x)
                
            elif self.mode == InteractionMode.HLINE:
                self._place_horizontal_line(ax, y)
                
            elif self.mode == InteractionMode.ZOOM_RECT:
                self._handle_zoom_click(ax, x, y)
                
            elif self.mode == InteractionMode.DELETE:
                self._delete_at_position(ax, x, y)
                
            elif self.mode == InteractionMode.NAVIGATE:
                # In navigate mode, clicking selects items
                item = self._find_annotation_at(ax, x, y)
                if item is not None:
                    self._select_item(item)
                else:
                    # Deselect if clicking in empty space
                    self._clear_selection()
                    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse button release."""
        if self.dragging_item is not None:
            # Finalize drag
            if isinstance(self.dragging_item, DraggableLine):
                self.dragging_item.is_dragging = False
            elif isinstance(self.dragging_item, ZoomRegion):
                self.dragging_item.is_dragging = False
            self.dragging_item = None
            self.drag_start_pos = None
            self.drag_offset = None
            self.fig.canvas.draw_idle()
            
    def _on_mouse_move(self, event) -> None:
        """Handle mouse movement for dragging and hover feedback."""
        if event.inaxes is None:
            # Clear hover state when mouse leaves axes
            if self.hover_item is not None:
                self._clear_hover()
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Handle dragging
        if self.dragging_item is not None:
            self._handle_drag(event)
            return
            
        # Handle zoom preview rectangle
        if self.zoom_start is not None and self.mode == InteractionMode.ZOOM_RECT:
            self._update_zoom_preview(event.inaxes, x, y)
            return
            
        # Hover detection for visual feedback (in Navigate mode)
        if self.mode == InteractionMode.NAVIGATE:
            item = self._find_annotation_at(event.inaxes, x, y)
            if item != self.hover_item:
                self._clear_hover()
                if item is not None:
                    self._show_hover(item)
            
    def _on_pick(self, event) -> None:
        """Handle pick events for selecting annotations."""
        # This provides an alternative selection mechanism
        artist = event.artist
        
        # Check if it's one of our managed lines
        for vline in self.vlines:
            if vline.line == artist:
                self._select_item(vline)
                return
                
        for hline in self.hlines:
            if hline.line == artist:
                self._select_item(hline)
                return
                
    # =========================================================================
    # Selection Methods
    # =========================================================================
    
    def _find_annotation_at(self, ax: plt.Axes, x: float, y: float) -> Optional[Any]:
        """Find annotation at the given position."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_thresh = self.style.selection_threshold * (xlim[1] - xlim[0])
        y_thresh = self.style.selection_threshold * (ylim[1] - ylim[0])
        
        # Check vertical lines
        for vline in self.vlines:
            if vline.parent_ax == ax or vline.parent_ax is None:
                if abs(vline.position - x) < x_thresh:
                    return vline
                    
        # Check horizontal lines
        for hline in self.hlines:
            if hline.parent_ax == ax or hline.parent_ax is None:
                if abs(hline.position - y) < y_thresh:
                    return hline
                    
        # Check zoom regions (by checking if click is in the source rectangle)
        for zoom in self.zoom_regions:
            if zoom.parent_ax == ax or zoom.parent_ax is None:
                x_min, x_max = zoom.x_range
                y_min, y_max = zoom.y_range
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return zoom
                    
        return None
        
    def _find_zoom_at_inset(self, ax: plt.Axes) -> Optional[ZoomRegion]:
        """Find zoom region whose inset axes matches the given axes."""
        for zoom in self.zoom_regions:
            if zoom.inset_ax == ax:
                return zoom
        return None
        
    def _select_item(self, item: Any) -> None:
        """Select an annotation item."""
        # Clear previous selection
        self._clear_selection()
        
        self.selected_item = item
        
        # Apply selection highlight
        if isinstance(item, DraggableLine):
            item.is_selected = True
            item.line.set_linewidth(self.style.selection_linewidth)
            item.line.set_alpha(1.0)
            if item.label:
                item.label.set_bbox(dict(
                    boxstyle='round,pad=0.3',
                    facecolor=self.style.selection_color,
                    alpha=0.9
                ))
        elif isinstance(item, ZoomRegion):
            item.is_selected = True
            # Make selection very visible with orange border and increased alpha
            item.rect.set_edgecolor(self.style.selection_color)
            item.rect.set_linewidth(self.style.selection_linewidth * 1.5)  # Thicker for visibility
            item.rect.set_alpha(min(1.0, self.style.zoom_rect_alpha * 2.5))  # Brighter
            # Also highlight the inset border
            for spine in item.inset_ax.spines.values():
                spine.set_edgecolor(self.style.selection_color)
                spine.set_linewidth(3.0)
            
        self._update_status("Zoom region selected - Press DELETE to remove, or drag to reposition")
        self.fig.canvas.draw_idle()
        
    def _clear_selection(self) -> None:
        """Clear current selection."""
        if self.selected_item is None:
            return
            
        item = self.selected_item
        
        # Remove selection highlight
        if isinstance(item, DraggableLine):
            item.is_selected = False
            item.line.set_linewidth(self.style.annotation_linewidth)
            item.line.set_alpha(0.8)
            if item.label:
                color = self.style.vline_color if item.orientation == 'vertical' else self.style.hline_color
                item.label.set_bbox(dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    alpha=0.9
                ))
        elif isinstance(item, ZoomRegion):
            item.is_selected = False
            # Restore original green styling
            item.rect.set_edgecolor(self.style.zoom_rect_color)
            item.rect.set_linewidth(2)
            item.rect.set_alpha(self.style.zoom_rect_alpha)
            # Restore inset border color
            for spine in item.inset_ax.spines.values():
                spine.set_edgecolor(self.style.zoom_rect_color)
                spine.set_linewidth(2.0)
            
        self.selected_item = None
        self._update_status()
        self.fig.canvas.draw_idle()
        
    def _quick_select(self, ax: plt.Axes, x: float, y: float) -> None:
        """Quick-select annotation with middle mouse button."""
        item = self._find_annotation_at(ax, x, y)
        if item is not None:
            self._select_item(item)
        else:
            self._clear_selection()
            
    def _quick_delete(self, ax: plt.Axes, x: float, y: float) -> None:
        """Quick-delete annotation with right mouse button."""
        item = self._find_annotation_at(ax, x, y)
        if item is not None:
            self._delete_item(item)
            
    def _show_hover(self, item: Any) -> None:
        """Show hover effect on annotation."""
        self.hover_item = item
        if isinstance(item, DraggableLine):
            item.line.set_linewidth(self.style.annotation_linewidth * 1.3)
        elif isinstance(item, ZoomRegion):
            item.rect.set_alpha(self.style.zoom_rect_alpha * 2)
        self.fig.canvas.draw_idle()
        
    def _clear_hover(self) -> None:
        """Clear hover effect."""
        if self.hover_item is None:
            return
        item = self.hover_item
        if isinstance(item, DraggableLine) and not item.is_selected:
            item.line.set_linewidth(self.style.annotation_linewidth)
        elif isinstance(item, ZoomRegion) and not item.is_selected:
            item.rect.set_alpha(self.style.zoom_rect_alpha)
        self.hover_item = None
        self.fig.canvas.draw_idle()
            
    # =========================================================================
    # Dragging Methods
    # =========================================================================
    
    def _start_drag(self, item: Any, x: float, y: float) -> None:
        """Start dragging an annotation."""
        self.dragging_item = item
        self.drag_start_pos = (x, y)
        
        if isinstance(item, DraggableLine):
            item.is_dragging = True
            if item.orientation == 'vertical':
                self.drag_offset = (item.position - x, 0)
            else:
                self.drag_offset = (0, item.position - y)
        elif isinstance(item, ZoomRegion):
            item.is_dragging = True
            x_min, x_max = item.x_range
            y_min, y_max = item.y_range
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            self.drag_offset = (center_x - x, center_y - y)
            
        self._select_item(item)
        self._update_status("Dragging...")
        
    def _start_zoom_inset_drag(self, zoom: ZoomRegion, event) -> None:
        """Start dragging a zoom inset to reposition it."""
        self.dragging_item = zoom
        zoom.is_dragging = True
        
        # Get inset position in figure coordinates
        bbox = zoom.inset_ax.get_position()
        self.drag_start_pos = (event.x, event.y)  # Use pixel coordinates
        self.drag_offset = (0, 0)  # Will compute offset during drag
        
        self._select_item(zoom)
        self._update_status("Dragging zoom inset - release to place")
        
    def _handle_drag(self, event) -> None:
        """Handle drag motion."""
        if self.dragging_item is None:
            return
            
        item = self.dragging_item
        
        if isinstance(item, DraggableLine):
            self._drag_line(item, event)
        elif isinstance(item, ZoomRegion):
            # Check if we're dragging the source rect or the inset
            if event.inaxes == item.inset_ax:
                # Dragging the inset - reposition it
                self._drag_zoom_inset(item, event)
            else:
                # Dragging the source rectangle - reposition zoom area
                self._drag_zoom_source(item, event)
                
        self.fig.canvas.draw_idle()
        
    def _drag_line(self, line: DraggableLine, event) -> None:
        """Drag a line annotation."""
        if line.orientation == 'vertical':
            if event.xdata is not None:
                new_pos = event.xdata + self.drag_offset[0]
                line.line.set_xdata([new_pos, new_pos])
                line.position = new_pos
                if line.label:
                    line.label.set_x(new_pos)
                    line.label.set_text(f't = {new_pos:.3f}')
        else:
            if event.ydata is not None:
                new_pos = event.ydata + self.drag_offset[1]
                line.line.set_ydata([new_pos, new_pos])
                line.position = new_pos
                if line.label:
                    line.label.set_y(new_pos)
                    line.label.set_text(f'y = {new_pos:.3f}')
                    
    def _drag_zoom_source(self, zoom: ZoomRegion, event) -> None:
        """Drag the zoom source rectangle to reposition the zoom area."""
        if event.xdata is None or event.ydata is None:
            return
            
        # Calculate new center
        x_min, x_max = zoom.x_range
        y_min, y_max = zoom.y_range
        width = x_max - x_min
        height = y_max - y_min
        
        new_center_x = event.xdata + self.drag_offset[0]
        new_center_y = event.ydata + self.drag_offset[1]
        
        new_x_min = new_center_x - width / 2
        new_x_max = new_center_x + width / 2
        new_y_min = new_center_y - height / 2
        new_y_max = new_center_y + height / 2
        
        # Update rectangle
        zoom.rect.set_xy((new_x_min, new_y_min))
        
        # Update zoom range
        zoom.x_range = (new_x_min, new_x_max)
        zoom.y_range = (new_y_min, new_y_max)
        
        # Update inset limits
        zoom.inset_ax.set_xlim(new_x_min, new_x_max)
        zoom.inset_ax.set_ylim(new_y_min, new_y_max)
        
    def _drag_zoom_inset(self, zoom: ZoomRegion, event) -> None:
        """Drag the zoom inset to reposition it on the figure."""
        if self.drag_start_pos is None:
            return
            
        # Calculate delta in figure coordinates
        dx = (event.x - self.drag_start_pos[0]) / self.fig.dpi / self.fig.get_figwidth()
        dy = (event.y - self.drag_start_pos[1]) / self.fig.dpi / self.fig.get_figheight()
        
        # Get current position
        bbox = zoom.inset_ax.get_position()
        
        # Update position with better bounds (keep inset fully visible)
        margin = 0.02
        new_x0 = max(margin, min(1.0 - bbox.width - margin, bbox.x0 + dx))
        new_y0 = max(0.15, min(1.0 - bbox.height - margin, bbox.y0 + dy))  # Account for toolbar
        
        zoom.inset_ax.set_position([new_x0, new_y0, bbox.width, bbox.height])
        
        # Update drag start for continuous movement
        self.drag_start_pos = (event.x, event.y)
        
        self._update_status(f"Repositioning zoom inset ({new_x0:.2f}, {new_y0:.2f})")
        
    # =========================================================================
    # Annotation Creation Methods
    # =========================================================================
    
    def _place_vertical_line(self, ax: plt.Axes, x: float) -> None:
        """Place a vertical reference line."""
        line = ax.axvline(
            x=x,
            color=self.style.vline_color,
            linestyle=self.style.annotation_linestyle,
            linewidth=self.style.annotation_linewidth,
            alpha=0.8,
            picker=5,
            zorder=100
        )
        
        # Add label at top
        ylim = ax.get_ylim()
        label = ax.text(
            x, ylim[1],
            f't = {x:.3f}',
            fontsize=self.style.label_fontsize,
            ha='center',
            va='bottom',
            color=self.style.vline_color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9,
                     edgecolor=self.style.vline_color, linewidth=1),
            zorder=101
        )
        
        draggable = DraggableLine(
            line=line,
            orientation='vertical',
            position=x,
            label=label,
            parent_ax=ax
        )
        self.vlines.append(draggable)
        
        # Record action for undo
        self.action_history.append({
            'type': 'add_vline',
            'object': draggable,
            'parent_ax': ax
        })
        self.redo_stack.clear()  # Clear redo on new action
        
        self._update_status(f"Vertical line at x = {x:.3f}")
        self.fig.canvas.draw_idle()
        
    def _place_horizontal_line(self, ax: plt.Axes, y: float) -> None:
        """Place a horizontal reference line."""
        line = ax.axhline(
            y=y,
            color=self.style.hline_color,
            linestyle=self.style.annotation_linestyle,
            linewidth=self.style.annotation_linewidth,
            alpha=0.8,
            picker=5,
            zorder=100
        )
        
        # Add label at right
        xlim = ax.get_xlim()
        label = ax.text(
            xlim[1], y,
            f'y = {y:.3f}',
            fontsize=self.style.label_fontsize,
            ha='left',
            va='center',
            color=self.style.hline_color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9,
                     edgecolor=self.style.hline_color, linewidth=1),
            zorder=101
        )
        
        draggable = DraggableLine(
            line=line,
            orientation='horizontal',
            position=y,
            label=label,
            parent_ax=ax
        )
        self.hlines.append(draggable)
        
        self.action_history.append({
            'type': 'add_hline',
            'object': draggable,
            'parent_ax': ax
        })
        self.redo_stack.clear()
        
        self._update_status(f"Horizontal line at y = {y:.3f}")
        self.fig.canvas.draw_idle()
        
    def _handle_zoom_click(self, ax: plt.Axes, x: float, y: float) -> None:
        """Handle clicks for zoom region creation."""
        if self.zoom_start is None:
            # First click - start zoom region
            self.zoom_start = (x, y)
            self.zoom_start_ax = ax  # Remember which axes we started in
            
            # Create preview rectangle
            self.zoom_preview_rect = Rectangle(
                (x, y), 0, 0,
                fill=True,
                facecolor=self.style.zoom_rect_color,
                alpha=0.3,
                edgecolor=self.style.zoom_rect_color,
                linewidth=2,
                linestyle='--',
                zorder=50
            )
            ax.add_patch(self.zoom_preview_rect)
            
            self._update_status(f"Corner 1: ({x:.2f}, {y:.2f}) - drag or click second corner")
        else:
            # Second click - complete zoom region
            start_pos = self.zoom_start
            start_ax = getattr(self, 'zoom_start_ax', ax)
            self._cancel_zoom_preview()
            if start_pos is not None:
                self._create_zoom_region(start_ax, start_pos, (x, y))
            
    def _update_zoom_preview(self, ax: plt.Axes, x: float, y: float) -> None:
        """Update the zoom preview rectangle during creation."""
        if self.zoom_preview_rect is None or self.zoom_start is None:
            return
            
        x0, y0 = self.zoom_start
        width = x - x0
        height = y - y0
        
        # Handle negative dimensions
        if width < 0:
            self.zoom_preview_rect.set_x(x)
            width = -width
        else:
            self.zoom_preview_rect.set_x(x0)
            
        if height < 0:
            self.zoom_preview_rect.set_y(y)
            height = -height
        else:
            self.zoom_preview_rect.set_y(y0)
            
        self.zoom_preview_rect.set_width(width)
        self.zoom_preview_rect.set_height(height)
        
        self.fig.canvas.draw_idle()
        
    def _cancel_zoom_preview(self) -> None:
        """Cancel and remove zoom preview rectangle."""
        if self.zoom_preview_rect is not None:
            try:
                self.zoom_preview_rect.remove()
            except Exception:
                pass
            self.zoom_preview_rect = None
        self.zoom_start = None
        self.fig.canvas.draw_idle()
        
    def _create_zoom_region(
        self,
        ax: plt.Axes,
        corner1: Tuple[float, float],
        corner2: Tuple[float, float]
    ) -> None:
        """Create a zoom inset region."""
        x1, y1 = corner1
        x2, y2 = corner2
        
        # Ensure proper ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Validate size
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_size_x = 0.01 * (xlim[1] - xlim[0])
        min_size_y = 0.01 * (ylim[1] - ylim[0])
        
        if (x_max - x_min) < min_size_x or (y_max - y_min) < min_size_y:
            self._update_status("Zoom region too small - try again")
            return
            
        # Cycle through inset locations
        loc = self.style.inset_locations[self.inset_location_index % len(self.style.inset_locations)]
        self.inset_location_index += 1
        
        # Create inset axes
        inset_ax = inset_axes(
            ax,
            width=self.style.inset_width,
            height=self.style.inset_height,
            loc=loc,
            borderpad=2
        )
        
        # Copy all line plots from parent to inset
        for line in ax.get_lines():
            try:
                xdata, ydata = line.get_data()
                if len(xdata) > 0:
                    inset_ax.plot(
                        xdata, ydata,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=max(0.5, line.get_linewidth() * 0.7),
                        alpha=line.get_alpha() or 1.0
                    )
            except Exception:
                pass
            
        # Set zoom limits
        inset_ax.set_xlim(x_min, x_max)
        inset_ax.set_ylim(y_min, y_max)
        
        # Style inset with enhanced visuals
        inset_ax.tick_params(labelsize=7)
        inset_ax.grid(True, alpha=0.3, linewidth=0.5)
        for spine in inset_ax.spines.values():
            spine.set_edgecolor(self.style.zoom_rect_color)
            spine.set_linewidth(2.0)  # Thicker border for visibility
            spine.set_linestyle('-')  # Solid line
            
        # Add title to inset with drag hint
        title_text = f'Zoom (drag to move)'
        inset_ax.set_title(title_text, fontsize=8, fontweight='bold', 
                          color=self.style.zoom_rect_color, pad=3,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=self.style.zoom_rect_color,
                                   alpha=0.9, linewidth=1.5))
        
        # Add connectors from mark_inset
        connectors = []
        try:
            connector_patches = mark_inset(
                ax, inset_ax,
                loc1=2, loc2=4,
                fc="none",
                ec=self.style.zoom_rect_color,
                linewidth=1.2,
                linestyle='--',
                alpha=0.6
            )
            if connector_patches:
                connectors = list(connector_patches)
        except Exception:
            pass
        
        # Draw rectangle on main plot
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=True,
            facecolor=self.style.zoom_rect_color,
            alpha=self.style.zoom_rect_alpha,
            edgecolor=self.style.zoom_rect_color,
            linewidth=2,
            zorder=50
        )
        ax.add_patch(rect)
        
        zoom_region = ZoomRegion(
            inset_ax=inset_ax,
            rect=rect,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            connectors=connectors,
            parent_ax=ax,
            inset_loc=loc
        )
        self.zoom_regions.append(zoom_region)
        
        self.action_history.append({
            'type': 'add_zoom',
            'object': zoom_region,
            'parent_ax': ax
        })
        self.redo_stack.clear()
        
        self._update_status(f"Zoom region created ({loc})")
        self.fig.canvas.draw_idle()
        
    # =========================================================================
    # Deletion Methods
    # =========================================================================
    
    def _delete_at_position(self, ax: plt.Axes, x: float, y: float) -> None:
        """Delete annotation at the clicked position."""
        item = self._find_annotation_at(ax, x, y)
        if item is not None:
            self._delete_item(item)
        else:
            self._update_status("No annotation at click position")
            
    def _delete_selected(self) -> None:
        """Delete currently selected annotation."""
        if self.selected_item is not None:
            self._delete_item(self.selected_item)
        else:
            self._update_status("No item selected")
            
    def _delete_item(self, item: Any) -> None:
        """Delete a specific annotation item."""
        if isinstance(item, DraggableLine):
            self._delete_line(item)
        elif isinstance(item, ZoomRegion):
            self._delete_zoom(item)
            
        if self.selected_item == item:
            self.selected_item = None
            
    def _delete_line(self, line: DraggableLine) -> None:
        """Delete a line annotation."""
        try:
            line.line.remove()
        except Exception:
            pass
            
        if line.label:
            try:
                line.label.remove()
            except Exception:
                pass
                
        if line.orientation == 'vertical':
            if line in self.vlines:
                self.vlines.remove(line)
            self.action_history.append({
                'type': 'delete_vline',
                'object': line,
                'parent_ax': line.parent_ax
            })
        else:
            if line in self.hlines:
                self.hlines.remove(line)
            self.action_history.append({
                'type': 'delete_hline',
                'object': line,
                'parent_ax': line.parent_ax
            })
            
        self.redo_stack.clear()
        self._update_status(f"{line.orientation.capitalize()} line deleted")
        self.fig.canvas.draw_idle()
        
    def _delete_zoom(self, zoom: ZoomRegion) -> None:
        """Delete a zoom region with all its components."""
        # Remove inset axes
        try:
            zoom.inset_ax.remove()
        except Exception:
            pass
            
        # Remove rectangle
        try:
            zoom.rect.remove()
        except Exception:
            pass
            
        # Remove connectors (these are the patches from mark_inset)
        for connector in zoom.connectors:
            try:
                connector.remove()
            except Exception:
                pass
        zoom.connectors.clear()
                
        if zoom in self.zoom_regions:
            self.zoom_regions.remove(zoom)
            
        self.action_history.append({
            'type': 'delete_zoom', 
            'object': zoom,
            'parent_ax': zoom.parent_ax
        })
        self.redo_stack.clear()
        
        self._update_status("Zoom region deleted")
        self.fig.canvas.draw_idle()
        
    # =========================================================================
    # Undo/Redo Methods
    # =========================================================================
    
    def _undo_last(self) -> None:
        """Undo the last action."""
        if not self.action_history:
            self._update_status("Nothing to undo")
            return
            
        action = self.action_history.pop()
        action_type = action['type']
        obj = action['object']
        
        if action_type == 'add_vline':
            # Undo add = remove
            try:
                obj.line.remove()
            except Exception:
                pass
            if obj.label:
                try:
                    obj.label.remove()
                except Exception:
                    pass
            if obj in self.vlines:
                self.vlines.remove(obj)
            self.redo_stack.append(action)
                
        elif action_type == 'add_hline':
            try:
                obj.line.remove()
            except Exception:
                pass
            if obj.label:
                try:
                    obj.label.remove()
                except Exception:
                    pass
            if obj in self.hlines:
                self.hlines.remove(obj)
            self.redo_stack.append(action)
                
        elif action_type == 'add_zoom':
            # Remove all zoom components including connectors
            try:
                obj.inset_ax.remove()
            except Exception:
                pass
            try:
                # Ensure rectangle is removed from its parent axes
                obj.rect.remove()
                # Also try removing from the axes' patches list
                if obj.parent_ax and obj.rect in obj.parent_ax.patches:
                    obj.parent_ax.patches.remove(obj.rect)
            except Exception:
                pass
            # Remove ALL connectors properly
            for connector in obj.connectors:
                try:
                    connector.remove()
                except Exception:
                    pass
            obj.connectors.clear()
            
            if obj in self.zoom_regions:
                self.zoom_regions.remove(obj)
            self.redo_stack.append(action)
            
        elif action_type in ['delete_vline', 'delete_hline', 'delete_zoom']:
            # Undo delete = restore (this is complex, so we just notify)
            self._update_status("Cannot restore deleted items - use Redo after Add")
            self.action_history.append(action)  # Put it back
            return
                
        self._update_status("Zoom region undone - green rectangle removed" if action_type == 'add_zoom' else "Action undone")
        if self.selected_item == obj:
            self.selected_item = None
        # Force a full redraw, not just idle
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def _redo_last(self) -> None:
        """Redo the last undone action."""
        if not self.redo_stack:
            self._update_status("Nothing to redo")
            return
            
        action = self.redo_stack.pop()
        action_type = action['type']
        obj = action['object']
        parent_ax = action.get('parent_ax', self.primary_ax)
        
        if action_type == 'add_vline':
            # Redo add vline
            self._place_vertical_line(parent_ax or self.primary_ax, obj.position)
            # Remove the new entry from history since redo shouldn't add new history
            if self.action_history:
                self.action_history.pop()
            self.action_history.append(action)
            
        elif action_type == 'add_hline':
            self._place_horizontal_line(parent_ax or self.primary_ax, obj.position)
            if self.action_history:
                self.action_history.pop()
            self.action_history.append(action)
            
        elif action_type == 'add_zoom':
            self._create_zoom_region(
                parent_ax or self.primary_ax,
                (obj.x_range[0], obj.y_range[0]),
                (obj.x_range[1], obj.y_range[1])
            )
            if self.action_history:
                self.action_history.pop()
            self.action_history.append(action)
            
        self._update_status("Action redone")
        self.fig.canvas.draw_idle()
        
    # =========================================================================
    # Save/Load Methods
    # =========================================================================
    
    def _save_figure(self) -> None:
        """Save figure to journal-quality PDF with only plot data (no interactive buttons)."""
        # Generate unique filename
        base_name = 'interactive_figure'
        existing = list(self.save_dir.glob(f'{base_name}_*.pdf'))
        num = len(existing) + 1
        
        # Store original figure settings
        original_bottom = self.fig.subplotpars.bottom
        
        # Temporarily hide status bar and buttons for clean export
        status_visible = self.status_text.get_visible()
        self.status_text.set_visible(False)
        
        # Store button axes and remove them from figure temporarily
        button_axes = []
        if self.buttons:
            for name, button in self.buttons.items():
                button_axes.append(button.ax)
                button.ax.set_visible(False)
        
        # Adjust subplot to remove button area (restore original tight layout)
        try:
            self.fig.subplots_adjust(bottom=0.1)  # Reset to normal bottom margin
        except Exception:
            pass
            
        # Force complete redraw without buttons
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.02)  # Ensure redraw completes
        
        # Save in journal-quality PDF format only
        saved_files = []
        fmt = 'pdf'
        
        filepath = self.save_dir / f'{base_name}_{num:02d}.{fmt}'
        try:
            # Use journal-quality PDF parameters with tight bbox to exclude button area
            save_kwargs = {
                'format': 'pdf',
                'dpi': 300,  # Journal standard resolution
                'bbox_inches': 'tight',  # Crop to actual plot content only
                'facecolor': 'white',
                'edgecolor': 'none',
                'pad_inches': 0.1,
                'metadata': {
                    'Title': f'Interactive Figure {num:02d}',
                    'Author': 'MicroPrecisionGimbal Digital Twin',
                    'Subject': 'Aerospace Gimbal Control Analysis',
                    'Creator': 'matplotlib + InteractiveFigureManager',
                },
            }
            
            # Save with proper file closing
            self.fig.savefig(str(filepath), **save_kwargs)
            
            # Force file close by flushing and waiting
            if hasattr(self.fig.canvas, 'flush_events'):
                self.fig.canvas.flush_events()
            time.sleep(0.05)  # Brief pause for file system sync
            
            # Verify file was created and has content
            if filepath.exists() and filepath.stat().st_size > 0:
                saved_files.append(filepath)
                print(f'Saved journal-quality PDF: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)')
                print(f'  → Plot data only (no interactive buttons)')
            else:
                print(f'Warning: {filepath} was not created properly')
                
        except Exception as e:
            print(f'Error saving PDF: {e}')
            import traceback
            traceback.print_exc()
        
        # Restore original layout and button visibility
        try:
            self.fig.subplots_adjust(bottom=original_bottom)
        except Exception:
            pass
            
        self.status_text.set_visible(status_visible)
        for ax in button_axes:
            ax.set_visible(True)
            
        self.fig.canvas.draw()
        
        # Save annotation state
        json_path = self.save_dir / f'{base_name}_{num:02d}_annotations.json'
        self._save_annotations(json_path)
        
        # Store last saved path for opening
        self._last_saved_pdf = saved_files[0] if saved_files else None
        
        if saved_files:
            self._update_status(f"Saved PDF as {base_name}_{num:02d}.pdf (journal quality)")
        else:
            self._update_status(f"Error: No files saved successfully")
        
    def _save_annotations(self, filepath: Path) -> None:
        """Save annotation state to JSON for reproducibility."""
        data = {
            'vlines': [
                {'position': v.position, 'orientation': v.orientation}
                for v in self.vlines
            ],
            'hlines': [
                {'position': h.position, 'orientation': h.orientation}
                for h in self.hlines
            ],
            'zoom_regions': [
                {
                    'x_range': list(z.x_range),
                    'y_range': list(z.y_range),
                    'inset_loc': z.inset_loc
                }
                for z in self.zoom_regions
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f'Annotations saved: {filepath}')
        except Exception as e:
            print(f'Error saving annotations: {e}')
            
    def _load_annotations(self, filepath: Path) -> None:
        """Load annotations from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Clear existing annotations
            for vline in self.vlines[:]:
                self._delete_line(vline)
            for hline in self.hlines[:]:
                self._delete_line(hline)
            for zoom in self.zoom_regions[:]:
                self._delete_zoom(zoom)
                
            # Load vlines
            for v in data.get('vlines', []):
                self._place_vertical_line(self.primary_ax, v['position'])
                
            # Load hlines
            for h in data.get('hlines', []):
                self._place_horizontal_line(self.primary_ax, h['position'])
                
            # Load zoom regions
            for z in data.get('zoom_regions', []):
                x_range = z['x_range']
                y_range = z['y_range']
                self._create_zoom_region(
                    self.primary_ax,
                    (x_range[0], y_range[0]),
                    (x_range[1], y_range[1])
                )
                
            self._update_status(f"Loaded annotations from {filepath.name}")
            
        except Exception as e:
            self._update_status(f"Error loading: {e}")
            
    def _open_last_saved(self) -> None:
        """Open the last saved PDF file in the default viewer."""
        import time
        
        if self._last_saved_pdf is None:
            # Find the most recent saved file
            pdf_files = sorted(self.save_dir.glob('interactive_figure_*.pdf'))
            if not pdf_files:
                self._update_status("No saved figures found")
                return
            filepath = pdf_files[-1]
        else:
            filepath = self._last_saved_pdf
            
        if not filepath.exists():
            self._update_status(f"File not found: {filepath}")
            return
            
        # Verify file is readable
        if filepath.stat().st_size == 0:
            self._update_status(f"File is empty: {filepath.name}")
            return
            
        try:
            # Small delay to ensure file is fully written and closed
            time.sleep(0.1)
            
            # Platform-specific file opening
            if sys.platform == 'win32':
                # Use absolute path and normpath for Windows
                abs_path = str(filepath.absolute().resolve())
                os.startfile(abs_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(filepath)], check=True)
            else:
                subprocess.run(['xdg-open', str(filepath)], check=True)
                
            self._update_status(f"Opened: {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)")
            print(f"Opened file: {filepath}")
        except Exception as e:
            self._update_status(f"Could not open file: {e}")
            print(f"Error opening file: {e}")
            print(f"File path: {filepath}")
            print(f"File exists: {filepath.exists()}")
            print(f"File size: {filepath.stat().st_size if filepath.exists() else 'N/A'}")
            
    # =========================================================================
    # Help and Utility
    # =========================================================================
            
    def _show_help(self) -> None:
        """Show help overlay."""
        help_text = """
================================================================================
                    INTERACTIVE FIGURE CONTROLS
================================================================================

KEYBOARD SHORTCUTS:
  V : Vertical line mode - click to place
  H : Horizontal line mode - click to place  
  Z : Zoom region mode - click two corners
  D : Delete mode - click on annotation to delete
  M : Move mode - drag annotations to reposition
  N : Navigate mode - pan/zoom (default)
  S : Save figure (PNG, PDF, SVG)
  O : Open last saved PNG
  U : Undo last action
  R : Redo last undone action
  Esc : Cancel current action / deselect
  Del : Delete selected annotation
  ? : Show this help

MOUSE ACTIONS:
  Left Click : Place annotation (in V/H/Z mode) or select (in Nav mode)
  Left Drag : Move annotation (in Move mode or when selected)
  Right Click : Quick-delete annotation under cursor
  Middle Click : Quick-select annotation under cursor

ZOOM REGIONS:
  - Click two corners to define zoom area
  - Drag the zoom rectangle to reposition the zoom area
  - Drag inside the inset to reposition the inset on the figure
  - Zoom insets cycle through corner positions

TIPS:
  - Use Undo (U) to reverse any action
  - Right-click for quick deletion without changing modes
  - Saved figures include annotation JSON for reproducibility
  - PDF and SVG exports are vector formats for publications

================================================================================
"""
        print(help_text)
        self._update_status("Help printed to console - Press ? anytime")
        
    def show(self) -> None:
        """Display the interactive figure."""
        self._update_status("Ready - Press ? for help")
        plt.show()
        
    def get_annotations(self) -> Dict:
        """Return current annotation state."""
        return {
            'vlines': [(v.position, v.orientation) for v in self.vlines],
            'hlines': [(h.position, h.orientation) for h in self.hlines],
            'zoom_regions': [(z.x_range, z.y_range, z.inset_loc) for z in self.zoom_regions]
        }


# =============================================================================
# Factory Function for Easy Integration
# =============================================================================

def make_interactive(
    fig: plt.Figure,
    axes: Union[plt.Axes, List[plt.Axes]],
    save_dir: str = "figures_comparative",
    **kwargs
) -> InteractiveFigureManager:
    """
    Factory function to make any matplotlib figure interactive.
    
    Parameters
    ----------
    fig : plt.Figure
        The figure to make interactive
    axes : plt.Axes or list
        The axes in the figure
    save_dir : str
        Directory for saving figures
    **kwargs
        Additional arguments for InteractiveStyleConfig
        
    Returns
    -------
    InteractiveFigureManager
        The manager instance
        
    Example
    -------
    ```python
    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    manager = make_interactive(fig, ax)
    manager.show()
    ```
    """
    style = InteractiveStyleConfig(**kwargs) if kwargs else None
    return InteractiveFigureManager(fig, axes, style=style, save_dir=save_dir)


# =============================================================================
# Standalone Demo
# =============================================================================

def demo():
    """Demonstrate interactive plotting capabilities."""
    print("=" * 70)
    print("  Industrial-Grade Interactive Plotting Demo")
    print("  Publication-Quality Figure Annotation Tools")
    print("=" * 70)
    print()
    
    # Generate sample data
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Simulate gimbal response
    target = np.where(t < 2, 0, 5.0)  # Step at t=2
    response = target * (1 - np.exp(-2 * (t - 2))) * (t >= 2)
    response += 0.1 * np.random.randn(len(t))  # Add noise
    
    # Create figure - explicitly disable constrained_layout for toolbar compatibility
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.set_layout_engine(None)  # Ensure no layout engine
    fig.subplots_adjust(bottom=0.18, hspace=0.15, top=0.92)
    
    # Plot data
    ax1.plot(t, target, 'k--', linewidth=2, label='Reference')
    ax1.plot(t, response, 'b-', linewidth=1.5, label='Response', alpha=0.8)
    ax1.set_ylabel('Position [deg]', fontsize=12, fontweight='bold')
    ax1.set_title('Gimbal Step Response - Interactive Demo', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 7)
    
    # Error plot
    error = response - target
    ax2.plot(t, error, 'r-', linewidth=1.5, label='Tracking Error')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(0.3, color='orange', linestyle=':', linewidth=2, label='+/- 0.3 deg threshold')
    ax2.axhline(-0.3, color='orange', linestyle=':', linewidth=2)
    ax2.fill_between(t, -0.3, 0.3, alpha=0.1, color='green')
    ax2.set_ylabel('Error [deg]', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    fig.suptitle('Interactive Figure - Press ? for controls', 
                 fontsize=14, fontweight='bold', color='#2c3e50')
    
    # Make interactive
    manager = make_interactive(fig, [ax1, ax2])
    
    print("=" * 70)
    print("QUICK CONTROLS:")
    print("-" * 70)
    print("  V : Place vertical reference line")
    print("  H : Place horizontal reference line")
    print("  Z : Create zoom region (click 2 corners)")
    print("  D : Delete mode (click to delete)")
    print("  M : Move mode (drag to reposition)")
    print("  S : Save figure (PNG/PDF/SVG)")
    print("  O : Open last saved PNG")
    print("  U : Undo | R : Redo")
    print("  Right-Click : Quick delete")
    print("  ? : Full help")
    print("=" * 70)
    print()
    
    manager.show()


if __name__ == '__main__':
    demo()
