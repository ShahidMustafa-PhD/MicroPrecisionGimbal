zzzzzzzzzzzzzz# Professional Interactive Figure Editor - Complete Guide

**Module:** `lasercom_digital_twin.core.plots.interactive_plotter`  
**Class:** `InteractiveFigureManager`  
**Version:** 3.0.0 - Professional Grade  
**Date:** January 30, 2026

---

## Overview

The `InteractiveFigureManager` class provides industrial-grade interactive annotation capabilities for matplotlib figures, designed for control engineering analysis and publication preparation.

## Professional Features

### 1. **Universal Mouse-Based Interaction**

All operations can be performed via mouse with intuitive controls:

| Mouse Action | Function | Context |
|-------------|----------|---------|
| **Left Click** | Place annotation | In V/H/Z modes |
| **Left Click** | Select annotation | In Navigate mode |
| **Left Drag** | Move annotation | In Move mode or on selected item |
| **Left Drag (in zoom inset)** | Reposition zoom window | Anywhere on figure |
| **Left Drag (on zoom rect)** | Reposition zoom area | Changes what's zoomed |
| **Right Click** | Quick-delete | Any mode, any annotation |
| **Middle Click** | Quick-select | Any annotation |
| **Hover** | Highlight preview | Visual feedback in Navigate mode |

### 2. **Zoom Region Controls**

#### Creating Zoom Regions
1. Press `Z` key to enter zoom mode
2. Click first corner of region to zoom
3. Drag or click second corner to complete
4. Zoom inset appears at auto-positioned location

#### Repositioning Zoom Area
- **Drag the green rectangle** on the main plot to move what area is being zoomed
- The zoom remains the same size but shows a different region

#### Repositioning Zoom Inset
- **Click and drag inside the zoom inset window** to move it anywhere on the figure
- Inset stays within figure bounds automatically
- Title shows "Zoom (drag to move)" as a hint

#### Deleting Zoom Regions

**Three Independent Methods Available:**

1. **Undo Method (Immediate):**
   - Create zoom with `Z` key
   - Immediately press `U` to undo
   - Removes: green rectangle, zoom inset, connector lines
   - Status: "Zoom region undone - green rectangle removed"
   - **Best for:** Just created, changed mind instantly

2. **Select + Delete Key (Precise Control):**
   - Create zoom region (green rectangle visible)
   - Click inside the green rectangle to select it
   - Rectangle border turns ORANGE (thick), inset spines turn orange
   - Status: "Zoom region selected - Press DELETE to remove..."
   - Press `Delete` key on keyboard
   - Everything disappears: rectangle, inset, connectors
   - Status: "Zoom region deleted"
   - **Best for:** Selective deletion, managing multiple annotations

3. **Right-Click (Fastest):**
   - Right-click anywhere inside the green rectangle
   - Instant deletion without selection step
   - Status: "Zoom region deleted"
   - **Best for:** Quick cleanup, single item removal

**All methods remove:**
- ✓ Green rectangle on main plot
- ✓ Zoom inset window  
- ✓ Connector lines (dashed lines from rectangle to inset)
- ✓ Entry from zoom regions list and action history

**Visual Feedback:**
- Normal: Green border (2px), semi-transparent fill
- Selected: Orange border (3px), increased opacity
- Deleted: Complete removal, forced canvas redraw

### 3. **Annotation Management**

#### Vertical and Horizontal Lines
- **Add:** Press `V` (vertical) or `H` (horizontal), then click position
- **Move:** Press `M`, then drag the line to new position
- **Delete:** Right-click or select + Delete key
- **Labels:** Auto-positioned with current value

#### Selection System
- **Visual Feedback:** Selected items highlighted in orange
- **Hover Effect:** Items brighten when mouse hovers over them
- **Multi-Mode:** Selection works in Navigate and Move modes
- **Deselect:** Click empty space or press `Esc`

### 4. **Professional File Operations**

#### Saving Figures
- **Keyboard:** Press `S`
- **Button:** Click "Save" button
- **Formats:** Saves PNG (300 DPI), PDF (vector), SVG (vector)
- **Annotations:** JSON file saved with coordinates for reproducibility
- **Clean Export:** Buttons and status bar hidden automatically
- **Verification:** File size displayed to confirm successful save

#### Opening Saved Figures
- **Keyboard:** Press `O`
- **Button:** Click "Open" button
- **Platform Support:** Works on Windows, macOS, and Linux
- **Smart Detection:** Opens last saved file or finds most recent
- **Error Handling:** Displays helpful error messages if file issues occur

#### PNG Save Fix
The PNG save mechanism has been completely redesigned:
- **Proper DPI:** 300 DPI enforced for publication quality
- **Optimization:** PIL optimize flag enabled for smaller files
- **File Verification:** Checks file exists and has content before reporting success
- **Timing:** Small delay before opening to ensure file is fully written
- **Path Handling:** Uses absolute resolved paths for Windows compatibility

### 5. **Keyboard Shortcuts**

| Key | Mode/Action | Description |
|-----|-------------|-------------|
| `V` | Vertical Line | Click to place vertical reference line |
| `H` | Horizontal Line | Click to place horizontal reference line |
| `Z` | Zoom Region | Click 2 corners to define zoom area |
| `D` | Delete Mode | Click any annotation to delete |
| `M` | Move Mode | Drag annotations to reposition |
| `N` | Navigate | Pan/zoom mode (default) |
| `S` | Save | Export to PNG/PDF/SVG |
| `O` | Open | Open last saved PNG |
| `U` | Undo | Reverse last action |
| `R` | Redo | Restore undone action |
| `Esc` | Cancel/Deselect | Cancel current mode or clear selection |
| `Delete` | Delete Selected | Remove currently selected annotation |
| `?` | Help | Print full help text to console |

### 6. **Status Bar Enhancements**

The status bar provides real-time feedback:
- **Current Mode:** Shows active interaction mode
- **Annotation Counts:** `[V:2 H:1 Z:1]` shows number of each type
- **Instructions:** Context-aware help for current operation
- **Selection Status:** Indicates when item is selected
- **Position Feedback:** Shows coordinates during zoom repositioning

### 7. **Undo/Redo System**

Full history management:
- **Undo:** Press `U` to reverse last action (add/delete)
- **Redo:** Press `R` to restore undone action
- **Stack Management:** Redo cleared on new action
- **Zoom Handling:** Completely removes all zoom components including connectors
- **Visual Update:** Immediate redraw after undo/redo

### 8. **Visual Feedback System**

#### Hover Effects
- **Lines:** Slightly thicker when mouse over
- **Zoom Regions:** Increased opacity on hover
- **Cursor Context:** Visual hint of what will be affected

#### Selection Highlighting
- **Lines:** Orange label background + thicker line
- **Zoom Regions:** Orange border + thicker edge
- **Persistent:** Remains until deselected

#### Drag Feedback
- **Active Dragging:** Status bar shows "Dragging..."
- **Zoom Inset:** Position coordinates displayed during drag
- **Preview:** Dashed rectangle during zoom creation

---

## Usage Examples

### Basic Usage

```python
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.plots import make_interactive

# Create your plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label='Data')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)

# Make it interactive
manager = make_interactive(fig, ax)
manager.show()
```

### Advanced Usage with Multiple Axes

```python
from lasercom_digital_twin.core.plots import make_interactive

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(t, position)
ax2.plot(t, error)

# Pass all axes for multi-axis interaction
manager = make_interactive(fig, [ax1, ax2])
manager.show()
```

### Custom Styling

```python
from lasercom_digital_twin.core.plots import make_interactive, InteractiveStyleConfig

style = InteractiveStyleConfig(
    vline_color='#ff0000',      # Red vertical lines
    hline_color='#0000ff',      # Blue horizontal lines
    zoom_rect_color='#00ff00',  # Green zoom regions
    save_dpi=600,               # Ultra-high resolution
    inset_width='40%',          # Larger zoom inset
    inset_height='35%'
)

manager = make_interactive(fig, ax, style=style)
manager.show()
```

---

## Workflow Recommendations

### For Publication Figures

1. **Create Base Plot:** Generate matplotlib figure with all data
2. **Activate Interactive:** Call `make_interactive(fig, ax)`
3. **Add Annotations:**
   - Press `V` to mark key time points
   - Press `H` to mark threshold levels
   - Press `Z` to create zoom regions of interest
4. **Position Elements:** Drag zoom insets to non-overlapping positions
5. **Save:** Press `S` to export (PNG for raster, PDF/SVG for vector)
6. **Verify:** Press `O` to open and check the saved figure

### For Data Analysis

1. **Zoom Exploration:** Create multiple zoom regions (`Z` key)
2. **Mark Features:** Add vertical lines at peaks/transitions (`V` key)
3. **Threshold Comparison:** Add horizontal lines (`H` key)
4. **Quick Adjustments:** Right-click to delete unwanted annotations
5. **Iterate:** Use Undo/Redo (`U`/`R`) to try different annotations

### For Presentations

1. **Clean Layout:** Ensure annotations don't overlap
2. **High Contrast:** Use default colors for visibility
3. **Large Features:** Annotations are thick and clear
4. **Export High-DPI:** 300 DPI PNG ensures clarity on projectors

---

## Technical Details

### Architecture

The `InteractiveFigureManager` uses:
- **Event-driven design:** Matplotlib event handlers for all interactions
- **State machine:** Mode-based operation (Navigate, VLine, HLine, Zoom, Delete, Move)
- **Object tracking:** Maintains lists of all annotations with metadata
- **History stack:** Full undo/redo with action recording

### Data Structures

```python
@dataclass
class DraggableLine:
    line: Line2D              # Matplotlib line object
    orientation: str          # 'vertical' or 'horizontal'
    position: float           # Current position
    label: Text               # Text annotation
    is_dragging: bool         # Drag state
    is_selected: bool         # Selection state
    parent_ax: Axes           # Parent axes reference

@dataclass
class ZoomRegion:
    inset_ax: Axes            # Inset axes for zoom
    rect: Rectangle           # Source rectangle on main plot
    x_range: Tuple[float, float]  # X limits
    y_range: Tuple[float, float]  # Y limits
    connectors: List[Patch]   # Connector lines from mark_inset
    is_selected: bool         # Selection state
    is_dragging: bool         # Drag state
    parent_ax: Axes           # Parent axes reference
    inset_loc: str            # Position label
```

### Performance Considerations

- **Efficient Redraw:** Uses `draw_idle()` to batch updates
- **Hover Throttling:** Only updates on actual item change
- **Minimal Overhead:** Event handlers check bounds before processing
- **Smart Saves:** Hides UI elements during export to avoid re-rendering

---

## Troubleshooting

### PNG Not Opening

**Fixed in v3.0:**
- File verification added (checks size > 0)
- Absolute paths used for Windows
- Small delay before opening
- Better error messages with diagnostics

If still having issues:
```python
# Check saved files manually
import os
files = list(Path('figures_comparative').glob('*.png'))
for f in files:
    print(f"{f.name}: {f.stat().st_size / 1024:.1f} KB")
```

### Zoom Inset Not Moving

**Requirements:**
- Must drag **inside** the inset axes, not on the border
- Use left mouse button
- Can be in Navigate or Move mode
- Inset title says "Zoom (drag to move)" when possible

### Selection Not Working

**Checklist:**
- Must be in Navigate mode (`N` key) for click-to-select
- In Move mode, drag starts immediately without selection
- Right-click always deletes, doesn't select
- Click empty space to deselect

### Undo Not Removing Zoom

**Fixed in v3.0:**
- All zoom components removed (inset, rectangle, connectors)
- `connectors.clear()` called to prevent stale references
- Complete state restoration

---

## Integration with Research Workflow

### With demo_feedback_linearization.py

The interactive plotter is already integrated:

```python
# At the end of demo_feedback_linearization.py
if __name__ == "__main__":
    # ... simulation runs ...
    # ... static figures saved ...
    
    # Interactive mode
    from lasercom_digital_twin.core.plots.interactive_plotter import make_interactive
    
    interactive_figures = [
        (fig_pos, [ax_az, ax_el]),
        (fig_error, [ax_err_az, ax_err_el]),
        (fig_los, ax_los)
    ]
    
    for fig, axes in interactive_figures:
        manager = make_interactive(fig, axes)
    
    plt.show()  # Show all interactive figures
```

### Annotation Persistence

Annotations are saved to JSON for reproducibility:

```json
{
  "vlines": [
    {"position": 2.5, "orientation": "vertical"}
  ],
  "hlines": [
    {"position": 0.1, "orientation": "horizontal"}
  ],
  "zoom_regions": [
    {
      "x_range": [2.0, 3.0],
      "y_range": [0.0, 0.5],
      "inset_loc": "upper left"
    }
  ]
}
```

Load annotations programmatically:

```python
manager._load_annotations(Path('figure_annotations.json'))
```

---

## Future Enhancements (Roadmap)

### Planned Features
- **Text Annotations:** Add custom text labels with drag
- **Arrow Annotations:** Point to features with arrows
- **Measurement Tools:** Click-to-measure distances
- **Multiple Zoom Types:** Box zoom, span zoom, dynamic zoom
- **Annotation Styling:** Change colors, line styles per annotation
- **Export Presets:** Quick save for different use cases (print, web, presentation)
- **Collaborative Annotations:** Save/load/merge annotation sets

### Under Consideration
- **Matplotlib Toolbar Integration:** Seamless integration with pan/zoom toolbar
- **Touch Support:** Multi-touch gestures for tablets
- **Animation Export:** Save annotated animations
- **LaTeX Math:** Add LaTeX equations as annotations

---

## Best Practices

### Do's
✅ Use keyboard shortcuts for speed  
✅ Right-click for quick operations  
✅ Save frequently with different names  
✅ Test zoom positions before final save  
✅ Use Undo liberally to experiment  
✅ Export to PDF for presentations (vector graphics)  
✅ Export to PNG for documents (300 DPI)  

### Don'ts
❌ Don't use in constrained_layout mode (use `fig.set_layout_engine(None)`)  
❌ Don't drag from inset border (drag from inside)  
❌ Don't forget to press `N` to return to Navigate mode  
❌ Don't overlap zoom insets with data  
❌ Don't use too many annotations (clutters figure)  

---

## Credits

**Author:** Dr. S. Shahid Mustafa  
**Project:** Lasercom Digital Twin - MicroPrecisionGimbal  
**Purpose:** Publication-quality interactive figure annotation for control systems research  
**License:** Project-specific (see main README)

---

## Changelog

### Version 3.0.0 (January 30, 2026)
- **Fixed:** PNG save/open mechanism with proper file verification
- **Enhanced:** Zoom inset repositioning with better bounds checking
- **Added:** Hover feedback system for visual preview
- **Added:** Position display during zoom inset drag
- **Improved:** Universal selection system (works in multiple modes)
- **Improved:** Status bar with real-time position feedback
- **Improved:** Zoom inset styling with drag hint in title
- **Fixed:** Click deselection in empty space
- **Fixed:** Zoom undo completely removes all connectors
- **Optimized:** File opening with timing and error diagnostics

### Version 2.0.0 (January 29, 2026)
- Full mouse-based interaction system
- Zoom inset dragging capability
- Redo support
- Open saved files functionality
- Selection highlighting
- Professional button toolbar

### Version 1.0.0 (Initial)
- Basic annotation placement
- Keyboard shortcuts
- Undo support
- Multi-format save

---

## Support

For issues or questions:
1. Check this guide first
2. Run the demo: `python -m lasercom_digital_twin.core.plots.interactive_plotter`
3. Press `?` in any interactive figure for help
4. Check console output for error messages

**Demo Command:**
```bash
python -c "from lasercom_digital_twin.core.plots.interactive_plotter import demo; demo()"
```
