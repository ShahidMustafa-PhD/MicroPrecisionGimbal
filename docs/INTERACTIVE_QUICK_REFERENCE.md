# Interactive Plotting Quick Reference

**Quick guide for using interactive features in demo plots.**

## Running the Demo

```bash
python demo_feedback_linearization.py
```

## Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| `Z` | Enter Zoom Mode | Enable rectangle drawing for zoom |
| `U` | Undo Last Zoom | Remove most recent zoom rectangle |
| `Delete` | Delete Selected | Remove orange-highlighted rectangle |
| `V` | Vertical Split | Split view into left/right panels |
| `H` | Horizontal Split | Split view into top/bottom panels |
| `S` | Single View | Return to single panel view |

## Mouse Actions

| Action | Effect |
|--------|--------|
| Click + Drag | Draw green zoom rectangle (in Z mode) |
| Click Rectangle | Select zoom (turns orange) |
| Right-Click Rectangle | Delete zoom instantly |
| Release Mouse | Apply zoom to selected region |

## Visual States

| Color | Meaning |
|-------|---------|
| Green | Zoom preview or active zoom rectangle |
| Orange | Selected zoom rectangle (ready to delete) |
| Crosshatch | Draw mode active |

## Common Workflows

### Basic Zoom
1. Press `Z`
2. Click-drag rectangle
3. Release to zoom in

### Undo Zoom
1. Press `U` to remove last zoom

### Delete Specific Zoom
1. Click zoom rectangle (turns orange)
2. Press `Delete`

### Quick Delete
1. Right-click zoom rectangle

## Tips

- Each figure has independent zoom history
- Undo removes zooms in reverse order (most recent first)
- Orange borders = selected and ready to delete
- All figures have these features automatically

## Figures Available

The demo creates **13 interactive figures**:

1. Angular Position Tracking
2. Tracking Error with FSM Handover
3. Control Effort & NDOB Estimation
4. Angular Velocities
5. Phase Plane
6. Controller Comparison Metrics
7. FSM Handover Analysis
8. Steady-State Error Analysis
9. Control Effort Efficiency
10. Torque Time History
11. EKF Diagnostics
12. Environmental Disturbances
13. Disturbance PSD

All support full interactive features.

## Troubleshooting

**Q: Green rectangles don't appear**  
A: Press `Z` key first to enter zoom mode

**Q: Delete key doesn't work**  
A: Click rectangle first (should turn orange) before pressing Delete

**Q: Zoom doesn't undo**  
A: Make sure you pressed `U` (uppercase U)

**Q: Right-click doesn't work (macOS)**  
A: Try Ctrl+Click instead

## For Developers

### Using Interactive Mode
```python
from lasercom_digital_twin.core.plots.research_comparison_plotter import ResearchComparisonPlotter

plotter = ResearchComparisonPlotter(interactive=True)
plotter.plot_all(results_pid, results_fbl, results_ndob, 
                 target_az_deg, target_el_deg)
```

### Disabling Interactive Mode
```python
plotter = ResearchComparisonPlotter(interactive=False)
```

### Accessing Managers
```python
manager = plotter.interactive_managers['angular_position']
print(f"Active zooms: {len(manager.zoom_rectangles)}")
```

---

**For full documentation, see:** [INTERACTIVE_IMPLEMENTATION_COMPLETE.md](INTERACTIVE_IMPLEMENTATION_COMPLETE.md)
