# Interactive Figure Editor - Quick Reference Card

## Essential Mouse Controls

| Action | What Happens |
|--------|--------------|
| **Left-click empty space** | Deselect annotation |
| **Left-click on annotation** | Select it (orange highlight) |
| **Left-drag annotation** | Move it (in Move mode or when selected) |
| **Left-drag in zoom inset** | Reposition inset window anywhere |
| **Left-drag on zoom rectangle** | Reposition what area is zoomed |
| **Right-click on annotation** | Delete immediately |
| **Hover over annotation** | Preview highlight |

## Essential Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `V` | Add vertical line (click position) |
| `H` | Add horizontal line (click position) |
| `Z` | Create zoom (click 2 corners) |
| `M` | Enter move mode (drag anything) |
| `D` | Enter delete mode (click to delete) |
| `N` | Return to navigate mode |
| `S` | **Save** PNG/PDF/SVG |
| `O` | **Open** last saved PNG |
| `U` | **Undo** last action |
| `R` | **Redo** undone action |
| `Del` | Delete selected annotation |
| `Esc` | Cancel or deselect |

## Zoom Region Workflow

1. **Create:** Press `Z`, click corner 1, click corner 2
2. **Move zoom area:** Drag the green rectangle on main plot
3. **Move inset window:** Drag inside the zoom inset
4. **Delete (3 methods):**
   - **Method 1:** Press `U` immediately after creation (undo)
   - **Method 2:** Click green rectangle (turns orange), press `Delete` key
   - **Method 3:** Right-click inside green rectangle (instant delete)

## Three Ways to Delete Zoom Rectangles

| Method | Steps | Best For |
|--------|-------|----------|
| **Undo** | Create zoom → Press `U` | Just created, changed mind |
| **Select + Delete** | Click rectangle (orange) → `Delete` key | Precise control, multiple items |
| **Right-click** | Right-click inside rectangle | Fastest, single item |

All methods remove:
- ✓ Green rectangle on main plot
- ✓ Zoom inset window
- ✓ Connector lines
- ✓ Entry from annotation list

## Common Workflows

### Annotate for Publication
1. Create figure → `make_interactive(fig, ax)`
2. Add reference lines: `V` for time markers, `H` for thresholds
3. Add zoom regions: `Z` to highlight details
4. Position zoom insets: Drag to non-overlapping spots
5. Save: `S` then `O` to verify

### Quick Exploration
1. Zoom interesting regions: `Z` key
2. Mark features: `V` key at peaks
3. Right-click to delete mistakes
4. Drag zoom insets out of the way

### Clean Up
1. Select annotation (click it)
2. Press `Delete` key
3. Or right-click for instant delete
4. Use `U` to undo if needed

## Status Bar Legend

```
Mode: Navigate [V:2 H:1 Z:1] | Item selected (Del to remove)
       ^^^^^^^^  ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       Current    Annotation    Additional context
       mode       counts
```

## Troubleshooting Tips

❓ **Green rectangle won't disappear with U (undo)?**  
→ Try Method 2: Click rectangle → press `Delete` key  
→ Or Method 3: Right-click inside rectangle for instant delete

❓ **Zoom inset won't move?**  
→ Drag **inside** the inset, not on border

❓ **Can't select annotation?**  
→ Press `N` for Navigate mode first

❓ **PNG won't open?**  
→ Check `figures_comparative/` folder, file should be >0 KB

❓ **Want to start over?**  
→ Delete all annotations with right-click, or close and reopen figure

## Pro Tips

💡 Right-click is fastest for deletion  
💡 Hover shows what you'll affect before clicking  
💡 Drag zoom insets to figure edges for clean look  
💡 Use PDF export for vector graphics (presentations)  
💡 Use PNG export for raster images (documents)  
💡 Undo/Redo work for everything  
💡 Annotations saved to JSON for reproducibility

---

**Press `?` in any interactive figure for full help**
