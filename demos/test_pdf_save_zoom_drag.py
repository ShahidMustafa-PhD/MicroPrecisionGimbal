"""
Test script to verify:
1. PDF-only save functionality (journal quality)
2. Zoom inset drag-to-move functionality
3. All interactive features working

Author: MicroPrecisionGimbal Digital Twin
Date: January 30, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.plots.interactive_plotter import make_interactive

def test_pdf_save_and_zoom_drag():
    """Test PDF save and zoom inset dragging."""
    
    print("=" * 70)
    print("TESTING: PDF Save & Zoom Inset Drag Functionality")
    print("=" * 70)
    
    # Create test figure with proper settings
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_layout_engine(None)
    fig.subplots_adjust(bottom=0.18)
    
    # Generate test data - damped oscillation
    t = np.linspace(0, 10, 1000)
    y1 = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t)
    y2 = np.cos(2*np.pi*0.5*t) * np.exp(-0.1*t)
    y3 = 0.5 * np.sin(2*np.pi*1.5*t) * np.exp(-0.15*t)
    
    # Plot multiple signals
    ax.plot(t, y1, 'b-', linewidth=2, label='Signal 1 (Sine)', alpha=0.8)
    ax.plot(t, y2, 'r--', linewidth=2, label='Signal 2 (Cosine)', alpha=0.8)
    ax.plot(t, y3, 'g:', linewidth=2, label='Signal 3 (High freq)', alpha=0.8)
    
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title('Test: PDF Save & Zoom Inset Dragging', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Make interactive
    manager = make_interactive(fig, ax)
    
    # Print instructions
    print("\n" + "=" * 70)
    print("INTERACTIVE FEATURES TO TEST:")
    print("=" * 70)
    print("\n1. ZOOM INSET DRAG-TO-MOVE:")
    print("   - Press 'Z' to enter Zoom mode")
    print("   - Click two corners to create a zoom region")
    print("   - Click INSIDE the zoom inset (magnified view)")
    print("   - Drag the inset to any position on the figure")
    print("   - Release mouse to place the inset")
    print("   - The inset should move smoothly to your chosen position")
    
    print("\n2. PDF SAVE (JOURNAL QUALITY):")
    print("   - Press 'S' to save the figure")
    print("   - Check console for 'Saved PDF:' message")
    print("   - File should be saved as .pdf in figures_comparative/")
    print("   - Press 'O' to open the saved PDF")
    print("   - Verify it opens in your default PDF viewer")
    print("   - PDF should be 300 DPI with metadata")
    
    print("\n3. OTHER FEATURES:")
    print("   - Press 'V' to add vertical lines")
    print("   - Double-click on vertical line to hide/show label")
    print("   - Press 'M' to enter Move mode")
    print("   - Drag any annotation to reposition")
    print("   - Press 'U' to undo")
    print("   - Press '?' for full help")
    
    print("\n" + "=" * 70)
    print("EXPECTED RESULTS:")
    print("=" * 70)
    print("✓ Zoom inset drags smoothly to any position")
    print("✓ Status bar shows coordinates during drag")
    print("✓ PDF saves with 'Saved PDF:' message (no PNG/SVG)")
    print("✓ PDF opens in default viewer when pressing 'O'")
    print("✓ File size shown in KB")
    print("✓ Toolbar buttons visible and working")
    print("=" * 70)
    
    manager.show()
    
    print("\n[OK] Test completed. Check if all features worked correctly.")

if __name__ == '__main__':
    test_pdf_save_and_zoom_drag()
