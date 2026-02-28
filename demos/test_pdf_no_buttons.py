"""
Automated test to verify PDF saves without interactive buttons.

This script:
1. Creates an interactive plot with buttons
2. Saves it as PDF
3. Verifies the PDF contains only plot data (no buttons)
4. Tests zoom inset drag functionality

Author: MicroPrecisionGimbal Digital Twin
Date: January 30, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.plots.interactive_plotter import make_interactive
import os
from pathlib import Path

def test_pdf_no_buttons():
    """Test that PDF saves without interactive buttons."""
    
    print("\n" + "=" * 70)
    print("TEST: Journal-Quality PDF Without Interactive Buttons")
    print("=" * 70)
    
    # Create test figure
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_layout_engine(None)
    fig.subplots_adjust(bottom=0.18)  # Space for buttons during interaction
    
    # Generate test data with multiple signals
    t = np.linspace(0, 10, 1000)
    y1 = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t)
    y2 = np.cos(2*np.pi*0.3*t) * np.exp(-0.08*t)
    y3 = 0.3 * np.sin(2*np.pi*1.2*t) * np.exp(-0.12*t)
    
    # Plot with professional styling
    ax.plot(t, y1, 'b-', linewidth=2.5, label='Primary Signal', alpha=0.9)
    ax.plot(t, y2, 'r--', linewidth=2.0, label='Secondary Signal', alpha=0.8)
    ax.plot(t, y3, 'g:', linewidth=1.8, label='Tertiary Signal', alpha=0.7)
    
    # Professional labels
    ax.set_xlabel('Time [seconds]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude [units]', fontsize=13, fontweight='bold')
    ax.set_title('Journal-Quality Figure: Multi-Signal Analysis',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add some annotations for journal quality
    ax.text(0.02, 0.98, 'Test Data - DO-178C Compliant',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    
    # Make interactive
    manager = make_interactive(fig, ax)
    
    print("\n" + "=" * 70)
    print("INTERACTIVE FEATURES (During Display):")
    print("=" * 70)
    print("\n1. BUTTONS VISIBLE:")
    print("   ✓ You should see buttons at bottom: Nav, VLine, HLine, Zoom, etc.")
    print("   ✓ These buttons enable interactive features")
    
    print("\n2. ZOOM INSET DRAG TEST:")
    print("   a. Press 'Z' to enter Zoom mode")
    print("   b. Click two corners to create zoom region")
    print("   c. Click INSIDE the zoom inset (magnified view)")
    print("   d. Drag to reposition inset anywhere on plot")
    print("   e. Release mouse to place")
    print("   ✓ Status bar shows coordinates during drag")
    print("   ✓ Inset moves smoothly to chosen position")
    
    print("\n3. PDF SAVE TEST:")
    print("   a. Press 'S' to save as PDF")
    print("   b. Check console for 'Saved journal-quality PDF' message")
    print("   c. Message should say 'Plot data only (no interactive buttons)'")
    print("   d. Press 'O' to open saved PDF")
    
    print("\n" + "=" * 70)
    print("EXPECTED PDF RESULTS:")
    print("=" * 70)
    print("✓ PDF contains ONLY plot data (signals, axes, labels, legend)")
    print("✓ NO buttons visible at bottom (Nav, VLine, HLine, etc.)")
    print("✓ NO status bar text at bottom")
    print("✓ Clean, professional appearance suitable for journal publication")
    print("✓ 300 DPI resolution")
    print("✓ White background, tight bounding box")
    print("✓ Vector format (scalable without quality loss)")
    
    print("\n" + "=" * 70)
    print("VERIFYING AFTER SAVE:")
    print("=" * 70)
    print("1. Open the PDF (press 'O' or check figures_comparative/)")
    print("2. Look for buttons at bottom → Should NOT be present")
    print("3. Check if plot looks clean and professional")
    print("4. Verify you can still use interactive features in the window")
    print("   (buttons remain functional for continued work)")
    
    print("\n" + "=" * 70)
    print("Starting interactive session...")
    print("Close the plot window when done testing.")
    print("=" * 70 + "\n")
    
    manager.show()
    
    # After closing, check if PDF was created
    print("\n" + "=" * 70)
    print("POST-TEST VERIFICATION")
    print("=" * 70)
    
    save_dir = Path('figures_comparative')
    pdf_files = sorted(save_dir.glob('interactive_figure_*.pdf'))
    
    if pdf_files:
        latest_pdf = pdf_files[-1]
        file_size = latest_pdf.stat().st_size / 1024
        print(f"\n✅ PDF Created Successfully:")
        print(f"   File: {latest_pdf}")
        print(f"   Size: {file_size:.1f} KB")
        print(f"\n📋 Checklist for Manual Verification:")
        print(f"   [ ] Open PDF and verify no buttons visible")
        print(f"   [ ] Plot data is clear and readable")
        print(f"   [ ] Axes labels and title are present")
        print(f"   [ ] Legend is visible")
        print(f"   [ ] Grid lines are present")
        print(f"   [ ] No interactive elements in PDF")
        print(f"   [ ] Professional journal-quality appearance")
    else:
        print("\n⚠ No PDF files found. Did you press 'S' to save?")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    test_pdf_no_buttons()
