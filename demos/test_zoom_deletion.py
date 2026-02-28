#!/usr/bin/env python3
"""
Test script for zoom rectangle deletion methods.

This script verifies that all three methods of deleting zoom rectangles work:
1. Undo (U key) - removes the last zoom region added
2. Select + Delete key - click on rectangle then press Delete
3. Right-click - instant deletion

Author: Dr. S. Shahid Mustafa
Date: January 30, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.plots.interactive_plotter import make_interactive


def create_test_figure():
    """Create a test figure with sample data."""
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_layout_engine(None)
    fig.subplots_adjust(bottom=0.18)
    
    # Generate test data
    t = np.linspace(0, 10, 500)
    y1 = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t)
    y2 = np.cos(2*np.pi*0.3*t) * np.exp(-0.15*t)
    
    ax.plot(t, y1, 'b-', linewidth=2, label='Damped sine', alpha=0.8)
    ax.plot(t, y2, 'r-', linewidth=2, label='Damped cosine', alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Zoom Rectangle Deletion Test - Enhanced Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig, ax


def print_instructions():
    """Print detailed test instructions."""
    print()
    print("="*70)
    print("  ZOOM RECTANGLE DELETION - VERIFICATION TEST")
    print("="*70)
    print()
    print("OBJECTIVE:")
    print("  Verify that green zoom rectangles can be deleted using all three methods")
    print()
    print("="*70)
    print("METHOD 1: UNDO WITH 'U' KEY")
    print("="*70)
    print("  Steps:")
    print("    1. Press 'Z' to enter zoom mode")
    print("    2. Click two corners to create a green zoom rectangle")
    print("    3. Immediately press 'U' to undo")
    print()
    print("  Expected Result:")
    print("    ✓ Green rectangle DISAPPEARS completely")
    print("    ✓ Zoom inset window is removed")
    print("    ✓ Connector lines disappear")
    print("    ✓ Status shows: 'Zoom region undone - green rectangle removed'")
    print()
    print("="*70)
    print("METHOD 2: SELECT + DELETE KEY")
    print("="*70)
    print("  Steps:")
    print("    1. Press 'Z' and create a zoom rectangle (2 clicks)")
    print("    2. Press 'N' to return to Navigate mode")
    print("    3. Click INSIDE the green rectangle to select it")
    print("    4. Observe: Rectangle border turns ORANGE (thick)")
    print("    5. Press DELETE key on keyboard")
    print()
    print("  Expected Result:")
    print("    ✓ Orange selection highlight appears when clicked")
    print("    ✓ Inset border also turns orange")
    print("    ✓ Status shows: 'Zoom region selected - Press DELETE to remove...'")
    print("    ✓ After DELETE: Everything disappears")
    print("    ✓ Status shows: 'Zoom region deleted'")
    print()
    print("="*70)
    print("METHOD 3: RIGHT-CLICK (QUICK DELETE)")
    print("="*70)
    print("  Steps:")
    print("    1. Press 'Z' and create a zoom rectangle")
    print("    2. Right-click INSIDE the green rectangle")
    print()
    print("  Expected Result:")
    print("    ✓ Instant deletion (no selection step needed)")
    print("    ✓ Everything disappears immediately")
    print("    ✓ Status shows: 'Zoom region deleted'")
    print()
    print("="*70)
    print("ADDITIONAL TESTS")
    print("="*70)
    print("  • Create multiple zoom rectangles and delete them individually")
    print("  • Test dragging zoom inset before deletion")
    print("  • Verify that undo history works correctly")
    print()
    print("="*70)
    print("KEYBOARD SHORTCUTS REFERENCE")
    print("="*70)
    print("  Z  = Zoom mode (create rectangle)")
    print("  N  = Navigate mode (pan/select)")
    print("  M  = Move mode (drag annotations)")
    print("  D  = Delete mode (click to delete)")
    print("  U  = Undo last action")
    print("  S  = Save figure")
    print("  DELETE = Delete selected item")
    print("  ESC = Cancel current action")
    print()
    print("="*70)
    print("TEST STATUS: Interactive window opened")
    print("="*70)
    print()


if __name__ == "__main__":
    # Create figure
    fig, ax = create_test_figure()
    
    # Make interactive
    manager = make_interactive(fig, ax)
    
    # Print instructions
    print_instructions()
    
    # Show interactive figure
    manager.show()
    
    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)
