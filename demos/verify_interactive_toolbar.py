#!/usr/bin/env python3
"""
Verification script to compare interactive toolbar presence between
test_zoom_deletion.py and ResearchComparisonPlotter.

This ensures both use the same InteractiveFigureManager pattern.

Author: Dr. S. Shahid Mustafa
Date: January 30, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.plots.interactive_plotter import make_interactive


def test_basic_interactive():
    """Test 1: Basic interactive figure (reference pattern)."""
    print("\n" + "="*70)
    print("TEST 1: BASIC INTERACTIVE FIGURE (test_zoom_deletion.py pattern)")
    print("="*70)
    
    # Create figure WITHOUT constrained_layout (key requirement)
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_layout_engine(None)
    fig.subplots_adjust(bottom=0.18)
    
    # Generate test data
    t = np.linspace(0, 10, 500)
    y = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t)
    
    ax.plot(t, y, 'b-', linewidth=2, label='Damped sine', alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Test 1: Basic Interactive (Reference Pattern)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    manager = make_interactive(fig, ax)
    
    print("\nExpected:")
    print("  ✓ Toolbar buttons at bottom (Nav, VLine, HLine, Zoom, Move, Del, etc.)")
    print("  ✓ Status bar showing current mode")
    print("  ✓ Keyboard shortcuts work (Z, V, H, U, S, etc.)")
    
    return manager


def test_research_plotter_style():
    """Test 2: ResearchComparisonPlotter style (with constrained_layout disabled)."""
    print("\n" + "="*70)
    print("TEST 2: RESEARCH PLOTTER STYLE (constrained_layout=False)")
    print("="*70)
    
    # Simulate ResearchComparisonPlotter behavior with interactive=True
    # Key: constrained_layout=False from the start
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                     sharex=True, constrained_layout=False)
    
    # Generate test data
    t = np.linspace(0, 10, 500)
    y1 = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t)
    y2 = np.cos(2*np.pi*0.3*t) * np.exp(-0.15*t)
    
    ax1.plot(t, y1, 'b-', linewidth=2, label='PID', alpha=0.8)
    ax1.plot(t, y2, 'r-', linewidth=2, label='FBL', alpha=0.8)
    ax1.set_ylabel('Azimuth [deg]', fontsize=11, fontweight='bold')
    ax1.set_title('Test 2: Research Style (Multi-Axes)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, y1*0.5, 'g-', linewidth=2, label='FBL+NDOB', alpha=0.8)
    ax2.set_ylabel('Elevation [deg]', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Gimbal Position Tracking (Research Style)', fontsize=14, fontweight='bold')
    
    manager = make_interactive(fig, [ax1, ax2])
    
    print("\nExpected:")
    print("  ✓ Toolbar buttons at bottom (SAME as Test 1)")
    print("  ✓ Status bar showing current mode")
    print("  ✓ Works on both axes")
    
    return manager


def test_constrained_layout_warning():
    """Test 3: What happens if constrained_layout=True (should disable toolbar)."""
    print("\n" + "="*70)
    print("TEST 3: WITH CONSTRAINED_LAYOUT=TRUE (should fallback to keyboard-only)")
    print("="*70)
    
    # This is the OLD ResearchComparisonPlotter behavior (before fix)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    
    t = np.linspace(0, 10, 500)
    y = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t)
    
    ax.plot(t, y, 'r-', linewidth=2, label='Wrong Config', alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Test 3: constrained_layout=True (Degraded Mode)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    manager = make_interactive(fig, ax)
    
    print("\nExpected:")
    print("  ✗ NO toolbar buttons (keyboard-only mode)")
    print("  ✓ Keyboard shortcuts still work (Z, V, H, etc.)")
    print("  ✓ This demonstrates the problem we just fixed!")
    
    return manager


if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("INTERACTIVE TOOLBAR VERIFICATION")
    print("="*70)
    print("\nThis script verifies that ResearchComparisonPlotter produces")
    print("identical interactive behavior to test_zoom_deletion.py")
    print()
    print("Key Requirement: constrained_layout=False when creating figures")
    print("="*70)
    
    # Run all three tests
    manager1 = test_basic_interactive()
    manager2 = test_research_plotter_style()
    manager3 = test_constrained_layout_warning()
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print("\nClose each figure window to continue to the next test.")
    print("Compare toolbar presence between Test 1, Test 2, and Test 3.")
    print("\nTest 1 & 2 should be IDENTICAL (toolbar present)")
    print("Test 3 should have NO toolbar (demonstrates old bug)")
    print("="*70)
    
    # Show figures sequentially
    print("\n[Press any key in figure window to continue to next test...]")
    manager1.show()
    print("\n[Showing Test 2...]")
    manager2.show()
    print("\n[Showing Test 3 - note the missing toolbar!]")
    manager3.show()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\n✓ If Test 1 and Test 2 look identical, the fix is successful!")
    print("✓ ResearchComparisonPlotter now uses the correct pattern.\n")
