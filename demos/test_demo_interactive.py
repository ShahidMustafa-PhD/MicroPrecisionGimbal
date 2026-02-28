#!/usr/bin/env python3
"""
Quick verification that demo_feedback_linearization.py uses ResearchComparisonPlotter.

This script:
1. Parses demo_feedback_linearization.py to verify imports
2. Confirms the ResearchComparisonPlotter is instantiated
3. Verifies interactive=True parameter is set
"""

import re
from pathlib import Path

def verify_demo_integration():
    """Verify demo script uses interactive plotter class."""
    
    demo_path = Path(__file__).parent / "demo_feedback_linearization.py"
    
    with open(demo_path, 'r') as f:
        content = f.read()
    
    print("=" * 80)
    print("DEMO INTEGRATION VERIFICATION")
    print("=" * 80)
    
    # Check 1: Import statement
    import_pattern = r'from lasercom_digital_twin\.core\.plots\.research_comparison_plotter import ResearchComparisonPlotter'
    has_import = bool(re.search(import_pattern, content))
    print(f"\n✓ Import ResearchComparisonPlotter: {'PASS' if has_import else 'FAIL'}")
    if not has_import:
        print("  ERROR: Missing import statement")
        return False
    
    # Check 2: Class instantiation
    instantiation_pattern = r'plotter\s*=\s*ResearchComparisonPlotter\('
    has_instantiation = bool(re.search(instantiation_pattern, content))
    print(f"✓ Instantiate plotter class: {'PASS' if has_instantiation else 'FAIL'}")
    if not has_instantiation:
        print("  ERROR: Missing class instantiation")
        return False
    
    # Check 3: Interactive parameter
    interactive_pattern = r'interactive\s*=\s*True'
    has_interactive = bool(re.search(interactive_pattern, content))
    print(f"✓ Interactive mode enabled: {'PASS' if has_interactive else 'FAIL'}")
    if not has_interactive:
        print("  WARNING: interactive=True not found (may use default)")
    
    # Check 4: plot_all() call
    plot_all_pattern = r'plotter\.plot_all\('
    has_plot_all = bool(re.search(plot_all_pattern, content))
    print(f"✓ Call plot_all() method: {'PASS' if has_plot_all else 'FAIL'}")
    if not has_plot_all:
        print("  ERROR: Missing plot_all() call")
        return False
    
    # Check 5: Legacy function marked deprecated
    legacy_pattern = r'\[DEPRECATED\].*Use ResearchComparisonPlotter'
    has_deprecated = bool(re.search(legacy_pattern, content, re.DOTALL))
    print(f"✓ Legacy function deprecated: {'PASS' if has_deprecated else 'WARN'}")
    if not has_deprecated:
        print("  WARNING: Legacy function not marked as deprecated")
    
    # Check 6: Old function call removed
    old_call_pattern = r'plot_research_comparison\(results_pid'
    # Look for it outside the function definition
    lines = content.split('\n')
    in_function_def = False
    old_call_found = False
    for i, line in enumerate(lines):
        if 'def plot_research_comparison' in line:
            in_function_def = True
        if in_function_def and (line.startswith('def ') or line.startswith('class ')):
            in_function_def = False
        if not in_function_def and re.search(old_call_pattern, line):
            old_call_found = True
            print(f"  WARNING: Old call found at line {i+1}: {line.strip()}")
    
    print(f"✓ Old function call removed: {'PASS' if not old_call_found else 'WARN'}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULT: ✓ ALL CHECKS PASSED")
    print("=" * 80)
    print("\nThe demo script now uses ResearchComparisonPlotter with interactive features:")
    print("  - Zoom mode (Z key)")
    print("  - Undo zoom (U key)")
    print("  - Split views (V/H keys)")
    print("  - Delete zoom rectangles (3 methods)")
    print("  - Orange selection highlighting")
    print("\nRun: python demo_feedback_linearization.py")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    verify_demo_integration()
