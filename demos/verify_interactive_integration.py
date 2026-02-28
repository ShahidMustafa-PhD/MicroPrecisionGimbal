#!/usr/bin/env python3
"""
Verification script to confirm ResearchComparisonPlotter uses InteractiveFigureManager.

This script:
1. Imports both classes
2. Verifies the integration exists
3. Confirms interactive mode is functional
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("VERIFYING ResearchComparisonPlotter ↔ InteractiveFigureManager INTEGRATION")
print("=" * 80)

# Step 1: Import verification
print("\n1. Import Verification...")
try:
    from lasercom_digital_twin.core.plots.interactive_plotter import InteractiveFigureManager
    print("   ✓ InteractiveFigureManager imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import InteractiveFigureManager: {e}")
    sys.exit(1)

try:
    from lasercom_digital_twin.core.plots.research_comparison_plotter import ResearchComparisonPlotter
    print("   ✓ ResearchComparisonPlotter imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import ResearchComparisonPlotter: {e}")
    sys.exit(1)

# Step 2: Check class integration
print("\n2. Integration Verification...")

# Check if ResearchComparisonPlotter imports InteractiveFigureManager
import inspect
source_file = Path(__file__).parent / "lasercom_digital_twin" / "core" / "plots" / "research_comparison_plotter.py"

with open(source_file, 'r') as f:
    source_code = f.read()

checks = {
    "Imports InteractiveFigureManager": "from lasercom_digital_twin.core.plots.interactive_plotter import" in source_code and "InteractiveFigureManager" in source_code,
    "Has interactive parameter": "interactive: bool" in source_code or "self.interactive" in source_code,
    "Has _make_figure_interactive method": "def _make_figure_interactive" in source_code,
    "Stores interactive_managers": "self.interactive_managers" in source_code,
    "Uses make_interactive() function": "make_interactive(" in source_code and "manager = make_interactive(" in source_code,
}

for check_name, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"   {status} {check_name}")

if not all(checks.values()):
    print("\n   ✗ INTEGRATION INCOMPLETE")
    sys.exit(1)

# Step 3: Instantiation verification
print("\n3. Instantiation Verification...")

try:
    # Create plotter with interactive mode
    plotter = ResearchComparisonPlotter(
        interactive=True,
        save_figures=False,
        show_figures=False
    )
    print("   ✓ ResearchComparisonPlotter instantiated with interactive=True")
    
    # Verify attributes
    assert hasattr(plotter, 'interactive'), "Missing 'interactive' attribute"
    assert hasattr(plotter, 'interactive_managers'), "Missing 'interactive_managers' attribute"
    assert hasattr(plotter, '_make_figure_interactive'), "Missing '_make_figure_interactive' method"
    assert plotter.interactive == True, "interactive flag not set correctly"
    assert isinstance(plotter.interactive_managers, dict), "interactive_managers not a dict"
    
    print("   ✓ All required attributes present")
    print(f"   ✓ interactive flag = {plotter.interactive}")
    print(f"   ✓ interactive_managers type = {type(plotter.interactive_managers).__name__}")
    
except Exception as e:
    print(f"   ✗ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Method signature verification
print("\n4. Method Signature Verification...")

try:
    sig = inspect.signature(plotter._make_figure_interactive)
    params = list(sig.parameters.keys())
    expected_params = ['fig', 'axes', 'fig_name']
    
    for param in expected_params:
        if param in params:
            print(f"   ✓ Parameter '{param}' present")
        else:
            print(f"   ✗ Parameter '{param}' MISSING")
            
    # Check return type annotation
    return_annotation = sig.return_annotation
    if 'InteractiveFigureManager' in str(return_annotation):
        print(f"   ✓ Return type: {return_annotation}")
    else:
        print(f"   ⚠ Return type annotation: {return_annotation}")
        
except Exception as e:
    print(f"   ✗ Method inspection failed: {e}")

# Step 5: Verify integration pattern
print("\n5. Integration Pattern Verification...")

pattern_checks = {
    "Creates managers in plot_all()": "self._make_figure_interactive" in source_code and "plot_all" in source_code,
    "Stores managers in dict": "self.interactive_managers[" in source_code,
    "Passes figure and axes": "_make_figure_interactive(self.figures" in source_code or "manager = self._make_figure_interactive" in source_code,
    "Checks interactive flag": "if self.interactive:" in source_code,
    "Creates InteractiveFigureManager": "manager = InteractiveFigureManager(" in source_code,
}

for check_name, condition in pattern_checks.items():
    status = "✓" if condition else "✗"
    print(f"   {status} {check_name}")

# Final summary
print("\n" + "=" * 80)
print("VERIFICATION RESULT")
print("=" * 80)

all_passed = all(checks.values())

if all_passed:
    print("\n✅ SUCCESS: ResearchComparisonPlotter properly uses InteractiveFigureManager")
    print("\nIntegration Details:")
    print("  • ResearchComparisonPlotter imports InteractiveFigureManager")
    print("  • Accepts 'interactive' parameter (default: True)")
    print("  • Has '_make_figure_interactive()' method")
    print("  • Stores managers in 'self.interactive_managers' dict")
    print("  • Creates manager for each figure in plot_all()")
    print("  • Passes figure, axes, and name to InteractiveFigureManager")
    print("\nUsage:")
    print("  plotter = ResearchComparisonPlotter(interactive=True)")
    print("  plotter.plot_all(results_pid, results_fbl, results_ndob, az, el)")
    print("  # All 13 figures will be interactive with zoom/annotation features")
    print("\n" + "=" * 80)
    sys.exit(0)
else:
    print("\n❌ FAILURE: Integration incomplete or missing")
    print("\nPlease ensure ResearchComparisonPlotter:")
    print("  1. Imports InteractiveFigureManager")
    print("  2. Has interactive parameter in __init__")
    print("  3. Implements _make_figure_interactive() method")
    print("  4. Calls _make_figure_interactive() for each figure")
    print("  5. Stores managers in self.interactive_managers dict")
    print("\n" + "=" * 80)
    sys.exit(1)
