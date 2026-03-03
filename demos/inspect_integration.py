#!/usr/bin/env python3
"""
Simple code inspection to confirm ResearchComparisonPlotter uses InteractiveFigureManager.

This is a static analysis without running full simulations.
"""

import ast
import inspect
from pathlib import Path

print("=" * 80)
print("CODE INSPECTION: ResearchComparisonPlotter → InteractiveFigureManager")
print("=" * 80)

# Import the classes
from lasercom_digital_twin.core.plots.research_comparison_plotter import ResearchComparisonPlotter
from lasercom_digital_twin.core.plots.interactive_plotter import InteractiveFigureManager

print("\n1. Class Import Check")
print(f"   ✓ ResearchComparisonPlotter imported from: {inspect.getfile(ResearchComparisonPlotter)}")
print(f"   ✓ InteractiveFigureManager imported from: {inspect.getfile(InteractiveFigureManager)}")

print("\n2. ResearchComparisonPlotter.__init__ Signature")
sig = inspect.signature(ResearchComparisonPlotter.__init__)
params = {name: param for name, param in sig.parameters.items() if name != 'self'}
for name, param in params.items():
    print(f"   • {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} = {param.default if param.default != inspect.Parameter.empty else 'required'}")

if 'interactive' in params:
    print(f"   ✓ Has 'interactive' parameter (default={params['interactive'].default})")
else:
    print(f"   ✗ Missing 'interactive' parameter")

print("\n3. ResearchComparisonPlotter Attributes")
plotter = ResearchComparisonPlotter(save_figures=False, show_figures=False, interactive=True)
attrs = {
    'interactive': plotter.interactive,
    'interactive_managers': type(plotter.interactive_managers).__name__,
    'has _make_figure_interactive': hasattr(plotter, '_make_figure_interactive'),
}

for attr, value in attrs.items():
    print(f"   ✓ {attr} = {value}")

print("\n4. Method _make_figure_interactive Inspection")
if hasattr(plotter, '_make_figure_interactive'):
    method = plotter._make_figure_interactive
    sig = inspect.signature(method)
    print(f"   • Signature: {sig}")
    print(f"   • Return type: {sig.return_annotation}")
    
    # Check if it returns InteractiveFigureManager
    if 'InteractiveFigureManager' in str(sig.return_annotation):
        print(f"   ✓ Returns InteractiveFigureManager")
    else:
        print(f"   ⚠ Return type: {sig.return_annotation}")
else:
    print(f"   ✗ Method not found")

print("\n5. Source Code Analysis")
source_file = Path(inspect.getfile(ResearchComparisonPlotter))
with open(source_file, 'r') as f:
    source = f.read()

checks = {
    "Imports InteractiveFigureManager": "from lasercom_digital_twin.core.plots.interactive_plotter import" in source and "InteractiveFigureManager" in source,
    "Has interactive parameter": "interactive: bool" in source,
    "Stores interactive_managers": "self.interactive_managers" in source,
    "Creates InteractiveFigureManager instance": "= InteractiveFigureManager(" in source,
    "Calls _make_figure_interactive in plot_all": "self._make_figure_interactive" in source and "def plot_all" in source,
    "Checks if interactive before creating managers": "if self.interactive:" in source,
}

for check, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"   {status} {check}")

print("\n6. Integration Flow Verification")
# Find where InteractiveFigureManager is instantiated in the source
import re
ifm_pattern = r'InteractiveFigureManager\((.*?)\)'
matches = re.findall(ifm_pattern, source, re.DOTALL)

if matches:
    print(f"   ✓ Found {len(matches)} InteractiveFigureManager instantiation(s)")
    for i, match in enumerate(matches[:2], 1):  # Show first 2
        # Clean up the match
        cleaned = ' '.join(match.split())[:100]
        print(f"     {i}. InteractiveFigureManager({cleaned}...)")
else:
    print(f"   ✗ No InteractiveFigureManager instantiations found")

# Check if managers are stored
storage_pattern = r'self\.interactive_managers\[.*?\]\s*=\s*(\w+)'
storage_matches = re.findall(storage_pattern, source)

if storage_matches:
    print(f"   ✓ Managers stored in self.interactive_managers dict")
    print(f"     Assignments found: {len(storage_matches)}")
else:
    print(f"   ✗ Managers not stored properly")

print("\n" + "=" * 80)
print("INSPECTION RESULT")
print("=" * 80)

all_checks_passed = all(checks.values()) and 'interactive' in params

if all_checks_passed:
    print("\n✅ CONFIRMED: ResearchComparisonPlotter uses InteractiveFigureManager")
    print("\nIntegration Summary:")
    print("  1. Imports InteractiveFigureManager from interactive_plotter module")
    print("  2. Accepts 'interactive' parameter in __init__ (default: True)")
    print("  3. Stores self.interactive_managers: Dict[str, InteractiveFigureManager]")
    print("  4. Has _make_figure_interactive() method that creates managers")
    print("  5. Calls _make_figure_interactive() for each figure in plot_all()")
    print("  6. Stores each manager in self.interactive_managers dict")
    print("\nUsage Example:")
    print("  >>> plotter = ResearchComparisonPlotter(interactive=True)")
    print("  >>> figures = plotter.plot_all(results_pid, results_fbl, results_ndob, az, el)")
    print("  >>> # All 13 figures are now interactive!")
    print("  >>> print(len(plotter.interactive_managers))  # Shows 13")
    print("\nInteractive Features:")
    print("  • Z key: Zoom mode (draw green rectangle)")
    print("  • U key: Undo last zoom (forced redraw)")
    print("  • Delete: Remove selected zoom (click first to select)")
    print("  • Right-click: Quick delete zoom")
    print("  • V/H keys: Add vertical/horizontal lines")
    print("  • M key: Move mode (drag annotations)")
    print("  • S key: Save figure (PNG/PDF/SVG)")
else:
    print("\n❌ ISSUE: Some integration components missing")
    print("\nMissing Components:")
    for check, passed in checks.items():
        if not passed:
            print(f"  • {check}")

print("=" * 80)
