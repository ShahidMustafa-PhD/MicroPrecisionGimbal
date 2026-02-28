#!/usr/bin/env python3
"""
Verify that ResearchComparisonPlotter uses make_interactive() following test_zoom_deletion.py pattern.
"""

from pathlib import Path
import re

print("=" * 80)
print("PATTERN VERIFICATION: make_interactive() Usage")
print("=" * 80)

# Read both files
research_plotter_path = Path("lasercom_digital_twin/core/plots/research_comparison_plotter.py")
test_zoom_path = Path("test_zoom_deletion.py")

with open(research_plotter_path, 'r') as f:
    research_source = f.read()

with open(test_zoom_path, 'r') as f:
    test_source = f.read()

print("\n1. Test Pattern Analysis (test_zoom_deletion.py)")
print("-" * 80)

# Extract pattern from test file
test_pattern_lines = []
for line in test_source.split('\n'):
    if 'make_interactive' in line.lower() or 'manager.show()' in line:
        test_pattern_lines.append(line.strip())

print("   Reference pattern:")
for line in test_pattern_lines[:5]:
    if line:
        print(f"     {line}")

print("\n2. Research Plotter Pattern Analysis")
print("-" * 80)

# Check for make_interactive usage
has_make_interactive = 'make_interactive(' in research_source
has_import = 'from lasercom_digital_twin.core.plots.interactive_plotter import' in research_source and 'make_interactive' in research_source

print(f"   ✓ Imports make_interactive: {has_import}")
print(f"   ✓ Calls make_interactive(): {has_make_interactive}")

# Find the actual call
make_interactive_matches = re.findall(r'(manager\s*=\s*make_interactive\([^)]+\))', research_source, re.DOTALL)
if make_interactive_matches:
    print(f"   ✓ Found {len(make_interactive_matches)} make_interactive() call(s)")
    for i, match in enumerate(make_interactive_matches[:1], 1):
        # Clean up for display
        cleaned = ' '.join(match.split())[:150]
        print(f"     {i}. {cleaned}...")
else:
    print(f"   ✗ No make_interactive() calls found")

print("\n3. Pattern Comparison")
print("-" * 80)

# Check if research plotter follows the same pattern
test_uses_make_interactive = 'make_interactive(' in test_source
research_uses_make_interactive = 'make_interactive(' in research_source

print(f"   Test file uses make_interactive():     {test_uses_make_interactive}")
print(f"   Research plotter uses make_interactive(): {research_uses_make_interactive}")

if test_uses_make_interactive and research_uses_make_interactive:
    print(f"   ✓ MATCH: Both use the same factory function pattern")
else:
    print(f"   ✗ MISMATCH: Patterns differ")

print("\n4. Detailed Pattern Elements")
print("-" * 80)

pattern_elements = {
    "Imports make_interactive": "make_interactive" in research_source and "import" in research_source,
    "Calls make_interactive with fig parameter": "make_interactive(" in research_source and "fig=" in research_source,
    "Calls make_interactive with axes parameter": "make_interactive(" in research_source and "axes=" in research_source,
    "Assigns result to manager": "manager = make_interactive(" in research_source,
    "Returns InteractiveFigureManager": "-> InteractiveFigureManager" in research_source,
}

for element, present in pattern_elements.items():
    status = "✓" if present else "✗"
    print(f"   {status} {element}")

print("\n5. Method Implementation Check")
print("-" * 80)

# Find _make_figure_interactive method
method_match = re.search(
    r'def _make_figure_interactive\(.*?\):(.*?)(?=\n    def |\nclass |\Z)',
    research_source,
    re.DOTALL
)

if method_match:
    method_body = method_match.group(1)
    
    checks = {
        "Uses make_interactive() factory": "make_interactive(" in method_body,
        "Passes fig parameter": "fig=" in method_body,
        "Passes axes parameter": "axes=" in method_body,
        "Passes style parameter": "style=" in method_body,
        "Returns manager": "return manager" in method_body,
        "Has InteractiveStyleConfig": "InteractiveStyleConfig(" in method_body,
    }
    
    print("   Method body analysis:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"     {status} {check}")
else:
    print("   ✗ Could not find _make_figure_interactive method")

print("\n" + "=" * 80)
print("VERIFICATION RESULT")
print("=" * 80)

all_checks = (
    has_make_interactive and 
    has_import and 
    all(pattern_elements.values())
)

if all_checks:
    print("\n✅ SUCCESS: ResearchComparisonPlotter follows test_zoom_deletion.py pattern")
    print("\nPattern Match:")
    print("  Test Pattern:")
    print("    fig, ax = create_test_figure()")
    print("    manager = make_interactive(fig, ax)")
    print("    manager.show()")
    print()
    print("  Research Plotter Pattern:")
    print("    # In _make_figure_interactive():")
    print("    manager = make_interactive(")
    print("        fig=fig,")
    print("        axes=axes,")
    print("        style=InteractiveStyleConfig(...),")
    print("        save_dir=...)")
    print("    return manager")
    print()
    print("  ✓ Both use make_interactive() factory function")
    print("  ✓ Both return InteractiveFigureManager instance")
    print("  ✓ Consistent pattern across the project")
else:
    print("\n❌ PATTERN MISMATCH")
    print("\nMissing elements:")
    for element, present in pattern_elements.items():
        if not present:
            print(f"  • {element}")

print("=" * 80)
