"""
Compile Graphviz .gv files to PDF and PNG formats.

This script compiles the lasercom gimbal system block diagram to publication-ready formats.
Requires Graphviz system binaries installed: https://graphviz.org/download/

Usage:
    python compile_diagrams.py
"""

import subprocess
import os
from pathlib import Path

def compile_diagram(gv_file: str, output_dir: str = "figures_diagrams"):
    """
    Compile a .gv file to both PDF and PNG formats.
    
    Args:
        gv_file: Path to .gv source file (without extension)
        output_dir: Directory containing .gv file and output
    """
    gv_path = Path(output_dir) / f"{gv_file}.gv"
    pdf_path = Path(output_dir) / f"{gv_file}.pdf"
    png_path = Path(output_dir) / f"{gv_file}.png"
    
    if not gv_path.exists():
        print(f"❌ Error: {gv_path} not found")
        return False
    
    print(f"\n📄 Compiling: {gv_path}")
    
    # Try to compile to PDF
    try:
        result_pdf = subprocess.run(
            ["dot", "-Tpdf", str(gv_path), "-o", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Generated PDF: {pdf_path}")
    except FileNotFoundError:
        print("\n❌ ERROR: Graphviz 'dot' command not found!")
        print("\n📦 Installation Required:")
        print("   Windows: https://graphviz.org/download/ (add to PATH)")
        print("   Linux:   sudo apt install graphviz")
        print("   macOS:   brew install graphviz")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ PDF compilation failed: {e.stderr}")
        return False
    
    # Try to compile to PNG (300 DPI)
    try:
        result_png = subprocess.run(
            ["dot", "-Tpng", "-Gdpi=300", str(gv_path), "-o", str(png_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Generated PNG: {png_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ PNG compilation failed: {e.stderr}")
        return False
    
    print(f"\n🎉 SUCCESS: Both PDF and PNG generated!")
    return True


def main():
    """Compile all diagrams."""
    print("=" * 60)
    print("Block Diagram Compiler")
    print("=" * 60)
    
    diagrams = [
        #"lasercom_gimbal_system",
        "robot_control_diagram",
        # Add more diagram names here as needed
    ]
    
    success_count = 0
    for diagram in diagrams:
        if compile_diagram(diagram):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Compiled {success_count}/{len(diagrams)} diagram(s)")
    print("=" * 60)
    
    if success_count == len(diagrams):
        print("\n✅ All diagrams compiled successfully!")
        print("\nFiles are in: figures_diagrams/")
    else:
        print("\n⚠ Some diagrams failed. Check Graphviz installation.")


if __name__ == "__main__":
    main()
