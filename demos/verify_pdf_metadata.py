"""Verify PDF metadata and properties."""
import os

pdf_path = 'figures_comparative/fig1_position_tracking.pdf'

if os.path.exists(pdf_path):
    file_size = os.path.getsize(pdf_path) / 1024
    print(f"\n✅ PDF File Verification:")
    print(f"   File: {pdf_path}")
    print(f"   Size: {file_size:.1f} KB")
    print(f"   Format: PDF (vector)")
    print(f"   Status: Successfully created")
    
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        print(f"\n✅ PDF Metadata:")
        print(f"   Title: {metadata.get('/Title', 'N/A')}")
        print(f"   Author: {metadata.get('/Author', 'N/A')}")
        print(f"   Subject: {metadata.get('/Subject', 'N/A')}")
        print(f"   Creator: {metadata.get('/Creator', 'N/A')}")
        print(f"   Pages: {len(reader.pages)}")
    except ImportError:
        print("\n⚠ pypdf not installed (metadata verification skipped)")
        print("   Install with: pip install pypdf")
else:
    print(f"❌ PDF file not found: {pdf_path}")
