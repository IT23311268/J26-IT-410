import sys
import glob
import os

sys.path.insert(0, "backend")
from services.reference_verifier import extract_references_from_pdf

pdfs = glob.glob("**/*.pdf", recursive=True)
print("PDFs found:", pdfs)

if pdfs:
    for pdf in pdfs:
        refs = extract_references_from_pdf(pdf)
        print(f"Extracted {len(refs)} references from {pdf}")
        if len(refs) > 0:
            print("  First ref:", refs[0][:100])
