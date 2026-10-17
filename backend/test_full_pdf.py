import os
import glob
from services.reference_verifier import verify_document_references

pdfs = glob.glob("uploads/*.pdf")
if pdfs:
    pdf_path = pdfs[0]
    print(f"Testing against {pdf_path}...")
    res = verify_document_references(pdf_path)
    print("\nSummary:")
    print(res["summary"])
else:
    print("PDF not found.")
