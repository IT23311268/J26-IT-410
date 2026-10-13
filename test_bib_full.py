import sys
sys.path.insert(0, "backend")
from services.reference_verifier import extract_references_from_pdf, verify_reference

main_pdf = "backend/uploads/Graphing_the_Truth_Structured_Visualizations_for_A_6cddbe09.pdf"
print(f"Testing: {main_pdf}")

refs = extract_references_from_pdf(main_pdf)
print(f"Extracted refs: {len(refs)}")

if len(refs) > 0:
    print("First ref:", refs[0][:120])
    print("\nVerifying first ref (quick test)...")
    v = verify_reference(refs[0], 1)
    print("Result keys:", list(v.keys()))
    for key in ['status', 'matched_source', 'matched_title', 'matched_authors', 'matched_doi', 'matched_arxiv', 'score']:
        print(f"  {key}: {v.get(key)}")
else:
    print("No refs extracted - checking raw text...")
    import pymupdf as fitz
    doc = fitz.open(main_pdf)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    lines = full_text.split("\n")
    print(f"Total lines: {len(lines)}")
    # Search for references section
    for i, line in enumerate(lines):
        if "reference" in line.lower() and len(line.strip()) < 30:
            print(f"  Line {i}: {repr(line)}")
