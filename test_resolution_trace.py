"""
Debug exactly what citation_text the claim extractor produces for each claim,
and trace how resolve_citation_to_reference handles them.
"""
import sys
import re
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

if len(sys.argv) < 2:
    print("Usage: python test_resolution_trace.py <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]

try:
    import pymupdf as fitz
except ImportError:
    import fitz

doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text() + "\n"
doc.close()

sys.path.insert(0, ".")
from services.analysis_pipeline import (
    extract_literature_review_section, extract_bibliography_section,
    parse_bibliography_entries, resolve_citation_to_reference
)
from services.claim_extractor import extract_claims

lr_text = extract_literature_review_section(full_text)
bib_text = extract_bibliography_section(full_text)
entries = parse_bibliography_entries(bib_text)

# Build the number->entry dict for quick reference
refs_by_num = {e["ref_num"]: e for e in entries}
print(f"Parsed reference numbers: {sorted(refs_by_num.keys(), key=int)}\n")

claims = extract_claims(lr_text)
print(f"Extracted {len(claims)} cited claims from Literature Review\n")

print("=" * 70)
for claim in claims:
    cid = claim["claim_id"]
    ctext = claim["claim_text"][:80]
    citation = claim["citation_text"]
    
    print(f"\n[REFERENCE RESOLUTION] claim_id={cid}")
    print(f"  citation_marker : {citation!r}")
    
    # Show what numbers the citation contains
    nums = re.findall(r"\d+", citation or "")
    print(f"  numbers found   : {nums}")
    
    resolved = resolve_citation_to_reference(citation, entries)
    if resolved:
        print(f"  reference_found : YES  [{resolved['ref_num']}] {resolved['ref_text'][:80]}")
    else:
        print(f"  reference_found : NO")
        # Explain why
        first_num = re.search(r"\[\s*(\d+)", citation or "")
        if first_num:
            n = first_num.group(1)
            if n in refs_by_num:
                print(f"  PROBLEM: number {n!r} IS in refs map but lookup FAILED!")
                print(f"  refs list ref_nums: {[e['ref_num'] for e in entries[:5]]}")
            else:
                print(f"  Reason: number {n!r} NOT in parsed refs map")
