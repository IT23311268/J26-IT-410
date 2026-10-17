"""
Diagnose bibliography parsing from the actual uploaded main paper.
Run: .\venv\Scripts\python.exe test_bib_diag.py uploads\<main_paper.pdf>
"""
import sys
import re
import os

# Force UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

if len(sys.argv) < 2:
    print("Usage: python test_bib_diag.py <path_to_main_paper.pdf>")
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

# ── Try to find References section by scanning backwards ──
lines = full_text.split("\n")
ref_start_idx = -1
for i in range(len(lines) - 1, max(-1, len(lines) - 2000), -1):
    line = lines[i].strip().lower()
    if line in ["references", "bibliography", "references.", "bibliography."]:
        ref_start_idx = i
        break
if ref_start_idx == -1:
    for i in range(len(lines) - 1, max(-1, len(lines) - 2000), -1):
        line = lines[i].strip().lower()
        if re.match(r"^\d*\.?\s*(references|bibliography)$", line):
            ref_start_idx = i
            break

print(f"References section starts at line index: {ref_start_idx}")
if ref_start_idx == -1:
    print("COULD NOT FIND REFERENCES SECTION!")
    sys.exit(1)

bib_raw = "\n".join(lines[ref_start_idx + 1:])

# Print the raw bib text with repr to see whitespace/newlines
print("\n=== RAW BIB TEXT (first 3000 chars, repr) ===")
print(repr(bib_raw[:3000]))

# ── Count markers ──
bracketed = re.findall(r"\[(\d+)\]", bib_raw)
print(f"\n=== [N] markers found in bib raw: {bracketed} ===")

numbered_dot = re.findall(r"^\s*\d+\.", bib_raw, re.MULTILINE)
print(f"=== 'N.' markers found: {len(numbered_dot)} ===")

# ── Import and run the actual parsers ──
sys.path.insert(0, ".")
from services.analysis_pipeline import parse_bibliography_entries, extract_bibliography_section

bib_section = extract_bibliography_section(full_text)
print(f"\n=== extract_bibliography_section output length: {len(bib_section)} chars ===")
print("First 1500 chars (repr):")
print(repr(bib_section[:1500]))

entries = parse_bibliography_entries(bib_section)
print(f"\n=== Parsed {len(entries)} bibliography entries ===")
ref_nums = [e['ref_num'] for e in entries]
print(f"Reference numbers: {ref_nums}")
print()
for e in entries:
    print(f"  [{e['ref_num']}] {e['ref_text'][:120]}")
