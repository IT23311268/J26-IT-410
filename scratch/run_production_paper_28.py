import sys, os, json
sys.path.insert(0, os.path.abspath("backend"))

from services.reference_verifier import verify_document_references

pdf_path = r"backend/uploads/framework-for-hallucination-detection-in-large-language-models-IJERTV15IS043963_f2da5c9d.pdf"
print(f"Running verification on {pdf_path}...")
res = verify_document_references(pdf_path)

print("\n" + "="*50)
print(f"TOTAL EXTRACTED = {res['total_extracted']}")
print(f"TOTAL ANALYZED = {res['total_analyzed']}")
print("FRONTEND SUMMARY:", res['summary'])

internal_counts = {"VERIFIED_REAL": 0, "REAL_METADATA_MISMATCH": 0, "UNCERTAIN": 0, "LIKELY_FABRICATED": 0}
for item in res['details']:
    det = item['verification'].get('verification_detail')
    internal_counts[det] = internal_counts.get(det, 0) + 1

print("INTERNAL BREAKDOWN:", internal_counts)
print("="*50)

# Check specifically R9, R11, R23:
for idx in [9, 11, 23]:
    if idx <= len(res['details']):
        item = res['details'][idx - 1]
        v = item['verification']
        print(f"\nR{idx}:")
        print(f"  Reference text: {item['reference_text'][:120]}")
        print(f"  Frontend status: {v['status']}")
        print(f"  Internal detail: {v.get('verification_detail')}")
        print(f"  Matched source:  {v.get('matched_source')}")
        print(f"  Matched title:   {v.get('matched_title')}")
        print(f"  Reason:          {v.get('reason')}")

with open(r"scratch\production_paper_28_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
