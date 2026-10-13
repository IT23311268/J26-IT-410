import sys, os, json
sys.path.insert(0, os.path.abspath("backend"))

from services.reference_verifier import verify_reference

with open(r"C:\Users\MMSSI\.gemini\antigravity\brain\bbc96363-f229-4c95-a88b-2df52d885822\scratch\paper_28_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

refs = [d["ref"] for d in data["details"]]
print(f"Total references loaded: {len(refs)}")

counts = {"REAL": 0, "UNCERTAIN": 0, "FAKE": 0}
internal_counts = {"VERIFIED_REAL": 0, "REAL_METADATA_MISMATCH": 0, "UNCERTAIN": 0, "LIKELY_FABRICATED": 0}

results = []
for i, ref in enumerate(refs, 1):
    ver = verify_reference(ref, ref_index=i)
    fe = ver["status"]
    intern = ver["verification_detail"]
    counts[fe] = counts.get(fe, 0) + 1
    internal_counts[intern] = internal_counts.get(intern, 0) + 1
    
    results.append({
        "index": i,
        "ref": ref,
        "frontend": fe,
        "internal": intern,
        "matched_source": ver.get("matched_source"),
        "matched_title": ver.get("matched_title"),
        "reason": ver.get("reason"),
        "confidence": ver.get("confidence")
    })
    sys.stdout.write(f"[{i:02d}] {fe:9s} ({intern:22s}) | {ver.get('reason','')[:65]}\n")
    sys.stdout.flush()

print("\n" + "="*50)
print(f"TOTAL REFERENCES = {len(refs)}")
print("FRONTEND COUNTS:")
print(f"REAL = {counts['REAL']}")
print(f"UNCERTAIN = {counts['UNCERTAIN']}")
print(f"FAKE = {counts['FAKE']}")
print("\nINTERNAL COUNTS:")
print(f"VERIFIED_REAL = {internal_counts['VERIFIED_REAL']}")
print(f"REAL_METADATA_MISMATCH = {internal_counts['REAL_METADATA_MISMATCH']}")
print(f"UNCERTAIN = {internal_counts['UNCERTAIN']}")
print(f"LIKELY_FABRICATED = {internal_counts['LIKELY_FABRICATED']}")
print("="*50)

with open(r"scratch\final_28_paper_results.json", "w", encoding="utf-8") as f:
    json.dump({"summary": counts, "internal": internal_counts, "details": results}, f, indent=2, ensure_ascii=False)
