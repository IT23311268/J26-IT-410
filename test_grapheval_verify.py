import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, "backend")

from services.reference_verifier import verify_reference

ref = 'H. Sansford, N. Richardson, H. P. Maretic, and J. N. Saada, "GraphEval: A Knowledge-Graph based LLM hallucination evaluation framework," 2024.'
print("=== Verifying GraphEval ===")
result = verify_reference(ref, ref_index=99)
print("\n=== RESULT ===")
for k, v in result.items():
    if k == "match":
        print(f"  {k}: (omitted)")
    else:
        print(f"  {k}: {v}")
