"""
Regression tests for cited-only claim extraction.
"""
from services.claim_extractor import extract_claims

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")

# ── Test 1: Cited sentence ──
print("\nTest 1 — cited sentence")
claims = extract_claims("LLM-as-a-Judge approaches rely on a separate model to evaluate factual consistency [5].")
check("claim extracted", len(claims) == 1)
check("citation [5] extracted", claims[0]["citation_text"] == "[5]" if claims else False)

# ── Test 2: Uncited sentence ──
print("\nTest 2 — uncited sentence")
claims = extract_claims("Hallucination detection research follows three major directions.")
check("claim NOT extracted", len(claims) == 0)

# ── Test 3: Heading ──
print("\nTest 3 — heading")
claims = extract_claims("2 Literature Review")
check("claim NOT extracted", len(claims) == 0)

# ── Test 4: Bibliography entry (should not become a claim) ──
print("\nTest 4 — bibliography entry")
claims = extract_claims("[5] P. Manakul, A. Liusie, and M.J.F. Gales, SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models, 2023.")
check("not treated as a claim", len(claims) == 0)

# ── Test 5: Multiple citations ──
print("\nTest 5 — multiple citations")
claims = extract_claims("Several methods have been proposed [5, 7, 9].")
check("one claim extracted", len(claims) == 1)
check("citation text correct", claims[0]["citation_text"] == "[5, 7, 9]" if claims else False)

# ── Test 6: Mixed input — only cited sentences survive ──
print("\nTest 6 — mixed Literature Review text")
mixed = (
    "Hallucination detection research follows three major directions. "
    "LLM-as-a-Judge approaches rely on a separate model to evaluate factual consistency [5]. "
    "Explainable AI techniques analyze model rationales [6]. "
    "Graph-based techniques convert source material into structured representations [7]. "
    "Several studies confirm this trend. "
    "These methods show promising results (Smith et al., 2024)."
)
claims = extract_claims(mixed)
check(f"only cited claims extracted (expected 4, got {len(claims)})", len(claims) == 4)
uncited = [c for c in claims if c["citation_text"] is None]
check("no uncited claims", len(uncited) == 0)

# ── Test 7: Author-year citations ──
print("\nTest 7 — author-year citation")
claims = extract_claims("Costa-jussà et al. (2022) built multilingual models.")
check("claim extracted", len(claims) == 1)

# ── Test 8: Claim IDs are sequential from 1 ──
print("\nTest 8 — sequential claim IDs")
claims = extract_claims(mixed)
ids = [c["claim_id"] for c in claims]
check(f"IDs are C1..C{len(claims)}", ids == [f"C{i}" for i in range(1, len(claims)+1)])

# ── Summary ──
print(f"\n{'='*50}")
print(f"PASSED: {PASS}   FAILED: {FAIL}")
if FAIL == 0:
    print("All tests passed.")
else:
    print("Some tests FAILED.")
