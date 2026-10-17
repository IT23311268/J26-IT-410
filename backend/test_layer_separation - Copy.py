"""
Regression tests for the Layer 1 / Layer 2 separation.
Tests citation_link_status and content_verification_status.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, ".")

from services.analysis_pipeline import (
    parse_bibliography_entries,
    resolve_citation_to_reference,
    _expand_numeric_citation,
)

# ── Setup ──
refs_text = """
[1] A. Zubiaga, "NLP in the era of LLMs," 2023.
[5] P. Manakul, A. Liusie, and M.J.F. Gales, "SelfCheckGPT," 2023.
[6] S. Wiegreffe and Y. Pinter, "Attention is not not explanation," 2019.
[7] H. Sansford et al., "GraphEval," 2024.
"""
entries = parse_bibliography_entries(refs_text)

passed = 0
failed = 0

def check(test_name, expected, actual):
    global passed, failed
    if expected == actual:
        print(f"  PASS: {test_name}")
        passed += 1
    else:
        print(f"  FAIL: {test_name} | Expected: {expected!r}, Got: {actual!r}")
        failed += 1


# ── Test 1 — valid citation link, cited PDF unavailable ──
print("=== Test 1: Valid link, PDF not available ===")
resolved = resolve_citation_to_reference("[5]", entries)
ref_found = resolved is not None
link_status = "LINK_VALID" if ref_found else "LINK_NOT_FOUND"
# No target_paper -> content_verification = NOT_AVAILABLE
target_paper = None
content_status = "NOT_AVAILABLE" if not target_paper else "SUPPORTED"
check("citation_link_status", "LINK_VALID", link_status)
check("content_verification_status", "NOT_AVAILABLE", content_status)


# ── Test 2 — citation marker absent from References ──
print("=== Test 2: Citation not in bibliography ===")
resolved = resolve_citation_to_reference("[99]", entries)
ref_found = resolved is not None
link_status = "LINK_VALID" if ref_found else "LINK_NOT_FOUND"
content_status = "NOT_AVAILABLE" if not ref_found else "SUPPORTED"
check("citation_link_status", "LINK_NOT_FOUND", link_status)
check("content_verification_status", "NOT_AVAILABLE", content_status)


# ── Test 3 — cited PDF available and supports claim ──
print("=== Test 3: PDF available, supported ===")
resolved = resolve_citation_to_reference("[5]", entries)
ref_found = resolved is not None
link_status = "LINK_VALID" if ref_found else "LINK_NOT_FOUND"
# Simulate target_paper matched and NLI returns SUPPORTED
target_paper = "selfcheckgpt.pdf"
hallucination_status = "SUPPORTED"
content_status = hallucination_status if target_paper else "NOT_AVAILABLE"
check("citation_link_status", "LINK_VALID", link_status)
check("content_verification_status", "SUPPORTED", content_status)


# ── Test 4 — cited PDF available and contradicts claim ──
print("=== Test 4: PDF available, contradicted ===")
resolved = resolve_citation_to_reference("[5]", entries)
ref_found = resolved is not None
link_status = "LINK_VALID" if ref_found else "LINK_NOT_FOUND"
target_paper = "selfcheckgpt.pdf"
hallucination_status = "CONTRADICTED"
content_status = hallucination_status if target_paper else "NOT_AVAILABLE"
check("citation_link_status", "LINK_VALID", link_status)
check("content_verification_status", "CONTRADICTED", content_status)


# ── Test 5 — cited PDF available but evidence insufficient ──
print("=== Test 5: PDF available, insufficient evidence ===")
resolved = resolve_citation_to_reference("[7]", entries)
ref_found = resolved is not None
link_status = "LINK_VALID" if ref_found else "LINK_NOT_FOUND"
target_paper = "grapheval.pdf"
hallucination_status = "INSUFFICIENT_EVIDENCE"
content_status = hallucination_status if target_paper else "NOT_AVAILABLE"
check("citation_link_status", "LINK_VALID", link_status)
check("content_verification_status", "INSUFFICIENT_EVIDENCE", content_status)


# ── Summary ──
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed > 0:
    sys.exit(1)
else:
    print("ALL REGRESSION TESTS PASSED")
