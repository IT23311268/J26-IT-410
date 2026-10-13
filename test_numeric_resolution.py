import sys
from services.analysis_pipeline import (
    parse_bibliography_entries,
    resolve_citation_to_reference,
    _expand_numeric_citation
)

refs_text = """
[1] A. Zubiaga, "Natural language processing", 2023.
[5] P. Manakul, A. Liusie, and M.J.F. Gales, "SelfCheckGPT", 2023.
[6] S. Wiegreffe and Y. Pinter, "Attention is not not explanation", 2019.
[7] H. Sansford et al., "GraphEval", 2024.
[8] Z. J. Wang et al., "WizMap", 2023.
[9] R. Qiu et al., "VADIS", 2025.
[10] P. Hase et al., "Methods for measuring", 2023.
"""

entries = parse_bibliography_entries(refs_text)

def check_test(name, expected, actual):
    if expected == actual:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} | Expected: {expected}, Got: {actual}")

print("=== Test 1: [5] ===")
res = resolve_citation_to_reference("[5]", entries)
check_test("entry found", True, res is not None)
check_test("resolved_reference_number", "5", res.get("ref_num") if res else None)

print("=== Test 2: [10] ===")
res = resolve_citation_to_reference("[10]", entries)
check_test("entry found", True, res is not None)
check_test("resolved_reference_number", "10", res.get("ref_num") if res else None)

print("=== Test 3: [5, 7] ===")
res = resolve_citation_to_reference("[5, 7]", entries)
check_test("entry found", True, res is not None)
all_res = res.get("all_resolved", [])
check_test("both resolved", ["5", "7"], [r["ref_num"] for r in all_res])

print("=== Test 4: [5-7] ===")
res = resolve_citation_to_reference("[5-7]", entries)
check_test("entry found", True, res is not None)
all_res = res.get("all_resolved", [])
check_test("resolve 5, 6, 7", ["5", "6", "7"], [r["ref_num"] for r in all_res])

print("=== Test 5: [99] ===")
res = resolve_citation_to_reference("[99]", entries)
check_test("entry found", False, res is not None)
