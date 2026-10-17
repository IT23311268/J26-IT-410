import sys
import json
from services.reference_verifier import verify_reference

tests = [
    {
        "name": "Test 1 — OpenAI system card",
        "ref": "OpenAI: GPT-4V(ision) System Card (2023). https://cdn.openai.com/papers/GPTV_System_Card.pdf",
        "expected": "VERIFIED_REAL"
    },
    {
        "name": "Test 2 — textbook",
        "ref": "Gonzalez, R.C., Woods, R.E.: Digital Image Processing (3rd Edition). Prentice-Hall, Inc., USA (2006).",
        "expected": "VERIFIED_REAL"
    },
    {
        "name": "Test 3 — OpenReview/ICLR",
        "ref": "He, P., Gao, J., Chen, W.: DeBERTaV3: Improving deBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In: The Eleventh International Conference on Learning Representations (2023).",
        "expected": "VERIFIED_REAL"
    },
    {
        "name": "Test 4 — real arXiv paper",
        "ref": "Hamadi, R.: Large language models meet computer vision: A brief survey (2023) arXiv:2311.16673",
        "expected": "VERIFIED_REAL"
    },
    {
        "name": "Test 5 — real paper with wrong identifier",
        "ref": "Smith, J. A real title about deep learning that is true (2023) https://doi.org/10.1007/s00521-024-10827-6",
        "expected": "REAL_METADATA_MISMATCH"
    },
    {
        "name": "Test 6 — fabricated reference",
        "ref": "Dumbledore, A. Magic in Deep Learning. Journal of Hogwarts Science, 2024.",
        "expected": "LIKELY_FABRICATED"
    },
    {
        "name": "Test 7 (R9) — ACL Anthology title variant (NLI corpus)",
        "ref": "A. Williams, N. Nangia, and S. Bowman, \u201cA Broad-Coverage Challenge Corpus for Natural Language Inference,\u201d NAACL, 2018.",
        "expected": "VERIFIED_REAL"
    },
    {
        "name": "Test 8 (R11) — HaluEval with wrong authors",
        "ref": "P. Minervini et al., \u201cHaluEval: A Large-Scale Hallucination Evaluation Benchmark,\u201d 2023.",
        "expected": "REAL_METADATA_MISMATCH"
    },
    {
        "name": "Test 9 (R23) — GPT-2 OpenAI report (no arXiv/Crossref)",
        "ref": "A. Radford et al., \u201cLanguage Models are Unsupervised Multitask Learners,\u201d OpenAI, 2019.",
        "expected": "VERIFIED_REAL"
    }
]

failed = False

for t in tests:
    print(f"\n{'='*50}\nTesting {t['name']}")
    res = verify_reference(t["ref"], ref_index=1)
    
    status = res["verification_detail"]
    expected = t["expected"]
    
    if isinstance(expected, list):
        success = status in expected
        exp_str = " or ".join(expected)
    else:
        success = status == expected
        exp_str = expected
        
    print(f"Result: {status} | Expected: {exp_str}")
    if not success:
        print(">>> FAILED <<<")
        failed = True
    else:
        print(">>> PASSED <<<")

if failed:
    sys.exit(1)
else:
    print("\nAll tests PASSED!")
    sys.exit(0)
