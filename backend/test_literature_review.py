import sys
from services.analysis_pipeline import run_analysis_pipeline

def test_pipeline():
    # Mocking the uploaded text, representing the main paper.
    review_text = """
    1. Introduction
    2. Literature Review
    The creation of monolingual and mined bilingual data enables the development of models for hundreds of languages (Costa-jussà et al., 2022).
    The proposed approach enables models for only one language (Costa-jussà et al., 2022).
    Some claim (Wiegreffe & Pinter, 2019).
    This claim has no citation.
    3. Methodology
    4. References
    [1] Costa-jussà, M.R. et al.: No Language Left Behind. (2022).
    [2] Wiegreffe, S., Pinter, Y. (2019).
    """
    
    # We will use the existing uploads dir to see what it does.
    # Costa-jussa paper is NOT uploaded in uploads/ so it should return CITATION_NOT_FOUND.
    
    res = run_analysis_pipeline(review_text, "uploads")
    
    print("\n--- Summary ---")
    for k, v in res["summary"].items():
        print(f"{k}: {v}")
        
    print("\n--- Claims ---")
    for c in res["claims"]:
        print(f"\nClaim: {c['claim_text']}")
        print(f"Citation: {c['citation_text']}")
        print(f"Resolved Ref: {c.get('resolved_reference')}")
        print(f"Hallucination: {c['hallucination_status']} ({c['hallucination_reason']})")
        print(f"Citation Status: {c['citation_status']} ({c['citation_reason']})")
        print(f"Evidence chunks: {len(c['evidence'])}")

if __name__ == "__main__":
    test_pipeline()
