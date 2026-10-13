import sys
sys.path.insert(0, "backend")

from services.reference_verifier import verify_reference

def test_grapheval():
    ref_text = 'H. Sansford, N. Richardson, H. P. Maretic, and J. N. Saada, "GraphEval: A Knowledge-Graph based LLM hallucination evaluation framework," 2024.'
    
    result = verify_reference(ref_text)
    
    assert result["matched_source"] == "arXiv", f"Expected source to be arXiv, got {result['matched_source']}"
    assert result["status"] == "REAL", f"Expected status to be REAL, got {result['status']}"
    # Remove version suffix if present for the check
    arxiv_id = result["matched_arxiv"].split('v')[0] if result["matched_arxiv"] else None
    assert arxiv_id == "2407.10793", f"Expected arxiv ID 2407.10793, got {result['matched_arxiv']}"
    
    print("GraphEval Regression Test Passed!")

if __name__ == "__main__":
    test_grapheval()
