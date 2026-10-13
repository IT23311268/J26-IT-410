import sys
sys.path.insert(0, "backend")

import json
from services.analysis_pipeline import run_analysis_pipeline

def main():
    with open("backend/data/literature_reviews.json", "r", encoding="utf-8") as f:
        reviews = json.load(f)
    if not reviews:
        print("No reviews")
        return
    lr_id = reviews[-1]["id"]
    print(f"Using LR {lr_id}")
    
    try:
        res = run_analysis_pipeline("test-123", lr_id, ["2023.eacl-main.199_70877ba7.pdf", "Graphing_the_Truth_Structured_Visualizations_for_A_6cddbe09.pdf"])
        print(f"Result keys: {list(res.keys())}")
        if 'status' in res:
            print(f"Status: {res['status']}")
            print(f"Message: {res.get('message')}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

main()
