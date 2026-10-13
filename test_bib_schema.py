import sys
import json
sys.path.insert(0, "backend")

from services.reference_verifier import verify_document_references

pdf_path = "backend/uploads/Graphing_the_Truth_Structured_Visualizations_for_A_6cddbe09.pdf"
print(f"Testing verify_document_references on: {pdf_path}")

result = verify_document_references(pdf_path)

print(f"\ntotal_extracted: {result['total_extracted']}")
print(f"total_analyzed: {result['total_analyzed']}")
print(f"summary: {result['summary']}")
print(f"details count: {len(result['details'])}")

if result['details']:
    first = result['details'][0]
    print(f"\nFirst detail reference_text: {first['reference_text'][:80]}")
    print(f"First detail verification keys: {list(first['verification'].keys())}")
    
    # Check if schema validation will work
    from models.schemas import ReferenceVerificationDetail, ReferenceVerificationResult
    try:
        detail_obj = ReferenceVerificationDetail(**first['verification'])
        result_obj = ReferenceVerificationResult(reference_text=first['reference_text'], verification=detail_obj)
        print("Schema validation: PASSED")
    except Exception as e:
        print(f"Schema validation FAILED: {e}")
