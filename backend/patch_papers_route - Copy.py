import os

with open("routes/papers.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add imports for reference checking
import_statement = """from models.schemas import (
    PapersUploadResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummary,
    ReferenceVerificationResponse
)
from services.reference_verifier import verify_document_references"""

content = content.replace("""from models.schemas import (
    PapersUploadResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummary,
)""", import_statement)

# If it didn't replace, it means I need to inject it. Let's do it carefully.
if "ReferenceVerificationResponse" not in content:
    content = content.replace("from services.analysis_pipeline import run_analysis_pipeline", "from services.analysis_pipeline import run_analysis_pipeline\nfrom services.reference_verifier import verify_document_references\nfrom models.schemas import ReferenceVerificationResponse")

new_route = """

@router.get("/papers/{filename}/verify-references", response_model=ReferenceVerificationResponse)
async def verify_paper_references(filename: str):
    \"\"\"
    Check whether the references included in a specific uploaded research paper are real or fake.
    \"\"\"
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Paper '{filename}' not found.")
        
    try:
        result = verify_document_references(file_path)
        return ReferenceVerificationResponse(
            paper_filename=filename,
            total_extracted=result["total_extracted"],
            total_analyzed=result["total_analyzed"],
            summary=result["summary"],
            details=result["details"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying references: {str(e)}")
"""

if "verify_paper_references" not in content:
    content += new_route

with open("routes/papers.py", "w", encoding="utf-8") as f:
    f.write(content)

print("routes/papers.py patched successfully.")
