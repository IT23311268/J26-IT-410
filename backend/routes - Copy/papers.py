"""
Routes for reference paper (PDF) upload, listing, and analysis.
Stage 3: returns full hallucination + citation verification results.
"""

import os
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import (
    PapersUploadResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummary,
    ReferenceVerificationResponse
)
from services.reference_verifier import verify_document_references
from services.paper_service import PaperService
from services.literature_service import LiteratureService
from services.analysis_pipeline import run_analysis_pipeline

UPLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads"
)

router = APIRouter()
paper_service = PaperService(UPLOADS_DIR)
literature_service = LiteratureService()


@router.post("/papers/upload", response_model=PapersUploadResponse)
async def upload_papers(files: List[UploadFile] = File(...)):
    """Upload one or more reference research papers in PDF format."""
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are accepted.",
            )
    try:
        result = await paper_service.save_papers(files)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")


@router.get("/papers")
async def list_papers():
    """List all uploaded reference papers."""
    papers = paper_service.list_papers()
    return {"total": len(papers), "papers": papers}


@router.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest):
    """
    Run the full Stage 3 analysis pipeline:
      1. Retrieve literature review text
      2. Extract claims and citations
      3. Process PDFs → chunks → embeddings → FAISS evidence retrieval
      4. NLI claim-evidence verification → hallucination status
      5. Citation verification → citation status
      6. Return structured results with dashboard summary
    """
    # Validate literature review exists
    review = literature_service.get_literature_review(request.literature_review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Literature review not found.")

    # Validate papers exist
    available_papers = paper_service.list_papers()
    available_filenames = [p["saved_filename"] for p in available_papers]
    for filename in request.paper_filenames:
        if filename not in available_filenames:
            raise HTTPException(
                status_code=404,
                detail=f"Paper '{filename}' not found in uploads.",
            )

    # Run the full pipeline
    try:
        pipeline_result = run_analysis_pipeline(
            literature_review=review["literature_review"],
            uploads_dir=UPLOADS_DIR,
            top_k=3,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {str(e)}",
        )

    summary_data = pipeline_result["summary"]

    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        status=pipeline_result["status"],
        message=(
            f"Analysis complete. Extracted {pipeline_result['total_claims']} claims from "
            f"{pipeline_result['total_papers_processed']} paper(s) "
            f"({pipeline_result['total_chunks_indexed']} chunks indexed) "
            f"in {pipeline_result['processing_time_seconds']}s."
        ),
        literature_review_id=request.literature_review_id,
        papers_analyzed=pipeline_result["total_papers_processed"],
        total_claims=pipeline_result["total_claims"],
        total_chunks_indexed=pipeline_result["total_chunks_indexed"],
        processing_time_seconds=pipeline_result["processing_time_seconds"],
        summary=AnalysisSummary(**summary_data),
        claims=pipeline_result["claims"],
    )


@router.get("/papers/{filename}/verify-references", response_model=ReferenceVerificationResponse)
async def verify_paper_references(filename: str):
    """
    Check whether the references included in a specific uploaded research paper are real or fake.
    """
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
