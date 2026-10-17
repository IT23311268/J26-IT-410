"""
Routes for Literature Review submission.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import LiteratureReviewRequest, LiteratureReviewResponse
from services.literature_service import LiteratureService

router = APIRouter()
literature_service = LiteratureService()


@router.post("/literature-review", response_model=LiteratureReviewResponse)
async def submit_literature_review(request: LiteratureReviewRequest):
    """Submit a literature review for analysis."""
    try:
        result = literature_service.save_literature_review(request.literature_review)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/literature-review/{review_id}")
async def get_literature_review(review_id: str):
    """Retrieve a previously submitted literature review."""
    result = literature_service.get_literature_review(review_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Literature review not found.")
    return result
