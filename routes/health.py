"""
Health check endpoint.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Returns the health status of the API."""
    return {
        "status": "healthy",
        "service": "Hallucination Detection & Citation Verification API",
        "version": "1.0.0",
    }
