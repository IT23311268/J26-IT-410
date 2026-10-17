"""
Main entry point for the Hallucination Detection & Citation Verification API.

PP1 Stage 3:
- Literature Review submission
- Reference PDF upload
- Claim-level hallucination detection
- Citation verification
"""

import logging
import os
import sys
import io

# Force UTF-8 on Windows to prevent 'charmap' codec errors from Unicode
# characters in PDF text (e.g. \u223c, \u2019, etc.)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.health import router as health_router
from routes.literature import router as literature_router
from routes.papers import router as papers_router


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Upload directory
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOADS_DIR, exist_ok=True)


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Hallucination Detection & Citation Verification API",
    description=(
        "PP1: Claim-Level Hallucination Detection "
        "and Citation Verification System"
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Routers
# -------------------------------------------------------------------

app.include_router(
    health_router,
    prefix="/api",
    tags=["Health"],
)

app.include_router(
    literature_router,
    prefix="/api",
    tags=["Literature Review"],
)

app.include_router(
    papers_router,
    prefix="/api",
    tags=["Reference Papers"],
)


# -------------------------------------------------------------------
# Startup model preloading
# -------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Pre-load embedding and NLI models once when the backend starts.

    This avoids loading large models every time the user clicks Analyze.
    """
    logger.info("Pre-loading models at startup to optimize runtime...")

    try:
        from services.embedding_service import _get_model
        _get_model()

        logger.info("Embedding model pre-loaded successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to preload embedding model: %s",
            exc,
        )
        raise

    try:
        from services.verifier import _get_nli_pipeline
        _get_nli_pipeline()

        logger.info("NLI model pre-loaded successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to preload NLI model: %s",
            exc,
        )
        raise

    logger.info("Models pre-loaded successfully.")


# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": (
            "Claim-Level Hallucination Detection "
            "& Citation Verification API"
        ),
    }


# -------------------------------------------------------------------
# Local development server
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )