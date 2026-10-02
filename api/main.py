"""
FastAPI app — Member 1 ingestion service.

Endpoints:
  GET  /health                           liveness check
  POST /ingest                           upload a PDF -> baseline extract + rasterise -> save + return IngestionResult
  GET  /paper/{paper_id}                 fetch a previously ingested paper's IngestionResult
  GET  /paper/{paper_id}/page/{n}        the rendered PNG of page n (Box 2 output)
  GET  /papers                           list all ingested paper_ids

Run locally:
    uvicorn api.main:app --reload --port 8000 \
        --reload-dir api --reload-dir ingestion --reload-dir schema

The --reload-dir flags matter. A bare --reload watches the whole repo,
including data/processed/ — so every ingest wrote page images into the
watched tree and restarted the server mid-request. Watching only the code
directories keeps the reloader useful without it reacting to our output.

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI —
Members 2/3/4 can try real requests there without writing any client code.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api import storage
from ingestion.baseline_extractor import extract_baseline
from ingestion.rasterise import DEFAULT_DPI, rasterise_pdf, resolve_image_path
from schema.ingestion_schema_v1 import SCHEMA_VERSION, IngestionResult

app = FastAPI(
    title="J26-IT-410 Ingestion Service",
    description="Member 1 — layout-aware multimodal document ingestion engine (Week 1: PyMuPDF flat baseline).",
    version=SCHEMA_VERSION,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@app.post("/ingest", response_model=IngestionResult)
async def ingest(
    file: UploadFile = File(...),
    rasterise: bool = Query(
        True, description="Also render each page to a PNG (pipeline Box 2)."
    ),
    dpi: int = Query(DEFAULT_DPI, gt=0, le=600, description="Page render resolution."),
) -> IngestionResult:
    if file.content_type not in ("application/pdf", "application/x-pdf") and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / (file.filename or "upload.pdf")
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = extract_baseline(tmp_path)
        except Exception as exc:  # noqa: BLE001 — surface as a clean 400, don't 500 on a bad PDF
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

        # Rasterise while the upload is still on disk — the temp dir is gone
        # once this block exits.
        if rasterise:
            try:
                result.pages = rasterise_pdf(
                    tmp_path,
                    paper_id=result.paper.paper_id,
                    out_root=storage.PROCESSED_DIR,
                    dpi=dpi,
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"Failed to render pages: {exc}"
                ) from exc

    storage.save(result)
    return result


def _load_or_404(paper_id: str) -> IngestionResult:
    """Shared lookup: 404 when never ingested, 500 with an actionable
    message when the record on disk is damaged."""
    try:
        result = storage.load(paper_id)
    except storage.CorruptRecord as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"No ingested paper with id '{paper_id}'")
    return result


@app.get("/paper/{paper_id}", response_model=IngestionResult)
def get_paper(paper_id: str) -> IngestionResult:
    return _load_or_404(paper_id)


@app.get(
    "/paper/{paper_id}/page/{page_index}",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
)
def get_page_image(paper_id: str, page_index: int) -> FileResponse:
    """The rendered bitmap of one page — open this in a browser to see
    exactly what the layout detector (Box 3) will be looking at."""
    result = _load_or_404(paper_id)

    page = next((p for p in result.pages if p.page_index == page_index), None)
    if page is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' has no rendered image for page {page_index}. "
            f"It has {len(result.pages)} page image(s). Re-ingest with rasterise=true.",
        )

    path = resolve_image_path(storage.PROCESSED_DIR, page)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Page image '{page.image_path}' is recorded but missing from disk.",
        )
    return FileResponse(path, media_type="image/png")


@app.get("/papers")
def list_papers() -> JSONResponse:
    return JSONResponse(content={"paper_ids": storage.list_paper_ids()})
