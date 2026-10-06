"""
FastAPI app — Member 1 ingestion service.

Endpoints:
  GET  /health                           liveness check
  POST /ingest                           upload a PDF -> baseline extract + rasterise -> save + return IngestionResult
  GET  /paper/{paper_id}                 fetch a previously ingested paper's IngestionResult
  GET  /paper/{paper_id}/page/{n}        the rendered PNG of page n (Box 2 output)
  GET  /paper/{paper_id}/page/{n}/layout the same page with detected regions drawn on it (Box 3)
  GET  /gallery/{paper_id}               every rendered page as one contact sheet (demo view)
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

import html
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from api import storage
from api.gallery import render_gallery
from api.overlay import render_overlay
from ingestion.baseline_extractor import extract_baseline
from ingestion.layout import extract_regions
from ingestion.sections import assign_sections
from ingestion.rasterise import DEFAULT_DPI, rasterise_pdf, resolve_image_path
from schema.ingestion_schema_v1 import SCHEMA_VERSION, IngestionResult, PageImage

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
    detect_layout: bool = Query(
        True, description="Also detect layout regions and reading order (pipeline Box 3)."
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

        if detect_layout:
            try:
                # Box 3 finds the regions, Box 4 says which section each
                # one sits in. Two calls, so a failure in either is
                # obvious and Box 3 stays usable on its own.
                regions = extract_regions(tmp_path, result.paper.paper_id)
                result.regions = assign_sections(regions)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"Failed to detect layout: {exc}"
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


def _page_or_404(result: IngestionResult, paper_id: str, page_index: int) -> PageImage:
    page = next((p for p in result.pages if p.page_index == page_index), None)
    if page is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' has no rendered image for page {page_index}. "
            f"It has {len(result.pages)} page image(s). Re-ingest with rasterise=true.",
        )
    return page


@app.get(
    "/paper/{paper_id}/page/{page_index}",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
)
def get_page_image(paper_id: str, page_index: int) -> FileResponse:
    """The rendered bitmap of one page — open this in a browser to see
    exactly what the layout detector (Box 3) will be looking at."""
    result = _load_or_404(paper_id)
    page = _page_or_404(result, paper_id, page_index)

    path = resolve_image_path(storage.PROCESSED_DIR, page)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Page image '{page.image_path}' is recorded but missing from disk.",
        )
    return FileResponse(path, media_type="image/png")


@app.get(
    "/paper/{paper_id}/page/{page_index}/layout",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def get_layout_overlay(paper_id: str, page_index: int) -> Response:
    """The page image with Box 3's detected regions drawn on it.

    The numbered badges are reading order. On a two-column page they
    should run down the left column before crossing to the right.
    """
    result = _load_or_404(paper_id)
    page = _page_or_404(result, paper_id, page_index)

    path = resolve_image_path(storage.PROCESSED_DIR, page)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Page image '{page.image_path}' is recorded but missing from disk.",
        )

    page_regions = [r for r in result.regions if r.bbox.page == page_index]
    if not page_regions:
        raise HTTPException(
            status_code=404,
            detail=f"No layout regions for page {page_index} of '{paper_id}'. "
            "Re-ingest with detect_layout=true.",
        )

    png = render_overlay(path, page_regions, page.scale)
    return Response(content=png, media_type="image/png")


@app.get("/gallery/{paper_id}", response_class=HTMLResponse, include_in_schema=False)
def gallery(paper_id: str) -> HTMLResponse:
    """Every rendered page of one paper, as a single contact sheet.

    A demo surface for progress reviews — Members 2/3/4 read the JSON, not
    this. Excluded from the OpenAPI schema so it doesn't clutter /docs.
    """
    return HTMLResponse(render_gallery(_load_or_404(paper_id)))


@app.get("/papers")
def list_papers() -> JSONResponse:
    return JSONResponse(content={"paper_ids": storage.list_paper_ids()})
