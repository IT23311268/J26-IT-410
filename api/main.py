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
from api.compare import render_comparison
from api.gallery import render_gallery
from api.home import PaperCard, render_home
from api.overlay import render_overlay
from api.pipeline import render_pipeline
from ingestion.artifacts import extract_artifacts
from ingestion.baseline_extractor import extract_baseline
from ingestion.layout import extract_regions
from ingestion.chunking import chunk_regions
from ingestion.sections import assign_sections, paper_title
from ingestion.rasterise import DEFAULT_DPI, rasterise_pdf, resolve_image_path
from schema.ingestion_schema_v1 import (
    SCHEMA_VERSION,
    ExtractionMethod,
    IngestionResult,
    PageImage,
)

app = FastAPI(
    title="J26-IT-410 Ingestion Service",
    description="Member 1 — layout-aware multimodal document ingestion engine (Week 1: PyMuPDF flat baseline).",
    version=SCHEMA_VERSION,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    """Upload a PDF, or open one already ingested.

    `/docs` is the contract and stays where it is. This is the page to
    open at a progress review, where the question is what the service
    does rather than what its request bodies look like.
    """
    cards = []
    for paper_id in storage.list_paper_ids():
        try:
            record = storage.load(paper_id)
        except storage.CorruptRecord:
            record = None
        if record is None:
            # A damaged or half-written record still gets a row. Hiding
            # it would make a paper look like it was never ingested.
            cards.append(PaperCard(paper_id=paper_id))
        else:
            cards.append(
                PaperCard(
                    paper_id=paper_id,
                    title=record.paper.title or "",
                    page_count=record.paper.page_count,
                    chunk_count=len(record.chunks),
                    artifact_count=len(record.artifacts),
                )
            )
    return HTMLResponse(render_home(cards))


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
                # one sits in, Box 5 cuts them into chunks. Separate
                # calls, so a failure in one is obvious and each stage
                # stays usable on its own.
                regions = extract_regions(tmp_path, result.paper.paper_id)
                result.regions = assign_sections(regions)
                # The baseline cannot find a title; Box 4 can, now that
                # the front matter is marked. Only fill it if the
                # baseline left it empty — a title read out of the PDF's
                # own metadata is better evidence than our guess.
                if not result.paper.title:
                    result.paper.title = paper_title(result.regions) or None
                # The layout-aware chunks replace the baseline's
                # fixed-size ones. `extraction_method` on the paper says
                # which path produced them, so the two runs stay
                # comparable in the same output format.
                result.chunks = chunk_regions(
                    result.regions, result.paper.paper_id
                )
                # Box 6 cuts each figure, table and equation out of the
                # page images Box 2 rendered. It needs `result.pages`,
                # so it has to run after rasterisation — with
                # rasterise=false it still returns the artifacts, just
                # without pictures.
                result.artifacts = extract_artifacts(
                    result.regions,
                    result.pages,
                    result.paper.paper_id,
                    storage.PROCESSED_DIR,
                )
                result.paper.extraction_method = ExtractionMethod.LAYOUT_AWARE_V1
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"Failed to detect layout: {exc}"
                ) from exc

        # Keep the PDF: /compare re-runs the baseline from it, and a
        # re-ingest after a bug fix needs no second upload.
        storage.save_source(result.paper.paper_id, tmp_path)

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


@app.get(
    "/paper/{paper_id}/artifact/{index}",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
)
def get_artifact_image(paper_id: str, index: int) -> FileResponse:
    """One cropped figure, table or equation — Box 6's output.

    Indexed by position in `artifacts`, which is reading order, so
    artifact 0 is the first one in the paper.
    """
    result = _load_or_404(paper_id)
    if not 0 <= index < len(result.artifacts):
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' has {len(result.artifacts)} artifact(s); "
            f"no artifact {index}.",
        )

    artifact = result.artifacts[index]
    if artifact.image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact {index} of '{paper_id}' was detected but never "
            "cropped — the paper was ingested with rasterise=false. "
            "Re-ingest with rasterise=true.",
        )

    path = storage.PROCESSED_DIR / artifact.image_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact image '{artifact.image_path}' is recorded but "
            "missing from disk.",
        )
    return FileResponse(path, media_type="image/png")


@app.get("/gallery/{paper_id}", response_class=HTMLResponse, include_in_schema=False)
def gallery(paper_id: str) -> HTMLResponse:
    """Every rendered page of one paper, as a single contact sheet.

    A demo surface for progress reviews — Members 2/3/4 read the JSON, not
    this. Excluded from the OpenAPI schema so it doesn't clutter /docs.
    """
    return HTMLResponse(render_gallery(_load_or_404(paper_id)))


@app.get("/pipeline/{paper_id}", response_class=HTMLResponse, include_in_schema=False)
def pipeline(paper_id: str) -> HTMLResponse:
    """Every box's output for one paper, in pipeline order, on one page.

    The baseline is re-run here for the last section, the same way
    /compare does it, so the comparison is never a stale copy. When the
    source PDF is gone the page still renders — that one section says
    why instead of the whole page 409-ing.
    """
    result = _load_or_404(paper_id)

    source = storage.source_path(paper_id)
    baseline = None
    source_bytes = None
    if source is not None:
        source_bytes = source.stat().st_size
        try:
            baseline = extract_baseline(source)
        except Exception:  # noqa: BLE001 — the page is worth more than the section
            baseline = None

    return HTMLResponse(render_pipeline(result, baseline, source_bytes))


@app.get("/compare/{paper_id}", response_class=HTMLResponse, include_in_schema=False)
def compare(paper_id: str) -> HTMLResponse:
    """The same PDF through both paths, side by side.

    Every other surface shows the pipeline runs. This one shows it beats
    the alternative, which is the claim the proposal actually makes. The
    baseline is re-run here rather than stored, so it always reflects the
    baseline as it stands today — a stale copy would quietly flatter us.
    """
    layout = _load_or_404(paper_id)

    source = storage.source_path(paper_id)
    if source is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No stored PDF for '{paper_id}', so the baseline cannot be "
                "re-run. Papers ingested before sources were kept need one "
                "more trip through POST /ingest."
            ),
        )

    baseline = extract_baseline(source)
    return HTMLResponse(render_comparison(baseline, layout))


@app.get("/papers")
def list_papers() -> JSONResponse:
    return JSONResponse(content={"paper_ids": storage.list_paper_ids()})
