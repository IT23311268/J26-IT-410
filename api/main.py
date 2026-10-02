"""
FastAPI app — Member 1 ingestion service.

Endpoints (Week 1 scope):
  GET  /health              liveness check
  POST /ingest               upload a PDF -> runs the baseline extractor -> saves + returns IngestionResult
  GET  /paper/{paper_id}     fetch a previously ingested paper's IngestionResult
  GET  /papers                list all ingested paper_ids

Run locally:
    uvicorn api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI —
Members 2/3/4 can try real requests there without writing any client code.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api import storage
from ingestion.baseline_extractor import extract_baseline
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
async def ingest(file: UploadFile = File(...)) -> IngestionResult:
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

    storage.save(result)
    return result


@app.get("/paper/{paper_id}", response_model=IngestionResult)
def get_paper(paper_id: str) -> IngestionResult:
    result = storage.load(paper_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No ingested paper with id '{paper_id}'")
    return result


@app.get("/papers")
def list_papers() -> JSONResponse:
    return JSONResponse(content={"paper_ids": storage.list_paper_ids()})
