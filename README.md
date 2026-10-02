# J26-IT-410 — Multimodal Ingestion Engine (Member 1)

Layout-aware multimodal document ingestion engine — the foundational
layer for the J26-IT-410 GraphRAG pipeline. Parses reference PDFs and
produces the structured `IngestionResult` JSON that Members 2 (grounding
verification), 3 (knowledge graph & GNN) and 4 (orchestration & IEEE
output) all build against.

**Branch:** `feature/m1-ingestion`
**Owner:** Senadhira D.O.V (IT23311268)

## What's here (Week 1)

- `schema/ingestion_schema_v1.py` — **the locked contract.** Pydantic
  models for `IngestionResult`, `Chunk`, `Artifact`, etc. Don't change
  field names without telling the group.
- `ingestion/baseline_extractor.py` — flat PyMuPDF extraction (no layout
  awareness, fixed-size chunking). This is the "baseline" every later
  metric in the proposal compares against.
- `ingestion/rasterise.py` — **pipeline Box 2.** Renders each page to a
  PNG. Boxes 3, 6 and 8 all work on these pixels, so the page is
  rendered once here and reused. `PageImage.scale` is the one place the
  points-to-pixels conversion is defined.
- `ingestion/layout.py` — **stub.** Box 3+ layout-aware parsing goes
  here (multi-column detection, section tree, artifact binding).
- `api/main.py` — FastAPI service: upload a PDF, get back structured
  JSON plus rendered page images.
- `scripts/generate_sample_pdf.py` — makes a synthetic test PDF so you
  can run everything before a real paper corpus is collected.
- `tests/` — pytest suite covering the schema, the baseline extractor,
  and the API end to end.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the tests

```bash
pytest -v
```

## Run the API locally

```bash
uvicorn api.main:app --reload --port 8000 \
  --reload-dir api --reload-dir ingestion --reload-dir schema
```

On Windows, as one line:

```
python -m uvicorn api.main:app --reload --port 8000 --reload-dir api --reload-dir ingestion --reload-dir schema
```

The `--reload-dir` flags are not optional. A bare `--reload` watches the
whole repo including `data/processed/`, so every ingest writes page
images into the watched tree and restarts the server mid-request.

Open **http://127.0.0.1:8000/docs** — interactive Swagger UI. Try
`POST /ingest` with any PDF (or generate one first, see below).

Then open `http://127.0.0.1:8000/paper/{paper_id}/page/0` in a browser to
see the rendered page image straight from the service.

## Generate a test PDF and ingest it manually

```bash
python scripts/generate_sample_pdf.py data/raw/sample_paper.pdf

curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@data/raw/sample_paper.pdf"
```

## Regenerate the sample schema output

```bash
python schema/ingestion_schema_v1.py
# writes schema/sample_output.json
```

## Pipeline status

The eight boxes from the component diagram in the proposal:

| # | Stage | State |
|---|-------|-------|
| 1 | PDF input | ✅ `POST /ingest` |
| 2 | Page rasterisation | ✅ `ingestion/rasterise.py` |
| 3 | Layout region detection | ⬜ next |
| 4 | Section identification | ⬜ |
| 5 | Section-aware chunking | ⬜ |
| 6 | Artifact extraction | ⬜ |
| 7 | Artifact binding | ⬜ after PP1 |
| 8 | Chunk + image store | ⬜ after PP1 (needs GPU for ColPali) |

Boxes 1–6 are the ~50% backend target for **Progress Presentation 1
(22–27 Oct 2026)**.
