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
- `ingestion/layout.py` — **stub.** Week 2+ layout-aware parsing goes
  here (multi-column detection, section tree, artifact binding).
- `api/main.py` — FastAPI service: upload a PDF, get back structured
  JSON.
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
uvicorn api.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs** — interactive Swagger UI. Try
`POST /ingest` with any PDF (or generate one first, see below).

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

## Roadmap (see project plan)

- **Week 1 (done in this scaffold):** JSON schema locked, API skeleton,
  PyMuPDF flat baseline, tests.
- **Week 2:** layout region detection, reading-order reconstruction,
  section tree, section-aware chunking (`ingestion/layout.py`).
- **Week 3:** figure/table extraction + artifact repository, e5 +
  Qdrant embeddings, `/paper/{id}` used by Members 2/3, ColPali spike.
- **Week 4:** freeze, eval script (baseline vs. layout-aware), demo
  rehearsal for Progress Presentation 1 (22–27 Oct 2026).
