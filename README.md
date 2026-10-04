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
- `ingestion/layout.py` — **pipeline Box 3.** Finds the blocks on each
  page, classifies them (heading / body / caption / figure / equation)
  and reconstructs reading order, so a two-column paper comes out in the
  order a human reads it instead of interleaved across the gutter. Pure
  geometry and font names — no model, no GPU.
- `scripts/inspect_fonts.py` — diagnostic. Prints every block on a page
  with the fonts it uses. The equation rules were built from its output
  on a real paper rather than from a guess.
- `api/overlay.py` — draws Box 3's regions on the page image, numbered in
  reading order. The way to *see* that detection worked.
- `api/main.py` — FastAPI service: upload a PDF, get back structured
  JSON plus rendered page images.
- `api/gallery.py` — contact-sheet view of every rendered page, for
  showing Box 2's output in one screen at a progress review. A demo
  surface only; Members 2/3/4 read the JSON.
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

Then, with a `paper_id` from `http://127.0.0.1:8000/papers`:

- `http://127.0.0.1:8000/gallery/{paper_id}` — **every page at once.** The
  view to open in a progress review.
- `http://127.0.0.1:8000/paper/{paper_id}/page/0` — one page, full size.
- `http://127.0.0.1:8000/paper/{paper_id}/page/0/layout` — **the same page
  with Box 3's detected regions drawn on it**, numbered in reading order.
  On a two-column paper the numbers should run down the left column
  before crossing to the right.

A `paper_id` derived from a filename with spaces must be percent-encoded
in the URL (`Linked%20Lists`). The gallery does this for you.

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
| 3 | Layout region detection | ✅ `ingestion/layout.py` |
| 4 | Section identification | ⬜ next |
| 5 | Section-aware chunking | ⬜ |
| 6 | Artifact extraction | ⬜ |
| 7 | Artifact binding | ⬜ after PP1 |
| 8 | Chunk + image store | ⬜ after PP1 (needs GPU for ColPali) |

Boxes 1–6 are the ~50% backend target for **Progress Presentation 1
(22–27 Oct 2026)**.
