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
  page, classifies them (heading / body / caption / figure / table /
  equation) and reconstructs reading order, so a two-column paper comes
  out in the order a human reads it instead of interleaved across the
  gutter. Figures, tables and equations absorb the text inside them, so
  axis labels and table cells do not scatter into the prose — and a
  figure is grown to its real extent first, since a chart's ticks and
  axis titles are printed outside its plot frame. Charts and tables are
  told apart by artwork: a cluster of straight lines is a frame or a set
  of rules, a figure has a filled shape or a curve in it — and then by
  their caption, which overrules the drawing, so a table pasted in as a
  screenshot is still filed as a table. Pure geometry, font names and
  the author's own labels — no model, no GPU.
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
- `ingestion/sections.py` — **pipeline Box 4.** Reads the HEADING
  regions, works out which section of the paper each one opens, and
  tags every region with the section it sits in. The same sentence
  means different things in Related Work and in Results, and Member 2
  cannot tell them apart without this label. Keyword order decides a
  heading that names two sections at once, so the order of
  `SECTION_KEYWORDS` is pinned by tests.
- `ingestion/chunking.py` — **pipeline Box 5.** Cuts the labelled
  regions into the `Chunk` objects Members 2 and 3 retrieve from. The
  cut falls at a section boundary always, and at a paragraph boundary
  when a section runs longer than one chunk — never inside a sentence,
  which is what the fixed-size baseline gets wrong.
- `ingestion/artifacts.py` — **pipeline Box 6.** Cuts each figure,
  table and equation out of the page image Box 2 rendered and writes it
  as a file of its own, with the caption the author printed beside it.
  Nothing here re-reads the PDF: the crop is the region's bbox times
  `PageImage.scale`. The crop margin was measured rather than guessed —
  ink runs 0.7 pt outside a bbox and the nearest caption sits 1.9 pt
  away, so each edge is padded by 6 pt *or half the distance to
  whatever is next to it*, whichever is smaller.
- `api/compare.py` — **baseline against layout-aware, side by side.**
  The same PDF through both paths, with the two numbers that carry the
  claim: how many chunks have no section, and how many start mid-word.
  Open `/compare/{paper_id}`.
- `api/pipeline.py` — **every box's output on one page**, in pipeline
  order, with the baseline comparison at the end. This is the view to
  open at a progress review: six separate demo URLs mean the panel has
  to map each screen onto the component diagram while the presenter
  talks. Open `/pipeline/{paper_id}`.
- `api/home.py` — the front door at `/`: drop a PDF in, or open one
  already ingested. `/docs` is still the contract for Members 2/3/4 and
  is linked from the footer.
- `api/ui.py` — the palette and base styles the three HTML surfaces
  share. No web fonts and no CDN: these pages are opened from localhost
  on a laptop plugged into a projector.
- `scripts/check_corpus.py` — runs Box 3 over a folder of papers and
  prints the pages that look wrong. Every threshold in `layout.py` was
  measured on one or two real pages, which is the honest way to pick a
  number but is not evidence that it holds across a corpus. This is how
  that gets checked:

  ```bash
  python scripts/check_corpus.py data/raw
  ```

  It does not know the right answer. It looks for the shapes every bug
  so far produced — short text scattered around a figure, a table and a
  figure overlapping, a page with no text layer — and names the pages
  worth opening in the `/layout` overlay. `tests/test_check_corpus.py`
  reintroduces each of those bugs and asserts it still notices.
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

Open **http://127.0.0.1:8000/** and drop a PDF on the page. It ingests
and takes you straight to the pipeline view for that paper. That is the
whole demo — one URL, one scroll.

`http://127.0.0.1:8000/docs` is still there: the interactive Swagger UI,
and the contract Members 2/3/4 write their clients against.

The individual views, with a `paper_id` from
`http://127.0.0.1:8000/papers`:

- `http://127.0.0.1:8000/pipeline/{paper_id}` — **every box's output in
  order on one page.** The view to open at a progress review.
- `http://127.0.0.1:8000/compare/{paper_id}` — **baseline against
  layout-aware**, the two runs side by side in full.
- `http://127.0.0.1:8000/paper/{paper_id}/artifact/0` — one cropped
  figure or table (Box 6), indexed in reading order.
- `http://127.0.0.1:8000/gallery/{paper_id}` — every page at once.
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
| 4 | Section identification | ✅ `ingestion/sections.py` |
| 5 | Section-aware chunking | ✅ `ingestion/chunking.py` |
| 6 | Artifact extraction | ✅ `ingestion/artifacts.py` |
| 7 | Artifact binding | ⬜ after PP1 |
| 8 | Chunk + image store | ⬜ after PP1 (needs GPU for ColPali) |

Boxes 1–6 are the ~50% backend target for **Progress Presentation 1
(22–27 Oct 2026)** — complete.

Box 7 is what fills `Artifact.linked_chunk_ids` and `Chunk.artifact_ids`,
which Box 6 deliberately leaves empty: the crop and its caption are one
job, deciding which passages discuss it is another.
