"""
Layout-aware extraction — WEEK 2+ WORK, NOT YET IMPLEMENTED.

This is where the real novelty lives: multi-column region detection,
reading-order reconstruction, IMRaD section classification, and
section-aware chunking that respects those boundaries — replacing the
fixed-size windows in baseline_extractor.py.

Planned approach (see project memory / implementation guide):
  1. Page rendering: rasterize each page (PyMuPDF get_pixmap) alongside
     the native text layer, so scanned and digital PDFs share one path.
  2. Layout region detection: classify blocks as title / body / figure /
     table / caption / equation. Start with PyMuPDF's block/line geometry
     (column-gap heuristics) as a fast v0; ColPali/ColQwen2 vision
     parsing is the v1 upgrade once GPU access is available.
  3. Reading-order reconstruction: multi-column pages re-sequenced into
     correct logical order (columns read top-to-bottom, left-to-right)
     before anything is chunked.
  4. Section tree construction: classify headings into SectionType
     (abstract / introduction / method / results / ...) via font-size +
     regex heuristics first, upgrade to a trained classifier later.
  5. Section-aware chunking: split within section boundaries instead of
     blind fixed-size windows.
  6. Artifact extraction & binding: crop figures/tables by bounding box,
     bind to captions, populate Artifact.linked_chunk_ids.

Do not implement ad hoc inside baseline_extractor.py — that function
must keep producing ExtractionMethod.PYMUPDF_FLAT_BASELINE output
unchanged, since the evaluation script (scripts/evaluate_baseline.py)
diffs this module's output against it.
"""

from __future__ import annotations

from pathlib import Path

from schema.ingestion_schema_v1 import IngestionResult


def extract_layout_aware(pdf_path: Path) -> IngestionResult:
    raise NotImplementedError(
        "Week 2 work. See module docstring for the planned approach. "
        "Until this lands, /ingest always uses the PyMuPDF flat baseline "
        "(ingestion/baseline_extractor.py)."
    )
