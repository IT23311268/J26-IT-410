"""
Baseline extractor — flat PyMuPDF text extraction.

This is deliberately naive: read pages top-to-bottom, fixed-size chunking,
no layout awareness, no section detection. It exists for two reasons:

1. It is a real, runnable Week-1 deliverable (upload a PDF, get chunks back).
2. It IS the "PyMuPDF flat extraction baseline" referenced throughout the
   proposal and presentation — every metric slide compares the real
   layout-aware pipeline against exactly this.

Week 2 replaces `extract_baseline()`'s internals with layout-aware parsing
(ingestion/layout.py, currently a stub) while keeping the same
IngestionResult shape, so nothing downstream breaks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf  # fitz

from schema.ingestion_schema_v1 import (
    Chunk,
    ExtractionMethod,
    IngestionResult,
    PaperMeta,
    SectionType,
)

DEFAULT_CHUNK_CHARS = 800
DEFAULT_CHUNK_OVERLAP = 100


def make_paper_id(pdf_path: Path) -> str:
    """Stable ID from file content, so re-ingesting the same PDF is a no-op
    downstream (Members 2/3/4 can dedupe on paper_id)."""
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:12]
    return f"{pdf_path.stem}-{digest}"


# PDF producers habitually stamp a generic string into the title metadata:
# Word writes "Microsoft Word - thesis_v3.docx", PowerPoint writes
# "PowerPoint Presentation", LaTeX tooling often leaves "untitled". Such a
# title is worse than none at all — Member 4's manuscript compiler would
# cite the reference as "PowerPoint Presentation". Treated as absent here,
# so the real title comes from the first-page TITLE section that Box 4
# identifies.
_GENERIC_TITLES = {
    "untitled",
    "untitled document",
    "document",
    "document1",
    "powerpoint presentation",
    "presentation",
    "slide 1",
    "no title",
}


def _clean_title(raw: str | None, source_filename: str) -> str | None:
    if not raw or not raw.strip():
        return None
    title = raw.strip()
    folded = title.casefold()

    if folded in _GENERIC_TITLES:
        return None
    if folded.startswith("microsoft word - "):
        return None
    # some producers just echo the filename back, which source_filename
    # already records
    if folded in {source_filename.casefold(), Path(source_filename).stem.casefold()}:
        return None
    return title


def _fixed_size_chunks(text: str, size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def extract_baseline(
    pdf_path: Path,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestionResult:
    """Flat extraction: every page's text is concatenated, then split into
    fixed-size overlapping windows. No section awareness — everything is
    tagged SectionType.UNKNOWN. No figures or tables are extracted."""

    doc = pymupdf.open(pdf_path)
    paper_id = make_paper_id(pdf_path)

    full_text_parts: list[str] = []
    page_boundaries: list[tuple[int, int]] = []  # (char_start, page_index) per page
    cursor = 0
    for page_index, page in enumerate(doc):
        page_text = page.get_text("text")
        page_boundaries.append((cursor, page_index))
        full_text_parts.append(page_text)
        cursor += len(page_text)
    full_text = "".join(full_text_parts)

    def page_for_offset(offset: int) -> int:
        page_idx = 0
        for start, idx in page_boundaries:
            if start <= offset:
                page_idx = idx
            else:
                break
        return page_idx

    raw_chunks = _fixed_size_chunks(full_text, chunk_chars, chunk_overlap)
    chunks: list[Chunk] = []
    offset = 0
    step = max(chunk_chars - chunk_overlap, 1)
    for i, raw in enumerate(raw_chunks):
        start_offset = i * step
        end_offset = start_offset + len(raw)
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}::chunk::{i}",
                paper_id=paper_id,
                section=SectionType.UNKNOWN,
                section_title_raw="",
                text=raw.strip(),
                page_start=page_for_offset(start_offset),
                page_end=page_for_offset(max(end_offset - 1, start_offset)),
                bbox=None,
                order_index=i,
                token_count=len(raw.split()),
                artifact_ids=[],
            )
        )

    title = _clean_title((doc.metadata or {}).get("title"), pdf_path.name)

    result = IngestionResult(
        paper=PaperMeta(
            paper_id=paper_id,
            source_filename=pdf_path.name,
            title=title,
            page_count=doc.page_count,
            extraction_method=ExtractionMethod.PYMUPDF_FLAT_BASELINE,
        ),
        chunks=chunks,
        artifacts=[],
    )
    doc.close()
    return result
