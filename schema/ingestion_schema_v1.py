"""
J26-IT-410 — Ingestion Output Schema, v1
=========================================

This is THE CONTRACT between Member 1 (ingestion) and Members 2, 3, 4.
Nobody edits this file without posting in the group chat first — every
downstream component (grounding gatekeeper, knowledge graph, orchestrator)
is built against these exact field names and types.

If a field must change: bump SCHEMA_VERSION, keep the old field working
if at all possible (add, don't rename), and tell the group.

Run this file directly to (re)generate schema/sample_output.json:
    python schema/ingestion_schema_v1.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.1.0"
# Changelog
#   1.0.0  initial contract (paper, chunks, artifacts)
#   1.1.0  added PageImage + IngestionResult.pages (additive — 1.0.0
#          readers keep working, the new field just defaults to [])


class SectionType(str, Enum):
    """IMRaD-style section labels. UNKNOWN is allowed — never crash on a
    heading we can't classify yet, just tag it and move on."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


class ArtifactType(str, Enum):
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"


class BoundingBox(BaseModel):
    """PDF coordinate space: origin top-left, units = points, page-relative."""

    page: int = Field(..., ge=0, description="0-indexed page number")
    x0: float
    y0: float
    x1: float
    y1: float


class Chunk(BaseModel):
    """A section-aware text chunk. This is the unit Member 2 verifies
    claims against, and the unit Member 3 turns into graph nodes."""

    chunk_id: str = Field(..., description="Stable ID: '{paper_id}::chunk::{n}'")
    paper_id: str
    section: SectionType
    section_title_raw: str = Field(
        "", description="Heading text as it appeared in the PDF, pre-classification"
    )
    text: str
    page_start: int = Field(..., ge=0)
    page_end: int = Field(..., ge=0)
    bbox: Optional[BoundingBox] = None
    order_index: int = Field(..., description="Reading-order position within the paper")
    token_count: Optional[int] = None
    artifact_ids: list[str] = Field(
        default_factory=list, description="Artifacts (figures/tables) this chunk references or contains"
    )


class Artifact(BaseModel):
    """A figure, table, or equation extracted as its own object, bound
    back to the chunk(s) that discuss it."""

    artifact_id: str = Field(..., description="Stable ID: '{paper_id}::artifact::{n}'")
    paper_id: str
    artifact_type: ArtifactType
    caption: str = ""
    bbox: BoundingBox
    image_path: Optional[str] = Field(
        None, description="Relative path into the artifact repository, e.g. 'artifacts/{paper_id}/fig_02.png'"
    )
    linked_chunk_ids: list[str] = Field(default_factory=list)


class PageImage(BaseModel):
    """A rendered bitmap of one PDF page — pipeline Box 2.

    Layout region detection (Box 3), figure cropping (Box 6) and the
    ColPali visual retriever (Box 8) all reason over pixels, not PDF
    drawing operators. So each page is rasterised once, here, and every
    later stage reuses the same image instead of re-rendering.
    """

    page_index: int = Field(..., ge=0, description="0-indexed page number")
    image_path: str = Field(
        ...,
        description="Path relative to data/processed/, e.g. '{paper_id}/pages/page_0000.png'",
    )
    width_px: int = Field(..., gt=0)
    height_px: int = Field(..., gt=0)
    dpi: int = Field(..., gt=0, description="Render resolution used for this page")

    @property
    def scale(self) -> float:
        """Pixels per PDF point, i.e. dpi / 72.

        Use this for every conversion between the two coordinate spaces
        instead of hard-coding the factor. A detector that returns pixel
        boxes converts back to BoundingBox points by dividing by `scale`;
        going the other way, multiply. Getting this backwards is the
        easiest way to misplace every figure crop in the paper.
        """
        return self.dpi / 72.0


class ExtractionMethod(str, Enum):
    """How this paper was parsed. Lets the eval script separate baseline
    runs from the real layout-aware pipeline in the same output format."""

    PYMUPDF_FLAT_BASELINE = "pymupdf_flat_baseline"
    LAYOUT_AWARE_V1 = "layout_aware_v1"


class PaperMeta(BaseModel):
    paper_id: str = Field(..., description="Stable ID, e.g. sha1 of filename or arXiv id")
    source_filename: str
    title: Optional[str] = None
    page_count: int = Field(..., ge=0)
    extraction_method: ExtractionMethod
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionResult(BaseModel):
    """Top-level object returned by POST /ingest and stored at
    data/processed/{paper_id}.json. This is what Members 2/3/4 read."""

    schema_version: str = SCHEMA_VERSION
    paper: PaperMeta
    chunks: list[Chunk]
    artifacts: list[Artifact] = Field(default_factory=list)
    pages: list[PageImage] = Field(
        default_factory=list,
        description="Rendered page bitmaps, in page order. Empty on a "
        "text-only baseline run — downstream code must tolerate that.",
    )


# ---------------------------------------------------------------------------
# Sample generation — run `python schema/ingestion_schema_v1.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from pathlib import Path

    sample = IngestionResult(
        paper=PaperMeta(
            paper_id="demo-0001",
            source_filename="demo-0001.pdf",
            title="A Demonstration Paper for Schema Purposes",
            page_count=6,
            extraction_method=ExtractionMethod.PYMUPDF_FLAT_BASELINE,
        ),
        chunks=[
            Chunk(
                chunk_id="demo-0001::chunk::0",
                paper_id="demo-0001",
                section=SectionType.ABSTRACT,
                section_title_raw="Abstract",
                text="This paper demonstrates the locked v1 ingestion schema used across J26-IT-410.",
                page_start=0,
                page_end=0,
                bbox=BoundingBox(page=0, x0=72.0, y0=650.0, x1=540.0, y1=700.0),
                order_index=0,
                token_count=14,
                artifact_ids=[],
            ),
            Chunk(
                chunk_id="demo-0001::chunk::1",
                paper_id="demo-0001",
                section=SectionType.METHOD,
                section_title_raw="2. Method",
                text="Figure 1 shows the pipeline used to produce this schema.",
                page_start=1,
                page_end=1,
                bbox=BoundingBox(page=1, x0=72.0, y0=400.0, x1=540.0, y1=430.0),
                order_index=1,
                token_count=10,
                artifact_ids=["demo-0001::artifact::0"],
            ),
        ],
        artifacts=[
            Artifact(
                artifact_id="demo-0001::artifact::0",
                paper_id="demo-0001",
                artifact_type=ArtifactType.FIGURE,
                caption="Figure 1: Pipeline overview.",
                bbox=BoundingBox(page=1, x0=100.0, y0=200.0, x1=480.0, y1=390.0),
                image_path="artifacts/demo-0001/fig_00.png",
                linked_chunk_ids=["demo-0001::chunk::1"],
            )
        ],
    )

    out_path = Path(__file__).parent / "sample_output.json"
    out_path.write_text(sample.model_dump_json(indent=2))
    print(f"wrote {out_path}")
