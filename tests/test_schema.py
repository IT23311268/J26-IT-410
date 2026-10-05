"""
Schema contract tests. These exist so that a change to
schema/ingestion_schema_v1.py that would break Members 2/3/4 fails loudly
here, instead of silently in someone else's integration.
"""

import json
from pathlib import Path

from schema.ingestion_schema_v1 import (
    Artifact,
    ArtifactType,
    BoundingBox,
    Chunk,
    ExtractionMethod,
    IngestionResult,
    PaperMeta,
    SCHEMA_VERSION,
    SectionType,
)


def test_sample_output_matches_schema():
    """schema/sample_output.json (checked into git as the reference example)
    must always validate against the current schema."""
    sample_path = Path(__file__).resolve().parent.parent / "schema" / "sample_output.json"
    data = json.loads(sample_path.read_text())
    result = IngestionResult.model_validate(data)
    assert result.schema_version == SCHEMA_VERSION
    assert len(result.chunks) >= 1


def test_chunk_requires_core_fields():
    chunk = Chunk(
        chunk_id="p1::chunk::0",
        paper_id="p1",
        section=SectionType.METHOD,
        text="some text",
        page_start=0,
        page_end=0,
        order_index=0,
    )
    assert chunk.artifact_ids == []
    assert chunk.bbox is None


def test_artifact_links_back_to_chunk():
    artifact = Artifact(
        artifact_id="p1::artifact::0",
        paper_id="p1",
        artifact_type=ArtifactType.FIGURE,
        bbox=BoundingBox(page=0, x0=0, y0=0, x1=10, y1=10),
        linked_chunk_ids=["p1::chunk::0"],
    )
    assert artifact.artifact_type == ArtifactType.FIGURE
    assert "p1::chunk::0" in artifact.linked_chunk_ids


def test_ingestion_result_round_trips_through_json():
    result = IngestionResult(
        paper=PaperMeta(
            paper_id="p1",
            source_filename="p1.pdf",
            page_count=1,
            extraction_method=ExtractionMethod.PYMUPDF_FLAT_BASELINE,
        ),
        chunks=[],
        artifacts=[],
    )
    as_json = result.model_dump_json()
    reloaded = IngestionResult.model_validate_json(as_json)
    assert reloaded == result
