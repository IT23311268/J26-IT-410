"""Regression tests for the storage layer.

These exist because of a real failure: the dev server's auto-reloader
restarted the process while `save()` was writing a record in place,
leaving a zero-byte JSON file. Every later read of that paper then died
with a raw pydantic "Invalid JSON: EOF while parsing a value" and the API
returned an opaque 500.
"""

from pathlib import Path

import pytest

from api import storage
from schema.ingestion_schema_v1 import (
    Chunk,
    ExtractionMethod,
    IngestionResult,
    PaperMeta,
    SectionType,
)


def _result(paper_id: str = "demo-0001") -> IngestionResult:
    return IngestionResult(
        paper=PaperMeta(
            paper_id=paper_id,
            source_filename=f"{paper_id}.pdf",
            page_count=1,
            extraction_method=ExtractionMethod.PYMUPDF_FLAT_BASELINE,
        ),
        chunks=[
            Chunk(
                chunk_id=f"{paper_id}::chunk::0",
                paper_id=paper_id,
                section=SectionType.UNKNOWN,
                text="hello",
                page_start=0,
                page_end=0,
                order_index=0,
            )
        ],
    )


@pytest.fixture(autouse=True)
def scratch_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")


def test_save_then_load_round_trips():
    storage.save(_result())
    loaded = storage.load("demo-0001")
    assert loaded is not None
    assert loaded.paper.paper_id == "demo-0001"


def test_load_returns_none_for_unknown_paper():
    assert storage.load("never-ingested") is None


def test_empty_file_raises_corrupt_record_not_a_raw_validation_error():
    """The exact state the interrupted write left behind."""
    storage.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (storage.PROCESSED_DIR / "broken.json").write_text("", encoding="utf-8")

    with pytest.raises(storage.CorruptRecord) as excinfo:
        storage.load("broken")
    # the message has to tell the reader what to do about it
    assert "Re-ingest" in str(excinfo.value)


def test_truncated_file_raises_corrupt_record():
    storage.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (storage.PROCESSED_DIR / "half.json").write_text('{"paper": {"pap', encoding="utf-8")

    with pytest.raises(storage.CorruptRecord):
        storage.load("half")


def test_save_leaves_no_temp_files_behind():
    storage.save(_result())
    leftovers = [p.name for p in storage.PROCESSED_DIR.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_interrupted_save_leaves_the_previous_record_intact(monkeypatch):
    """Atomicity: if the write dies partway, the old record must survive
    rather than being replaced by a truncated one."""
    storage.save(_result())  # a good record exists

    original = storage.load("demo-0001")
    assert original is not None

    boom = _result()
    monkeypatch.setattr(
        type(boom), "model_dump_json", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("killed"))
    )
    with pytest.raises(RuntimeError):
        storage.save(boom)

    # still readable, still complete
    survivor = storage.load("demo-0001")
    assert survivor is not None
    assert survivor.paper.paper_id == "demo-0001"


def test_round_trips_non_ascii_text():
    """Real papers carry accented author names and typographic dashes.
    Windows defaults to cp1252, so the encoding has to be explicit."""
    result = _result("unicode-0001")
    result.chunks[0].text = "Lefèvre & Müller — “structure-aware” retrieval, 2024"
    storage.save(result)

    loaded = storage.load("unicode-0001")
    assert loaded is not None
    assert loaded.chunks[0].text == "Lefèvre & Müller — “structure-aware” retrieval, 2024"


def test_list_paper_ids_ignores_in_flight_temp_files():
    storage.save(_result("a"))
    (storage.PROCESSED_DIR / ".write-xyz.tmp").write_text("partial", encoding="utf-8")

    assert storage.list_paper_ids() == ["a"]
