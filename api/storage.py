"""
Minimal file-based store for ingestion results.

Week 1 scope: JSON files on disk under data/processed/. This is
intentionally boring — swapping it for a real DB later only touches this
file, because api/main.py only calls save() / load() / exists().
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from pydantic import ValidationError

from schema.ingestion_schema_v1 import IngestionResult

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# The uploaded PDF, kept so a stage can be re-run without the file
# being uploaded again. /compare needs it to run the baseline a
# second time, and it is what makes a re-ingest after a bug fix a
# one-line script rather than a trip through the browser.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class CorruptRecord(Exception):
    """A record exists on disk but could not be read back as an
    IngestionResult. Raised instead of letting a raw pydantic error
    surface as an opaque 500."""


def _path_for(paper_id: str) -> Path:
    return PROCESSED_DIR / f"{paper_id}.json"


def save(result: IngestionResult) -> Path:
    """Write the record atomically.

    The payload goes to a temporary file in the same directory and is then
    renamed over the target. `os.replace()` is atomic on both POSIX and
    Windows, so an interrupted save leaves either the previous complete
    file or the new complete one — never a half-written or zero-byte one.

    This is not hypothetical: writing in place meant that when the dev
    server's auto-reloader restarted the process mid-write (it fires
    because ingestion writes page images into the watched tree), the
    record was left empty, and every later read of it failed with
    "Invalid JSON: EOF while parsing a value".
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target = _path_for(result.paper.paper_id)
    payload = result.model_dump_json(indent=2)

    # encoding is explicit everywhere in this module: Python defaults to the
    # locale encoding on Windows (cp1252), which mangles or rejects the
    # accented names and typographic dashes that real papers are full of.
    fd, tmp_name = tempfile.mkstemp(dir=PROCESSED_DIR, prefix=".write-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def load(paper_id: str) -> IngestionResult | None:
    """Return the stored record, or None if this paper was never ingested.

    Raises CorruptRecord if the file is present but unreadable, so the
    caller can tell "never ingested" apart from "ingested, then damaged".
    """
    path = _path_for(paper_id)
    if not path.exists():
        return None

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise CorruptRecord(
            f"'{path.name}' is empty — the previous save was interrupted. Re-ingest this PDF."
        )
    try:
        return IngestionResult.model_validate_json(raw)
    except ValidationError as exc:
        raise CorruptRecord(
            f"'{path.name}' is not a valid IngestionResult. Re-ingest this PDF. ({exc.error_count()} error(s))"
        ) from exc


def exists(paper_id: str) -> bool:
    return _path_for(paper_id).exists()


def list_paper_ids() -> list[str]:
    if not PROCESSED_DIR.exists():
        return []
    # ".write-*.tmp" files never match "*.json", so an in-flight save is
    # invisible here.
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.json"))


def save_source(paper_id: str, pdf_path: Path) -> Path:
    """Keep the uploaded PDF under its paper_id."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"{paper_id}.pdf"
    shutil.copyfile(pdf_path, target)
    return target


def source_path(paper_id: str) -> Path | None:
    """Where this paper's PDF was kept, or None if it predates that."""
    candidate = RAW_DIR / f"{paper_id}.pdf"
    return candidate if candidate.exists() else None
