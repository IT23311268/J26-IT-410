"""
Minimal file-based store for ingestion results.

Week 1 scope: JSON files on disk under data/processed/. This is
intentionally boring — swapping it for a real DB later only touches this
file, because api/main.py only calls save() / load() / exists().
"""

from __future__ import annotations

import json
from pathlib import Path

from schema.ingestion_schema_v1 import IngestionResult

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _path_for(paper_id: str) -> Path:
    return PROCESSED_DIR / f"{paper_id}.json"


def save(result: IngestionResult) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(result.paper.paper_id)
    path.write_text(result.model_dump_json(indent=2))
    return path


def load(paper_id: str) -> IngestionResult | None:
    path = _path_for(paper_id)
    if not path.exists():
        return None
    return IngestionResult.model_validate_json(path.read_text())


def exists(paper_id: str) -> bool:
    return _path_for(paper_id).exists()


def list_paper_ids() -> list[str]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.json"))
