"""Recoverable within-source duplicate observations, separate from paper identity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def deterministic_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def record_fingerprint(record: dict, *, kind: str = "paper") -> str:
    """Use complete source values, not a paper-pair or author-name shortcut.

    A context's text, label, split and original endpoints remain distinct. An edge
    carrying different contexts remains distinct. No record is removed here.
    """
    source = record.get("raw_values")
    if source is None:
        source = {
            key: value
            for key, value in record.items()
            if key not in {"record_key", "provenance", "processing_timestamp"}
        }
    return hashlib.sha256(deterministic_json({"kind": kind, "source": source}).encode()).hexdigest()


class SourceDeduplicator:
    """SQLite registry retaining duplicate and source-ID conflict observations."""

    def __init__(self, path: str | Path | sqlite3.Connection):
        self._owns_connection = not isinstance(path, sqlite3.Connection)
        self.db = sqlite3.connect(path) if self._owns_connection else path
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS source_duplicate_records(
              kind TEXT,dataset TEXT,version TEXT,fingerprint TEXT,first_record_key TEXT,
              PRIMARY KEY(kind,dataset,version,fingerprint));
            CREATE TABLE IF NOT EXISTS source_duplicate_ids(
              kind TEXT,dataset TEXT,version TEXT,source_record_id TEXT,fingerprint TEXT,
              first_record_key TEXT,
              PRIMARY KEY(kind,dataset,version,source_record_id,fingerprint));
        """)

    def observe(self, record: dict, *, kind: str = "paper") -> dict:
        dataset = record.get("source_dataset") or "unknown"
        version = record.get("dataset_version") or "unknown"
        key = record["record_key"]
        fingerprint = record_fingerprint(record, kind=kind)
        existing = self.db.execute(
            "SELECT first_record_key FROM source_duplicate_records "
            "WHERE kind=? AND dataset=? AND version=? AND fingerprint=?",
            (kind, dataset, version, fingerprint),
        ).fetchone()
        self.db.execute(
            "INSERT OR IGNORE INTO source_duplicate_records VALUES (?,?,?,?,?)",
            (kind, dataset, version, fingerprint, key),
        )
        source_id = record.get("source_record_id")
        conflict = False
        if source_id is not None:
            conflict = bool(
                self.db.execute(
                    "SELECT 1 FROM source_duplicate_ids WHERE kind=? AND dataset=? AND version=? "
                    "AND source_record_id=? AND fingerprint<>? LIMIT 1",
                    (kind, dataset, version, str(source_id), fingerprint),
                ).fetchone()
            )
            self.db.execute(
                "INSERT OR IGNORE INTO source_duplicate_ids VALUES (?,?,?,?,?,?)",
                (kind, dataset, version, str(source_id), fingerprint, key),
            )
        return {
            "record_key": key,
            "kind": kind,
            "fingerprint": fingerprint,
            "duplicate": existing is not None,
            "duplicate_of": existing[0] if existing else None,
            "source_record_id_conflict": conflict,
            "raw_record_retained": True,
        }

    def close(self):
        self.db.commit()
        if self._owns_connection:
            self.db.close()
