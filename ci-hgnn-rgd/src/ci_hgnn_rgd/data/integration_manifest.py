"""Versioned, atomic Stage 03 publication and bounded portable table storage."""

import json
import os
import sqlite3
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

from ..pipeline import canonical_json, fingerprint
from .manifests import atomic_json, now
from .provenance import file_sha256

TABLES = (
    "cleaned_records",
    "quarantine",
    "papers",
    "source_local_papers",
    "authors",
    "authorships",
    "citations",
    "citation_contexts",
    "source_fields",
    "identifier_mappings",
    "match_candidates",
    "match_decisions",
    "unresolved_records",
    "temporal_eligibility",
    "excluded_records",
    "provenance",
    "source_local_nodes",
)

INDEX_FIELDS = (
    "record_key",
    "canonical_paper_id",
    "canonical_author_id",
    "context_id",
    "edge_id",
    "source_paper_id",
    "target_paper_id",
    "source_dataset",
    "source_split",
    "title",
    "abstract",
    "publication_date",
    "date_precision",
    "doi",
    "arxiv_id",
    "original_context",
    "normalized_context",
    "original_intent_label",
    "normalized_intent_label",
    "decision",
    "kind",
)


class TableStore:
    """JSONL audit rows plus optional partitioned Parquet with a stable Arrow schema.

    Every Parquet partition has queryable scalar columns and a lossless record_json
    column for nested source metadata. JSONL is retained for portable audit/recovery.
    One batch is held across all tables, rather than one unbounded batch per table.
    """

    def __init__(self, root: Path, *, batch_size=1000, storage_format="auto", memory_budget_mb=128):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_buffer_bytes = memory_budget_mb * 1024**2 // 4
        self.use_parquet = storage_format != "jsonl" and find_spec("pyarrow") is not None
        if storage_format == "parquet" and not self.use_parquet:
            raise ValueError("Parquet output requires the reviewed optional 'parquet' extra")
        self.counts = Counter()
        self.parts = Counter()
        self.streams = {}
        self.buffer = []
        self.buffer_bytes = 0
        self.db = sqlite3.connect(self.root / "trace.sqlite")
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("CREATE TABLE trace (table_name TEXT, record_id TEXT, payload TEXT)")
        self.db.execute("CREATE INDEX trace_id ON trace(record_id)")
        for table in TABLES:
            self.streams[table] = (self.root / (table + ".jsonl")).open("x", encoding="utf-8")

    def write(self, table: str, row: dict):
        if table not in self.streams:
            raise ValueError(f"unknown integration table: {table}")
        encoded = canonical_json(row)
        self.streams[table].write(encoded + "\n")
        self.counts[table] += 1
        keys = {
            str(row[key])
            for key in (
                "record_key",
                "canonical_paper_id",
                "canonical_author_id",
                "context_id",
                "edge_id",
                "left_record_key",
                "right_record_key",
            )
            if row.get(key) is not None
        }
        for reference in row.get("source_record_ids", row.get("source_records", [])):
            if isinstance(reference, dict) and reference.get("record_key"):
                keys.add(reference["record_key"])
        for key in keys:
            self.db.execute("INSERT INTO trace VALUES (?,?,?)", (table, key, encoded))
        if self.use_parquet:
            self.buffer.append((table, row, encoded))
            self.buffer_bytes += len(encoded.encode("utf-8"))
            if len(self.buffer) >= self.batch_size or self.buffer_bytes >= self.max_buffer_bytes:
                self.flush()
        elif sum(self.counts.values()) % self.batch_size == 0:
            self.db.commit()

    def flush(self):
        if self.buffer:
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pa.schema(
                [(field, pa.string()) for field in INDEX_FIELDS]
                + [
                    ("publication_year", pa.int64()),
                    ("eligible", pa.bool_()),
                    ("record_json", pa.large_string()),
                ]
            )
            grouped = {}
            for table, row, encoded in self.buffer:
                values = {
                    field: str(row[field]) if row.get(field) is not None else None
                    for field in INDEX_FIELDS
                }
                year = row.get("publication_year")
                values.update(
                    publication_year=year if type(year) is int else None,
                    eligible=row.get("eligible") if type(row.get("eligible")) is bool else None,
                    record_json=encoded,
                )
                grouped.setdefault(table, []).append(values)
            for table, rows in grouped.items():
                directory = self.root / (table + ".parquet")
                directory.mkdir(exist_ok=True)
                partition = directory / f"part-{self.parts[table]:06d}.parquet"
                pq.write_table(
                    pa.Table.from_pylist(rows, schema=schema), partition, compression="zstd"
                )
                self.parts[table] += 1
            self.buffer.clear()
            self.buffer_bytes = 0
        self.db.commit()

    def close(self):
        try:
            self.flush()
        finally:
            self.db.close()
            for stream in self.streams.values():
                if not stream.closed:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()


def iter_table(run_dir: Path, table: str):
    if table not in TABLES:
        raise ValueError("unknown integration table")
    with (Path(run_dir) / (table + ".jsonl")).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


class RunPublication:
    """A failed run remains recoverable and never replaces the previous latest pointer."""

    def __init__(self, output_dir: Path, config: dict):
        self.root = Path(output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = "run-" + now().replace(":", "-") + "-" + uuid4().hex[:8]
        self.pending = self.root / (".pending-" + self.run_id)
        self.final = self.root / self.run_id
        self.pending.mkdir(exist_ok=False)
        self.config = config

    def publish(self, manifest: dict):
        if not self.pending.resolve().is_relative_to(self.root) or self.final.exists():
            raise ValueError("run publication must use a new directory inside output root")
        manifest.update(
            run_id=self.run_id,
            output_dir=str(self.final),
            config=self.config,
            config_sha256=fingerprint(self.config),
            completed_at=now(),
            status="COMPLETE",
        )
        manifest["output_checksums"] = {
            str(path.relative_to(self.pending)): file_sha256(path)
            for path in sorted(self.pending.rglob("*"))
            if path.is_file()
        }
        atomic_json(self.pending / "integration_manifest.json", manifest)
        os.rename(self.pending, self.final)
        atomic_json(self.root / "latest.json", {"run_id": self.run_id, "run_dir": str(self.final)})
        return self.final

    def fail(self, error: Exception):
        atomic_json(
            self.pending / "failure.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "config_sha256": fingerprint(self.config),
                "failed_at": now(),
            },
        )


def resolve_run(output_dir: Path, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        pointer = Path(output_dir) / "latest.json"
        if not pointer.exists():
            raise ValueError("no Stage 03 run; execute data stage03-run or provide --run-dir")
        run_dir = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])
    path = Path(run_dir).resolve()
    manifest = path / "integration_manifest.json"
    if (
        not manifest.is_file()
        or json.loads(manifest.read_text(encoding="utf-8"))["status"] != "COMPLETE"
    ):
        raise ValueError("Stage 03 run is not completely published")
    return path


def trace_record(run_dir: Path, record_id: str, *, limit=100):
    if not 1 <= limit <= 1000:
        raise ValueError("trace limit must be between 1 and 1000")
    uri = (Path(run_dir) / "trace.sqlite").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute(
            "SELECT table_name,payload FROM trace WHERE record_id=? ORDER BY table_name LIMIT ?",
            (record_id, limit + 1),
        ).fetchall()
        # Canonical IDs lead to source keys, which carry the actual merge/review
        # decisions and raw field provenance. Expansion is bounded by trace limit.
        source_keys = set()
        for _, payload in rows[:limit]:
            record = json.loads(payload)
            for reference in record.get("source_record_ids", []):
                if reference.get("record_key"):
                    source_keys.add(reference["record_key"])
            if record.get("record_key") and record["record_key"] != record_id:
                source_keys.add(record["record_key"])
        seen = set(rows)
        for source_key in sorted(source_keys)[:limit]:
            remaining = limit + 1 - len(rows)
            if remaining <= 0:
                break
            for item in db.execute(
                "SELECT table_name,payload FROM trace WHERE record_id=? "
                "ORDER BY table_name LIMIT ?",
                (source_key, remaining),
            ):
                if item not in seen:
                    rows.append(item)
                    seen.add(item)
    if not rows:
        raise ValueError(f"record not present in this run: {record_id}")
    return {
        "record_id": record_id,
        "run_dir": str(run_dir),
        "truncated": len(rows) > limit,
        "records": [
            {"table": table, "record": json.loads(payload)} for table, payload in rows[:limit]
        ],
        "interpretation": (
            "Source values, transformations, decisions and temporal exclusions are retained; "
            "no network identity verification is implied."
        ),
    }
