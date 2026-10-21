"""Executed text inspections and canonical JSONL staging, retaining invalid-record locators."""

import json
from pathlib import Path
from uuid import uuid4

from ..pipeline import fingerprint
from .manifests import DatasetStatus, atomic_json, now
from .profiling import Profiler
from .provenance import file_sha256
from .sources import arxiv, scicite
from .streaming import Selection, events


def inspect_text(
    dataset,
    path: Path,
    output_dir: Path,
    *,
    max_records=None,
    sample_size=None,
    seed=42,
    version=None,
    synthetic=False,
    short_text=20,
    long_text=10000,
    max_line_bytes=4 * 1024**2,
):
    if dataset not in {"scicite", "arxiv-metadata"}:
        raise ValueError("text inspection supports scicite and arxiv-metadata")
    if not path.exists():
        raise FileNotFoundError(f"local dataset not found: {path}")
    run_dir = output_dir / "inspections" / dataset / str(uuid4())
    run_dir.mkdir(parents=True, exist_ok=False)
    index_path = run_dir / "identifiers.sqlite"
    staging_path = run_dir / "staging.jsonl"
    source_hash = (
        file_sha256(path)
        if path.is_file()
        else fingerprint(
            [
                (str(item.relative_to(path)), file_sha256(item))
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix in {".json", ".jsonl"}
            ]
        )
    )
    acquired_at = None
    sidecar = path.with_name(path.name + ".source.json")
    if path.is_file() and sidecar.exists():
        acquisition = json.loads(sidecar.read_text(encoding="utf-8"))
        if acquisition.get("sha256") != source_hash:
            raise ValueError("source changed since acquisition; checksum mismatch")
        acquired_at = acquisition.get("downloaded_at")
    config = {
        "dataset": dataset,
        "max_records": max_records,
        "sample_size": sample_size,
        "seed": seed,
        "version": version,
        "synthetic": synthetic,
        "short_text": short_text,
        "long_text": long_text,
        "max_line_bytes": max_line_bytes,
    }
    selection = Selection(max_records=max_records, sample_size=sample_size, seed=seed)
    profiler = Profiler(index_path, short_text=short_text, long_text=long_text)
    adapter = scicite if dataset == "scicite" else arxiv
    try:
        with staging_path.open("x", encoding="utf-8") as stream:
            for event in selection.select(events(path, dataset, max_line_bytes=max_line_bytes)):
                canonical = None
                issues = [event.error] if event.error else []
                if event.record is not None:
                    try:
                        canonical, adapter_issues = adapter.adapt(
                            event, version=version, synthetic=synthetic, acquired_at=acquired_at
                        )
                        issues += adapter_issues
                        json.dumps(canonical.model_dump(mode="json"), allow_nan=False)
                    except (ValueError, TypeError, OverflowError):
                        canonical = None
                        issues.append("canonical_adaptation_failed")
                profiler.observe(event, canonical, issues)
                envelope = {
                    "schema_version": "1.0",
                    "validation_status": "invalid" if issues else "valid",
                    "issues": issues,
                    "source_locator": f"{event.source}:line:{event.line}",
                    "source_record_sha256": event.sha256,
                    "record": canonical.model_dump(mode="json") if canonical else None,
                }
                stream.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
        quality = profiler.report()
        schema = profiler.schema.report()
    finally:
        profiler.close()
    if not quality["raw_record_count"]:
        raise ValueError("no source records were inspected")
    partial = (
        not selection.exhausted
        or sample_size is not None
        or quality["invalid_record_count"] > 0
        or quality["cross_split_exact_text_overlap_groups"] > 0
        or quality["duplicate_records"] > 0
    )
    result = {
        "schema_version": "1.0",
        "dataset_id": dataset,
        "verified_at": now(),
        "source_path": str(path.resolve()),
        "source_sha256": source_hash,
        "source_version": version,
        "unknown_upstream_version": version is None,
        "synthetic": synthetic,
        "status": DatasetStatus.PARTIALLY_VALIDATED.value
        if partial
        else DatasetStatus.VALIDATED.value,
        "scope": selection.describe(),
        "inspection_config": config,
        "inspection_config_sha256": fingerprint(config),
        "observed_schema": schema,
        "quality": quality,
        "staging_path": str(staging_path.resolve()),
        "index_path": str(index_path.resolve()),
        "limitations": [
            (
                "Source availability and historical temporal eligibility are not established "
                "by ingestion."
            ),
            "Identifier syntax and presence do not establish external resolvability.",
            "No source records were cleaned, dropped, merged, or resplit.",
        ],
    }
    if dataset == "scicite" and "string" in schema["fields"] and "context" not in schema["fields"]:
        result["documentation_discrepancies"] = [
            "README prose mentions context, but inspected records expose string."
        ]
    if sample_size:
        result["limitations"].append(
            "Quality/overlap results cover only the deterministic sample, not unselected records."
        )
    atomic_json(run_dir / "inspection.json", result)
    result["inspection_path"] = str((run_dir / "inspection.json").resolve())
    return result
