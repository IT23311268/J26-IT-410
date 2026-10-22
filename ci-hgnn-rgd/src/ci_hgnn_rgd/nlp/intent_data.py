"""Stage 04 modelling views derived from immutable, published Stage 03 contexts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..data.integration_manifest import resolve_run
from ..data.provenance import file_sha256
from ..data.split_management import SplitAuditor, normalized_context
from ..data.temporal_integrity import assess_temporal, parse_cutoff
from ..pipeline import fingerprint
from .intent_config import Stage04Config
from .intent_contracts import (
    INTENT_LABELS,
    LABEL_TO_ID,
    IntentDataError,
    IntentDatasetRecord,
    normalize_context_text,
    normalize_intent_label,
)

SourceSplit = Literal["train", "dev", "test"]
_SPLITS: tuple[SourceSplit, ...] = ("train", "dev", "test")


@dataclass(frozen=True)
class PreparedIntentData:
    """A deterministic, non-mutating view over the Stage 03 context table."""

    stage03_run_dir: Path
    records: dict[SourceSplit, tuple[IntentDatasetRecord, ...]]
    dataset_manifest: dict
    dataset_profile: dict
    leakage_audit: dict
    temporal_validation: dict
    exclusions: tuple[dict, ...]

    @property
    def all_records(self) -> tuple[IntentDatasetRecord, ...]:
        return tuple(record for split in _SPLITS for record in self.records[split])

    def split(self, name: SourceSplit) -> tuple[IntentDatasetRecord, ...]:
        return self.records[name]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_stage03_manifest(run_dir: Path) -> dict:
    path = run_dir / "integration_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntentDataError(
            f"cannot read Stage 03 integration manifest: {type(error).__name__}"
        ) from None
    if manifest.get("status") != "COMPLETE":
        raise IntentDataError("Stage 03 run is not completely published")
    return manifest


def _verify_table(run_dir: Path, manifest: dict, table: str) -> Path:
    path = run_dir / (table + ".jsonl")
    if not path.is_file():
        raise IntentDataError(f"Stage 03 context table is missing: {path.name}")
    relative = str(path.relative_to(run_dir))
    checksums = manifest.get("output_checksums", {})
    expected = checksums.get(relative) or checksums.get(relative.replace("/", "\\"))
    if expected is None:
        raise IntentDataError("Stage 03 manifest does not contain a checksum for citation contexts")
    actual = file_sha256(path)
    if actual != expected:
        raise IntentDataError("Stage 03 citation-context checksum does not match its manifest")
    return path


def _record_from_stage03(row: dict) -> IntentDatasetRecord:
    if not isinstance(row, dict):
        raise IntentDataError("citation-context row must be an object")
    for field in ("provenance", "transformations"):
        values = row.get(field)
        if values is not None and (
            not isinstance(values, list) or any(not isinstance(item, dict) for item in values)
        ):
            raise IntentDataError(f"{field} must be a list of objects")
    if row.get("temporal_eligibility") is not None and not isinstance(
        row["temporal_eligibility"], dict
    ):
        raise IntentDataError("temporal_eligibility must be an object")
    temporal = row.get("temporal_eligibility") or {}
    if not isinstance(temporal.get("evidence", {}), dict):
        raise IntentDataError("temporal evidence must be an object")
    context_id = row.get("context_id")
    if not isinstance(context_id, str) or not context_id.strip():
        raise IntentDataError("missing citation_context_id")
    source_dataset = row.get("source_dataset")
    if source_dataset != "scicite":
        raise IntentDataError("Stage 04 only accepts SciCite citation contexts")
    split = row.get("source_split")
    if split not in _SPLITS:
        raise IntentDataError("missing or unsupported official source split")
    if row.get("quarantine"):
        raise IntentDataError("quarantined Stage 03 record")
    original = row.get("original_context")
    if not isinstance(original, str) or not original.strip():
        raise IntentDataError("missing or empty citation context")
    model_text = normalize_context_text(original)
    if not model_text:
        raise IntentDataError("citation context is empty after whitespace normalization")
    original_label = row.get("original_intent_label")
    label = normalize_intent_label(original_label)
    staged_label = row.get("normalized_intent_label")
    if staged_label is not None and normalize_intent_label(staged_label) != label:
        raise IntentDataError("Stage 03 original and normalized citation-intent labels disagree")
    stored_normalized = row.get("normalized_context")
    transformations = list(row.get("transformations") or [])
    if isinstance(stored_normalized, str) and stored_normalized != model_text:
        transformations.append(
            {
                "field": "original_context",
                "operation": "stage04_nfc_whitespace_normalization",
                "reason": "Stage 03 normalized context differed from deterministic Stage 04 view",
            }
        )
    provenance = tuple(row.get("provenance") or [])
    identity = row.get("identity_resolution_status")
    return IntentDatasetRecord(
        citation_context_id=context_id,
        source_dataset=source_dataset,
        source_record_id=row.get("source_record_id"),
        dataset_version=row.get("dataset_version"),
        source_split=split,
        original_text=original,
        model_text=model_text,
        original_label=original_label,
        normalized_label=label,
        label_id=LABEL_TO_ID[label],
        citing_paper_id=row.get("citing_paper_id"),
        cited_paper_id=row.get("cited_paper_id"),
        citing_identifier=row.get("citing_identifier"),
        cited_identifier=row.get("cited_identifier"),
        identity_resolution_status=identity
        if isinstance(identity, str) and identity
        else "UNRESOLVED",
        provenance=provenance,
        temporal_eligibility=row.get("temporal_eligibility") or {},
        transformations=tuple(transformations),
    )


def audit_intent_leakage(
    records: Iterable[IntentDatasetRecord],
    config: Stage04Config,
    *,
    sqlite_path: Path,
) -> dict:
    """Run the shared bounded Stage 03 split auditor without changing assignments."""
    sqlite_path = Path(sqlite_path)
    if sqlite_path.exists():
        raise IntentDataError(
            "Stage 04 leakage audit database already exists; use a fresh audit path"
        )
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with SplitAuditor(
        sqlite_path,
        near_threshold=config.evaluation.near_duplicate_threshold,
        max_candidates_per_context=config.evaluation.near_candidates_per_context,
        max_comparisons=config.evaluation.max_near_comparisons,
    ) as auditor:
        for record in records:
            auditor.add_context(
                {
                    "context_id": record.citation_context_id,
                    "source_split": record.source_split,
                    "original_context": record.original_text,
                    "normalized_context": record.model_text,
                    "citing_paper_id": record.citing_paper_id,
                    "cited_paper_id": record.cited_paper_id,
                    "citing_identifier": record.citing_identifier,
                    "cited_identifier": record.cited_identifier,
                    "source_dataset": record.source_dataset,
                    "dataset_version": record.dataset_version,
                    # SciCite does not provide a guaranteed globally canonical
                    # record ID.  This deliberately scoped digest lets the shared
                    # auditor report reuse of the raw source record identifier
                    # across official splits without promoting it to a paper ID.
                    "source_record_sha256": _sha256_text(
                        "\x1f".join(
                            (
                                record.source_dataset,
                                record.dataset_version or "unknown",
                                str(record.source_record_id),
                            )
                        )
                    )
                    if record.source_record_id is not None
                    else None,
                    "provenance": list(record.provenance),
                }
            )
        report = auditor.report()
        findings = {}
        for name, column in (
            ("source_record_id", "record_hash"),
            ("exact_text", "raw_hash"),
            ("normalized_text", "norm_hash"),
        ):
            # Only constant column names above enter this SQL. Include every
            # affected context, rather than relying on a truncated examples list.
            findings[name] = [
                {"citation_context_id": context_id, "source_split": split, "group_sha256": key}
                for context_id, split, key in auditor.db.execute(
                    f"SELECT context_id,split,{column} FROM split_contexts "
                    f"WHERE {column} IN (SELECT {column} FROM split_contexts "
                    f"WHERE {column} IS NOT NULL AND split IS NOT NULL "
                    f"GROUP BY {column} HAVING COUNT(DISTINCT split)>1) ORDER BY seq"
                )
            ]
        findings["near_duplicate_pairs"] = [
            {
                "left_context_id": left,
                "right_context_id": right,
                "left_split": ls,
                "right_split": rs,
                "similarity": similarity,
            }
            for left, right, ls, rs, similarity in auditor.db.execute(
                "SELECT l.context_id,r.context_id,l.split,r.split,n.similarity "
                "FROM split_near n JOIN split_contexts l ON l.seq=n.left_seq "
                "JOIN split_contexts r ON r.seq=n.right_seq ORDER BY n.left_seq,n.right_seq"
            )
        ]
        report["affected_records"] = findings
    report["scope"] = "nonquarantined Stage 03 SciCite contexts admitted to the Stage 04 model view"
    report["benchmark_policy"] = (
        "Official SciCite train/dev/test assignments are retained. Findings are reported, not "
        "silently filtered from benchmark evaluation."
    )
    report["shared_source_record_identifier_overlap_groups"] = report["exact_record_overlap_groups"]
    report["exact_record_overlap_definition"] = (
        "scoped source_dataset, dataset_version, and source_record_id digest"
    )
    return report


def _temporal_validation(records: Iterable[IntentDatasetRecord], stage03_manifest: dict) -> dict:
    rows = tuple(records)
    source_temporal = stage03_manifest.get("temporal", {})
    dated = 0
    eligible = 0
    for row in rows:
        evidence = row.temporal_eligibility.get("evidence", {})
        if evidence.get("publication_date") or evidence.get("publication_year"):
            dated += 1
        eligible += bool(row.temporal_eligibility.get("eligible"))
    return {
        "status": "NOT_EVALUABLE",
        "official_split_evaluation": "supported_nonchronological_benchmark",
        "records_with_publication_date_or_year": dated,
        "records_with_strict_temporal_eligibility": eligible,
        "records": len(rows),
        "stage03_temporal_summary": source_temporal,
        "reason": (
            "SciCite official source splits are preserved benchmark assignments, not verified "
            "chronological splits. The Stage 03 source records lack sufficient verified historical "
            "availability and publication coverage for a temporal generalization claim."
        ),
    }


def chronological_split_if_supported(
    records: Iterable[IntentDatasetRecord],
    *,
    train_end: datetime | str,
    dev_end: datetime | str,
    test_end: datetime | str,
) -> dict:
    """Build an opt-in chronological allocation only from verified dated records.

    The function does not substitute ingestion timestamps for publication dates and
    fails closed when the supplied Stage 03 view cannot support the requested
    evaluation.
    """
    train_boundary = parse_cutoff(train_end)
    dev_boundary = parse_cutoff(dev_end)
    test_boundary = parse_cutoff(test_end)
    if not train_boundary < dev_boundary < test_boundary:
        raise IntentDataError("chronological boundaries must satisfy train < dev < test")
    allocated: dict[str, list[IntentDatasetRecord]] = {"train": [], "dev": [], "test": []}
    unknown = []
    after_window = []
    for record in records:
        temporal = record.temporal_eligibility
        evidence = temporal.get("evidence", {})
        value = evidence.get("publication_date") or evidence.get("publication_year")
        if not value or not temporal.get("eligible"):
            unknown.append(record.citation_context_id)
            continue
        moment = parse_cutoff(value)
        name, boundary = (
            ("train", train_boundary)
            if moment < train_boundary
            else ("dev", dev_boundary)
            if moment < dev_boundary
            else ("test", test_boundary)
        )
        if moment >= test_boundary:
            after_window.append(record.citation_context_id)
            continue
        # An eligibility flag computed against a later Stage 03 cutoff cannot
        # authorize future metadata in an earlier evaluation window.
        assessment = assess_temporal(evidence, boundary, kind="citation_context")
        if not assessment["strict_historical_claim"]:
            unknown.append(record.citation_context_id)
            continue
        allocated[name].append(record)
    if unknown or not all(allocated.values()):
        raise IntentDataError(
            "chronological evaluation is not supported by complete verified temporal metadata"
        )
    return {
        "split_construction": "verified_publication_date_and_stage03_temporal_eligibility",
        "records": {name: tuple(values) for name, values in allocated.items()},
        "official_splits_replaced": False,
        "excluded_after_test_window": after_window,
        "boundaries": {
            "train_end": train_boundary.isoformat(),
            "dev_end": dev_boundary.isoformat(),
            "test_end": test_boundary.isoformat(),
        },
    }


def prepare_intent_data(
    config: Stage04Config,
    project_root: Path,
    *,
    stage03_run: Path | None = None,
    audit_sqlite_path: Path | None = None,
) -> PreparedIntentData:
    """Read and validate genuine Stage 03 data without modifying any source table."""
    project_root = Path(project_root).resolve()
    run = resolve_run(config.dataset.stage03_output_dir, stage03_run or config.dataset.stage03_run)
    manifest = _load_stage03_manifest(run)
    real_sources = set(manifest.get("real_sources", []))
    if "scicite" not in real_sources and not config.dataset.allow_synthetic:
        raise IntentDataError("real SciCite Stage 03 source data are required for Stage 04")
    table = _verify_table(run, manifest, config.dataset.citation_context_table)
    grouped: dict[SourceSplit, list[IntentDatasetRecord]] = {split: [] for split in _SPLITS}
    exclusions = []
    total_rows = 0
    with table.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            total_rows += 1
            try:
                row = json.loads(line)
                record = _record_from_stage03(row)
            except (IntentDataError, ValueError, TypeError, json.JSONDecodeError) as error:
                # The initial parse can itself have failed, so never attempt a
                # second unguarded decode while constructing an exclusion record.
                try:
                    raw = json.loads(line) if line.lstrip().startswith("{") else {}
                except json.JSONDecodeError:
                    raw = {}
                if not isinstance(raw, dict):
                    raw = {}
                exclusions.append(
                    {
                        "line": line_number,
                        "citation_context_id": raw.get("context_id"),
                        "source_record_id": raw.get("source_record_id"),
                        "source_split": raw.get("source_split"),
                        "reason": str(error),
                    }
                )
                continue
            grouped[record.source_split].append(record)
    if not all(grouped[split] for split in _SPLITS):
        missing = [split for split in _SPLITS if not grouped[split]]
        raise IntentDataError("missing official Stage 04 split(s): " + ", ".join(missing))
    ordered = {
        split: tuple(sorted(rows, key=lambda row: row.citation_context_id))
        for split, rows in grouped.items()
    }
    all_records = tuple(record for split in _SPLITS for record in ordered[split])
    if audit_sqlite_path is None:
        with tempfile.TemporaryDirectory(prefix="ci-hgnn-rgd-stage04-") as directory:
            leakage = audit_intent_leakage(
                all_records, config, sqlite_path=Path(directory) / "leakage.sqlite"
            )
    else:
        leakage = audit_intent_leakage(all_records, config, sqlite_path=audit_sqlite_path)
    label_distribution = {
        split: {
            label: sum(row.normalized_label == label for row in ordered[split])
            for label in INTENT_LABELS
        }
        for split in _SPLITS
    }
    records_fingerprint = fingerprint(
        [
            {
                "citation_context_id": row.citation_context_id,
                "split": row.source_split,
                "label_id": row.label_id,
                "text_sha256": _sha256_text(row.original_text),
                "source_record_id": str(row.source_record_id)
                if row.source_record_id is not None
                else None,
            }
            for row in all_records
        ]
    )
    profile = {
        "source": "published_stage03_citation_contexts",
        "stage03_run_id": manifest.get("run_id"),
        "stage03_run_dir": str(run),
        "source_table": table.name,
        "source_table_sha256": file_sha256(table),
        "stage03_total_cleaned_records": manifest.get("table_counts", {}).get("cleaned_records"),
        "stage03_nonquarantined_citation_contexts": manifest.get("table_counts", {}).get(
            "citation_contexts"
        ),
        "stage03_quarantined_records": manifest.get("table_counts", {}).get("quarantine", 0),
        "rows_read": total_rows,
        "records_admitted": len(all_records),
        "records_excluded_by_stage04": len(exclusions),
        "split_counts": {split: len(ordered[split]) for split in _SPLITS},
        "label_distribution": label_distribution,
        "label_mapping": dict(LABEL_TO_ID),
        "dataset_fingerprint": records_fingerprint,
        "original_text_preserved_in_stage03": True,
        "model_text_normalization": "NFC plus whitespace collapse only",
        "paper_identity_linkage": {
            "linked_contexts": sum(
                bool(record.citing_paper_id and record.cited_paper_id) for record in all_records
            ),
            "unlinked_contexts": sum(
                not (record.citing_paper_id and record.cited_paper_id) for record in all_records
            ),
            "interpretation": "Missing endpoints remain null and are not manufactured.",
        },
    }
    dataset_manifest = {
        "schema_version": "1.0",
        "stage": 4,
        "source_dataset": "scicite",
        "stage03_run_id": manifest.get("run_id"),
        "stage03_run_dir": str(run),
        "stage03_manifest_sha256": file_sha256(run / "integration_manifest.json"),
        "citation_context_table": table.name,
        "citation_context_table_sha256": profile["source_table_sha256"],
        "dataset_fingerprint": records_fingerprint,
        "label_mapping": dict(LABEL_TO_ID),
        "official_splits_preserved": True,
        "split_counts": profile["split_counts"],
        "label_distribution": label_distribution,
        "real_sources": manifest.get("real_sources", []),
        "synthetic_sources": manifest.get("synthetic_sources", []),
        "excluded_records": exclusions,
        "source_provenance_policy": "Stage 03 provenance is copied into every prediction export.",
    }
    return PreparedIntentData(
        stage03_run_dir=run,
        records=ordered,
        dataset_manifest=dataset_manifest,
        dataset_profile=profile,
        leakage_audit={
            "stage04_model_view": leakage,
            "stage03_source_view": manifest.get("split_audit", {}),
        },
        temporal_validation=_temporal_validation(all_records, manifest),
        exclusions=tuple(exclusions),
    )


def quick_training_subset(
    records: Iterable[IntentDatasetRecord], *, limit: int
) -> tuple[IntentDatasetRecord, ...]:
    """Choose a deterministic label-balanced development subset from train-only rows."""
    rows = tuple(sorted(records, key=lambda row: row.citation_context_id))
    if limit < 1:
        raise IntentDataError("quick training limit must be positive")
    if len(rows) <= limit:
        return rows
    buckets: dict[str, list[IntentDatasetRecord]] = defaultdict(list)
    for row in rows:
        buckets[row.normalized_label].append(row)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for label in INTENT_LABELS:
            if buckets[label] and len(selected) < limit:
                selected.append(buckets[label].pop(0))
    return tuple(selected)


def leakage_controlled_train_view(
    train: Iterable[IntentDatasetRecord], evaluation: Iterable[IntentDatasetRecord]
) -> tuple[IntentDatasetRecord, ...]:
    """Offer an opt-in sensitivity view without changing official benchmark assignments."""
    evaluation_hashes = {
        _sha256_text(normalized_context(record.original_text)) for record in evaluation
    }
    return tuple(
        record
        for record in train
        if _sha256_text(normalized_context(record.original_text)) not in evaluation_hashes
    )
