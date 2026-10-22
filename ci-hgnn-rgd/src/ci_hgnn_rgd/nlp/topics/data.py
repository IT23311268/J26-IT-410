"""Checksum-verified Stage 03 paper views, bounded sampling and temporal reassessment."""

import heapq
import json
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ...data.integration_manifest import resolve_run
from ...data.provenance import file_sha256
from ...data.temporal_integrity import assess_temporal
from ...pipeline import fingerprint
from ..intent_contracts import normalize_context_text
from .config import Stage05Config
from .contracts import PreparedPaper


@dataclass(frozen=True)
class CorpusView:
    papers: tuple[PreparedPaper, ...]
    manifest: dict
    profile: dict
    exclusions: tuple[dict, ...]
    temporal_snapshot: dict


def temporal_evidence(row: dict) -> tuple[dict, ...]:
    assessment = row.get("temporal_eligibility") or {}
    if assessment.get("contributors"):
        return tuple(
            (item.get("eligibility") or {}).get("evidence", {})
            for item in assessment["contributors"]
        )
    if isinstance(assessment.get("evidence"), dict):
        return (assessment["evidence"],)
    # Stage 03 canonical contributors are preferred. Missing availability is
    # deliberately retained; ingestion is never substituted for availability.
    return (
        {
            field: row.get(field)
            for field in (
                "publication_date",
                "publication_year",
                "date_precision",
                "available_at",
                "version_date",
                "update_date",
                "feature_available_at",
                "enrichment_date",
            )
        },
    )


def prepare_paper(row: dict, config: Stage05Config, *, policy: str, cutoff) -> PreparedPaper:
    if not isinstance(row, dict) or row.get("quarantine"):
        raise ValueError("malformed_or_quarantined_paper")
    title, abstract = row.get("title"), row.get("abstract")
    if title is not None and not isinstance(title, str):
        raise ValueError("invalid_title_type")
    if abstract is not None and not isinstance(abstract, str):
        raise ValueError("invalid_abstract_type")
    title = title or ""
    clean_title = normalize_context_text(title)
    clean_abstract = normalize_context_text(abstract) if abstract else ""
    if not clean_title and not clean_abstract:
        raise ValueError("empty_scientific_text")
    if not clean_abstract and not config.dataset.allow_title_only:
        raise ValueError("missing_abstract_title_only_disabled")
    if not clean_title and not config.dataset.allow_abstract_only:
        raise ValueError("missing_title_abstract_only_disabled")
    if "\ufffd" in title + (abstract or "") or any(
        unicodedata.category(char) == "Cs" for char in title + (abstract or "")
    ):
        raise ValueError("unrecoverable_encoding_requires_review")
    provenance = row.get("provenance") or []
    if (
        not isinstance(provenance, list)
        or not provenance
        or any(not isinstance(item, dict) for item in provenance)
    ):
        raise ValueError("missing_or_invalid_source_provenance")
    language = row.get("language")
    if config.dataset.allowed_languages and language not in config.dataset.allowed_languages:
        raise ValueError("unsupported_or_unknown_language")
    if len(title) + len(abstract or "") > config.dataset.max_text_characters:
        raise ValueError("text_exceeds_character_budget_not_silently_truncated")
    refs = row.get("source_record_ids") or []
    ref = refs[0] if refs else {}
    dataset = row.get("source_dataset") or ref.get("source_dataset")
    if not dataset:
        raise ValueError("missing_source_dataset")
    source_id = row.get("source_record_id", ref.get("source_record_id"))
    version = row.get("dataset_version", ref.get("dataset_version"))
    canonical_id = row.get("canonical_paper_id")
    source_key = row.get("record_key") or ref.get("record_key")
    if canonical_id is None and source_id is None and source_key is None:
        raise ValueError("missing_source_record_identifier")
    paper_key = canonical_id or "source-local:" + fingerprint(
        [dataset, version, source_id, source_key]
    )
    evidences = temporal_evidence(row)
    assessments = [assess_temporal(evidence, cutoff, policy) for evidence in evidences]
    if any(a["invalid_temporal_fields"] for a in assessments):
        raise ValueError("invalid_source_temporal_metadata")
    if policy != "EXPLORATORY_NO_TEMPORAL_CLAIM" and not all(a["eligible"] for a in assessments):
        reasons = sorted({r for a in assessments for r in a["reasons"]})
        raise ValueError("temporally_ineligible:" + ",".join(reasons))
    text = clean_title + " [SEP] " + clean_abstract if clean_abstract else clean_title
    mode = (
        "title_abstract"
        if clean_title and clean_abstract
        else "title_only"
        if clean_title
        else "abstract_only"
    )
    year = row.get("publication_year")
    if year is not None and (type(year) is not int or not 1000 <= year <= 9999):
        raise ValueError("invalid_publication_year")
    year = year if type(year) is int and 1000 <= year <= 9999 else None
    return PreparedPaper(
        paper_key=paper_key,
        canonical_paper_id=canonical_id,
        source_dataset=dataset,
        source_record_id=source_id,
        dataset_version=version,
        title=clean_title,
        abstract=clean_abstract or None,
        model_text=text,
        original_title=title,
        original_abstract=abstract,
        text_mode=mode,
        publication_year=year,
        publication_date=row.get("publication_date"),
        date_precision=row.get("date_precision") or "unknown",
        categories=tuple(row.get("categories", row.get("subject_categories")) or []),
        authors=tuple(row.get("authors") or []),
        external_ids=row.get("external_ids"),
        language=language,
        identity_resolution_status=row.get(
            "identity_resolution_status", row.get("identity_status", "SOURCE_LOCAL_UNVERIFIED")
        ),
        provenance=tuple(provenance),
        source_temporal_evidence=evidences,
        temporal_assessment={
            "policy": policy,
            "cutoff": str(cutoff),
            "assessments": assessments,
            "strict_historical_claim": all(a["strict_historical_claim"] for a in assessments),
        },
        text_sha256=fingerprint([title, abstract, "NFC_whitespace_v1"]),
    )


def prepare_corpus(
    config: Stage05Config, *, exclusions_path: Path | None = None, allow_synthetic_fixtures=False
) -> CorpusView:
    run = resolve_run(config.dataset.stage03_root, config.dataset.stage03_run)
    manifest = json.loads((run / "integration_manifest.json").read_text(encoding="utf-8"))
    temporal_config = manifest.get("config", {}).get("temporal_integrity", {})
    policy = config.temporal.policy or temporal_config.get("policy", "STRICT_POINT_IN_TIME")
    cutoff = config.temporal.cutoff or temporal_config.get("cutoff")
    if policy != "EXPLORATORY_NO_TEMPORAL_CLAIM" and cutoff is None:
        # Some older runs persist cutoff in the source assessments only.
        cutoff = manifest.get("temporal", {}).get("cutoff")
        if cutoff is None:
            raise ValueError("historical Stage 05 needs --temporal-cutoff; Stage 03 has no cutoff")
    counts = Counter()
    samples = []
    heap = []
    tables = []
    if exclusions_path:
        exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    stream = exclusions_path.open("x", encoding="utf-8") if exclusions_path else None

    def excluded(row, reason):
        counts[reason.split(":")[0]] += 1
        record = {
            "paper_key": row.get("canonical_paper_id", row.get("record_key"))
            if isinstance(row, dict)
            else None,
            "reason": reason,
        }
        if len(samples) < 100:
            samples.append(record)
        if stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        with tempfile.TemporaryDirectory(prefix="stage05-identity-") as directory:
            with closing(sqlite3.connect(Path(directory) / "seen.sqlite")) as db:
                db.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")
                for table in ("papers", "source_local_papers"):
                    path = run / (table + ".jsonl")
                    if not path.is_file():
                        continue
                    expected = manifest.get("output_checksums", {}).get(path.name)
                    if not expected or file_sha256(path) != expected:
                        raise ValueError(
                            f"Stage 03 checksum mismatch or missing checksum: {path.name}"
                        )
                    tables.append({"path": path.name, "sha256": expected})
                    with path.open(encoding="utf-8") as source:
                        for line in source:
                            counts["rows_read"] += 1
                            row = None
                            try:
                                row = json.loads(line)
                                paper = prepare_paper(row, config, policy=policy, cutoff=cutoff)
                                if not allow_synthetic_fixtures and any(
                                    ref.get("synthetic") for ref in paper.provenance
                                ):
                                    raise ValueError("synthetic_source_not_research_corpus")
                                if db.execute(
                                    "SELECT 1 FROM seen WHERE id=?", (paper.paper_key,)
                                ).fetchone():
                                    excluded(row, "duplicate_paper_identity")
                                    continue
                                db.execute("INSERT INTO seen VALUES (?)", (paper.paper_key,))
                            except (ValueError, TypeError) as error:
                                excluded(row, str(error))
                                continue
                            counts["eligible_unique_papers"] += 1
                            rank = int(fingerprint([config.seed, paper.paper_key]), 16)
                            item = (-rank, paper.paper_key, paper)
                            if len(heap) < config.dataset.max_records:
                                heapq.heappush(heap, item)
                            elif item[:2] > heap[0][:2]:
                                removed = heapq.heapreplace(heap, item)[2]
                                excluded(
                                    {"canonical_paper_id": removed.paper_key},
                                    "excluded_by_bounded_sampling",
                                )
                            else:
                                excluded(row, "excluded_by_bounded_sampling")
    finally:
        if stream:
            stream.close()
    papers = tuple(sorted((item[2] for item in heap), key=lambda p: p.paper_key))
    corpus_hash = fingerprint([(p.paper_key, p.text_sha256) for p in papers])
    snapshot = {
        "policy": policy,
        "cutoff": str(cutoff) if cutoff else None,
        "snapshot_id": "snapshot:" + fingerprint([policy, str(cutoff), corpus_hash])[:20],
        "strict_historical_claim": bool(papers)
        and all(p.temporal_assessment["strict_historical_claim"] for p in papers),
        "corpus_frozen": True,
    }
    return CorpusView(
        papers,
        {
            "stage03_run_id": manifest["run_id"],
            "stage03_run_dir": str(run),
            "stage03_temporal_audit": manifest.get("temporal", {}),
            "tables": tables,
            "corpus_sha256": corpus_hash,
            "sampling": "seeded_hash_bottom_k",
            "max_records": config.dataset.max_records,
        },
        {
            "counts": dict(counts),
            "modelling_papers": len(papers),
            "text_modes": dict(Counter(p.text_mode for p in papers)),
            "source_datasets": dict(Counter(p.source_dataset for p in papers)),
            "status": "READY" if papers else "BLOCKED_NO_ELIGIBLE_PAPER_TEXT",
            "blocker": None
            if papers
            else (
                "Supply Stage 03 canonical papers with titles/abstracts; "
                "citation contexts are not paper abstracts."
            ),
            "exclusions_sampled": len(samples),
            "full_exclusions_path": str(exclusions_path) if exclusions_path else None,
        },
        tuple(samples),
        snapshot,
    )
