"""Stage 03 reports derived exclusively from executed run observations."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .integration_manifest import TABLES, iter_table
from .manifests import atomic_json, now
from .provenance import file_sha256

REPORTS = (
    "stage03_cleaning_report",
    "stage03_entity_resolution",
    "stage03_integration_coverage",
    "stage03_temporal_leakage",
    "stage03_graph_integrity",
    "stage03_readiness",
)


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def make_reports(root: Path, manifest: dict, *, run_dir: Path):
    sources = manifest["sources"]
    counts = manifest["table_counts"]
    linked = manifest["linkage"]
    context_count = linked.get("citation_contexts", 0)
    canonical = counts.get("papers", 0)
    metadata = {
        "schema_version": "1.0",
        "generated_at": now(),
        "run_dir": str(run_dir),
        "action": manifest["action"],
        "real_sources": manifest["real_sources"],
        "synthetic_sources": manifest["synthetic_sources"],
        "observation_kind": "mixed_real_and_synthetic"
        if manifest["real_sources"] and manifest["synthetic_sources"]
        else "real_observed"
        if manifest["real_sources"]
        else "synthetic_software_fixture",
        "blocked_sources": {
            key: value for key, value in sources.items() if value["status"] == "BLOCKED"
        },
        "assumptions": manifest["limitations"],
        "unverified": [
            "External network identity authentication",
            "Unknown dataset releases",
            "Historical availability without record-specific evidence",
        ],
        "scientific_stages_executed": [],
        "no_model_results": True,
    }
    clusters = cross_source_clusters = cross_source_occurrences = 0
    if manifest["completed_phase"] >= 4:
        for paper in iter_table(root, "papers"):
            clusters += 1
            datasets = {ref["source_dataset"] for ref in paper.get("source_record_ids", [])}
            if len(datasets) > 1:
                cross_source_clusters += 1
                cross_source_occurrences += len(paper["source_record_ids"])
    coverage = {
        "records_ingested_per_source": {
            key: value["observed_records"]
            for key, value in sources.items()
            if value["status"] == "PROCESSED"
        },
        "table_counts": {table: counts.get(table, 0) for table in TABLES},
        "canonical_paper_clusters": clusters,
        "verified_cross_source_clusters": cross_source_clusters,
        "paper_record_occurrences_in_cross_source_clusters": cross_source_occurrences,
        "cross_dataset_coverage": cross_source_occurrences / counts["identifier_mappings"]
        if counts.get("identifier_mappings")
        else None,
        "cross_dataset_coverage_denominator": (
            "processed valid paper-record occurrences with identity mappings; "
            "unavailable source datasets excluded"
        ),
        "source_local_papers": counts.get("source_local_papers", 0),
        "resolved_citation_endpoints": linked.get("resolved_citation_endpoints", 0),
        "unresolved_citation_endpoints": linked.get("unresolved_citation_endpoints", 0),
        "context_linkage": {
            "linked": linked.get("linked_contexts", 0),
            "denominator": context_count,
            "rate": linked.get("linked_contexts", 0) / context_count if context_count else None,
            "scope": "nonquarantined contexts retained for independent intent data",
        },
        "heterogeneous_graph_linkage_coverage": None,
        "heterogeneous_graph_linkage_reason": (
            "no verified cross-source heterogeneous mapping supplied"
        )
        if not cross_source_clusters
        else "paper identity coverage does not prove heterogeneous node linkage",
    }
    readiness = {
        "stage03": "PARTIALLY_READY"
        if manifest["real_sources"] and manifest["completed_phase"] >= 6
        else "NOT_VALIDATED",
        "stage04_citation_intent": "PARTIALLY_READY"
        if counts.get("citation_contexts", 0) and "scicite" in manifest["real_sources"]
        else "BLOCKED",
        "stage05_paper_metadata": "PARTIALLY_READY"
        if canonical and manifest["real_sources"]
        else "BLOCKED",
        "verified_graph_relationships": "PARTIALLY_READY"
        if manifest["graph_integrity"].get("valid_edge_occurrences", 0)
        else "BLOCKED",
        "strict_forecasting_readiness": "NOT_ESTABLISHED",
        "stage04_checklist": {
            "contexts_and_observed_three_class_labels_retained": bool(
                counts.get("citation_contexts", 0)
            ),
            "official_splits_preserved": not manifest["split_audit"].get(
                "official_splits_modified", False
            ),
            "quarantine_recoverable": True,
            "resolve_cross_split_exact_and_near_overlap_policy": "PENDING_REVIEW",
            "verify_source_release_and_licence_attribution": "PENDING_REVIEW",
            "document_train_only_fit_and_independent_evaluation_protocol": (
                "PENDING_STAGE04_APPROVAL"
            ),
            "citation_intent_is_not_direct_investigation": True,
        },
        "later_stages_started": False,
    }
    documents = {
        "stage03_cleaning_report": manifest["cleaning"],
        "stage03_entity_resolution": manifest["identity"],
        "stage03_integration_coverage": coverage,
        "stage03_temporal_leakage": {
            "temporal": manifest["temporal"],
            "split_audit": manifest["split_audit"],
        },
        "stage03_graph_integrity": {
            **manifest["graph_integrity"],
            "independent_context_edges_retained": linked.get(
                "edges_retained_outside_canonical_graph", 0
            ),
            "scope": (
                "canonical graph audits include only resolved context edges; "
                "unresolved independent context links are counted separately"
            ),
        },
        "stage03_readiness": readiness,
    }
    for name, observed in documents.items():
        document = {**metadata, "observed": observed}
        atomic_json(root / (name + ".json"), document)
        body = "# " + name.replace("_", " ") + "\n\n"
        body += f"Run: `{run_dir.name}`. Evidence: **{metadata['observation_kind']}**.\n\n"
        body += (
            "Counts cover only the configured and actually processed inputs; "
            "missing data are not zero-overlap findings.\n\n"
        )
        body += "```json\n" + json.dumps(observed, ensure_ascii=False, indent=2) + "\n```\n\n"
        if name == "stage03_temporal_leakage":
            body += (
                "Publication and ingestion dates do not establish historical citation or feature "
                "visibility. Excluded records remain recoverable. Official source splits were "
                "retained; shared paper identities are an experiment-dependent risk. "
                "Near-duplicate retrieval is bounded and approximate.\n\n"
            )
        if name == "stage03_entity_resolution":
            body += (
                "EXACT_VERIFIED refers to compatible source-asserted authoritative identifiers "
                "or documented mappings. Heuristic scores rank candidates and are not calibrated "
                "probabilities. Review-only matches remain separate.\n\n"
            )
        body += "Blocked sources:\n\n"
        body += (
            "\n".join(
                f"- **{key}**: {value['reason']}"
                for key, value in metadata["blocked_sources"].items()
            )
            + "\n\n"
        )
        body += (
            "Limitations:\n\n" + "\n".join("- " + text for text in manifest["limitations"]) + "\n"
        )
        atomic_text(root / (name + ".md"), body)


def publish_reports(run_dir: Path, report_dir: Path):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in REPORTS:
        atomic_json(
            report_dir / (name + ".json"),
            json.loads((run_dir / (name + ".json")).read_text(encoding="utf-8")),
        )
        atomic_text(
            report_dir / (name + ".md"), (run_dir / (name + ".md")).read_text(encoding="utf-8")
        )
    return {name: str(report_dir / (name + ".json")) for name in REPORTS}


def validate_run(run_dir: Path):
    manifest = json.loads((run_dir / "integration_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest["output_checksums"].items():
        path = (run_dir / relative).resolve()
        if (
            not path.is_relative_to(run_dir.resolve())
            or not path.is_file()
            or file_sha256(path) != expected
        ):
            mismatches.append(relative)
    quarantined = manifest["table_counts"].get("quarantine", 0)
    invalid_edges = manifest["graph_integrity"].get("invalid_edge_occurrences", 0)
    temporal_violations = manifest["temporal"].get("eligible_with_future_violation", 0)
    return {
        "run_dir": str(run_dir),
        "checksum_mismatches": mismatches,
        "quarantined_records": quarantined,
        "invalid_canonical_graph_edges": invalid_edges,
        "eligible_future_violations": temporal_violations,
        "valid": not (mismatches or quarantined or invalid_edges or temporal_violations),
        "interpretation": (
            "structural/data-quality checks; exclusions and partial linkage "
            "remain scientifically explicit"
        ),
    }
