"""Evidence-backed schema, quality, compatibility and readiness reports in JSON/Markdown."""

import json
from pathlib import Path
from uuid import uuid4

from ..pipeline import fingerprint
from .identifier_audit import audit_indices
from .manifests import DatasetStatus, atomic_json, now

TARGETS = {
    "scicite": "Stage 04 citation-intent training",
    "arxiv-metadata": "Stage 05 topic identification",
    "mag-year-h2gb": "Stage 06 heterogeneous graph (requested readiness label)",
    "ogbn-arxiv": "Stage 07 baseline/prototype (requested readiness label)",
}


def _write_pair(root, name, document, body):
    atomic_json(root / (name + ".json"), document)
    # The immutable generation JSON is the audit source; Markdown contains interpretation.
    target = root / (name + ".md")
    import os
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def generate_reports(service):
    root = service.output_dir
    root.mkdir(parents=True, exist_ok=True)
    reports = service.inspection_reports()
    metadata = {
        "generated_at": now(),
        "generation_id": str(uuid4()),
        "config": service.config.model_dump(mode="json"),
        "config_sha256": fingerprint(service.config),
        "data_dir": str(service.data_dir),
        "output_dir": str(root),
        "scientific_stages_executed": [],
        "no_empirical_model_results": True,
    }
    schemas, quality, readiness = {}, {}, {}
    schema_md = [
        "# Dataset schema report",
        "",
        "Only file/API observations are measured schema. Documentation is separately labelled.",
        "",
    ]
    quality_md = [
        "# Data quality report",
        "",
        "Inspection counts are scoped to the selected records, never extrapolated.",
        "",
    ]
    ready_md = [
        "# Stage 02 readiness",
        "",
        "Readiness is for data handoff, not model validity or verified research gaps.",
        "",
        "| Dataset | Readiness | Loading status | Downstream use |",
        "| --- | --- | --- | --- |",
    ]
    from .registry import builtin_registry

    catalogue = builtin_registry()
    for dataset, state in service.registry.states.items():
        observed = reports.get(dataset)
        schemas[dataset] = {
            "documented_expectations": catalogue.get(dataset).schema_expectations,
            "observed": observed.get("observed_schema", observed) if observed else None,
            "unavailable_or_unverified": list(state.errors)
            + ([] if observed else ["actual dataset schema not inspected"]),
        }
        quality[dataset] = {
            "status": state.status.value,
            "observed": (state.quality or None) if observed else None,
            "scope": observed.get("scope") if observed else None,
            "limitations": list(state.limitations),
            "errors": list(state.errors),
        }
        if state.status == DatasetStatus.BLOCKED:
            ready = "BLOCKED"
            reason = "; ".join(state.errors)
        elif state.status == DatasetStatus.FAILED or observed is None or observed.get("synthetic"):
            ready, reason = "NOT_VALIDATED", "No real source inspection available."
        elif state.status != DatasetStatus.VALIDATED or state.version_id is None:
            ready, reason = (
                "PARTIALLY_READY",
                (
                    "Inspected evidence exists; resolve version, quality, scope and temporal "
                    "limitations before modeling."
                ),
            )
        else:
            ready, reason = (
                "READY",
                (
                    "Executed structural checks passed for the declared scope; scientific "
                    "validation remains separate."
                ),
            )
        readiness[dataset] = {
            "readiness": ready,
            "reason": reason,
            "loading_status": state.status.value,
            "target": TARGETS.get(dataset, "optional metadata enrichment"),
            "source": state.source_url,
            "licence_notes": state.licence_notes,
            "resource_requirements": "text streams bounded by record size; SQLite grows on disk"
            if dataset in {"scicite", "arxiv-metadata"}
            else (
                "graph arrays require a reviewed dependency and memory budget; no graph "
                "loaded unless explicitly configured"
            ),
            "integration_limitations": list(state.limitations),
            "inspection_path": state.inspection_path,
        }
        ready_md.append(
            f"| {dataset} | {ready} | {state.status.value} | {TARGETS.get(dataset, 'optional')} |"
        )
        schema_md.extend(
            [
                f"## {dataset}",
                "",
                f"Source: {state.source_url}",
                "",
                "Documented expectations: " + ", ".join(catalogue.get(dataset).schema_expectations),
                "",
            ]
        )
        if observed:
            fields = observed.get("observed_schema", {}).get("fields", {})
            schema_md.extend(
                [
                    "Observed fields/types: "
                    + (
                        ", ".join(
                            f"{field} ({'/'.join(info['types'])})" for field, info in fields.items()
                        )
                        if fields
                        else "see observed graph schema in JSON"
                    ),
                    "",
                    "Scope: `"
                    + json.dumps(
                        observed.get("scope", observed.get("inspection_config")), sort_keys=True
                    )
                    + "`",
                    "",
                ]
            )
            for discrepancy in observed.get("documentation_discrepancies", []):
                schema_md.extend(["Documentation discrepancy: " + discrepancy, ""])
        else:
            schema_md.extend(["Actual fields, node/edge types and counts: **unverified**.", ""])
        quality_md.extend([f"## {dataset}", "", f"Status: {state.status.value}", ""])
        if observed and "raw_record_count" in state.quality:
            q = state.quality
            quality_md.extend(
                [
                    (
                        f"Inspected {q['raw_record_count']} records: "
                        f"{q['valid_record_count']} structurally valid, "
                        f"{q['invalid_record_count']} invalid/flagged."
                    ),
                    "",
                    f"Issues: `{json.dumps(q['issues'], sort_keys=True)}`",
                    "",
                    (
                        f"Exact duplicate records: {q['duplicate_records']}; repeated text "
                        f"occurrences: {q['duplicate_contexts']}; cross-split exact-text "
                        f"groups: {q['cross_split_exact_text_overlap_groups']}."
                    ),
                    "",
                    (
                        "Risk: shared split text can inflate later classification evaluation "
                        "(high severity). Preserve splits and define an independent "
                        "leakage-handling protocol before training."
                    )
                    if q["cross_split_exact_text_overlap_groups"]
                    else "No cross-split overlap was observed within this inspection scope.",
                    "",
                    (
                        "Repeated raw IDs are candidate-key findings, not proof of duplicate "
                        "scholarly entities; examine their source grain before Stage 03."
                    ),
                    "",
                ]
            )
        quality_md.extend(list(state.errors) or ["No additional failure details."])
        quality_md.append("")
    indices = {
        dataset: Path(state.index_path)
        for dataset, state in service.registry.states.items()
        if state.index_path
        and Path(state.index_path).exists()
        and dataset in reports
        and not reports[dataset].get("synthetic")
    }
    audit = audit_indices(indices, root / ("identifier-audit-" + str(uuid4()) + ".sqlite"))
    audit["missing_mappings"] = [dataset for dataset in TARGETS if dataset not in indices]
    audit_md = [
        "# Identifier compatibility",
        "",
        audit["scope"],
        "",
        audit["exact_meaning"],
        "",
        "Audited datasets: " + (", ".join(audit["datasets_audited"]) or "none"),
        "",
    ]
    if len(indices) < 2:
        audit_md.extend(
            [
                (
                    "Cross-dataset overlap is **not measured**: fewer than two real source "
                    "identifier indexes are available."
                ),
                "",
            ]
        )
    for pair in audit["pairs"]:
        audit_md.extend(
            [
                f"## {pair['left']} / {pair['right']}",
                "",
                (
                    f"Exact shared unique identifier keys: {pair['EXACT_VERIFIED']}; "
                    f"ambiguous keys: {pair['AMBIGUOUS']}; title-only candidates: "
                    f"{pair['CANDIDATE_title_keys']}."
                ),
                "",
            ]
        )
    audit_md.extend(audit["limitations"])
    ready_md.extend(["", "## Reasons and next actions", ""])
    for dataset, result in readiness.items():
        ready_md.append(f"- **{dataset}**: {result['reason']}")
    ready_md.extend(
        [
            "",
            (
                "The attachment's downstream stage numbers differ from the existing proposed "
                "pipeline. Requested readiness labels are retained here; no later stage was "
                "renumbered or executed."
            ),
            "",
            (
                "Before Stage 03: approve temporal policies, examine invalid records and "
                "candidate keys, verify actual mappings and snapshot versions. No cleaning, "
                "merging, resplitting or training was performed."
            ),
        ]
    )
    documents = {
        "dataset_schema_report": ({**metadata, "datasets": schemas}, "\n".join(schema_md)),
        "data_quality_report": ({**metadata, "datasets": quality}, "\n".join(quality_md)),
        "identifier_compatibility": ({**metadata, **audit}, "\n".join(audit_md)),
        "stage02_readiness": ({**metadata, "datasets": readiness}, "\n".join(ready_md)),
    }
    for name, (document, body) in documents.items():
        _write_pair(root, name, document, body)
    return {
        "reports": {name: str(root / (name + ".json")) for name in documents},
        "readiness": readiness,
        "generation_id": metadata["generation_id"],
    }
