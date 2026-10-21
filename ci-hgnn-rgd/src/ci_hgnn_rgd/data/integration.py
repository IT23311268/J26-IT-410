"""Offline Stage 03 orchestration: preserve evidence, resolve conservatively, publish atomically."""

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .. import __version__
from ..pipeline import canonical_json, fingerprint
from .integration_manifest import RunPublication, TableStore
from .manifests import now
from .provenance import file_sha256
from .stage03_config import InputSource, Stage03Config

ALIASES = {"arxiv": "arxiv-metadata", "mag_year": "mag-year-h2gb", "ogbn_arxiv": "ogbn-arxiv"}
DATASETS = (
    "scicite",
    "arxiv-metadata",
    "mag-year-h2gb",
    "ogbn-arxiv",
    "openalex",
    "semantic-scholar",
)
PHASES = {
    "clean": 1,
    "normalize-identifiers": 2,
    "generate-match-candidates": 3,
    "resolve-entities": 4,
    "deduplicate": 5,
    "integrate": 6,
    "stage03-run": 6,
}
LOGGER = logging.getLogger("ci_hgnn_rgd")


def select_sources(project, config: Stage03Config, *, dataset=None, path=None):
    """Read existing registry state without mutating Stage 02 or acquiring data."""
    requested = ALIASES.get(dataset, dataset)
    if requested is not None and requested not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    registry_path = project.paths.reports / "dataset_registry.json"
    states = (
        json.loads(registry_path.read_text(encoding="utf-8"))["datasets"]
        if registry_path.exists()
        else {}
    )
    configured = {source.dataset: source for source in config.integration.inputs}
    selected, statuses = [], {}
    for name in DATASETS:
        if requested is not None and name != requested:
            statuses[name] = {"status": "NOT_SELECTED", "reason": "not selected for this run"}
            continue
        source = configured.get(name)
        state = states.get(name, {})
        if requested == name and path is not None:
            source = InputSource(
                dataset=name,
                path=Path(path).resolve(),
                version=source.version if source else state.get("version_id"),
                synthetic=source.synthetic if source else False,
                format=source.format if source else "raw",
            )
        if source is None and not configured and config.integration.discover_stage02:
            local = state.get("local_path")
            source = InputSource(
                dataset=name, path=Path(local) if local else None, version=state.get("version_id")
            )
        if source is None or source.path is None or not source.path.exists():
            reason = "no configured local source file"
            if state.get("errors"):
                reason += "; " + "; ".join(state["errors"])
            statuses[name] = {"status": "BLOCKED", "reason": reason}
            if requested == name or (source is not None and source.path is not None):
                raise ValueError(f"{name}: {reason}")
            continue
        if name in {"ogbn-arxiv", "mag-year-h2gb"} and source.format != "graph_tables":
            statuses[name] = {
                "status": "BLOCKED",
                "reason": (
                    "graph integration requires an explicit JSONL graph_tables export; "
                    "binary caches are not deserialized"
                ),
            }
            if requested == name:
                raise ValueError(statuses[name]["reason"])
            continue
        selected.append(source)
        statuses[name] = {
            "status": "AVAILABLE",
            "path": str(source.path),
            "version": source.version,
            "format": source.format,
            "synthetic": source.synthetic,
            "licence_notes": state.get("licence_notes", "verify chosen source release terms"),
            "citation_requirements": state.get(
                "citation_requirements", "retain upstream attribution and source release citation"
            ),
            "source_url": state.get("source_url"),
        }
    if not selected:
        raise ValueError("no genuine or explicitly synthetic local inputs configured for Stage 03")
    return selected, statuses


def _input_hash(path: Path):
    if path.is_file():
        return file_sha256(path)
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix in {".jsonl", ".json"}
    )
    return fingerprint([(str(item.relative_to(path)), file_sha256(item)) for item in files])


def _temporal_evidence(row):
    from .temporal_integrity import parse_cutoff

    evidence = {
        field: row.get(field)
        for field in (
            "publication_date",
            "publication_year",
            "date_precision",
            "version_date",
            "update_date",
            "snapshot_date",
            "available_at",
            "citation_observed_at",
            "feature_available_at",
            "enrichment_date",
        )
    }
    provenances = row.get("provenance") or []
    evidence["snapshot_date"] = evidence.get("snapshot_date") or row.get("dataset_snapshot_date")
    evidence["update_date"] = evidence.get("update_date") or row.get("source_updated_date")
    evidence["training_data_cutoff"] = row.get("training_data_cutoff")
    # All contributing versions must be eligible. The latest known date is conservative.
    for key, aliases in {
        "available_at": ("available_at",),
        "snapshot_date": ("snapshot_date", "dataset_snapshot_date"),
        "ingested_at": ("acquired_at", "inspected_at"),
    }.items():
        values = [
            str(p[alias])
            for p in provenances
            for alias in aliases
            if isinstance(p, dict) and p.get(alias)
        ]
        if values and evidence.get(key) is None:
            try:
                evidence[key] = max(values, key=parse_cutoff)
            except (ValueError, TypeError, OverflowError):
                evidence[key] = "invalid_temporal_provenance"
    raw = row.get("raw_values") or {}
    # Later metadata/version dates may never be silently reused as earlier features.
    for key in (
        "update_date",
        "version_date",
        "enrichment_date",
        "citation_observed_at",
        "feature_available_at",
    ):
        if evidence.get(key) is None and isinstance(raw, dict):
            evidence[key] = raw.get(key, raw.get("updated_date") if key == "update_date" else None)
    versions = row.get("versions") or []
    if versions and evidence.get("version_date") is None:
        from email.utils import parsedate_to_datetime

        dates = []
        for version in versions:
            if isinstance(version, dict) and version.get("created"):
                try:
                    dates.append(parsedate_to_datetime(version["created"]).isoformat())
                except (ValueError, TypeError, OverflowError):
                    dates.append(str(version["created"]))
        if dates:
            try:
                evidence["version_date"] = max(dates, key=parse_cutoff)
            except (ValueError, TypeError, OverflowError):
                evidence["version_date"] = "invalid_source_version_date"
    return evidence


def _scoped_id(identifier, dataset, version):
    from .normalization import identifier_token

    if not identifier:
        return None
    token = identifier_token(identifier)
    return canonical_json(token) if token is not None else None


def _lookup(db, identifier, dataset, version):
    scoped = _scoped_id(identifier, dataset, version)
    if scoped is None:
        return None
    results = db.execute(
        "SELECT DISTINCT canonical_id FROM endpoint_ids WHERE scoped_id=? LIMIT 2", (scoped,)
    ).fetchall()
    return results[0][0] if len(results) == 1 else None


def run_stage03(
    project, config: Stage03Config, *, action="stage03-run", dataset=None, path=None, dry_run=False
):
    """Execute a declared prefix of Stage 03; all rows remain outside later ML stages."""
    from .entity_resolution import IdentityEngine
    from .graph_integrity import GraphAuditor
    from .integration_adapters import iter_source_records
    from .split_management import SplitAuditor
    from .stage03_reporting import make_reports, publish_reports
    from .temporal_integrity import assess_temporal

    if action not in PHASES:
        raise ValueError("unknown Stage 03 processing action")
    phase = PHASES[action]
    inputs, statuses = select_sources(project, config, dataset=dataset, path=path)
    if dry_run:
        return {
            "dry_run": True,
            "action": action,
            "sources": statuses,
            "output_dir": str(config.integration.output_dir),
            "network_access": False,
            "scientific_stages_executed": [],
            "record_counts": "not measured in dry run",
        }
    publication = RunPublication(config.integration.output_dir, config.model_dump(mode="json"))
    store = engine = splits = graph = db = None
    start = now()
    input_hashes = {}
    counts = defaultdict(Counter)
    transformations, quarantine_reasons, missing_fields, temporal_counts = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    duplicate_count = 0
    linkage = Counter()
    try:
        store = TableStore(
            publication.pending,
            batch_size=config.integration.batch_size,
            storage_format=config.integration.storage_format,
            memory_budget_mb=config.integration.memory_budget_mb,
        )
        db = sqlite3.connect(publication.pending / "processing.sqlite")
        db.executescript("""
            PRAGMA cache_size=-4096; PRAGMA temp_store=FILE;
            CREATE TABLE records(record_key TEXT PRIMARY KEY,kind TEXT,payload TEXT);
            CREATE INDEX records_kind ON records(kind);
            CREATE TABLE raw_hashes(
              dataset TEXT,hash TEXT,record_key TEXT,PRIMARY KEY(dataset,hash));
            CREATE TABLE endpoint_ids(scoped_id TEXT,canonical_id TEXT);
            CREATE INDEX endpoint_lookup ON endpoint_ids(scoped_id);
        """)
        engine = IdentityEngine(
            publication.pending / "identity.sqlite",
            max_candidates=config.entity_resolution.max_candidates,
            memory_budget_mb=config.integration.memory_budget_mb,
            batch_size=config.integration.batch_size,
            title_review_threshold=config.entity_resolution.probable_title_threshold,
            material_title_conflict_threshold=config.entity_resolution.material_title_conflict_threshold,
            year_tolerance=config.entity_resolution.year_tolerance,
            candidate_methods=config.entity_resolution.candidate_methods,
            max_cluster_size=config.entity_resolution.max_cluster_size,
        )
        splits = SplitAuditor(
            publication.pending / "splits.sqlite",
            near_threshold=config.temporal_integrity.near_duplicate_threshold,
            max_candidates_per_context=config.temporal_integrity.near_candidates_per_context,
            max_comparisons=config.temporal_integrity.max_near_comparisons,
        )
        graph = GraphAuditor(publication.pending / "graph-integrity.sqlite")
        cutoff = config.temporal_integrity.cutoff or project.run.cutoff
        for source in inputs:
            input_hashes[source.dataset] = _input_hash(source.path)
            LOGGER.info(
                "Stage 03 reading dataset=%s max_records=%s",
                source.dataset,
                config.integration.max_records,
            )
            for row in iter_source_records(
                source.dataset,
                source.path,
                version=source.version,
                max_records=config.integration.max_records,
                synthetic=source.synthetic,
                max_line_bytes=project.data.max_line_bytes,
                format=source.format,
            ):
                count = counts[source.dataset]
                count["input_records"] += 1
                quarantined = bool(row.get("quarantine"))
                count["quarantined_records" if quarantined else "valid_records"] += 1
                if quarantined:
                    count["invalid_records"] += 1
                    quarantine_reasons.update(str(issue) for issue in row.get("issues", []))
                if row.get("transformations"):
                    count["corrected_records"] += 1
                    for transform in row["transformations"]:
                        transformations[
                            str(transform.get("operation", transform.get("method", transform)))
                            if isinstance(transform, dict)
                            else str(transform)
                        ] += 1
                missing_fields.update(str(field) for field in row.get("missing_fields", []))
                store.write("cleaned_records", row)
                store.write(
                    "provenance",
                    {
                        "record_key": row["record_key"],
                        "source_dataset": source.dataset,
                        "provenance": row.get("provenance", []),
                        "raw_values": row.get("raw_values"),
                        "transformations": row.get("transformations", []),
                        "input_sha256": input_hashes[source.dataset],
                    },
                )
                raw_hash = (
                    (row.get("provenance") or [{}])[0].get("record_sha256")
                    if row["kind"] == "invalid"
                    else fingerprint(row.get("raw_values"))
                )
                raw_hash = raw_hash or fingerprint([row["record_key"], row.get("raw_values")])
                prior = db.execute(
                    "SELECT record_key FROM raw_hashes WHERE dataset=? AND hash=?",
                    (source.dataset, raw_hash),
                ).fetchone()
                if prior:
                    duplicate_count += 1
                    count["duplicate_records"] += 1
                    store.write(
                        "match_decisions",
                        {
                            "kind": "within_source_duplicate",
                            "record_key": row["record_key"],
                            "duplicate_of": prior[0],
                            "decision": "RETAINED_OCCURRENCE",
                            "rule": "exact_raw_record_hash",
                            "sha256": raw_hash,
                        },
                    )
                else:
                    db.execute(
                        "INSERT INTO raw_hashes VALUES(?,?,?)",
                        (source.dataset, raw_hash, row["record_key"]),
                    )
                eligibility = assess_temporal(
                    _temporal_evidence(row),
                    cutoff,
                    config.temporal_integrity.policy,
                    kind=row["kind"],
                )
                eligibility.update(
                    record_key=row["record_key"], source_dataset=source.dataset, kind=row["kind"]
                )
                row["temporal_eligibility"] = eligibility
                temporal_counts["records_assessed"] += 1
                temporal_counts["eligible" if eligibility["eligible"] else "excluded"] += 1
                temporal_counts["unknown_availability"] += int(eligibility["unknown_availability"])
                temporal_counts["eligible_with_future_violation"] += int(
                    eligibility["eligible"]
                    and any("future" in reason for reason in eligibility["reasons"])
                )
                for reason in eligibility["reasons"]:
                    temporal_counts["reason:" + reason] += 1
                store.write("temporal_eligibility", eligibility)
                if not eligibility["eligible"]:
                    store.write(
                        "excluded_records",
                        {
                            "record_key": row["record_key"],
                            "kind": row["kind"],
                            "source_dataset": source.dataset,
                            "temporal_eligibility": eligibility,
                            "recovery_table": "cleaned_records.jsonl",
                        },
                    )
                if quarantined:
                    store.write("quarantine", row)
                    store.write(
                        "unresolved_records",
                        {
                            "record_key": row["record_key"],
                            "source_dataset": source.dataset,
                            "reason": "quarantine",
                            "issues": row.get("issues", []),
                            "recovery_table": "quarantine.jsonl",
                        },
                    )
                else:
                    db.execute(
                        "INSERT INTO records VALUES(?,?,?)",
                        (row["record_key"], row["kind"], canonical_json(row)),
                    )
                    if row["kind"] == "paper" and phase >= 3:
                        engine.add(row)
                if count["input_records"] % config.integration.batch_size == 0:
                    db.commit()
                    LOGGER.info(
                        "Stage 03 dataset=%s records=%s", source.dataset, count["input_records"]
                    )
            statuses[source.dataset].update(
                status="PROCESSED",
                input_sha256=input_hashes[source.dataset],
                observed_records=counts[source.dataset]["input_records"],
                max_records=config.integration.max_records,
            )
        db.commit()
        if sum(item["input_records"] for item in counts.values()) == 0:
            raise ValueError("configured inputs contained no readable source records")
        identity_summary = {
            "status": "NOT_EXECUTED",
            "reason": "requested cleaning/normalization prefix only",
        }
        if phase >= 3:
            for candidate in engine.iter_candidates():
                store.write("match_candidates", candidate)
        if phase >= 4:
            identity_summary = engine.finalize()
            for decision in engine.iter_decisions():
                store.write("match_decisions", decision)
            for mapping in engine.iter_mappings():
                store.write("identifier_mappings", mapping)
                key = mapping["record_key"]
                record = db.execute(
                    "SELECT payload FROM records WHERE record_key=?", (key,)
                ).fetchone()
                if record:
                    row = json.loads(record[0])
                    for identifier in row.get("identifiers", []):
                        if identifier.get("valid", True):
                            scoped = _scoped_id(
                                identifier, row["source_dataset"], row.get("dataset_version")
                            )
                            if scoped:
                                db.execute(
                                    "INSERT INTO endpoint_ids VALUES(?,?)",
                                    (scoped, mapping["canonical_paper_id"]),
                                )
            for paper in engine.iter_papers():
                store.write("papers", paper)
                if len({ref["source_dataset"] for ref in paper.get("source_record_ids", [])}) <= 1:
                    store.write("source_local_papers", paper)
                if not paper.get("temporal_eligibility", {}).get("eligible", False):
                    store.write(
                        "excluded_records",
                        {
                            "canonical_paper_id": paper["canonical_paper_id"],
                            "kind": "canonical_paper",
                            "reason": "one_or_more_contributing_versions_not_historically_eligible",
                            "temporal_eligibility": paper.get("temporal_eligibility"),
                            "recovery_table": "papers.jsonl",
                        },
                    )
                graph.add_node(
                    {
                        "node_id": paper["canonical_paper_id"],
                        "namespace": "canonical",
                        "node_type": "paper",
                        "dataset_version": None,
                    }
                )
            db.commit()
        if phase >= 6:
            # Register source-local nonpaper nodes before any relation lookup.
            for (raw_row,) in db.execute(
                "SELECT payload FROM records WHERE kind IN ('author','source_field','graph_node') "
                "ORDER BY record_key"
            ):
                node = json.loads(raw_row)
                node_id = "source-node:" + fingerprint(
                    [node["source_dataset"], node.get("dataset_version"), node["record_key"]]
                )
                node["canonical_node_id"] = node_id
                node_type = node.get("node_type", "author" if node["kind"] == "author" else "field")
                if node["kind"] == "author":
                    node.update(
                        canonical_author_id=node_id, identity_status="SOURCE_LOCAL_UNVERIFIED"
                    )
                    store.write("authors", node)
                elif node["kind"] == "source_field":
                    store.write("source_fields", node)
                else:
                    store.write("source_local_nodes", node)
                graph.add_node(
                    {
                        "node_id": node_id,
                        "namespace": "canonical",
                        "node_type": node_type,
                        "dataset_version": None,
                    }
                )
                for identifier in node.get("identifiers", []):
                    scoped = _scoped_id(
                        identifier, node["source_dataset"], node.get("dataset_version")
                    )
                    if scoped:
                        db.execute("INSERT INTO endpoint_ids VALUES(?,?)", (scoped, node_id))
                if node.get("source_split") and node.get("node_identifier"):
                    splits.add_official_index(
                        node["source_dataset"],
                        node.get("dataset_version"),
                        node["node_identifier"]["namespace"],
                        node["node_identifier"].get("normalized"),
                        node["source_split"],
                    )
            for (raw_row,) in db.execute("SELECT payload FROM records ORDER BY record_key"):
                row = json.loads(raw_row)
                if row["kind"] == "paper":
                    _paper_relations(row, engine, store, graph)
                    if row.get("source_split") and row.get("node_identifier"):
                        splits.add_official_index(
                            row["source_dataset"],
                            row.get("dataset_version"),
                            row["node_identifier"]["namespace"],
                            row["node_identifier"].get("normalized"),
                            row["source_split"],
                        )
                elif row["kind"] == "citation_context":
                    _context_relations(row, db, store, splits, graph, linkage)
                elif row["kind"] in {"citation_edge", "relation"}:
                    _source_edge(row, db, store, graph, linkage)
        # Audit quarantined contexts too: exclusions must not conceal official-split overlap.
        if phase >= 6:
            from .integration_manifest import iter_table

            store.streams["quarantine"].flush()
            for row in iter_table(publication.pending, "quarantine"):
                if row.get("kind") == "citation_context":
                    splits.add_context(_audit_context(row))
        split_report = splits.report() if phase >= 6 else {"status": "NOT_EXECUTED"}
        graph_report = graph.report() if phase >= 6 else {"status": "NOT_EXECUTED"}
        manifest = {
            "schema_version": "1.0",
            "stage": 3,
            "component_version": __version__,
            "started_at": start,
            "action": action,
            "completed_phase": phase,
            "scientific_stages_executed": [],
            "no_empirical_model_results": True,
            "real_sources": [source.dataset for source in inputs if not source.synthetic],
            "synthetic_sources": [source.dataset for source in inputs if source.synthetic],
            "sources": statuses,
            "input_checksums": input_hashes,
            "cleaning": {
                "by_source": {key: dict(value) for key, value in counts.items()},
                "transformations": dict(transformations),
                "missing_fields": dict(missing_fields),
                "quarantine_reasons": dict(quarantine_reasons),
                "duplicate_records": duplicate_count,
                "raw_records_deleted": 0,
            },
            "identity": identity_summary,
            "table_counts": dict(store.counts),
            "linkage": dict(linkage),
            "temporal": {
                **dict(temporal_counts),
                "cutoff": str(cutoff),
                "policy": config.temporal_integrity.policy,
                "historical_snapshot_constructed": False,
            },
            "split_audit": split_report,
            "graph_integrity": graph_report,
            "storage": {
                "format": "parquet_and_jsonl" if store.use_parquet else "jsonl",
                "partition_rows_max": config.integration.batch_size,
                "nested_parquet_data": "record_json",
                "fallback_reason": None
                if store.use_parquet
                else "JSONL explicitly selected or optional PyArrow unavailable",
            },
            "limitations": [
                (
                    "Shared identifiers establish declared-source identity evidence, "
                    "not external network authentication."
                ),
                (
                    "Historical availability is not inferred from publication, ingestion "
                    "or a current graph."
                ),
                (
                    "Exact matching and bounded title blocks do not establish "
                    "exhaustive entity-resolution recall."
                ),
                (
                    "Canonical IDs are repeatable for identical record keys/configuration; "
                    "incremental additions can change cluster anchors."
                ),
                (
                    "No learned aggregate, model, topic-investigation edge or research-gap "
                    "ground truth was created."
                ),
                (
                    "Missing graph sources and authoritative mappings prevent measured "
                    "heterogeneous linkage."
                ),
            ],
        }
        store.close()
        store = None
        db.close()
        db = None
        engine.close()
        engine = None
        splits.close()
        splits = None
        graph.close()
        graph = None
        make_reports(publication.pending, manifest, run_dir=publication.final)
        final = publication.publish(manifest)
        publish_reports(final, config.integration.report_dir)
        return {
            "run_dir": str(final),
            "run_id": manifest["run_id"],
            "action": action,
            "table_counts": manifest["table_counts"],
            "sources": statuses,
            "temporal": manifest["temporal"],
            "reports": str(config.integration.report_dir),
            "status": "COMPLETE",
            "scientific_stages_executed": [],
        }
    except Exception as error:
        for resource in (store, engine, splits, graph, db):
            if resource is not None:
                try:
                    resource.close()
                except (OSError, sqlite3.Error, ValueError):
                    pass
        publication.fail(error)
        raise


def _paper_relations(row, engine, store, graph):
    paper_id = engine.lookup(row["record_key"])
    for position, name in enumerate(row.get("authors", [])):
        author_id = "author:" + fingerprint(
            [row["source_dataset"], row.get("dataset_version"), row["record_key"], position]
        )
        author = {
            "canonical_author_id": author_id,
            "display_name": name,
            "source_author_ids": [],
            "verified_external_ids": [],
            "affiliation_data": [],
            "provenance": row.get("provenance", []),
            "identity_status": "SOURCE_LOCAL_NAME_ONLY",
            "record_key": row["record_key"],
            "source_dataset": row["source_dataset"],
        }
        store.write("authors", author)
        graph.add_node(
            {
                "node_id": author_id,
                "namespace": "canonical",
                "node_type": "author",
                "dataset_version": None,
            }
        )
        relation = {
            "record_key": row["record_key"],
            "source_id": paper_id,
            "target_id": author_id,
            "source_node_type": "paper",
            "target_node_type": "author",
            "edge_type": "authored_by",
            "identifier_namespace": "canonical",
            "author_position": position,
            "temporal_eligibility": row["temporal_eligibility"],
            "provenance": row.get("provenance", []),
        }
        store.write("authorships", relation)
        graph.add_edge(relation)
    for category in row.get("categories", []):
        store.write(
            "source_fields",
            {
                "source_field_id": "field:" + fingerprint([row["source_dataset"], category]),
                "field_name": category,
                "field_taxonomy": "source_categories",
                "source_dataset": row["source_dataset"],
                "canonical_paper_id": paper_id,
                "record_key": row["record_key"],
                "provenance": row.get("provenance", []),
                "interpretation": (
                    "source category, not a validated topic or direct-investigation label"
                ),
            },
        )


def _audit_context(row):
    provenance = row.get("provenance") or [{}]
    return {
        **row,
        "context_id": row.get("context_id", row["record_key"]),
        "source_record_sha256": provenance[0].get("record_sha256"),
    }


def _context_relations(row, db, store, splits, graph, linkage):
    source = _lookup(
        db, row.get("citing_identifier"), row["source_dataset"], row.get("dataset_version")
    )
    target = _lookup(
        db, row.get("cited_identifier"), row["source_dataset"], row.get("dataset_version")
    )
    row.update(
        citing_paper_id=source,
        cited_paper_id=target,
        identity_resolution_status=(
            "EXACT_VERIFIED"
            if all(
                identifier and identifier.get("verified")
                for identifier in (row.get("citing_identifier"), row.get("cited_identifier"))
            )
            else "SOURCE_LOCAL_LINKED"
        )
        if source and target
        else "UNRESOLVED",
    )
    store.write("citation_contexts", row)
    splits.add_context(_audit_context(row))
    linkage["citation_contexts"] += 1
    linkage["linked_contexts"] += int(bool(source and target))
    linkage["resolved_citation_endpoints"] += int(source is not None) + int(target is not None)
    linkage["unresolved_citation_endpoints"] += int(source is None) + int(target is None)
    context_id = row.get("context_id", row["record_key"])
    edge = {
        "edge_id": "citation:" + fingerprint([row["record_key"], context_id]),
        "record_key": row["record_key"],
        "source_dataset": row["source_dataset"],
        "source_paper_id": source,
        "target_paper_id": target,
        "original_source_id": row.get("citing_identifier"),
        "original_target_id": row.get("cited_identifier"),
        "identifier_namespace": "canonical"
        if source and target
        else "unresolved-source-identifiers",
        "edge_type": "cites",
        "edge_provenance": row.get("provenance", []),
        "citation_context_ids": [context_id],
        "citation_intent_status": "observed_source_label",
        "citation_intent": row.get("normalized_intent_label"),
        "source_split": row.get("source_split"),
        "temporal_status": row["temporal_eligibility"],
        "temporal_eligibility": row["temporal_eligibility"],
    }
    store.write("citations", edge)
    if source and target:
        graph.add_edge(edge)
    else:
        linkage["edges_retained_outside_canonical_graph"] += 1
        store.write(
            "unresolved_records",
            {
                "record_key": row["record_key"],
                "edge_id": edge["edge_id"],
                "source_dataset": row["source_dataset"],
                "reason": "unresolved_context_paper_endpoints",
                "original_source_id": edge["original_source_id"],
                "original_target_id": edge["original_target_id"],
            },
        )


def _source_edge(row, db, store, graph, linkage):
    source = _lookup(
        db, row.get("source_identifier"), row["source_dataset"], row.get("dataset_version")
    )
    target = _lookup(
        db, row.get("target_identifier"), row["source_dataset"], row.get("dataset_version")
    )
    edge = {
        **row,
        "edge_id": row.get("edge_id", "edge:" + fingerprint(row["record_key"])),
        "source_paper_id": source,
        "target_paper_id": target,
        "original_source_id": row.get("source_identifier"),
        "original_target_id": row.get("target_identifier"),
        "identifier_namespace": "canonical"
        if source and target
        else "unresolved-source-identifiers",
        "source_id": source,
        "target_id": target,
        "source_dataset_version": None,
        "target_dataset_version": None,
        "citation_intent": None,
        "citation_intent_status": "unavailable_without_observed_context",
        "citation_context_ids": [],
    }
    store.write("citations", edge)
    linkage["resolved_citation_endpoints"] += int(source is not None) + int(target is not None)
    linkage["unresolved_citation_endpoints"] += int(source is None) + int(target is None)
    graph.add_edge(edge)
