"""Stage 06 orchestration: validated inputs, versioned publication and audit APIs."""

import importlib.metadata
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..data.manifests import atomic_json
from ..data.provenance import file_sha256
from ..pipeline import fingerprint
from . import assembly
from .audit import reports, validate_store
from .inputs import document, input_report, inspect_inputs
from .pyg import export_heterodata, load_hetero_graph, make_heterodata
from .store import REVERSES, GraphStore, write_sidecar

LOGGER = logging.getLogger(__name__)


def publish_directory(pending, final):
    if final.exists():
        raise ValueError("validated graph versions cannot be overwritten")
    for attempt in range(8):
        try:
            os.rename(pending, final)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.5))


def estimate_resources(config, bundle=None):
    bundle = bundle or inspect_inputs(config)
    counts = bundle.stage03_manifest.get("table_counts", {})
    papers = min(config.max_papers, counts.get("papers", 0) + counts.get("source_local_papers", 0))
    dimension = 0
    if bundle.embedding_run:
        dimension = document(bundle.embedding_run / "embedding_model_manifest.json")["encoder"][
            "dimension"
        ]
    result = {
        "paper_count_upper_bound": papers,
        "embedding_dimension": dimension,
        "paper_embedding_tensor_bytes_upper_bound": papers * dimension * 4,
        "max_tensor_bytes": config.features.max_tensor_bytes,
        "sqlite_cache_mib": config.sqlite_cache_mib,
        "estimate_semantics": "heuristic_payload_bound_not_peak_process_RSS",
        "cpu_only_construction": True,
        "gpu_required": False,
    }
    try:
        import psutil

        result.update(
            available_ram_bytes=psutil.virtual_memory().available,
            cpu_logical_count=psutil.cpu_count(),
        )
    except ImportError:
        result["available_ram_bytes"] = None
    import shutil

    result["disk_free_bytes"] = shutil.disk_usage(Path.cwd()).free
    return result


def _mask_relations(store, config):
    relations = set(config.masked_relations)
    if "cites" in relations:
        relations.update({"originates_from", "refers_to", "expresses_intent_about"})
    for relation in relations:
        for row in store.edge_rows(relation):
            store.evidence("held_out_relations", row["edge_id"], row)
        store.db.execute("DELETE FROM edges WHERE rel=?", (relation,))


def build_graph(config, *, graph_version=None, dry_run=False):
    bundle = inspect_inputs(config)
    if dry_run:
        return {
            "status": "DRY_RUN",
            **input_report(bundle),
            "resource_estimate": estimate_resources(config, bundle),
        }
    now = datetime.now(UTC)
    version = graph_version or "graph-" + now.isoformat().replace(":", "-") + "-" + uuid4().hex[:8]
    if Path(version).name != version or version.startswith("."):
        raise ValueError("graph version must be a plain nonhidden directory name")
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending, final = output / (".pending-" + version), output / version
    if pending.exists() or final.exists():
        raise ValueError("graph version already exists; choose a new version")
    pending.mkdir()
    started = time.perf_counter()
    store = GraphStore(pending / "graph.sqlite", config)
    closed = False
    try:
        atomic_json(pending / "config.json", config.model_dump(mode="json"))
        atomic_json(pending / "input_manifest.json", input_report(bundle))
        for name, function in (
            ("paper nodes", assembly.paper_nodes),
            ("authorship", assembly.author_nodes_and_edges),
            ("citations", assembly.citation_edges),
            ("topics", assembly.topic_nodes_and_edges),
            ("investigation evidence", assembly.investigation_edges),
            ("citation intent", assembly.citation_intent),
            ("research fields", assembly.research_fields),
            ("embedding index", assembly.embedding_index),
        ):
            LOGGER.info("Stage 06 constructing %s", name)
            function(store, bundle, config)
            store.db.commit()
        store.index_nodes()
        _mask_relations(store, config)
        assembly.reverse_relations(store, config)
        data, features = make_heterodata(store, bundle, config)
        integrity = validate_store(store, data)
        measured = reports(store, bundle, config, data, features, integrity)
        export_heterodata(data, pending / "tensors")
        for (kind,) in store.db.execute("SELECT DISTINCT kind FROM nodes ORDER BY kind"):
            write_sidecar(
                pending / "nodes" / (kind + ".parquet"), store.node_rows(kind), config.batch_size
            )
            write_sidecar(
                pending / "mappings" / (kind + "_index.parquet"),
                (
                    {k: row[k] for k in ("stable_id", "graph_index", "identifier_namespace")}
                    for row in store.node_rows(kind)
                ),
                config.batch_size,
            )
        for (relation,) in store.db.execute("SELECT DISTINCT rel FROM edges ORDER BY rel"):
            write_sidecar(
                pending / "edges" / (relation + ".parquet"),
                store.edge_rows(relation),
                config.batch_size,
            )
        for (kind,) in store.db.execute("SELECT DISTINCT kind FROM evidence ORDER BY kind"):
            write_sidecar(
                pending / "evidence" / (kind + ".parquet"),
                store.evidence_rows(kind),
                config.batch_size,
            )
        for name, report in measured.items():
            atomic_json(pending / "reports" / (name + ".json"), report)
        for name in (
            "graph_schema",
            "graph_statistics",
            "temporal_integrity",
            "graph_integrity",
            "stage06_readiness",
        ):
            text = (
                "# "
                + name.replace("_", " ").title()
                + "\n\n```json\n"
                + json.dumps(measured[name], indent=2)
                + "\n```\n"
            )
            (pending / "reports" / (name + ".md")).write_text(text, encoding="utf-8")
        for name, report in (
            ("temporal_manifest", measured["temporal_integrity"]),
            ("feature_manifest", features),
            ("provenance_manifest", measured["provenance_audit"]),
        ):
            atomic_json(pending / "manifests" / (name + ".json"), report)
        try:
            import psutil

            resources = psutil.Process().memory_info()._asdict()
        except ImportError:
            resources = {"peak_memory": "not_measured_psutil_unavailable"}
        atomic_json(pending / "reports/resource_usage.json", resources)
        store.close()
        closed = True
        versions = {}
        for package in ("torch", "torch-geometric", "numpy", "pyarrow", "safetensors"):
            versions[package] = importlib.metadata.version(package)
        manifest = {
            "stage": 6,
            "schema_version": "stage06-v1",
            "graph_version": version,
            "status": "COMPLETE",
            "snapshot_id": bundle.snapshot_id,
            "created_at": now.isoformat(),
            "policy": bundle.policy,
            "cutoff": bundle.cutoff,
            "config_sha256": fingerprint(config),
            "runtime_seconds": time.perf_counter() - started,
            "inputs": input_report(bundle),
            "input_checksums": bundle.checksums,
            "statistics": measured["graph_statistics"],
            "environment": versions,
            "reverse_relations": {
                REVERSES.get(rel, "rev_" + rel): rel
                for rel in measured["graph_statistics"]["scholarly_edge_counts"]
            },
            "masked_relations": list(config.masked_relations),
            "artifact_checksums": {
                p.relative_to(pending).as_posix(): file_sha256(p)
                for p in sorted(pending.rglob("*"))
                if p.is_file()
            },
            "gnn_training_executed": False,
            "gap_scoring_executed": False,
        }
        atomic_json(pending / "manifests/graph_manifest.json", manifest)
        publish_directory(pending, final)
        atomic_json(output / "latest.json", {"graph_version": version, "graph_dir": str(final)})
        # Report versions are separate from immutable graph versions.
        import shutil

        report_path = config.report_dir / version
        report_warning = None
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(final / "reports", report_path)
            atomic_json(
                config.report_dir / "latest.json",
                {"graph_version": version, "report_dir": str(report_path), "graph_dir": str(final)},
            )
        except OSError as error:
            report_warning = str(error)
            LOGGER.warning("Graph published; use its internal reports: %s", error)
            report_path = final / "reports"
        return {
            "status": "COMPLETE",
            "graph_version": version,
            "graph_dir": str(final),
            "report_dir": str(report_path),
            "report_copy_warning": report_warning,
            "statistics": measured["graph_statistics"],
            "integrity": integrity,
            "runtime_seconds": manifest["runtime_seconds"],
            "heterodata": str(final / "tensors/heterodata.safetensors"),
        }
    except Exception as error:
        if not closed:
            store.close()
            closed = True
        if pending.exists():
            atomic_json(
                pending / "failure.json",
                {"status": "FAILED", "error": str(error), "type": type(error).__name__},
            )
        raise
    finally:
        if not closed:
            store.close()


def graph_path(config, version=None):
    version = version or document(config.output_dir / "latest.json")["graph_version"]
    path = (config.output_dir / version).resolve()
    if not path.is_relative_to(config.output_dir.resolve()) or path.name.startswith(".pending-"):
        raise ValueError("invalid graph version path")
    if (path / "failure.json").exists():
        raise ValueError("failed graph is not published")
    manifest = document(path / "manifests/graph_manifest.json")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("graph is not complete")
    return path


def trace_record(config, identifier, *, version=None, edge=False):
    import sqlite3

    graph = graph_path(config, version)
    with sqlite3.connect(graph.as_uri() + "/graph.sqlite?mode=ro", uri=True) as db:
        if edge:
            result = db.execute("SELECT payload FROM edges WHERE id=?", (identifier,)).fetchall()
        else:
            result = db.execute("SELECT payload FROM nodes WHERE key=?", (identifier,)).fetchall()
        if not result:
            raise ValueError("node/edge ID is absent from this graph version")
        records = [json.loads(item[0]) for item in result]
        feature = None
        contexts = []
        if not edge:
            value = db.execute(
                "SELECT payload FROM evidence WHERE kind='feature_provenance' AND id=?",
                (identifier,),
            ).fetchone()
            feature = json.loads(value[0]) if value else None
        else:
            source_edge = records[0].get("derived_from_edge_id", identifier)
            contexts = [
                json.loads(value[0])
                for value in db.execute(
                    "SELECT e.payload FROM intent_contributions c JOIN evidence e "
                    "ON e.kind='citation_intent_links' AND e.id=c.context_id "
                    "WHERE c.edge_id=? ORDER BY c.context_id",
                    (source_edge,),
                )
            ]
        return {
            "graph_version": graph.name,
            "records": records,
            "embedding_feature_provenance": feature,
            "supporting_context_predictions": contexts,
        }


def validate_graph(config, version=None):
    graph = graph_path(config, version)
    data = load_hetero_graph(graph)
    from .pyg import validate_heterodata

    return {
        "graph_version": graph.name,
        **validate_heterodata(data),
        "sidecar_checksums_verified": True,
    }
