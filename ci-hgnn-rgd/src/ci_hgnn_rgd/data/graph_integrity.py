"""Source-local graph validation, preserving edge multiplicity and provenance."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _endpoint(row: dict, side: str) -> dict:
    alias = "source" if side == "source" else "target"
    paper_alias = "citing_paper_id" if side == "source" else "cited_paper_id"
    return {
        "node_id": row.get(
            f"{alias}_id",
            row.get(f"{alias}_paper_id", row.get(f"{alias}_node_id", row.get(paper_alias))),
        ),
        "namespace": row.get(f"{alias}_namespace", row.get("identifier_namespace", "canonical")),
        "dataset_version": row.get(f"{alias}_dataset_version", row.get("dataset_version")),
        "node_type": row.get(f"{alias}_node_type", "paper"),
    }


def _node_key(node: dict) -> str:
    return _key(
        [
            node.get("namespace", "canonical"),
            node.get("dataset_version"),
            node.get("node_type", "paper"),
            str(node.get("node_id")),
        ]
    )


def _mapping_verified(row: dict) -> bool:
    evidence = row.get("cross_source_mapping_provenance", row.get("mapping_provenance"))
    if isinstance(evidence, dict):
        return bool(
            evidence.get("verified") is True
            and (
                evidence.get("source_record_id")
                or evidence.get("source_url")
                or evidence.get("evidence")
            )
        )
    if isinstance(evidence, list | tuple):
        return any(_mapping_verified({"mapping_provenance": item}) for item in evidence)
    return False


def _intrinsic_edge_checks(edge: dict) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    source, target = _endpoint(edge, "source"), _endpoint(edge, "target")
    edge_type = edge.get("edge_type", "cites")
    if edge_type == "investigates_topic":
        reasons.append("stage03_investigates_topic_not_permitted")
    for side, endpoint in (("source", source), ("target", target)):
        value = endpoint["node_id"]
        if value is None or isinstance(value, bool) or not isinstance(value, int | str):
            reasons.append(f"invalid_{side}_identifier")
        if not endpoint["namespace"]:
            reasons.append(f"missing_{side}_namespace")
        # Dataset-local integer IDs require an explicit dataset version, including
        # the honest marker "unknown" when upstream publishes no release identifier.
        if isinstance(value, int) and endpoint["dataset_version"] is None:
            reasons.append(f"unscoped_{side}_internal_index")
        bound = edge.get(f"{side}_node_count")
        index = edge.get(f"{side}_index", value if isinstance(value, int) else None)
        if bound is not None:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(bound, bool)
                or not isinstance(bound, int)
                or not 0 <= index < bound
            ):
                reasons.append(f"{side}_index_out_of_range")
    same_namespace = source["namespace"] == target["namespace"]
    same_version = source["dataset_version"] == target["dataset_version"]
    if (not same_namespace or not same_version) and not _mapping_verified(edge):
        reasons.append("cross_namespace_or_version_without_verified_mapping")
    if edge_type in {"cites", "citation"}:
        if source["node_type"] != "paper" or target["node_type"] != "paper":
            reasons.append("citation_requires_paper_endpoints")
        if _node_key(source) == _node_key(target):
            warnings.append("self_citation_preserved")
    expected_types = {
        "authored_by": ("paper", "author"),
        "has_author": ("paper", "author"),
        "writes": ("author", "paper"),
        "author_of": ("author", "paper"),
    }
    if edge_type in expected_types:
        if (source["node_type"], target["node_type"]) != expected_types[edge_type]:
            reasons.append("authorship_endpoint_type_mismatch")
    intent = edge.get("citation_intent", edge.get("intent_label", edge.get("intent")))
    if intent is not None:
        contexts = edge.get("citation_context_ids", edge.get("context_ids", []))
        status = edge.get("citation_intent_status")
        observed = edge.get("observed_citation_context") is True or status in {
            "observed",
            "source_observed",
            "observed_context",
            "observed_source_label",
        }
        if not contexts or not observed:
            reasons.append("citation_intent_without_observed_context_evidence")
    temporal = edge.get("temporal_eligibility", edge.get("temporal_status"))
    if isinstance(temporal, dict):
        if temporal.get("eligible") is False:
            warnings.append("temporally_excluded_edge_preserved_outside_historical_snapshot")
        if temporal.get("unknown_availability"):
            warnings.append("unknown_edge_availability")
        if any(reason.startswith("future_") for reason in temporal.get("reasons", [])):
            warnings.append("future_temporal_evidence_detected")
    elif isinstance(temporal, str) and temporal in {"EXCLUDED", "INELIGIBLE"}:
        warnings.append("temporally_excluded_edge_preserved_outside_historical_snapshot")
    else:
        warnings.append("edge_temporal_eligibility_not_verified")
    return reasons, warnings


def validate_edge(edge: dict, nodes: dict | None = None) -> dict:
    """Validate a single edge; optional nodes may be keyed by scoped tuple or ID.

    An ID-only lookup still checks the stored node namespace/version/type. It can
    never authorize equating equal integers from distinct source graphs.
    """
    reasons, warnings = _intrinsic_edge_checks(edge)
    for side in ("source", "target"):
        endpoint = _endpoint(edge, side)
        if nodes is not None:
            scoped = (
                endpoint["namespace"],
                endpoint["dataset_version"],
                endpoint["node_type"],
                endpoint["node_id"],
            )
            node = nodes.get(scoped, nodes.get(endpoint["node_id"]))
            if node is None:
                reasons.append(f"missing_{side}_endpoint")
            elif _node_key(node) != _node_key(endpoint):
                reasons.append(f"{side}_endpoint_namespace_mismatch")
    return {
        "valid": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "source_endpoint": _endpoint(edge, "source"),
        "target_endpoint": _endpoint(edge, "target"),
    }


class GraphAuditor:
    """SQLite endpoint/duplicate indexes, keeping every observed edge occurrence.

    Edges may be added before nodes. Final endpoint existence is assessed only by
    ``report``. Source-specific relation types are retained; unknown semantics are
    reported rather than rewritten into a universal graph taxonomy.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA temp_store=FILE;
            CREATE TABLE IF NOT EXISTS graph_nodes(
                node_key TEXT PRIMARY KEY,node_id TEXT,namespace TEXT,version TEXT,
                node_type TEXT,payload TEXT,occurrences INTEGER DEFAULT 1);
            CREATE TABLE IF NOT EXISTS graph_edges(
                seq INTEGER PRIMARY KEY,edge_id TEXT,source_key TEXT,target_key TEXT,
                edge_type TEXT,pair_key TEXT,record_hash TEXT,payload TEXT,
                reasons TEXT,warnings TEXT,contexts INTEGER);
            CREATE INDEX IF NOT EXISTS graph_edge_pair ON graph_edges(pair_key);
            CREATE INDEX IF NOT EXISTS graph_edge_record ON graph_edges(record_hash);
            CREATE INDEX IF NOT EXISTS graph_edge_source ON graph_edges(source_key);
            CREATE INDEX IF NOT EXISTS graph_edge_target ON graph_edges(target_key);
            CREATE TABLE IF NOT EXISTS graph_node_conflicts(node_key TEXT,payload TEXT);
        """)

    def add_node(self, row: dict) -> None:
        if row.get("node_id") is None:
            raise ValueError("nodes require an explicit source or canonical node_id")
        payload, key = _key(row), _node_key(row)
        prior = self.db.execute(
            "SELECT payload FROM graph_nodes WHERE node_key=?", (key,)
        ).fetchone()
        if prior and prior[0] != payload:
            self.db.execute("INSERT INTO graph_node_conflicts VALUES (?,?)", (key, payload))
        self.db.execute(
            "INSERT INTO graph_nodes(node_key,node_id,namespace,version,node_type,payload) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(node_key) DO UPDATE SET occurrences=occurrences+1",
            (
                key,
                str(row["node_id"]),
                row.get("namespace", "canonical"),
                row.get("dataset_version"),
                row.get("node_type", "paper"),
                payload,
            ),
        )

    def add_edge(self, row: dict) -> dict:
        result = validate_edge(row)
        source = _node_key(result["source_endpoint"])
        target = _node_key(result["target_endpoint"])
        edge_type = row.get("edge_type", "cites")
        pair_key = hashlib.sha256(_key([source, target, edge_type]).encode()).hexdigest()
        contexts = row.get("citation_context_ids", row.get("context_ids", []))
        context_count = len(contexts) if isinstance(contexts, list | tuple) else 0
        self.db.execute(
            "INSERT INTO graph_edges(edge_id,source_key,target_key,edge_type,pair_key,"
            "record_hash,payload,reasons,warnings,contexts) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                row.get("edge_id"),
                source,
                target,
                edge_type,
                pair_key,
                row.get("source_record_sha256"),
                _key(row),
                _key(result["reasons"]),
                _key(result["warnings"]),
                context_count,
            ),
        )
        return result

    def report(self) -> dict:
        self.db.commit()
        reasons, warnings = Counter(), Counter()
        invalid = valid = resolved = unresolved = contexts = 0
        examples = []
        for (
            seq,
            edge_id,
            raw_reasons,
            raw_warnings,
            source_found,
            target_found,
            n_context,
        ) in self.db.execute(
            "SELECT e.seq,e.edge_id,e.reasons,e.warnings,s.node_key,t.node_key,e.contexts "
            "FROM graph_edges e LEFT JOIN graph_nodes s ON s.node_key=e.source_key "
            "LEFT JOIN graph_nodes t ON t.node_key=e.target_key ORDER BY e.seq"
        ):
            issues = json.loads(raw_reasons)
            for side, found in (("source", source_found), ("target", target_found)):
                if found is None:
                    issues.append(f"missing_{side}_endpoint")
                    unresolved += 1
                else:
                    resolved += 1
            if issues:
                invalid += 1
                if len(examples) < 20:
                    examples.append({"occurrence": seq, "edge_id": edge_id, "reasons": issues})
            else:
                valid += 1
            reasons.update(set(issues))
            warnings.update(json.loads(raw_warnings))
            contexts += n_context

        def duplicate_stats(column):
            groups, repeats = self.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM graph_edges "
                "WHERE " + column + " IS NOT NULL GROUP BY " + column + " HAVING COUNT(*)>1)"
            ).fetchone()
            return {"groups": groups, "additional_occurrences": repeats}

        return {
            "nodes": self.db.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0],
            "node_occurrences": self.db.execute(
                "SELECT COALESCE(SUM(occurrences),0) FROM graph_nodes"
            ).fetchone()[0],
            "node_conflicting_metadata_occurrences": self.db.execute(
                "SELECT COUNT(*) FROM graph_node_conflicts"
            ).fetchone()[0],
            "node_types": dict(
                self.db.execute("SELECT node_type,COUNT(*) FROM graph_nodes GROUP BY node_type")
            ),
            "edge_occurrences": valid + invalid,
            "valid_edge_occurrences": valid,
            "invalid_edge_occurrences": invalid,
            "resolved_endpoint_occurrences": resolved,
            "unresolved_endpoint_occurrences": unresolved,
            "edge_types": dict(
                self.db.execute("SELECT edge_type,COUNT(*) FROM graph_edges GROUP BY edge_type")
            ),
            "issue_counts": dict(sorted(reasons.items())),
            "warning_counts": dict(sorted(warnings.items())),
            "repeated_endpoint_relation_pairs": duplicate_stats("pair_key"),
            "exact_source_record_duplicates": duplicate_stats("record_hash"),
            "citation_context_references_preserved": contexts,
            "multiplicity_policy": "all_occurrences_preserved_not_automatically_collapsed",
            "self_citations_policy": "reported_not_automatically_invalid",
            "invalid_examples": examples,
            "unknown_relation_semantics": [
                row[0]
                for row in self.db.execute(
                    "SELECT DISTINCT edge_type FROM graph_edges ORDER BY edge_type"
                )
                if row[0]
                not in {"cites", "citation", "authored_by", "has_author", "writes", "author_of"}
            ],
        }

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
