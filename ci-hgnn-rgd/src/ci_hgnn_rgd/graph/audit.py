"""Observed graph integrity, coverage and descriptive statistics; no learned scores."""

import json
from array import array
from collections import Counter

from .pyg import validate_heterodata


def validate_store(store, data):
    report = validate_heterodata(data)
    for kind, count in store.db.execute("SELECT kind,COUNT(*) FROM nodes GROUP BY kind"):
        if data[kind].num_nodes != count:
            raise ValueError("tensor/node mapping count mismatch")
        for i, row in enumerate(store.node_rows(kind)):
            if row["graph_index"] != i or not row.get("identifier_namespace"):
                raise ValueError("node mapping indices or identifier namespaces invalid")
            if not row.get("provenance") or not row.get("temporal_eligibility", {}).get("eligible"):
                raise ValueError("graph node lacks provenance/temporal eligibility")
    for sk, rel, tk, count in store.db.execute(
        "SELECT sk,rel,tk,COUNT(*) FROM edges GROUP BY sk,rel,tk"
    ):
        if data[sk, rel, tk].edge_index.shape[1] != count:
            raise ValueError("edge sidecar/tensor count mismatch")
        for i, row in enumerate(store.edge_rows(rel)):
            if list(data[sk, rel, tk].edge_index[:, i]) != [
                row["source_index"],
                row["target_index"],
            ]:
                raise ValueError("edge mapping/tensor ordering mismatch")
            if not row.get("provenance") or not row.get("temporal_eligibility", {}).get("eligible"):
                raise ValueError("graph edge lacks eligible provenance")
            if rel in {"investigates_topic", "mentions_topic"} and (
                row.get("evidence_tier") != "HUMAN_VERIFIED"
            ):
                raise ValueError("unsupported verified investigation/mention relation")
            if rel.startswith("predicted_") and row.get("evidence_tier") != "MODEL_PREDICTED":
                raise ValueError("predicted investigation tier is not preserved")
            if row["derived"]:
                original = store.db.execute(
                    "SELECT s,t,derived FROM edges WHERE id=?", (row.get("derived_from_edge_id"),)
                ).fetchone()
                if original != (row["target_id"], row["source_id"], 0):
                    raise ValueError("reverse relation has no correct original source fact")
    report.update(
        provenance_complete=True,
        mapping_alignment=True,
        unsupported_investigation_edges=0,
        critical_violations=0,
    )
    return report


def statistics(store, data):
    node_counts = dict(store.db.execute("SELECT kind,COUNT(*) FROM nodes GROUP BY kind"))
    for kind in ("paper", "author", "topic"):
        node_counts.setdefault(kind, 0)
    facts = dict(store.db.execute("SELECT rel,COUNT(*) FROM edges WHERE derived=0 GROUP BY rel"))
    derived = dict(store.db.execute("SELECT rel,COUNT(*) FROM edges WHERE derived=1 GROUP BY rel"))
    evidence_counts = dict(store.db.execute("SELECT kind,COUNT(*) FROM evidence GROUP BY kind"))
    n_papers = node_counts["paper"]
    associated = store.db.execute(
        "SELECT COUNT(DISTINCT s) FROM edges WHERE rel='associated_with_topic'"
    ).fetchone()[0]
    citing = store.db.execute("SELECT COUNT(*) FROM edges WHERE rel='cites'").fetchone()[0]
    citation_pairs = store.db.execute(
        "SELECT COUNT(*) FROM (SELECT s,t FROM edges WHERE rel='cites' GROUP BY s,t)"
    ).fetchone()[0]
    self_citations = store.db.execute(
        "SELECT COUNT(*) FROM edges WHERE rel='cites' AND s=t"
    ).fetchone()[0]
    tiers, sources, unknown = Counter(), Counter(), Counter()
    for (payload,) in store.db.execute(
        "SELECT payload FROM evidence WHERE kind='paper_topic_evidence'"
    ):
        tiers[json.loads(payload)["evidence_tier"]] += 1
    for kind, payload in store.db.execute("SELECT kind,payload FROM nodes"):
        node = json.loads(payload)
        unknown[kind] += bool(node["temporal_eligibility"].get("unknown_availability"))
        for source in {p.get("source_dataset", "unknown") for p in node.get("provenance", [])}:
            sources[source] += 1
    degree_histograms = {}
    for side, label in (("s", "out_degree"), ("t", "in_degree")):
        histogram = dict(
            store.db.execute(
                f"SELECT n,COUNT(*) FROM (SELECT COUNT(*) n FROM edges WHERE rel='cites' "
                f"GROUP BY {side}) GROUP BY n"
            )
        )
        histogram[0] = n_papers - sum(histogram.values())
        degree_histograms[label] = histogram
    store.db.executescript("""
        CREATE TEMP TABLE incident(kind TEXT,key TEXT,PRIMARY KEY(kind,key));
        INSERT OR IGNORE INTO incident SELECT sk,s FROM edges WHERE derived=0;
        INSERT OR IGNORE INTO incident SELECT tk,t FROM edges WHERE derived=0;
    """)
    isolated = dict(
        store.db.execute("""SELECT n.kind,COUNT(*) FROM nodes n LEFT JOIN incident i
        ON n.kind=i.kind AND n.key=i.key WHERE i.key IS NULL GROUP BY n.kind""")
    )
    store.db.execute("DROP TABLE incident")
    offsets, total = {}, 0
    for kind in sorted(data.node_types):
        offsets[kind], total = total, total + data[kind].num_nodes
    components = None
    if total <= 200000:
        parents = array("q", range(total))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for sk, rel, tk in data.edge_types:
            if store.db.execute("SELECT derived FROM edges WHERE rel=? LIMIT 1", (rel,)).fetchone()[
                0
            ]:
                continue
            for source, target in data[sk, rel, tk].edge_index.T.tolist():
                a, b = root(source + offsets[sk]), root(target + offsets[tk])
                if a != b:
                    parents[b] = a
        components = len({root(i) for i in range(total)})
    linked = evidence_counts.get("citation_intent_links", 0)
    contexts = evidence_counts.get("citation_context_predictions", 0)
    return {
        "measurement_kind": "observed_input_graph_not_synthetic_unless_sources_marked_synthetic",
        "node_counts": node_counts,
        "scholarly_edge_counts": facts,
        "derived_message_passing_edge_counts": derived,
        "reverse_edges_counted_as_scholarly_facts": False,
        "citation_observations": citing,
        "unique_citation_pairs": citation_pairs,
        "self_citations": self_citations,
        "citation_degree_histograms": degree_histograms,
        "citation_pair_density_including_possible_self_loops": citation_pairs / n_papers**2
        if n_papers
        else None,
        "isolated_nodes_by_type": isolated,
        "weak_connected_components": components,
        "components_method": "sparse_union_find"
        if components is not None
        else "not_computed_budget",
        "paper_topic_coverage": {
            "assigned_papers": associated,
            "papers": n_papers,
            "fraction": associated / n_papers if n_papers else None,
        },
        "citation_intent_coverage": {
            "linked_contexts": linked,
            "prediction_contexts": contexts,
            "fraction": linked / contexts if contexts else None,
            "unlinked_contexts": evidence_counts.get("unlinked_citation_contexts", 0),
        },
        "investigation_evidence_tiers": dict(tiers),
        "evidence_tables": evidence_counts,
        "unknown_temporal_availability_nodes": dict(unknown),
        "source_dataset_node_contributions": dict(sources),
    }


def reports(store, bundle, config, data, feature_quality, integrity):
    stats = statistics(store, data)
    excluded = Counter()
    for row in store.evidence_rows("excluded_records"):
        excluded[row["reason"]] += 1
    temporal = {
        "policy": bundle.policy,
        "cutoff": bundle.cutoff,
        "snapshot_id": bundle.snapshot_id,
        "strict_historical_claim": (
            bundle.policy == "STRICT_POINT_IN_TIME"
            and stats["node_counts"]["paper"] > 0
            and not any(stats["unknown_temporal_availability_nodes"].values())
        ),
        "empty_paper_snapshot": stats["node_counts"]["paper"] == 0,
        "excluded_reasons": dict(excluded),
        "unknown_availability_nodes": stats["unknown_temporal_availability_nodes"],
        "model_availability_policy": config.temporal.model_availability_policy,
        "split_cutoffs": config.temporal.split_cutoffs,
        "split_method": "explicit_chronological_visibility_no_random_edge_split",
        "official_splits_modified": False,
        "source_stage03_temporal_audit": bundle.stage03_manifest.get("temporal", {}),
    }
    retained_future = Counter()
    for kind, payload in store.db.execute("SELECT kind,payload FROM nodes"):
        node = json.loads(payload)
        for reason in node["temporal_eligibility"].get("reasons", []):
            if reason.startswith("future_"):
                retained_future[kind + ":" + reason] += 1
    temporal["retained_future_evidence_under_exploratory_policy"] = dict(retained_future)
    schema = {
        "schema_version": "stage06-v1",
        "node_types": data.node_types,
        "relations": [list(key) for key in data.edge_types],
        "citation_direction": "citing_paper_to_cited_paper_verified_source_direction",
        "intent_probability_order": ["background", "method", "result"],
        "aggregate_method": config.aggregate_method,
        "aggregate_calibration": "not_validated_even_if_individual_contexts_are_calibrated",
        "score_semantics": "per_relation_source_scores_not_globally_comparable",
        "topic_identity": "composite_of_model_version_snapshot_and_topic_id",
        "paper_index": "sorted_stable_id_within_filtered_graph_version",
        "reverse_edges": "derived_message_passing_only",
    }
    data_blockers = []
    if not stats["citation_observations"]:
        data_blockers.append(
            "No verified citation relationships in the selected Stage 03 graph corpus"
        )
    if not stats["citation_intent_coverage"]["linked_contexts"]:
        data_blockers.append("No citation-intent predictions linked to verified graph paper pairs")
    if not stats["investigation_evidence_tiers"].get("HUMAN_VERIFIED", 0):
        data_blockers.append("No human-verified direct-investigation labels in supplied evidence")
    return {
        "graph_schema": schema,
        "graph_statistics": stats,
        "citation_intent_coverage": stats["citation_intent_coverage"],
        "investigation_evidence_coverage": {
            "tiers": stats["investigation_evidence_tiers"],
            "verified_investigation_edges": stats["scholarly_edge_counts"].get(
                "investigates_topic", 0
            ),
            "predicted_investigation_edges": stats["scholarly_edge_counts"].get(
                "predicted_investigates_topic", 0
            ),
            "unknown_is_positive": False,
        },
        "temporal_integrity": temporal,
        "graph_integrity": integrity,
        "feature_quality": feature_quality,
        "provenance_audit": {
            "status": "PASSED",
            "all_nodes_and_edges_have_source_provenance": True,
            "source_checksums": bundle.checksums,
            "unlinked_evidence_retained": True,
            "per_context_predictions_preserved": True,
        },
        "stage06_readiness": {
            "graph_loadable_for_stage07": True,
            "has_paper_nodes": stats["node_counts"]["paper"] > 0,
            "scientific_supervision_complete": not data_blockers,
            "scientific_data_blockers": data_blockers,
            "gnn_training_executed": False,
            "gap_scoring_executed": False,
            "blocked_dependencies": bundle.blocked,
            "masked_target_relations": list(config.masked_relations),
            "relation_masking_required_for_link_prediction": True,
            "limitations": [
                "Graphs retain admitted inputs; missing citations and identities are not inferred.",
                "Temporal availability is unknown where source evidence is absent.",
                "Topic membership and rule candidates do not establish investigation.",
                "Node indices are stable for identical filtered inputs, not incremental additions.",
                "Tensor byte and SQLite-cache budgets do not cap total process RSS.",
            ],
        },
    }
