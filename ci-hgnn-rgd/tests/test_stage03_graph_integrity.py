"""Tiny synthetic graphs validate structure; no real graph integration claimed."""

from ci_hgnn_rgd.data.graph_integrity import GraphAuditor, validate_edge


def node(index, namespace="synthetic", version="1", node_type="paper"):
    return {
        "node_id": index,
        "namespace": namespace,
        "dataset_version": version,
        "node_type": node_type,
    }


def edge(source=0, target=1, **extra):
    return {
        "source_id": source,
        "target_id": target,
        "identifier_namespace": "synthetic",
        "dataset_version": "1",
        **extra,
    }


def test_edges_before_nodes_are_resolved_only_at_final_audit(tmp_path):
    with GraphAuditor(tmp_path / "graph.sqlite") as audit:
        audit.add_edge(edge())
        audit.add_node(node(0))
        audit.add_node(node(1))
        audit.add_edge(edge(target=99))
        result = audit.report()
    assert result["valid_edge_occurrences"] == 1
    assert result["invalid_edge_occurrences"] == 1
    assert result["resolved_endpoint_occurrences"] == 3
    assert result["unresolved_endpoint_occurrences"] == 1


def test_cross_namespace_and_cross_version_indices_never_equated():
    result = validate_edge(edge(target_namespace="another"))
    assert not result["valid"]
    assert "cross_namespace_or_version_without_verified_mapping" in result["reasons"]
    assert not validate_edge(edge(target_dataset_version="2"))["valid"]
    # A boolean without attributable mapping evidence is insufficient.
    assert not validate_edge(
        edge(target_namespace="another", mapping_provenance={"verified": True})
    )["valid"]


def test_verified_cross_source_mapping_requires_explicit_provenance():
    result = validate_edge(
        edge(
            target_namespace="another",
            mapping_provenance={"verified": True, "source_record_id": "synthetic-mapping"},
        )
    )
    assert result["valid"]


def test_node_lookup_checks_namespace_even_when_ids_match():
    result = validate_edge(edge(), {0: node(0, "another"), 1: node(1)})
    assert not result["valid"]
    assert "source_endpoint_namespace_mismatch" in result["reasons"]


def test_index_bounds_and_version_scope_are_required():
    result = validate_edge(edge(target=-1, source_node_count=2, target_node_count=2))
    assert not result["valid"]
    assert "target_index_out_of_range" in result["reasons"]
    assert not validate_edge(edge(dataset_version=None))["valid"]


def test_self_citations_and_context_multiplicity_are_preserved(tmp_path):
    with GraphAuditor(tmp_path / "graph.sqlite") as audit:
        audit.add_node(node(0))
        audit.add_edge(edge(target=0, citation_context_ids=["context1"]))
        audit.add_edge(edge(target=0, citation_context_ids=["context2"]))
        result = audit.report()
    assert result["valid_edge_occurrences"] == 2
    assert result["citation_context_references_preserved"] == 2
    assert result["repeated_endpoint_relation_pairs"] == {"groups": 1, "additional_occurrences": 1}
    assert result["warning_counts"]["self_citation_preserved"] == 2


def test_intent_requires_observed_citation_context_and_no_investigation_edges():
    assert not validate_edge(edge(citation_intent="method"))["valid"]
    assert not validate_edge(edge(citation_intent="method", citation_context_ids=["c1"]))["valid"]
    assert validate_edge(
        edge(
            citation_intent="method", citation_context_ids=["c1"], citation_intent_status="observed"
        )
    )["valid"]
    assert not validate_edge(edge(edge_type="investigates_topic"))["valid"]


def test_authorship_direction_and_node_type_checked():
    assert validate_edge(edge(edge_type="authored_by", target_node_type="author"))["valid"]
    assert not validate_edge(edge(edge_type="writes", target_node_type="author"))["valid"]


def test_temporally_excluded_edges_are_flagged_without_deleting_raw_edge(tmp_path):
    with GraphAuditor(tmp_path / "graph.sqlite") as audit:
        for index in (0, 1):
            audit.add_node(node(index))
        audit.add_edge(edge(temporal_eligibility={"eligible": False, "unknown_availability": True}))
        result = audit.report()
    assert result["edge_occurrences"] == 1
    assert (
        result["warning_counts"]["temporally_excluded_edge_preserved_outside_historical_snapshot"]
        == 1
    )
    assert result["warning_counts"]["unknown_edge_availability"] == 1
