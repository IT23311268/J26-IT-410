"""Offline synthetic contract and graph tests; fixture results are not scholarly measurements."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch_geometric")
pytest.importorskip("pyarrow")
import torch  # noqa: E402
from stage06_helpers import change_config, fixture_config, refresh, write_jsonl  # noqa: E402

from ci_hgnn_rgd.graph.inputs import inspect_inputs, rows  # noqa: E402
from ci_hgnn_rgd.graph.pyg import (  # noqa: E402
    load_hetero_graph,
    temporal_view,
    validate_heterodata,
)
from ci_hgnn_rgd.graph.workflow import (  # noqa: E402
    build_graph,
    publish_directory,
    trace_record,
)


@pytest.mark.parametrize("newer_compatible", [False, True])
def test_auto_topic_selection_uses_latest_matching_stage03_corpus(tmp_path, newer_compatible):
    config = fixture_config(tmp_path)
    original = config.inputs.topic_run
    expected = original
    if newer_compatible:
        expected = original.parent / "run-y-compatible"
        shutil.copytree(original, expected)
    foreign = original.parent / "run-z-foreign"
    shutil.copytree(original, foreign)
    (foreign / "training_corpus_manifest.json").write_text(
        json.dumps({"stage03_run_id": "run-other-corpus"}), encoding="utf-8"
    )
    refresh(foreign, "stage05_manifest.json")

    bundle = inspect_inputs(change_config(config, inputs={"topic_run": None}))
    assert bundle.topic_run == expected
    assert bundle.embedding_run == expected
    with pytest.raises(ValueError, match="topic corpus belongs to a different Stage 03 run"):
        inspect_inputs(change_config(config, inputs={"topic_run": foreign}))


@pytest.fixture
def built(tmp_path):
    config = fixture_config(tmp_path)
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    return config, result, data


def test_stage03_04_05_contracts_and_real_pyg_counts(built):
    _, result, data = built
    assert data["paper"].num_nodes == 3 and data["author"].num_nodes == 2
    assert data["topic"].num_nodes == 2
    stats = result["statistics"]
    assert stats["scholarly_edge_counts"]["cites"] == 3
    assert stats["scholarly_edge_counts"]["associated_with_topic"] == 2
    assert stats["investigation_evidence_tiers"] == {
        "HUMAN_VERIFIED": 1,
        "MODEL_PREDICTED": 1,
        "UNKNOWN": 1,
    }
    assert result["integrity"]["critical_violations"] == 0


@pytest.mark.parametrize("kind", ["paper", "author", "topic"])
def test_index_mapping_is_stable_across_versions_and_chunk_sizes(tmp_path, kind):
    config = fixture_config(tmp_path)
    first = build_graph(config)
    second = build_graph(change_config(config, batch_size=256))

    def mapping(result):
        path = Path(result["graph_dir"]) / "mappings" / (kind + "_index.parquet")
        return [(row["stable_id"], row["graph_index"]) for row in rows(path)]

    assert mapping(first) == mapping(second)


def test_citation_direction_self_citations_and_duplicate_observation_provenance(built):
    _, result, data = built
    edges = list(rows(Path(result["graph_dir"]) / "edges/cites.parquet"))
    assert edges[0]["source_id"] == "p:1" and edges[0]["target_id"] == "p:2"
    assert result["statistics"]["unique_citation_pairs"] == 2
    assert result["statistics"]["self_citations"] == 1
    assert data["paper", "cites", "paper"].edge_index.tolist() == [[0, 0, 0], [1, 1, 0]]
    assert edges[0]["original"]["edge_id"] == "c:0"


def test_context_multiplicity_unlinked_preservation_and_intent_mean(built):
    _, result, data = built
    edge = data["paper", "cites", "paper"]
    assert torch.allclose(edge.edge_attr[0, :3], torch.tensor([0.3, 0.5, 0.2]))
    assert edge.edge_attr[0, 3] == 2
    assert edge.missing_intent.tolist() == [False, False, True]
    coverage = result["statistics"]["citation_intent_coverage"]
    assert coverage["linked_contexts"] == 2 and coverage["unlinked_contexts"] == 1
    root = Path(result["graph_dir"])
    unlinked = list(rows(root / "evidence/unlinked_citation_contexts.parquet"))
    assert unlinked[0]["citing_paper_id"] is None
    assert len(list(rows(root / "evidence/citation_context_predictions.parquet"))) == 3


@pytest.mark.parametrize("representation", ["edge_attributes", "context_nodes", "both"])
def test_context_node_representations_never_fabricate_endpoints(tmp_path, representation):
    config = change_config(fixture_config(tmp_path), citation_intent_representation=representation)
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    if representation == "edge_attributes":
        assert "citation_context" not in data.node_types
    else:
        assert data["citation_context"].num_nodes == 2
        assert data["citation_context", "originates_from", "paper"].edge_index.shape == (2, 2)
    if representation == "context_nodes":
        assert data["paper", "cites", "paper"].missing_intent.all()


def test_topic_outlier_association_and_evidence_tier_separation(built):
    _, result, data = built
    assert data["paper", "associated_with_topic", "topic"].edge_index.shape[1] == 2
    assert data["paper", "investigates_topic", "topic"].edge_index.shape[1] == 1
    assert data["paper", "predicted_investigates_topic", "topic"].edge_index.shape[1] == 1
    assert ("paper", "mentions_topic", "topic") not in data.edge_types
    assert result["statistics"]["paper_topic_coverage"]["assigned_papers"] == 2
    # Same author name never becomes a global identity.
    authors = list(rows(Path(result["graph_dir"]) / "nodes/author.parquet"))
    assert authors[0]["display_name"] == authors[1]["display_name"]
    assert authors[0]["stable_id"] != authors[1]["stable_id"]


def test_membership_and_unknown_do_not_create_investigation(tmp_path):
    def mutate(values):
        values["evidence"] = []

    result = build_graph(fixture_config(tmp_path, mutate=mutate))
    assert "investigates_topic" not in result["statistics"]["scholarly_edge_counts"]


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda v: v["assignments"][0].update(topic_model_version="wrong"), "namespace|version"),
        (lambda v: v["topics"][0].update(model_version="other-model"), "selected fitted model"),
        (lambda v: v["evidence"][0].update(evidence_source="unknown"), "candidate signals"),
        (lambda v: v["evidence"][0].update(evidence_text_span="badxxx"), "span"),
        (lambda v: v["citations"][0].update(target_namespace="other"), "namespace"),
        (lambda v: v["authorships"][0].update(target_id="missing"), "missing.*author"),
        (lambda v: v["intents"][0].update(prob_background=0.7), "sum"),
        (lambda v: v["intents"][0].update(model_version="wrong"), "model version"),
    ],
)
def test_invalid_contracts_fail_publication(tmp_path, mutation, error):
    config = fixture_config(tmp_path, mutate=mutation)
    with pytest.raises(ValueError, match=error):
        build_graph(config)
    assert not list(config.output_dir.glob("graph-*"))
    assert list(config.output_dir.glob(".pending-*/failure.json"))


def test_absent_citation_pair_never_creates_intent_edge(tmp_path):
    def mutate(values):
        values["citations"] = []

    result = build_graph(fixture_config(tmp_path, mutate=mutate))
    assert result["statistics"]["citation_intent_coverage"]["linked_contexts"] == 0
    assert "cites" not in result["statistics"]["scholarly_edge_counts"]


def test_optional_source_fields_are_separate_from_topics(tmp_path):
    config = change_config(fixture_config(tmp_path), include_research_fields=True)
    result = build_graph(config)
    assert result["statistics"]["node_counts"]["research_field"] == 1
    assert result["statistics"]["node_counts"]["topic"] == 2


def test_embedding_alignment_by_identity_even_when_index_is_shuffled(tmp_path):
    config = fixture_config(tmp_path)
    root = config.inputs.topic_run
    index = list(rows(root / "paper_embedding_index.jsonl"))
    write_jsonl(root / "paper_embedding_index.jsonl", list(reversed(index)))
    refresh(root, "stage05_manifest.json")
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    assert data["paper"].x.tolist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert data["paper"].has_embedding.all()


@pytest.mark.parametrize(
    "case,error",
    [("dimension", "dimension"), ("nan", "non-finite"), ("text", "text hash"), ("row", "duplicat")],
)
def test_invalid_embedding_features_rejected(tmp_path, case, error):
    config = fixture_config(tmp_path)
    root = config.inputs.topic_run
    if case in {"dimension", "nan"}:
        values = np.ones((3, 4 if case == "dimension" else 2), dtype=np.float32)
        if case == "nan":
            values[0, 0] = np.nan
        np.save(root / "paper_embeddings.npy", values, allow_pickle=False)
    else:
        values = list(rows(root / "paper_embedding_index.jsonl"))
        if case == "text":
            values[0]["text_sha256"] = "wrong"
        else:
            values[0]["row"] = values[1]["row"]
        write_jsonl(root / "paper_embedding_index.jsonl", values)
    refresh(root, "stage05_manifest.json")
    with pytest.raises(ValueError, match=error):
        build_graph(config)


def test_missing_features_use_masked_zeros(tmp_path):
    config = change_config(fixture_config(tmp_path), features={"paper_embedding": "none"})
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    assert not data["paper"].has_embedding.any()
    assert data["paper"].x.shape == (3, 0)


@pytest.mark.parametrize("bad", ["edge_dtype", "edge_bounds", "feature_rows", "edge_attrs"])
def test_pyg_integrity_rejects_invalid_tensor_structures(built, bad):
    data = built[2].clone()
    edge = data["paper", "cites", "paper"]
    if bad == "edge_dtype":
        edge.edge_index = edge.edge_index.float()
    elif bad == "edge_bounds":
        edge.edge_index[0, 0] = 99
    elif bad == "feature_rows":
        data["paper"].x = data["paper"].x[:1]
    else:
        edge.edge_attr = edge.edge_attr[:1]
    with pytest.raises((ValueError, IndexError)):
        validate_heterodata(data)


def test_reverse_edges_are_derived_and_not_additional_citations(built):
    _, result, data = built
    assert torch.equal(
        data["paper", "rev_cites", "paper"].edge_index,
        data["paper", "cites", "paper"].edge_index.flip(0),
    )
    assert result["statistics"]["citation_observations"] == 3
    assert data["paper", "rev_cites", "paper"].derived.all()


def test_target_relation_masks_hide_inverse_edges_and_counts(tmp_path):
    config = fixture_config(tmp_path)
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"], masked_relations=("cites",))
    assert ("paper", "cites", "paper") not in data.edge_types
    assert ("paper", "rev_cites", "paper") not in data.edge_types
    masked = build_graph(change_config(config, masked_relations=("associated_with_topic",)))
    data = load_hetero_graph(masked["graph_dir"])
    assert data["topic"].x.sum() == 0


def test_temporal_snapshot_excludes_future_papers_citations_and_models(tmp_path):
    def mutate(values):
        values["papers"][2].update(
            publication_date="2024-01-01",
            publication_year=2024,
            temporal_eligibility={
                "evidence": {"publication_date": "2024-01-01", "available_at": "2024-01-01"}
            },
        )
        values["citations"][1]["temporal_eligibility"] = {
            "evidence": {"citation_observed_at": "2024-01-01", "available_at": "2024-01-01"}
        }

    config = change_config(
        fixture_config(tmp_path, mutate=mutate, produced="2025-01-01T00:00:00Z"),
        temporal={"policy": "STRICT_POINT_IN_TIME", "cutoff": "2022-12-31"},
    )
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    assert data["paper"].num_nodes == 2
    assert data["paper", "cites", "paper"].edge_index.shape[1] == 2
    assert "topic" not in data.node_types
    assert not data["paper"].has_embedding.any()


@pytest.mark.parametrize(
    "policy,count",
    [
        ("STRICT_POINT_IN_TIME", 0),
        ("PUBLICATION_DATE_PROXY", 3),
        ("EXPLORATORY_NO_TEMPORAL_CLAIM", 3),
    ],
)
def test_unknown_availability_policy_is_explicit(tmp_path, policy, count):
    def mutate(values):
        for paper in values["papers"]:
            paper["temporal_eligibility"] = {"evidence": {"publication_date": "2020-01-01"}}

    config = change_config(
        fixture_config(tmp_path, mutate=mutate), temporal={"policy": policy, "cutoff": "2022-12-31"}
    )
    result = build_graph(config)
    assert result["statistics"]["node_counts"]["paper"] == count
    temporal = json.loads(
        (Path(result["graph_dir"]) / "reports/temporal_integrity.json").read_text()
    )
    assert temporal["policy"] == policy


def test_temporal_feature_masks_and_aggregates_do_not_leak_future_models(tmp_path):
    config = change_config(
        fixture_config(tmp_path),
        temporal={
            "policy": "PUBLICATION_DATE_PROXY",
            "cutoff": "2022-12-31",
            "split_cutoffs": {"train": "2020-12-31", "dev": "2021-12-31"},
        },
    )
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    train = temporal_view(data, "train")
    dev = temporal_view(data, "dev")
    assert not train["paper"].has_embedding.any() and train["paper"].x.sum() == 0
    assert train["topic"].x.sum() == 0 and dev["topic"].x.sum() == 2
    assert dev["paper"].has_embedding.all()
    assert train["paper", "cites", "paper"].missing_intent.all()
    assert not dev["paper", "cites", "paper"].missing_intent[:2].any()
    assert not any(name.startswith("dev_") for name in train["paper", "cites", "paper"])
    assert "original_index" in train["paper"]


def test_provenance_trace_and_official_split_preservation(built):
    config, result, _ = built
    node = trace_record(config, "p:1")["records"][0]
    edge = trace_record(config, "c:0", edge=True)["records"][0]
    assert node["source_split"] == "train"
    assert edge["original"]["source_dataset"] == "synthetic"
    assert edge["provenance"][0]["synthetic"] is True
    feature = trace_record(config, "p:1")["embedding_feature_provenance"]
    assert feature["encoder"]["model_id"] == "synthetic"
    support = trace_record(config, "c:0", edge=True)["supporting_context_predictions"]
    assert len(support) == 2 and support[0]["citation_context_id"] == "ctx:1"
    manifest = json.loads((Path(result["graph_dir"]) / "manifests/graph_manifest.json").read_text())
    assert not manifest["gnn_training_executed"] and not manifest["gap_scoring_executed"]


def test_checksum_tampering_and_snapshot_mismatch_are_rejected(built):
    _, result, _ = built
    with pytest.raises(ValueError, match="snapshot"):
        load_hetero_graph(result["graph_dir"], snapshot_id="other")
    path = Path(result["graph_dir"]) / "tensors/heterodata.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_hetero_graph(result["graph_dir"])


def test_versioned_atomic_publication_and_pending_rejection(tmp_path):
    config = fixture_config(tmp_path)
    build_graph(config, graph_version="fixed")
    with pytest.raises(ValueError, match="already exists"):
        build_graph(config, graph_version="fixed")
    with pytest.raises(ValueError, match="not atomically published"):
        load_hetero_graph(config.output_dir / ".pending-any")
    pending, final = tmp_path / ".pending-lock", tmp_path / "published"
    pending.mkdir()
    import os

    rename, calls = os.rename, []

    def locked(source, target):
        calls.append(1)
        if len(calls) == 1:
            error = PermissionError("synthetic scanner lock")
            error.winerror = 32
            raise error
        rename(source, target)

    with (
        patch("ci_hgnn_rgd.graph.workflow.os.rename", side_effect=locked),
        patch("ci_hgnn_rgd.graph.workflow.time.sleep"),
    ):
        publish_directory(pending, final)
    assert len(calls) == 2 and final.exists()


def test_dry_run_is_read_only_and_no_network_is_needed(tmp_path):
    config = fixture_config(tmp_path)
    with patch("socket.socket", side_effect=AssertionError("offline build must not use network")):
        result = build_graph(config, dry_run=True)
    assert result["status"] == "DRY_RUN" and not config.output_dir.exists()


def test_tensor_resource_guard_and_stage03_checksum_failure(tmp_path):
    config = change_config(fixture_config(tmp_path), features={"max_tensor_bytes": 1})
    with pytest.raises(ValueError, match="byte budget"):
        build_graph(config)
    path = config.inputs.stage03_run / "papers.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        inspect_inputs(config)


def test_source_local_internal_indices_are_isolated_by_dataset_and_version():
    from ci_hgnn_rgd.graph.assembly import paper_key

    first = {"source_dataset": "ogbn-arxiv", "dataset_version": "1", "source_record_id": 123}
    assert paper_key(first) != paper_key({**first, "source_dataset": "mag-year"})
    assert paper_key(first) != paper_key({**first, "dataset_version": "2"})


def test_exact_repeated_source_observation_has_recoverable_duplicate_audit(tmp_path):
    def mutate(values):
        values["citations"].append(dict(values["citations"][0]))

    result = build_graph(fixture_config(tmp_path, mutate=mutate))
    assert result["statistics"]["citation_observations"] == 3
    assert result["statistics"]["evidence_tables"]["duplicate_source_observations"] == 1


def test_partial_embedding_coverage_has_zero_vectors_and_masks(tmp_path):
    config = fixture_config(tmp_path)
    root = config.inputs.topic_run
    index = list(rows(root / "paper_embedding_index.jsonl"))[:2]
    write_jsonl(root / "paper_embedding_index.jsonl", index)
    np.save(root / "paper_embeddings.npy", np.eye(2, dtype=np.float32), allow_pickle=False)
    refresh(root, "stage05_manifest.json")
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    assert data["paper"].has_embedding.tolist() == [True, True, False]
    assert not data["paper"].x[2].any()


def test_strict_missing_embedding_strategy_can_fail_instead_of_zero_fill(tmp_path):
    config = change_config(
        fixture_config(tmp_path),
        features={"paper_embedding": "none", "missing_embedding_strategy": "error"},
    )
    with pytest.raises(ValueError, match="embedding missing"):
        build_graph(config)


def test_context_paths_do_not_reveal_masked_citation_targets(tmp_path):
    config = change_config(fixture_config(tmp_path), citation_intent_representation="both")
    result = build_graph(config)
    masked = load_hetero_graph(result["graph_dir"], masked_relations=("cites", "authored_by"))
    assert not masked["author"].x.any()
    assert not any(
        key[1]
        in {
            "cites",
            "rev_cites",
            "refers_to",
            "originates_from",
            "rev_refers_to",
            "rev_originates_from",
        }
        for key in masked.edge_types
    )


def test_rule_candidate_policy_keeps_candidates_distinct_from_verified_relations(tmp_path):
    def mutate(values):
        evidence = values["evidence"][0]
        evidence.update(
            evidence_source="rule_candidate",
            review_status="unreviewed",
            evidence_label="UNKNOWN",
            candidate_label="INVESTIGATES_TOPIC",
        )

    config = change_config(fixture_config(tmp_path, mutate=mutate), allow_rule_candidate_edges=True)
    result = build_graph(config)
    facts = result["statistics"]["scholarly_edge_counts"]
    assert facts["candidate_investigates_topic"] == 1
    assert "investigates_topic" not in facts


def test_graph_construction_itself_is_offline(tmp_path):
    config = fixture_config(tmp_path)
    with patch("socket.socket", side_effect=AssertionError("no downloads allowed")):
        result = build_graph(config)
    assert result["status"] == "COMPLETE"


def test_indexing_twenty_thousand_nodes_has_bounded_sql_work(tmp_path):
    from ci_hgnn_rgd.graph.stage06_config import Stage06Config
    from ci_hgnn_rgd.graph.store import GraphStore

    store = GraphStore(tmp_path / "index.sqlite", Stage06Config())
    n = 20000
    try:
        store.db.executemany(
            "INSERT INTO nodes VALUES(?,?,NULL,?)",
            (("paper", f"synthetic:{i:08}", "{}") for i in reversed(range(n))),
        )
        callbacks = 0

        def bounded_work():
            nonlocal callbacks
            callbacks += 1
            return int(callbacks * 10000 > n * 5000)

        store.db.set_progress_handler(bounded_work, 10000)
        store.index_nodes()
        store.db.set_progress_handler(None, 0)
        assert store.db.execute(
            "SELECT idx FROM nodes WHERE key='synthetic:00000000'"
        ).fetchone() == (0,)
        assert store.db.execute("SELECT MAX(idx) FROM nodes").fetchone() == (n - 1,)
    finally:
        store.close()


def test_embeddings_remain_available_when_topic_nodes_are_disabled(tmp_path):
    config = change_config(fixture_config(tmp_path), include_topics=False)
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    assert "topic" not in data.node_types and data["paper"].has_embedding.all()


def test_duplicate_prediction_id_is_an_informative_contract_failure(tmp_path):
    def mutate(values):
        values["intents"].append(dict(values["intents"][0]))

    config = fixture_config(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="duplicate citation-context prediction"):
        build_graph(config)


def test_conflicting_canonical_paper_rows_cannot_be_silently_coalesced(tmp_path):
    def mutate(values):
        values["papers"].append({**values["papers"][0], "title": "Conflicting paper identity"})

    config = fixture_config(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="canonical paper identity has conflicting"):
        build_graph(config)


def test_shared_verified_author_supports_independently_dated_authorships(tmp_path):
    def mutate(values):
        values["authors"][0]["identity_status"] = "VERIFIED"
        values["authors"][0]["temporal_eligibility"] = {"evidence": {"available_at": "2020-01-01"}}
        values["authors"][0]["verified_external_ids"] = [
            {"namespace": "synthetic-orcid", "value": "A1"}
        ]
        values["authorships"][1]["target_id"] = "a:1"
        values["authorships"][1]["source_dataset"] = "synthetic-second-source"
        values["authorships"][1]["provenance"] = [
            {
                "source_dataset": "synthetic-second-source",
                "source_record_id": "verified-authorship",
                "synthetic": True,
            }
        ]
        values["authorships"][1]["mapping_provenance"] = {
            "verified": True,
            "source_record_id": "synthetic-verified-canonical-mapping",
        }
        values["authorships"][1]["temporal_eligibility"] = {
            "evidence": {"publication_date": "2020-01-01", "citation_observed_at": "2021-01-01"}
        }

    config = change_config(
        fixture_config(tmp_path, mutate=mutate),
        temporal={
            "policy": "PUBLICATION_DATE_PROXY",
            "cutoff": "2022-12-31",
            "split_cutoffs": {"train": "2020-12-31", "dev": "2021-12-31"},
        },
    )
    result = build_graph(config)
    data = load_hetero_graph(result["graph_dir"])
    assert data["author"].num_nodes == 1 and data["author"].x.sum() == 2
    assert temporal_view(data, "train")["author"].x.sum() == 1
    assert temporal_view(data, "dev")["author"].x.sum() == 2
