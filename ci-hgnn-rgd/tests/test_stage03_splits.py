"""Synthetic splits test audit mechanics, never benchmark performance."""

import pytest

from ci_hgnn_rgd.data.split_management import SplitAuditor, group_aware_splits, temporal_split


def context(key, text, split, **extra):
    return {"context_id": key, "original_context": text, "source_split": split, **extra}


def test_exact_normalized_and_paper_overlap_preserve_official_splits(tmp_path):
    original = [
        context(
            "c1",
            "SYNTHETIC Unicode α context",
            "train",
            citing_paper_id="paper1",
            source_record_sha256="same",
        ),
        context(
            "c2",
            "SYNTHETIC Unicode α context",
            "dev",
            citing_paper_id="paper1",
            source_record_sha256="same",
        ),
        context("c3", " SYNTHETIC  UNICODE α context ", "test"),
    ]
    with SplitAuditor(tmp_path / "split.sqlite") as audit:
        for row in original:
            audit.add_context(row)
        result = audit.report()
    assert result["exact_record_overlap_groups"] == 1
    assert result["exact_text_overlap_groups"] == 1
    assert result["normalized_text_overlap_groups"] == 1
    assert result["paper_identity_overlap_groups"] == 1
    assert not result["official_splits_modified"]
    assert [row["source_split"] for row in original] == ["train", "dev", "test"]


def test_near_duplicate_detection_and_explicit_comparison_budget(tmp_path):
    text = "SYNTHETIC citation context with an important scientific method in this long example."
    with SplitAuditor(tmp_path / "split.sqlite", near_threshold=0.9, max_comparisons=1) as audit:
        audit.add_context(context("c1", text, "train"))
        audit.add_context(context("c2", text.replace("important", "essential"), "dev"))
        audit.add_context(context("c3", text.replace("important", "significant"), "test"))
        result = audit.report()
    assert result["near_duplicate_cross_split_pairs"] == 1
    assert result["near_audit"]["comparisons"] == 1
    assert result["near_audit"]["budget_skipped_contexts"] > 0
    assert not result["near_audit"]["exhaustive"]


def test_source_paper_namespaces_are_isolated(tmp_path):
    with SplitAuditor(tmp_path / "split.sqlite") as audit:
        audit.add_context(
            context(
                "c1",
                "SYNTHETIC A",
                "train",
                citing_paper_id=1,
                paper_identifier_namespace="ogbn:version1",
            )
        )
        audit.add_context(
            context(
                "c2",
                "SYNTHETIC B",
                "test",
                citing_paper_id=1,
                paper_identifier_namespace="mag:version1",
            )
        )
        assert audit.report()["paper_identity_overlap_groups"] == 0


def test_official_graph_node_splits_and_dataset_versions_preserved(tmp_path):
    with SplitAuditor(tmp_path / "split.sqlite") as audit:
        audit.add_official_index("ogbn", "1", "ogbn-node", 0, "train")
        audit.add_official_index("ogbn", "1", "ogbn-node", 1, "valid")
        audit.add_official_index("ogbn", "1", "ogbn-node", 2, "test")
        audit.add_official_index("mag", "1", "mag-node", 0, "test")
        result = audit.report()
    assert len(result["official_node_splits"]) == 4
    assert result["official_node_overlap_groups"] == 0
    assert not result["official_splits_modified"]


def test_group_splits_repeat_deterministically_and_preserve_sources():
    rows = [context(str(i), "SYNTHETIC", "official", group=f"g{i // 3}") for i in range(30)]
    first = list(group_aware_splits(rows, group_field="group", seed=7))
    second = list(group_aware_splits(reversed(rows), group_field="group", seed=7))
    assert {r["context_id"]: r["experiment_split"] for r in first} == {
        r["context_id"]: r["experiment_split"] for r in second
    }
    groups = {}
    for row in first:
        assert row["source_split"] == "official"
        groups.setdefault(row["group"], set()).add(row["experiment_split"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_temporal_windows_are_ordered_and_unknown_records_unassigned():
    options = {
        "training_cutoff": "2018-12-31",
        "validation_end": "2019-12-31",
        "test_end": "2020-12-31",
    }
    assert temporal_split("2018-01-01", **options) == "train"
    assert temporal_split("2019-01-01", **options) == "dev"
    assert temporal_split("2020-01-01", **options) == "test"
    assert temporal_split("2021-01-01", **options) is None
    assert temporal_split(None, **options) is None
    with pytest.raises(ValueError, match="train < validation < test"):
        temporal_split("2020-01-01", **{**options, "training_cutoff": "2020-12-31"})


def test_missing_group_never_falls_back_to_context_id():
    with pytest.raises(ValueError, match="grouping field"):
        list(group_aware_splits([{"context_id": "c1"}], group_field="paper_group"))


def test_unresolved_paper_id_overlap_is_source_local_and_archive_scoped(tmp_path):
    def unresolved(
        key, split, archive="snapshot.tar.gz", dataset="scicite", namespace="scicite-paper-id"
    ):
        return context(
            key,
            "SYNTHETIC " + key,
            split,
            source_dataset=dataset,
            citing_identifier={"namespace": namespace, "raw": "source-paper-1"},
            provenance=[
                {
                    "source_dataset": dataset,
                    "dataset_version": None,
                    "source_file": archive + "!" + split + ".jsonl",
                }
            ],
        )

    with SplitAuditor(tmp_path / "split.sqlite") as audit:
        audit.add_context(unresolved("a", "train"))
        audit.add_context(unresolved("b", "dev"))
        audit.add_context(unresolved("c", "test", archive="different.tar.gz"))
        audit.add_context(unresolved("d", "test", dataset="different"))
        audit.add_context(unresolved("e", "test", namespace="different-paper-id"))
        report = audit.report()
    assert report["source_local_paper_overlap_groups"] == 1
    assert report["source_local_paper_identifier_occurrences"] == 5
    assert report["resolved_paper_identity_overlap_groups"] == 0
    assert report["paper_identity_overlap_groups"] == 0


def test_source_local_overlap_requires_scope_but_known_versions_isolate_releases(tmp_path):
    with SplitAuditor(tmp_path / "split.sqlite") as audit:
        for key, split, version in [
            ("a", "train", "1"),
            ("b", "dev", "1"),
            ("c", "test", "2"),
            ("d", "test", None),
        ]:
            audit.add_context(
                context(
                    key,
                    "SYNTHETIC " + key,
                    split,
                    source_dataset="scicite",
                    dataset_version=version,
                    cited_identifier={"namespace": "scicite-paper-id", "raw": "source-paper-2"},
                )
            )
        report = audit.report()
    assert report["source_local_paper_overlap_groups"] == 1
    assert report["source_local_identifiers_missing_scope"] == 1


def test_cleaner_supplied_context_normalization_still_gets_casefold_audit_key(tmp_path):
    with SplitAuditor(tmp_path / "split.sqlite") as audit:
        audit.add_context(
            context(
                "a",
                "SYNTHETIC Original α Context",
                "train",
                normalized_context="SYNTHETIC Original α Context",
            )
        )
        audit.add_context(
            context(
                "b",
                "synthetic original α context",
                "test",
                normalized_context="synthetic original α context",
            )
        )
        report = audit.report()
    assert report["exact_text_overlap_groups"] == 0
    assert report["normalized_text_overlap_groups"] == 1
