"""Runner-level SYNTHETIC fixtures, never genuine scholarly integration evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ci_hgnn_rgd.data.integration import run_stage03
from ci_hgnn_rgd.data.integration_manifest import iter_table
from ci_hgnn_rgd.data.stage03_config import Stage03Config


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def settings(tmp_path, inputs, *, policy="STRICT_POINT_IN_TIME"):
    return Stage03Config.model_validate(
        {
            "integration": {
                "inputs": [{"synthetic": True, **source} for source in inputs],
                "discover_stage02": False,
                "output_dir": tmp_path / "runs",
                "report_dir": tmp_path / "reports",
                "storage_format": "jsonl",
                "batch_size": 2,
                "memory_budget_mb": 16,
            },
            "temporal_integrity": {
                "policy": policy,
                "cutoff": datetime(2020, 12, 31, 23, 59, 59, tzinfo=UTC),
                "max_near_comparisons": 20,
            },
        }
    )


def execute(project, stage03):
    result = run_stage03(project, stage03)
    root = Path(result["run_dir"])
    manifest = json.loads((root / "integration_manifest.json").read_text(encoding="utf-8"))
    return root, manifest


def test_partial_scicite_run_retains_unlinked_contexts_without_inventing_papers(tmp_path, config):
    path = write_rows(
        tmp_path / "train.jsonl",
        [
            {
                "id": "SYNTHETIC-c1",
                "string": "SYNTHETIC scientific α citation context.",
                "label": "method",
                "citingPaperId": "SYNTHETIC-source-p1",
                "citedPaperId": "SYNTHETIC-source-p2",
            }
        ],
    )
    root, manifest = execute(config, settings(tmp_path, [{"dataset": "scicite", "path": path}]))
    contexts = list(iter_table(root, "citation_contexts"))
    assert len(contexts) == 1
    assert contexts[0]["citing_paper_id"] is None
    assert contexts[0]["cited_paper_id"] is None
    assert contexts[0]["original_intent_label"] == "method"
    assert contexts[0]["normalized_intent_label"] == "Method"
    assert contexts[0]["source_split"] == "train"
    assert list(iter_table(root, "papers")) == []
    assert manifest["linkage"]["linked_contexts"] == 0
    assert manifest["linkage"]["unresolved_citation_endpoints"] == 2
    assert manifest["graph_integrity"]["edge_occurrences"] == 0
    assert manifest["split_audit"]["split_counts"] == {"train": 1}
    assert manifest["real_sources"] == []
    assert manifest["scientific_stages_executed"] == []


def test_observed_graph_indices_keep_splits_provenance_and_no_inferred_intent(tmp_path, config):
    path = write_rows(
        tmp_path / "ogbn.jsonl",
        [
            {
                "kind": "node",
                "node_id": 0,
                "node_type": "paper",
                "source_split": "train",
                "title": "SYNTHETIC source-local paper zero",
                "publication_year": 2018,
            },
            {
                "kind": "node",
                "node_id": 1,
                "node_type": "paper",
                "source_split": "test",
                "title": "SYNTHETIC source-local paper one",
                "publication_year": 2019,
            },
            {
                "kind": "citation_edge",
                "source": 0,
                "target": 1,
                "edge_type": "cites",
                "citation_intent": "method",
            },
        ],
    )
    root, manifest = execute(
        config,
        settings(
            tmp_path,
            [
                {
                    "dataset": "ogbn-arxiv",
                    "path": path,
                    "version": "v1",
                    "format": "graph_tables",
                }
            ],
        ),
    )
    cleaned_nodes = [r for r in iter_table(root, "cleaned_records") if r["kind"] == "paper"]
    assert {r["source_split"] for r in cleaned_nodes} == {"train", "test"}
    assert {r["source_record_id"] for r in cleaned_nodes} == {0, 1}
    assert all(r["dataset_version"] == "v1" for r in cleaned_nodes)
    citations = list(iter_table(root, "citations"))
    assert len(citations) == 1
    assert citations[0]["original_source_id"]["raw"] == 0
    assert citations[0]["original_target_id"]["raw"] == 1
    assert citations[0]["citation_intent"] is None
    assert citations[0]["citation_context_ids"] == []
    assert manifest["graph_integrity"]["valid_edge_occurrences"] == 1
    assert manifest["graph_integrity"]["invalid_edge_occurrences"] == 0
    assert manifest["split_audit"]["official_node_overlap_groups"] == 0
    assert {row["source_split"] for row in manifest["split_audit"]["official_node_splits"]} == {
        "train",
        "test",
    }
    assert not manifest["split_audit"]["official_splits_modified"]


def test_heterogeneous_paper_author_relation_resolves_only_scoped_local_nodes(tmp_path, config):
    path = write_rows(
        tmp_path / "mag.jsonl",
        [
            {
                "kind": "node",
                "node_id": 0,
                "node_type": "paper",
                "title": "SYNTHETIC heterogeneous paper",
            },
            {
                "kind": "node",
                "node_id": 0,
                "node_type": "author",
                "display_name": "SYNTHETIC Unverified Name",
            },
            {
                "kind": "relation",
                "source": 0,
                "target": 0,
                "source_node_type": "paper",
                "target_node_type": "author",
                "edge_type": "authored_by",
            },
        ],
    )
    root, manifest = execute(
        config,
        settings(
            tmp_path,
            [
                {
                    "dataset": "mag-year-h2gb",
                    "path": path,
                    "version": "v1",
                    "format": "graph_tables",
                }
            ],
        ),
    )
    authors = list(iter_table(root, "authors"))
    assert len(authors) == 1
    assert authors[0]["identity_status"] == "SOURCE_LOCAL_UNVERIFIED"
    relation = next(iter_table(root, "citations"))
    assert relation["source_id"] != relation["target_id"]
    assert relation["source_node_type"] == "paper"
    assert relation["target_node_type"] == "author"
    assert manifest["graph_integrity"]["invalid_edge_occurrences"] == 0
    assert manifest["graph_integrity"]["valid_edge_occurrences"] == 1


def test_strict_canonical_metadata_retains_future_and_unknown_availability_exclusions(
    tmp_path, config
):
    path = write_rows(
        tmp_path / "openalex.jsonl",
        [
            {
                "id": "https://openalex.org/W100",
                "title": "SYNTHETIC old publication future metadata",
                "abstract": "SYNTHETIC later abstract version",
                "publication_date": "2018-01-01",
                "updated_date": "2024-01-01",
                "available_at": "2018-01-01",
            },
            {
                "id": "https://openalex.org/W101",
                "title": "SYNTHETIC unknown metadata availability",
                "publication_date": "2018-01-01",
            },
        ],
    )
    root, _ = execute(config, settings(tmp_path, [{"dataset": "openalex", "path": path}]))
    papers = list(iter_table(root, "papers"))
    assert len(papers) == 2
    assert all(not paper["temporal_eligibility"]["eligible"] for paper in papers)
    future = next(paper for paper in papers if "future metadata" in paper["title"])
    assert "future_update_date" in future["temporal_eligibility"]["reasons"]
    unknown = next(paper for paper in papers if "unknown metadata" in paper["title"])
    assert unknown["temporal_eligibility"]["unknown_availability"]
    excluded = [
        row for row in iter_table(root, "excluded_records") if row["kind"] == "canonical_paper"
    ]
    assert {row["canonical_paper_id"] for row in excluded} == {
        row["canonical_paper_id"] for row in papers
    }


def test_exact_cross_source_doi_merge_preserves_both_source_mappings(tmp_path, config):
    left = write_rows(
        tmp_path / "openalex.jsonl",
        [
            {
                "id": "https://openalex.org/W100",
                "title": "SYNTHETIC shared scholarly identity",
                "doi": "https://doi.org/10.1234/SYNTHETIC",
                "publication_year": 2018,
            }
        ],
    )
    right = write_rows(
        tmp_path / "semantic.jsonl",
        [
            {
                "paperId": "a" * 40,
                "title": "SYNTHETIC shared scholarly identity",
                "year": 2018,
                "externalIds": {"DOI": "10.1234/synthetic"},
            }
        ],
    )
    root, manifest = execute(
        config,
        settings(
            tmp_path,
            [
                {"dataset": "openalex", "path": left},
                {"dataset": "semantic-scholar", "path": right},
            ],
        ),
    )
    papers = list(iter_table(root, "papers"))
    assert len(papers) == 1
    assert papers[0]["identity_status"] == "EXACT_VERIFIED"
    assert {item["source_dataset"] for item in papers[0]["source_record_ids"]} == {
        "openalex",
        "semantic-scholar",
    }
    assert len(list(iter_table(root, "identifier_mappings"))) == 2
    assert manifest["identity"]["automatic_cluster_merges"] == 1
    assert manifest["real_sources"] == []


def test_explicit_missing_source_fails_before_publishing_any_run(tmp_path, config):
    configured = settings(
        tmp_path, [{"dataset": "arxiv-metadata", "path": tmp_path / "absent.jsonl"}]
    )
    with pytest.raises(ValueError, match="no configured local source file"):
        run_stage03(config, configured)
    assert not configured.integration.output_dir.exists()
    assert not configured.integration.report_dir.exists()


def test_unsupported_scicite_label_is_quarantined_and_never_fabricated(tmp_path, config):
    path = write_rows(
        tmp_path / "train.jsonl",
        [
            {
                "id": "SYNTHETIC-invalid-label",
                "string": "SYNTHETIC context lacking a supported label.",
                "label": "invented-investigates",
            }
        ],
    )
    root, manifest = execute(config, settings(tmp_path, [{"dataset": "scicite", "path": path}]))
    quarantined = list(iter_table(root, "quarantine"))
    assert len(quarantined) == 1
    assert quarantined[0]["raw_values"]["label"] == "invented-investigates"
    assert quarantined[0]["normalized_intent_label"] is None
    assert not list(iter_table(root, "citation_contexts"))
    assert manifest["scientific_stages_executed"] == []
