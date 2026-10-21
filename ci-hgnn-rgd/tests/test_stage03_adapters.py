"""Synthetic local adapter cases; genuine integrations are tested separately."""

import json
from hashlib import sha256

import pytest
from stage02_helpers import arxiv_record, scicite_record

from ci_hgnn_rgd.data.inspection import inspect_text
from ci_hgnn_rgd.data.integration_adapters import iter_source_records


def write_records(path, records):
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    return path


def test_scicite_preserves_raw_context_split_and_source_local_ids(tmp_path):
    raw = scicite_record(string="  Synthetic\ncontext [7] α².  ")
    path = write_records(tmp_path / "train.jsonl", [raw])
    record = next(iter_source_records("scicite", path, synthetic=True))
    assert record["original_context"] == raw["string"]
    assert record["raw_values"] == raw
    assert record["source_split"] == "train"
    assert record["citing_identifier"]["namespace"] == "scicite-paper-id"
    assert not record["citing_identifier"]["verified"]
    assert record["provenance"][0]["synthetic"]
    assert record["original_intent_label"] == "background"


def test_scicite_same_paper_pair_does_not_collapse_distinct_contexts(tmp_path):
    path = write_records(
        tmp_path / "test.jsonl",
        [
            scicite_record(string="Synthetic first context."),
            scicite_record(string="Synthetic second context."),
        ],
    )
    records = list(iter_source_records("scicite", path, synthetic=True))
    assert len(records) == 2
    assert records[0]["record_key"] != records[1]["record_key"]


def test_scicite_nonstandard_constants_remain_recoverable_and_quarantined(tmp_path):
    path = write_records(tmp_path / "dev.jsonl", [scicite_record(sectionName=float("nan"))])
    record = next(iter_source_records("scicite", path, synthetic=True))
    assert record["quarantine"]
    assert record["raw_values"]["sectionName"] == {"__source_nonfinite__": "NaN"}
    assert record["original_context"]
    assert "nonstandard_json_constant" in record["issues"]
    assert record["provenance"][0]["record_sha256"]
    json.dumps(record, allow_nan=False)


def test_malformed_source_rows_are_yielded_as_recoverable_quarantines(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_bytes(b'{bad json\n{"string":"Synthetic", "label":"method"}\n')
    records = list(iter_source_records("scicite", path, synthetic=True))
    assert len(records) == 2
    assert records[0]["kind"] == "invalid"
    assert records[0]["quarantine"]
    assert records[0]["provenance"][0]["locator"].endswith(":line:1")
    assert records[1]["normalized_intent_label"] == "Method"


def test_arxiv_metadata_preserves_versions_author_order_categories_and_dates(tmp_path):
    raw = arxiv_record(title="  Synthetic α: $x_i$  ")
    path = write_records(tmp_path / "snapshot.jsonl", [raw])
    record = next(iter_source_records("arxiv", path, version="synthetic-v1", synthetic=True))
    assert record["title"] == "Synthetic α: $x_i$"
    assert record["raw_values"]["title"] == raw["title"]
    assert record["publication_date"] == "2020-01-01"
    assert record["date_precision"] == "day"
    assert record["update_date"] == raw["update_date"]
    assert record["versions"] == raw["versions"]
    identifiers = {item["namespace"]: item for item in record["identifiers"]}
    assert identifiers["doi"]["verified"]
    assert identifiers["arxiv"]["normalized"] == "2001.00001"
    assert identifiers["arxiv"]["version"] == "v2"


def test_explicit_stage02_adapter_keeps_original_locator_and_api(tmp_path):
    path = write_records(tmp_path / "train.jsonl", [scicite_record()])
    inspected = inspect_text("scicite", path, tmp_path / "inspection", synthetic=True)
    record = next(
        iter_source_records(
            "scicite",
            inspected["staging_path"],
            format="stage02",
            synthetic=True,
        )
    )
    assert record["original_context"] == scicite_record()["string"]
    assert record["source_split"] == "train"
    assert len(record["provenance"]) == 2
    assert "original_raw_fields_in_source_locator_not_envelope" in record["issues"]


def test_openalex_metadata_uses_explicit_external_namespace_and_retains_raw(tmp_path):
    raw = {
        "id": "https://openalex.org/W123",
        "title": " Synthetic Work ",
        "ids": {
            "doi": "https://doi.org/10.1234/SYNTHETIC",
            "openalex": "https://openalex.org/W123",
        },
        "authorships": [
            {"author": {"display_name": "Second"}},
            {"author": {"display_name": "First"}},
        ],
        "publication_date": "2020-03",
        "publication_year": 2020,
        "abstract_inverted_index": {"Synthetic": [0]},
    }
    path = write_records(tmp_path / "metadata.jsonl", [raw])
    record = next(iter_source_records("openalex", path, synthetic=True))
    assert record["authors"] == ["Second", "First"]
    assert record["abstract"] is None
    assert record["raw_values"]["abstract_inverted_index"] == {"Synthetic": [0]}
    assert record["date_precision"] == "month"
    assert {item["namespace"] for item in record["identifiers"]} == {"doi", "openalex"}


def test_semantic_scholar_metadata_does_not_invent_contexts_or_labels(tmp_path):
    raw = {
        "paperId": "a" * 40,
        "title": "Synthetic paper",
        "year": 2018,
        "externalIds": {"ArXiv": "1801.00001v2", "MAG": "123"},
        "authors": [{"name": "Synthetic Author", "authorId": "44"}],
    }
    path = write_records(tmp_path / "s2.jsonl", [raw])
    record = next(iter_source_records("semantic-scholar", path, synthetic=True))
    assert record["authors"] == ["Synthetic Author"]
    assert record["publication_date"] is None
    assert "original_context" not in record
    assert {item["namespace"] for item in record["identifiers"]} == {
        "arxiv",
        "mag",
        "semantic_scholar",
    }


def test_graph_exports_preserve_source_namespace_version_and_split(tmp_path):
    path = write_records(
        tmp_path / "nodes.jsonl",
        [
            {"kind": "node", "node_id": 123, "node_type": "paper", "split": "train"},
            {"kind": "edge", "source": 123, "target": 124, "edge_type": "cites"},
        ],
    )
    records = list(iter_source_records("ogbn-arxiv", path, version="v1", synthetic=True))
    assert records[0]["source_split"] == "train"
    assert records[0]["node_identifier"]["namespace"] == "internal_graph"
    assert records[0]["node_identifier"]["original_namespace"] == "ogbn-arxiv-node-index"
    assert records[0]["node_identifier"]["scope"]["dataset"] == "ogbn-arxiv"
    assert records[1]["original_source_id"] == 123
    assert records[1]["citation_intent_status"] == "UNOBSERVED"
    assert records[1]["source_identifier"]["scope"]["version"] == "v1"


def test_graph_namespace_never_promotes_identical_index_across_sources(tmp_path):
    path = write_records(
        tmp_path / "nodes.jsonl",
        [
            {"kind": "node", "node_id": 123, "node_type": "paper"},
        ],
    )
    ogb = next(iter_source_records("ogbn-arxiv", path, version="v1", synthetic=True))
    mag = next(iter_source_records("mag-year", path, version="v1", synthetic=True))
    assert ogb["identifiers"][0]["scope"]["dataset"] != mag["identifiers"][0]["scope"]["dataset"]


def test_unsafe_graph_files_and_missing_inputs_fail_explicitly(tmp_path):
    path = tmp_path / "graph.pt"
    path.write_bytes(b"not executed")
    with pytest.raises(ValueError, match="safe JSONL"):
        list(iter_source_records("ogbn-arxiv", path))
    with pytest.raises(FileNotFoundError):
        list(iter_source_records("arxiv", tmp_path / "missing.jsonl"))


def test_max_records_is_bounded_and_record_keys_are_repeatable(tmp_path):
    path = write_records(
        tmp_path / "train.jsonl", [scicite_record(id=str(index)) for index in range(5)]
    )
    first = list(iter_source_records("scicite", path, max_records=2, synthetic=True))
    second = list(iter_source_records("scicite", path, max_records=2, synthetic=True))
    assert len(first) == 2
    assert [row["record_key"] for row in first] == [row["record_key"] for row in second]


def test_acquisition_timestamp_requires_matching_source_checksum(tmp_path):
    path = write_records(tmp_path / "train.jsonl", [scicite_record()])
    sidecar = path.with_name(path.name + ".source.json")
    sidecar.write_text(
        json.dumps(
            {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "downloaded_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    record = next(iter_source_records("scicite", path, synthetic=True))
    assert record["provenance"][0]["acquired_at"] == "2026-01-01T00:00:00Z"
    assert record["provenance"][0]["available_at"] is None
    path.write_text(json.dumps(scicite_record(id="changed")))
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(iter_source_records("scicite", path, synthetic=True))


def test_graph_tables_format_preserves_nonpaper_source_local_nodes(tmp_path):
    path = write_records(
        tmp_path / "nodes.jsonl",
        [
            {"kind": "node", "node_type": "institution", "node_id": 12},
        ],
    )
    record = next(
        iter_source_records(
            "mag-year",
            path,
            version="synthetic-v1",
            format="graph_tables",
            synthetic=True,
        )
    )
    assert record["kind"] == "graph_node"
    assert record["node_identifier"]["scope"]["node_type"] == "institution"
    assert record["source_record_id"] == 12


def test_openalex_updated_date_is_separate_from_publication(tmp_path):
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Synthetic paper",
        "publication_date": "2018-01-01",
        "updated_date": "2026-01-01",
    }
    path = write_records(tmp_path / "metadata.jsonl", [raw])
    record = next(iter_source_records("openalex", path, synthetic=True))
    assert record["publication_date"] == "2018-01-01"
    assert record["source_updated_date"] == record["update_date"] == "2026-01-01"


def test_declared_graph_author_and_field_records_preserve_source_metadata(tmp_path):
    path = write_records(
        tmp_path / "nodes.jsonl",
        [
            {"kind": "author", "node_id": 2, "name": "  Synthetic Author  "},
            {
                "kind": "source_field",
                "source_field_id": 3,
                "field_name": "Synthetic field",
                "field_taxonomy": "observed-source-taxonomy",
            },
        ],
    )
    author, field = iter_source_records(
        "mag-year",
        path,
        version="synthetic-v1",
        format="graph_tables",
        synthetic=True,
    )
    assert author["kind"] == "author"
    assert author["display_name"] == "Synthetic Author"
    assert author["identity_status"] == "SOURCE_LOCAL_UNVERIFIED"
    assert field["kind"] == "source_field"
    assert field["source_field_id"] == 3
    assert field["field_taxonomy"] == "observed-source-taxonomy"
