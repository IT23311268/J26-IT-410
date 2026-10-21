import pytest
from stage02_helpers import Array, OGBFixture

from ci_hgnn_rgd.data.download_manager import AcquisitionBlocked
from ci_hgnn_rgd.data.sources.ogbn_arxiv import OGBNArxivAdapter, validate_splits


def test_miniature_official_interface_shapes_and_direction():
    report = OGBNArxivAdapter().inspect(OGBFixture())
    assert not report["issues"]
    assert report["node_types"] == ["paper"]
    assert report["feature_shape"] == (3, 2)
    assert report["edge_count"] == 2
    assert report["official_splits"]["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert report["identifier_namespace"] == "ogbn-arxiv-node-index"


def test_invalid_edge_endpoints_and_row_counts():
    dataset = OGBFixture()
    dataset.graph["edge_index"] = Array([[0, 3], [1, -1]])
    dataset.labels = Array([[0]])
    report = OGBNArxivAdapter().inspect(dataset)
    assert "edge_endpoint_out_of_bounds" in report["issues"]
    assert "labels_row_count_mismatch_or_missing" in report["issues"]


def test_invalid_duplicate_overlapping_splits():
    splits = {"train": Array([0, 0]), "valid": Array([0, 4]), "test": Array([2])}
    issues = validate_splits(splits, 3)["issues"]
    assert "duplicate_split_index:train" in issues
    assert "overlapping_split_index:valid" in issues
    assert "split_index_out_of_bounds:valid" in issues


def test_bounded_edge_scope_is_explicit():
    report = OGBNArxivAdapter().inspect(OGBFixture(), max_edges=1)
    assert report["partial"] and report["edge_integrity"]["checked_edges"] == 1


def test_official_constructor_arguments_are_preserved(tmp_path):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return OGBFixture()

    dataset = OGBNArxivAdapter().load(tmp_path, factory=factory)
    assert seen == {"name": "ogbn-arxiv", "root": str(tmp_path)}
    assert not OGBNArxivAdapter().inspect(dataset)["issues"]


def test_missing_graph_environment_is_blocked_not_mock_validated(tmp_path, monkeypatch):
    monkeypatch.setattr("ci_hgnn_rgd.data.sources.ogbn_arxiv.find_spec", lambda _: None)
    with pytest.raises(AcquisitionBlocked, match="optional"):
        OGBNArxivAdapter().load(tmp_path)
