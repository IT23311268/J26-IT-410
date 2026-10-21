import pytest
from stage02_helpers import Array, HeteroFixture

from ci_hgnn_rgd.data.download_manager import AcquisitionBlocked
from ci_hgnn_rgd.data.sources.mag_year import MAGYearAdapter


def test_only_observed_node_types_and_attributes_are_reported():
    report = MAGYearAdapter().inspect([HeteroFixture()])
    assert report["node_types"] == ["paper", "author"]
    assert "institution" not in report["nodes"]
    assert report["nodes"]["paper"]["year_shape"] is None
    assert not report["issues"]


def test_heterogeneous_edge_bounds_and_masks():
    graph = HeteroFixture()
    graph.stores["author", "writes", "paper"]["edge_index"] = Array([[5], [0]])
    graph.stores["paper"]["val_mask"] = Array([True, False], "bool")
    issues = MAGYearAdapter().inspect([graph])["issues"]
    assert "edge_endpoint_out_of_bounds" in issues
    assert "overlapping_split_masks:paper" in issues


def test_h2gb_constructor_never_requests_random_resplit(tmp_path):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return [HeteroFixture()]

    MAGYearAdapter().load(tmp_path, factory=factory)
    assert seen == {"root": str(tmp_path), "name": "mag-year", "rand_split": False}


def test_real_loading_missing_dependencies_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("ci_hgnn_rgd.data.sources.mag_year.find_spec", lambda _: None)
    with pytest.raises(AcquisitionBlocked, match="dependencies unavailable"):
        MAGYearAdapter().load(tmp_path)
