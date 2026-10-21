import pytest

from ci_hgnn_rgd.data.identifier_audit import (
    IdentifierIndex,
    audit_indices,
    normalize_arxiv,
    normalize_doi,
)
from ci_hgnn_rgd.schemas import ExternalIdentifier


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" DOI:10.1234/TEST ", "10.1234/test"),
        ("https://doi.org/10.1234/ABC", "10.1234/abc"),
        ("http://dx.doi.org/10.1234/ABC", "10.1234/abc"),
        ("not a DOI", None),
        (None, None),
    ],
)
def test_doi_normalization(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    "raw,base,version",
    [
        ("arXiv:2301.01234v3", "2301.01234", "v3"),
        ("https://arxiv.org/pdf/hep-th/9901001v2.pdf", "hep-th/9901001", "v2"),
        ("math.GT/0301001", "math.gt/0301001", None),
        ("0704.0001", "0704.0001", None),
    ],
)
def test_arxiv_normalization_and_version(raw, base, version):
    result = normalize_arxiv(raw)
    assert (result.base, result.version) == (base, version)


@pytest.mark.parametrize(
    "raw", ["2313.12345", "arxiv:invalid", "https://example.org/2301.12345", None]
)
def test_invalid_arxiv_not_reinterpreted(raw):
    assert normalize_arxiv(raw) is None


def test_exact_ambiguous_title_candidate_and_namespace_safety(tmp_path):
    paths = {name: tmp_path / f"{name}.sqlite" for name in ("left", "right")}
    for name, path in paths.items():
        index = IdentifierIndex(path)
        index.add_entity("exact", (ExternalIdentifier(namespace="doi", value="10.1234/shared"),))
        index.add_entity("candidate", (), title="SYNTHETIC same title")
        index.add_entity(
            "internal", (ExternalIdentifier(namespace=name + "-node-index", value="1"),)
        )
        index.add_entity("ambiguous", (ExternalIdentifier(namespace="arxiv", value="2001.00001"),))
        if name == "left":
            index.add_entity(
                "ambiguous2", (ExternalIdentifier(namespace="arxiv", value="2001.00001"),)
            )
        index.close()
    report = audit_indices(paths, tmp_path / "audit.sqlite")
    pair = report["pairs"][0]
    assert pair["EXACT_VERIFIED"] == 1
    assert pair["AMBIGUOUS"] == 1
    assert pair["CANDIDATE_title_keys"] == 1
    assert "left-node-index" not in pair["common_namespaces"]
    assert pair["coverage"]["left"]["exact_identifier_covered"] == 1
    assert pair["coverage"]["left"]["estimated_integration_coverage"] == 1 / 5


def test_no_dataset_pair_means_overlap_is_not_measured(tmp_path):
    assert audit_indices({}, tmp_path / "empty.sqlite")["pairs"] == []
