"""Synthetic identifier cases; these tests do not verify a live scholarly source."""

import pytest

from ci_hgnn_rgd.data.normalization import identifier_token, normalize_identifier


@pytest.mark.parametrize(
    ("namespace", "raw", "normalized", "version"),
    [
        ("doi", " DOI:10.1000/AbC.(d) ", "10.1000/abc.(d)", None),
        ("doi", "https://doi.org/10.1000/ABC%3Bdef", "10.1000/abc;def", None),
        ("arxiv", "arXiv:2301.01234v2", "2301.01234", "v2"),
        ("arxiv", "https://arxiv.org/pdf/hep-th/9901001v3.pdf", "hep-th/9901001", "v3"),
        ("arxiv", "cs.AI/9912001v1", "cs.ai/9912001", "v1"),
        ("openalex", "https://openalex.org/w123", "W123", None),
        ("openalex-work-id", "W123", "W123", None),
        ("semantic-scholar", "A" * 40, "a" * 40, None),
        ("mag-paper-id", "123456789", "123456789", None),
    ],
)
def test_source_identifier_normalization_preserves_raw(namespace, raw, normalized, version):
    result = normalize_identifier(namespace, raw, verified=True)
    assert result["raw"] == raw
    assert result["normalized"] == normalized
    assert result["version"] == version
    assert result["valid"] and result["verified"]


@pytest.mark.parametrize(
    ("namespace", "raw"),
    [
        ("doi", "10.1/invalid"),
        ("doi", "10.1000/has space"),
        ("arxiv", "2300.12345"),
        ("arxiv", "2301.12345v0"),
        ("openalex", "A123"),
        ("semantic_scholar", "arbitrary-hash"),
        ("mag", "-1"),
        ("mag", True),
        ("doi", None),
    ],
)
def test_malformed_identifiers_remain_raw_but_invalid(namespace, raw):
    result = normalize_identifier(namespace, raw, verified=True)
    assert result["raw"] == raw
    assert not result["valid"] and not result["verified"]
    assert identifier_token(result) is None


def test_internal_indices_are_dataset_version_and_node_type_scoped():
    def internal(dataset="ogbn-arxiv", version="v1", node_type="paper"):
        return normalize_identifier(
            "ogbn-arxiv-node-index",
            123,
            dataset=dataset,
            version=version,
            node_type=node_type,
        )

    baseline = internal()
    assert baseline["scope"] == {"dataset": "ogbn-arxiv", "version": "v1", "node_type": "paper"}
    assert baseline["raw"] == 123 and baseline["normalized"] == "123"
    assert not baseline["verified"]
    assert (
        len(
            {
                identifier_token(result)
                for result in (
                    baseline,
                    internal(dataset="mag-year"),
                    internal(version="v2"),
                    internal(node_type="author"),
                )
            }
        )
        == 4
    )
    assert not internal(version=None)["valid"]
    assert not internal(version="unknown")["valid"]
    assert not internal(version="latest")["valid"]
    assert not internal(node_type=None)["valid"]


def test_external_namespaces_and_scope_components_cannot_collide():
    mag = normalize_identifier("mag", "123", verified=True)
    local = normalize_identifier("source_local", "123", dataset="x", version="v1", verified=True)
    assert identifier_token(mag) != identifier_token(local)
    first = normalize_identifier("source_local", "1", dataset="a|b", version="c")
    second = normalize_identifier("source_local", "1", dataset="a", version="b|c")
    assert identifier_token(first) != identifier_token(second)


def test_syntax_validation_does_not_invent_source_verification():
    result = normalize_identifier("doi", "10.1000/abc")
    assert result["valid"]
    assert not result["verified"]
