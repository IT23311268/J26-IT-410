"""Opt-in actual scholarly graph round trip; never falls back to synthetic records."""

import os
from pathlib import Path

import pytest

from ci_hgnn_rgd.graph.pyg import load_hetero_graph, validate_heterodata


@pytest.mark.integration
def test_published_real_stage06_graph():
    configured = os.environ.get("CI_HGNN_STAGE06_GRAPH")
    if not configured:
        pytest.skip("set CI_HGNN_STAGE06_GRAPH to an actual published graph directory")
    root = Path(configured)
    import json

    manifest = json.loads((root / "manifests/graph_manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"]["node_counts"]["paper"] > 0
    from ci_hgnn_rgd.graph.inputs import rows

    for node in rows(root / "nodes/paper.parquet"):
        assert not any(item.get("synthetic") for item in node["provenance"])
    graph = load_hetero_graph(root)
    assert validate_heterodata(graph)["status"] == "PASSED"
