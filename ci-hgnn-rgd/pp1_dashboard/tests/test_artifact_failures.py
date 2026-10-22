"""Adversarial contract fixtures are synthetic; none are research findings."""

import hashlib
import json
import sqlite3

import pytest
from app.adapters.artifacts import ArtifactUnavailable, RealReader, verify
from app.main import create_app
from fastapi.testclient import TestClient


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_published_graph(root):
    base = root / "data/processed/stage06"
    graph = base / "graph-synthetic-contract-test"
    (graph / "manifests").mkdir(parents=True)
    (graph / "reports").mkdir()
    db = graph / "graph.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE nodes(kind TEXT,key TEXT,payload TEXT);
        CREATE TABLE edges(rel TEXT,derived INTEGER,payload TEXT);
        CREATE TABLE evidence(payload TEXT);
    """)
    conn.execute(
        "INSERT INTO nodes VALUES (?,?,?)",
        (
            "paper",
            "synthetic:test",
            json.dumps(
                {"title": "Synthetic contract fixture", "provenance": [{"synthetic": True}]}
            ),
        ),
    )
    conn.commit()
    conn.close()
    files = {
        "reports/graph_integrity.json": {"status": "PASSED", "critical_violations": 0},
        "reports/graph_statistics.json": {"node_counts": {"paper": 1}, "scholarly_edge_counts": {}},
    }
    for name, payload in files.items():
        (graph / name).write_text(json.dumps(payload))
    manifest = {
        "schema_version": "stage06-v1",
        "status": "COMPLETE",
        "graph_version": graph.name,
        "artifact_checksums": {name: hash_file(graph / name) for name in ("graph.sqlite", *files)},
        "input_checksums": {},
    }
    (graph / "manifests/graph_manifest.json").write_text(json.dumps(manifest))
    (base / "latest.json").write_text(json.dumps({"graph_version": graph.name}))
    return graph


def test_hidden_synthetic_provenance_rejects_real_graph(tmp_path):
    synthetic_published_graph(tmp_path)
    with pytest.raises(ArtifactUnavailable, match="Synthetic"):
        RealReader(tmp_path)
    response = TestClient(create_app(tmp_path)).get("/api/overview").json()
    assert response["mode"] == "real" and response["status"] == "unavailable"
    assert response["data"] is None


def test_tampered_database_fails_before_reading(tmp_path):
    graph = synthetic_published_graph(tmp_path)
    with (graph / "graph.sqlite").open("ab") as stream:
        stream.write(b"tampered")
    response = TestClient(create_app(tmp_path)).get("/api/graph/stats").json()
    assert response["status"] == "unavailable"
    assert "checksum" in response["notice"]


def test_incomplete_graph_is_not_accepted(tmp_path):
    graph = synthetic_published_graph(tmp_path)
    path = graph / "manifests/graph_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "PENDING"
    path.write_text(json.dumps(manifest))
    assert TestClient(create_app(tmp_path)).get("/api/overview").json()["status"] == "unavailable"


def test_cached_artifacts_must_remain_unchanged(tmp_path):
    path = tmp_path / "published.json"
    path.write_text('{"original":true}')
    reader = RealReader.__new__(RealReader)
    reader.watched = {}
    reader.watch(path)
    reader.assert_unchanged()
    path.write_text('{"changed":true,"new":true}')
    with pytest.raises(ArtifactUnavailable, match="changed"):
        reader.assert_unchanged()


def test_missing_and_mismatched_checksums_fail_closed(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    for expected in (None, "0" * 64):
        with pytest.raises(ArtifactUnavailable):
            verify(path, expected)
