"""API contracts, genuine artifact consistency and separated synthetic UI behavior."""

import json
import sqlite3

import pytest
from app.adapters.artifacts import RealReader, safe_path, synthetic
from app.schemas import DEMO_NOTICE, Envelope, Neighborhood, Overview
from conftest import DASHBOARD

ROUTES = [
    "overview",
    "datasets",
    "citation-intents",
    "topics",
    "graph/stats",
    "graph/nodes",
    "graph/neighborhood",
    "progress",
    "validation/status",
]


@pytest.mark.parametrize("route", ROUTES)
def test_missing_artifacts_never_fall_back(missing, route):
    response = missing.get("/api/" + route)
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "real"
    assert result["status"] == "unavailable"
    assert result["data"] is None
    assert result["notice"]


def test_explicit_demo_works_without_real_artifacts(missing):
    data = missing.get("/api/overview?mode=demo").json()
    Envelope[Overview].model_validate(data)
    assert data["status"] == "available"
    assert data["mode"] == "demo"
    assert data["notice"] == DEMO_NOTICE
    assert missing.get("/api/overview").json()["status"] == "unavailable"


@pytest.mark.parametrize("route", ROUTES)
def test_real_contract_and_provenance(real, route):
    response = real.get("/api/" + route)
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "real"
    assert result["status"] == "available"
    assert result["graph_version"].startswith("graph-")
    assert not synthetic(result["data"])
    assert '"demo:' not in response.text
    assert "Synthetic Demo Data" not in response.text


def test_measured_counts_match_published_artifact(real):
    reader = RealReader(DASHBOARD.parent)
    data = real.get("/api/overview").json()
    Envelope[Overview].model_validate(data)
    assert data["data"]["counts"]["papers"] == reader.stats_report["node_counts"]["paper"]
    assert data["data"]["counts"]["citations"] == reader.edge_counts.get("cites", 0)
    assert data["data"]["graph"]["edges"] == sum(reader.edge_counts.values())
    assert data["data"]["provenance"]
    with reader.connection() as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly|read.only"):
            conn.execute("CREATE TABLE forbidden_dashboard_write(x TEXT)")


@pytest.mark.parametrize("route", ROUTES)
def test_demo_always_identified(client, route):
    response = client.get("/api/" + route, params={"mode": "demo"})
    result = response.json()
    assert response.status_code == 200, response.text
    assert result["mode"] == "demo"
    assert result["notice"] == DEMO_NOTICE
    data = result["data"]
    for collection in ("items", "nodes", "edges", "stages"):
        for record in data.get(collection, []):
            assert record["synthetic"] is True
            assert record["notice"] == DEMO_NOTICE


def test_demo_contains_no_model_evaluation_or_confidence(client):
    result = client.get("/api/validation/status?mode=demo").json()["data"]
    assert result["metrics"] == []
    records = client.get("/api/citation-intents?mode=demo").json()["data"]["items"]
    assert all(r["confidence"] is None and r["model_id"] == "No model" for r in records)


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
        "offset=100001",
        "mode=auto",
        "label=investigation",
        "q=" + "x" * 161,
    ],
)
def test_invalid_pagination_mode_and_taxonomy(client, query):
    assert client.get("/api/citation-intents?" + query).status_code == 422


@pytest.mark.parametrize(
    "query",
    [
        "limit=151",
        "depth=0",
        "depth=3",
        "node_types=unknown",
        "edge_types=author_of",
        "seed=" + "x" * 301,
    ],
)
def test_graph_resource_limits(client, query):
    assert client.get("/api/graph/neighborhood?mode=demo&" + query).status_code == 422


@pytest.mark.parametrize("mode", ["real", "demo"])
def test_graph_filters_limits_and_details(real, mode):
    result = real.get(
        "/api/graph/neighborhood",
        params={
            "mode": mode,
            "depth": 2,
            "limit": 7,
            "node_types": "topic,paper",
            "edge_types": "associated_with_topic",
        },
    ).json()
    Envelope[Neighborhood].model_validate(result)
    data = result["data"]
    ids = {n["id"] for n in data["nodes"]}
    assert 0 < len(ids) <= 7
    assert len(data["edges"]) <= 400
    assert all(n["type"] in ("topic", "paper") for n in data["nodes"])
    assert all(
        e["relation"] == "associated_with_topic" and e["source"] in ids and e["target"] in ids
        for e in data["edges"]
    )
    assert (
        real.get("/api/graph/neighborhood", params={"mode": mode, "node_types": ""}).json()["data"][
            "nodes"
        ]
        == []
    )
    assert (
        real.get("/api/graph/neighborhood", params={"mode": mode, "edge_types": ""}).json()["data"][
            "edges"
        ]
        == []
    )
    node = data["nodes"][0]
    detail = real.get("/api/graph/nodes/" + node["id"], params={"mode": mode}).json()["data"]
    assert detail["node"]["id"] == node["id"]
    assert detail["node"]["provenance"]
    assert len(detail["relations"]) <= 50


def test_real_citation_text_and_model_identity(real):
    first = real.get("/api/citation-intents?limit=1").json()["data"]
    second = real.get("/api/citation-intents?limit=1&offset=1").json()["data"]
    assert first["total"] > 0
    assert first["items"][0]["id"] != second["items"][0]["id"]
    row = first["items"][0]
    assert row["text"]
    assert row["provenance"]
    assert row["label"] in ("background", "method", "result")
    filtered = real.get(
        "/api/citation-intents", params={"q": row["text"][:30], "label": row["label"]}
    ).json()["data"]
    assert row["id"] in [item["id"] for item in filtered["items"]]
    assert all(item["label"] == row["label"] for item in filtered["items"])


def test_topic_membership_not_investigation(real):
    topic = real.get("/api/topics?limit=1").json()["data"]["items"][0]
    detail = real.get("/api/topics/" + topic["id"]).json()["data"]
    assert detail["provenance"]
    assert len(detail["papers"]) <= 20 and len(detail["evidence"]) <= 20
    assert "does not establish" in detail["membership_note"]
    assert all(e["tier"] in ("UNKNOWN", "RULE_CANDIDATE") for e in detail["evidence"])


def test_unknown_records_and_get_only(client):
    for route in ("graph/nodes/missing", "topics/missing"):
        assert client.get("/api/" + route + "?mode=demo").status_code == 404
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)("/api/overview").status_code == 405


def test_future_stages_remain_planned(real):
    progress = real.get("/api/progress").json()["data"]
    assert all(s["status"] == "Not Yet Implemented" for s in progress["stages"] if s["number"] > 6)
    assert "percentage" in progress["engineering_note"]
    assert all(
        s["status"] != "Verified Data" for s in progress["stages"] if s["number"] in (1, 2, 3, 5)
    )


def test_two_hop_budget_retains_heterogeneous_neighbors(real):
    result = real.get("/api/graph/neighborhood?depth=2&limit=70").json()["data"]
    assert len(result["nodes"]) <= 70
    assert {"paper", "author", "topic"} <= {n["type"] for n in result["nodes"]}


def test_malformed_and_escaping_pointers_are_unavailable(tmp_path):
    from app.main import create_app
    from fastapi.testclient import TestClient

    root = tmp_path / "data/processed/stage06"
    root.mkdir(parents=True)
    for pointer in ({}, {"graph_version": "../../../../outside"}):
        (root / "latest.json").write_text(json.dumps(pointer))
        data = TestClient(create_app(tmp_path)).get("/api/overview").json()
        assert data["status"] == "unavailable"
    with pytest.raises(Exception, match="outside"):
        safe_path(tmp_path, "../escape")


def test_synthetic_nested_provenance_is_rejected():
    assert synthetic({"unseen": {"provenance": [{"synthetic": True}]}})
    assert synthetic({"synthetic_sources": ["fixture"]})
    assert not synthetic({"provenance": [{"synthetic": False}]})
