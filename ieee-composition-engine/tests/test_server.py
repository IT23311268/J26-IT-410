"""
Tests for server.py - the web API the React page talks to.

Run with:   pytest -q

The API must only CALL the engine: these tests compose the sample paper
through the API and check the answers match what the engine itself gives.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["IEEE_OUTPUT_DIR"] = str(tmp_path_factory.mktemp("server"))
    os.environ["GEMINI_API_KEY"] = ""                      # rules only
    import importlib
    import server
    importlib.reload(server)
    from starlette.testclient import TestClient
    return TestClient(server.app)


@pytest.fixture(scope="module")
def job(client):
    if not os.path.exists(os.path.join(ROOT, "data", "input", "sample_paper.pdf")):
        pytest.skip("run tools/make_sample_paper.py first")
    job_id = client.post("/api/jobs", json={"paper": "sample_paper.pdf", "gemini": False,
                                             "paper_size": "letter"}).json()["job_id"]
    for _ in range(300):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "done", state
    return job_id, state


def test_settings_lists_the_papers(client):
    data = client.get("/api/settings").json()
    assert data["gemini_key"] is False
    assert data["papers"][0]["name"] == "sample_paper.pdf"


def test_all_seven_stages_finish(job):
    _, state = job
    assert [s["state"] for s in state["stages"]] == ["done"] * 7


def test_result_matches_the_engine(client, job):
    job_id, _ = job
    data = client.get(f"/api/jobs/{job_id}/result").json()
    assert data["title"].startswith("Automated Detection of Pulmonary Nodules")
    assert data["score"]["after"] > data["score"]["before"]
    assert len(data["rubric"]) == len(data["rulebook"]["rules"])
    assert data["visuals"][0]["image"].endswith(".png")


def test_preview_highlights_a_rule(client, job):
    job_id, _ = job
    page = client.get(f"/api/jobs/{job_id}/preview", params={"rule": "FIG-02", "place": 0}).text
    assert '<mark class="hl hl-cur" id="hl-current">Fig. 1.</mark>' in page


def test_downloads(client, job):
    job_id, _ = job
    assert client.get(f"/api/jobs/{job_id}/file/docx").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/file/annotated").status_code == 200
    xlsx = client.post(f"/api/jobs/{job_id}/rubric.xlsx", json={"decisions": {"LAY-01": True}})
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"      # an Excel (zip) file


def test_upload_and_bad_input(client):
    saved = client.post("/api/upload?filename=my paper.docx",
                        content=open(os.path.join(ROOT, "data", "input", "sample_draft.docx"), "rb").read())
    assert saved.json()["name"] == "uploads/my_paper.docx"
    assert client.post("/api/upload?filename=virus.exe", content=b"x").status_code == 400
    assert client.post("/api/jobs", json={"paper": "../server.py"}).json()["error"] == "Unknown paper."
    assert client.get("/api/jobs/nope").status_code == 404
