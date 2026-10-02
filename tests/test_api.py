from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_then_fetch(tmp_path: Path, monkeypatch):
    # Redirect storage to a scratch dir so this test never touches the
    # real data/processed/ folder.
    from api import storage

    scratch = tmp_path / "processed"
    monkeypatch.setattr(storage, "PROCESSED_DIR", scratch)

    from scripts.generate_sample_pdf import build_sample_pdf

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)

    with pdf_path.open("rb") as f:
        resp = client.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["paper"]["page_count"] == 3
    assert len(body["chunks"]) > 0

    paper_id = body["paper"]["paper_id"]

    fetch_resp = client.get(f"/paper/{paper_id}")
    assert fetch_resp.status_code == 200
    assert fetch_resp.json()["paper"]["paper_id"] == paper_id

    list_resp = client.get("/papers")
    assert paper_id in list_resp.json()["paper_ids"]


def test_ingest_rejects_non_pdf():
    resp = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_fetch_unknown_paper_is_404():
    resp = client.get("/paper/does-not-exist")
    assert resp.status_code == 404
