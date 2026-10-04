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


def test_ingest_rasterises_pages_and_serves_them(tmp_path: Path, monkeypatch):
    from api import storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    from scripts.generate_sample_pdf import build_sample_pdf

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)

    with pdf_path.open("rb") as f:
        resp = client.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["pages"]) == 3
    assert [p["page_index"] for p in body["pages"]] == [0, 1, 2]

    paper_id = body["paper"]["paper_id"]

    img = client.get(f"/paper/{paper_id}/page/1")
    assert img.status_code == 200, img.text
    assert img.headers["content-type"] == "image/png"
    assert img.content[:8] == b"\x89PNG\r\n\x1a\n"

    missing = client.get(f"/paper/{paper_id}/page/99")
    assert missing.status_code == 404


def test_ingest_detects_layout_and_serves_the_overlay(tmp_path: Path, monkeypatch):
    from api import storage
    from scripts.generate_two_column_pdf import build_two_column_pdf

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    pdf_path = tmp_path / "two_column.pdf"
    build_two_column_pdf(pdf_path)

    with pdf_path.open("rb") as f:
        resp = client.post(
            "/ingest", files={"file": ("two_column.pdf", f, "application/pdf")}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["regions"]) > 0
    assert body["schema_version"] == "1.2.0"
    # reading order recovered: the left column before the right
    texts = [r["text"] for r in body["regions"]]
    joined = " ".join(texts)
    assert joined.index("ALPHA") < joined.index("BETA") < joined.index("GAMMA")

    paper_id = body["paper"]["paper_id"]

    overlay = client.get(f"/paper/{paper_id}/page/0/layout")
    assert overlay.status_code == 200, overlay.text
    assert overlay.headers["content-type"] == "image/png"
    assert overlay.content[:8] == b"\x89PNG\r\n\x1a\n"
    # the overlay is drawn on top of the page, so it is never smaller
    plain = client.get(f"/paper/{paper_id}/page/0")
    assert len(overlay.content) > len(plain.content) * 0.5


def test_overlay_404s_when_layout_was_not_run(tmp_path: Path, monkeypatch):
    from api import storage
    from scripts.generate_sample_pdf import build_sample_pdf

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)
    with pdf_path.open("rb") as f:
        resp = client.post(
            "/ingest",
            files={"file": ("sample.pdf", f, "application/pdf")},
            params={"detect_layout": "false"},
        )
    assert resp.json()["regions"] == []

    paper_id = resp.json()["paper"]["paper_id"]
    overlay = client.get(f"/paper/{paper_id}/page/0/layout")
    assert overlay.status_code == 404
    assert "detect_layout=true" in overlay.json()["detail"]


def test_ingest_can_skip_rasterisation(tmp_path: Path, monkeypatch):
    """Text-only mode stays available — it is the cheap path for the
    baseline-vs-layout-aware evaluation run."""
    from api import storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    from scripts.generate_sample_pdf import build_sample_pdf

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)

    with pdf_path.open("rb") as f:
        resp = client.post(
            "/ingest",
            files={"file": ("sample.pdf", f, "application/pdf")},
            params={"rasterise": "false"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["pages"] == []


def test_ingest_rejects_non_pdf():
    resp = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_fetch_unknown_paper_is_404():
    resp = client.get("/paper/does-not-exist")
    assert resp.status_code == 404


def test_corrupt_record_gives_an_actionable_message_not_a_bare_500(
    tmp_path: Path, monkeypatch
):
    """A zero-byte record used to surface as an opaque pydantic traceback.
    It must now come back as a message that says what to do."""
    from api import storage

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(storage, "PROCESSED_DIR", processed)
    (processed / "damaged.json").write_text("", encoding="utf-8")

    resp = client.get("/paper/damaged")
    assert resp.status_code == 500
    assert "Re-ingest" in resp.json()["detail"]

    page_resp = client.get("/paper/damaged/page/0")
    assert page_resp.status_code == 500
    assert "Re-ingest" in page_resp.json()["detail"]
