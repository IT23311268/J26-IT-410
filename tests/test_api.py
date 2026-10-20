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

    from schema.ingestion_schema_v1 import SCHEMA_VERSION

    assert len(body["regions"]) > 0
    # compared against the constant, not a literal: this test is about
    # layout and the overlay, and should not fail every time the schema
    # version moves for an unrelated reason
    assert body["schema_version"] == SCHEMA_VERSION
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


# --------------------------------------------------------------------------
# the React client
# --------------------------------------------------------------------------


def test_the_walkthrough_endpoint_carries_its_own_baseline(tmp_path: Path, monkeypatch):
    """One request, because the client needs both runs to draw the
    comparison and two round trips would let it render half of it.

    The baseline is re-run here rather than stored, the same way
    /pipeline and /compare do it. A stored copy would quietly flatter
    us as the baseline changed underneath it.
    """
    from api import storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    from scripts.generate_sample_pdf import build_sample_pdf

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)
    with pdf_path.open("rb") as f:
        paper_id = client.post(
            "/ingest", files={"file": ("sample.pdf", f, "application/pdf")}
        ).json()["paper"]["paper_id"]

    resp = client.get(f"/paper/{paper_id}/walkthrough")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert set(body) == {"result", "baseline", "source_bytes"}
    assert body["result"]["paper"]["paper_id"] == paper_id
    assert body["source_bytes"] == pdf_path.stat().st_size

    # The two runs have to be distinguishable, or the comparison at the
    # end of the walkthrough is comparing a thing with itself.
    assert body["baseline"] is not None
    assert len(body["baseline"]["chunks"]) != len(body["result"]["chunks"])


def test_the_walkthrough_still_answers_without_a_source_pdf(tmp_path: Path, monkeypatch):
    """Seven stages are worth more than the eighth.

    A paper ingested before sources were kept cannot have its baseline
    re-run. That is a state the client renders a sentence for, not an
    error — so this returns 200 with a null baseline rather than a 409
    that would blank the whole walkthrough.
    """
    from api import storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    from scripts.generate_sample_pdf import build_sample_pdf

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)
    with pdf_path.open("rb") as f:
        paper_id = client.post(
            "/ingest", files={"file": ("sample.pdf", f, "application/pdf")}
        ).json()["paper"]["paper_id"]

    monkeypatch.setattr(storage, "source_path", lambda _pid: None)

    resp = client.get(f"/paper/{paper_id}/walkthrough")
    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline"] is None
    assert body["source_bytes"] is None
    assert body["result"]["chunks"], "the seven stages that do not need a baseline"


def test_the_walkthrough_endpoint_404s_for_an_unknown_paper():
    assert client.get("/paper/not-a-paper/walkthrough").status_code == 404


def test_the_react_client_is_served_and_the_server_views_still_are():
    """Both front ends, on purpose.

    /pipeline needs no build step and cannot fail at a presentation;
    /app is the one the group develops from now on. Deleting either
    before the React view has been presented once would leave the demo
    with no fallback.
    """
    served = client.get("/app/")
    assert served.status_code in (200, 503), served.text
    if served.status_code == 503:
        # A checkout that has not run `npm run build` says which command
        # is missing instead of 404-ing, which would read as a broken route.
        assert "npm run build" in served.text
    else:
        assert '<div id="root">' in served.text

    # The server-rendered front door is untouched by any of this.
    assert client.get("/").status_code == 200
