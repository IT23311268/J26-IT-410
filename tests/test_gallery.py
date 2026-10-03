from pathlib import Path

from fastapi.testclient import TestClient

from api.gallery import render_gallery
from api.main import app
from schema.ingestion_schema_v1 import (
    ExtractionMethod,
    IngestionResult,
    PageImage,
    PaperMeta,
)

client = TestClient(app)


def _result(paper_id: str, title: str | None, n_pages: int) -> IngestionResult:
    return IngestionResult(
        paper=PaperMeta(
            paper_id=paper_id,
            source_filename=f"{paper_id}.pdf",
            title=title,
            page_count=n_pages,
            extraction_method=ExtractionMethod.PYMUPDF_FLAT_BASELINE,
        ),
        chunks=[],
        pages=[
            PageImage(
                page_index=i,
                image_path=f"{paper_id}/pages/page_{i:04d}.png",
                width_px=1700,
                height_px=2200,
                dpi=200,
            )
            for i in range(n_pages)
        ],
    )


def test_renders_one_card_per_page():
    out = render_gallery(_result("demo-1", "A Paper", 24))

    assert out.count('class="card"') == 24
    assert "Page 1<" in out
    assert "Page 24<" in out
    assert "Page 25<" not in out


def test_percent_encodes_ids_with_spaces_in_image_urls():
    """Real filenames have spaces. A raw space in src= breaks the image."""
    out = render_gallery(_result("Linked Lists-2026-S2-abc123", "Lecture 05", 2))

    assert "/paper/Linked%20Lists-2026-S2-abc123/page/0" in out
    assert "/paper/Linked Lists-2026-S2-abc123/page/0" not in out


def test_escapes_html_in_the_title():
    out = render_gallery(_result("demo-1", "Attention <script>alert(1)</script>", 1))

    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_falls_back_to_filename_when_the_pdf_has_no_title():
    out = render_gallery(_result("demo-1", None, 1))
    assert "demo-1.pdf" in out


def test_explains_itself_when_there_are_no_page_images():
    result = _result("demo-1", "A Paper", 0)
    out = render_gallery(result)

    assert 'class="card"' not in out
    assert "rasterise=false" in out


def test_gallery_endpoint_serves_html(tmp_path: Path, monkeypatch):
    from api import storage
    from scripts.generate_sample_pdf import build_sample_pdf

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")

    pdf_path = tmp_path / "sample.pdf"
    build_sample_pdf(pdf_path)
    with pdf_path.open("rb") as f:
        ingest = client.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})
    paper_id = ingest.json()["paper"]["paper_id"]

    resp = client.get(f"/gallery/{paper_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.text.count('class="card"') == 3


def test_gallery_404s_for_an_unknown_paper():
    assert client.get("/gallery/never-ingested").status_code == 404
