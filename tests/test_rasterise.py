from pathlib import Path

import pytest

from ingestion.rasterise import (
    DEFAULT_DPI,
    page_image_dir,
    rasterise_pdf,
    resolve_image_path,
)


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    from scripts.generate_sample_pdf import build_sample_pdf

    out_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = out_dir / "sample_paper.pdf"
    build_sample_pdf(pdf_path)
    return pdf_path


def test_rasterise_produces_one_image_per_page(sample_pdf: Path, tmp_path: Path):
    pages = rasterise_pdf(sample_pdf, "test-paper", tmp_path)

    assert len(pages) == 3
    for i, page in enumerate(pages):
        assert page.page_index == i
        assert page.dpi == DEFAULT_DPI
        assert page.width_px > 0 and page.height_px > 0


def test_rasterise_writes_real_files_to_disk(sample_pdf: Path, tmp_path: Path):
    pages = rasterise_pdf(sample_pdf, "test-paper", tmp_path)

    for page in pages:
        on_disk = resolve_image_path(tmp_path, page)
        assert on_disk.exists(), f"{page.image_path} was recorded but not written"
        # a real PNG, not a zero-byte placeholder
        assert on_disk.stat().st_size > 1000
        assert on_disk.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_image_paths_are_relative_not_absolute(sample_pdf: Path, tmp_path: Path):
    """Absolute paths would break the moment Members 2/3/4 read the JSON
    on a different machine, so the contract stores relative paths only."""
    pages = rasterise_pdf(sample_pdf, "test-paper", tmp_path)

    for page in pages:
        assert not Path(page.image_path).is_absolute()
        assert page.image_path.startswith("test-paper/pages/")
        assert str(tmp_path) not in page.image_path


def test_higher_dpi_gives_a_bigger_image(sample_pdf: Path, tmp_path: Path):
    low = rasterise_pdf(sample_pdf, "low", tmp_path, dpi=72)
    high = rasterise_pdf(sample_pdf, "high", tmp_path, dpi=200)

    assert high[0].width_px > low[0].width_px
    assert high[0].height_px > low[0].height_px


def test_scale_converts_points_to_pixels(sample_pdf: Path, tmp_path: Path):
    """PageImage.scale is what Box 3 will use to map detector output back
    into PDF-point BoundingBoxes. Pin the arithmetic down now."""
    pages = rasterise_pdf(sample_pdf, "test-paper", tmp_path, dpi=144)

    page = pages[0]
    assert page.scale == pytest.approx(2.0)  # 144 dpi / 72 points-per-inch

    # a 100pt-wide box occupies 200px at this scale, and back again
    assert 100 * page.scale == pytest.approx(200)
    assert 200 / page.scale == pytest.approx(100)


def test_rejects_nonsense_dpi(sample_pdf: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        rasterise_pdf(sample_pdf, "test-paper", tmp_path, dpi=0)


def test_rerunning_overwrites_rather_than_duplicating(sample_pdf: Path, tmp_path: Path):
    """Re-ingesting the same paper must not leave stale extra pages behind."""
    rasterise_pdf(sample_pdf, "test-paper", tmp_path)
    rasterise_pdf(sample_pdf, "test-paper", tmp_path)

    written = list(page_image_dir(tmp_path, "test-paper").glob("*.png"))
    assert len(written) == 3
