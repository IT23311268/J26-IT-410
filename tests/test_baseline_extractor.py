from pathlib import Path

import pytest

from ingestion.baseline_extractor import extract_baseline, make_paper_id
from schema.ingestion_schema_v1 import ExtractionMethod, SectionType


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    from scripts.generate_sample_pdf import build_sample_pdf

    out_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = out_dir / "sample_paper.pdf"
    build_sample_pdf(pdf_path)
    return pdf_path


def test_make_paper_id_is_stable(sample_pdf: Path):
    id_a = make_paper_id(sample_pdf)
    id_b = make_paper_id(sample_pdf)
    assert id_a == id_b
    assert sample_pdf.stem in id_a


def test_extract_baseline_produces_chunks(sample_pdf: Path):
    result = extract_baseline(sample_pdf)

    assert result.paper.page_count == 3
    assert result.paper.extraction_method == ExtractionMethod.PYMUPDF_FLAT_BASELINE
    assert len(result.chunks) > 0

    # baseline is flat: no section detection, no artifacts
    assert all(c.section == SectionType.UNKNOWN for c in result.chunks)
    assert result.artifacts == []

    # chunks stay in order and reference valid pages
    for i, chunk in enumerate(result.chunks):
        assert chunk.order_index == i
        assert 0 <= chunk.page_start < result.paper.page_count
        assert 0 <= chunk.page_end < result.paper.page_count
        assert chunk.text.strip() != ""


@pytest.mark.parametrize(
    "raw_title",
    [
        None,
        "",
        "   ",
        "untitled",
        "Untitled",
        "PowerPoint Presentation",
        "Microsoft Word - thesis_v3.docx",
        "Document1",
        "paper.pdf",  # producer echoed the filename
        "paper",  # ...or its stem
    ],
)
def test_generic_pdf_titles_are_treated_as_absent(raw_title):
    """A citation reading "PowerPoint Presentation" is worse than one with
    no title, so these never reach Members 2/3/4."""
    from ingestion.baseline_extractor import _clean_title

    assert _clean_title(raw_title, "paper.pdf") is None


def test_a_real_title_survives_cleaning():
    from ingestion.baseline_extractor import _clean_title

    assert (
        _clean_title("  Attention Is All You Need  ", "1706.03762.pdf")
        == "Attention Is All You Need"
    )


def test_extract_baseline_respects_chunk_size(sample_pdf: Path):
    small = extract_baseline(sample_pdf, chunk_chars=200, chunk_overlap=20)
    large = extract_baseline(sample_pdf, chunk_chars=5000, chunk_overlap=0)
    # smaller windows -> more chunks than one giant window
    assert len(small.chunks) > len(large.chunks)
