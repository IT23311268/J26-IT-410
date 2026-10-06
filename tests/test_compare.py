"""Tests for the baseline-vs-layout-aware page.

Every other demo surface shows the pipeline runs. This one is the
evidence for the claim the proposal makes, so the numbers on it have to
be right: a page that flatters the pipeline is worse than no page.
"""

from pathlib import Path

import pytest

from api.compare import _starts_mid_word, render_comparison
from ingestion.baseline_extractor import extract_baseline
from ingestion.chunking import chunk_regions
from ingestion.layout import extract_regions
from ingestion.sections import assign_sections
from schema.ingestion_schema_v1 import ExtractionMethod


@pytest.fixture
def two_column(tmp_path: Path) -> Path:
    import subprocess
    import sys

    pdf = tmp_path / "paper.pdf"
    subprocess.run(
        [sys.executable, "scripts/generate_two_column_pdf.py", str(pdf)],
        check=True,
        capture_output=True,
    )
    return pdf


@pytest.fixture
def both(two_column: Path):
    baseline = extract_baseline(two_column, chunk_chars=300, chunk_overlap=0)

    layout = extract_baseline(two_column)
    layout.regions = assign_sections(extract_regions(two_column, layout.paper.paper_id))
    layout.chunks = chunk_regions(layout.regions, layout.paper.paper_id)
    layout.paper.extraction_method = ExtractionMethod.LAYOUT_AWARE_V1
    return baseline, layout


# --------------------------------------------------------------------------
# the measure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("s and shreds every sentence.", True),   # tail of "columns"
        ("and no GPU.", True),                    # tail of a sentence
        ("2. Method", False),
        ("ALPHA. Parsing a two-column paper", False),
        ("", False),
        ("   ", False),
    ],
)
def test_mid_word_detection(text: str, expected: bool):
    """A chunk opening in lower case is the tail of a word the chunk
    before it kept the head of — the signature of fixed-size cutting."""
    assert _starts_mid_word(text) is expected


# --------------------------------------------------------------------------
# the claim, measured on real output
# --------------------------------------------------------------------------


def test_the_baseline_loses_every_section(both):
    baseline, _ = both
    assert all(c.section.value == "unknown" for c in baseline.chunks)


def test_the_layout_aware_run_loses_none(both):
    _, layout = both
    assert layout.chunks
    assert not any(c.section.value == "unknown" for c in layout.chunks)


def test_the_baseline_cuts_words_in_half(both):
    baseline, _ = both
    assert any(_starts_mid_word(c.text) for c in baseline.chunks)


def test_the_layout_aware_run_cuts_none(both):
    _, layout = both
    assert not any(_starts_mid_word(c.text) for c in layout.chunks)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def test_the_page_renders(both):
    baseline, layout = both
    html = render_comparison(baseline, layout)
    assert html.startswith("<!doctype html>")
    assert "Baseline" in html
    assert "Layout-aware" in html


def test_both_columns_carry_chunks(both):
    baseline, layout = both
    html = render_comparison(baseline, layout)
    assert "ALPHA" in html
    assert "introduction" in html


def test_the_limit_is_respected(both):
    """A hundred-page paper must not render a wall."""
    baseline, layout = both
    html = render_comparison(baseline, layout, limit=2)
    assert html.count('class="chunk"') == 4  # two a side


def test_text_is_escaped(two_column: Path):
    """Chunk text goes straight into the page; a paper containing < or &
    must not be able to break it."""
    baseline = extract_baseline(two_column)
    layout = extract_baseline(two_column)
    layout.chunks[0] = layout.chunks[0].model_copy(
        update={"text": "<script>alert('x')</script> & more"}
    )
    html = render_comparison(baseline, layout)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_an_empty_side_does_not_crash(two_column: Path):
    baseline = extract_baseline(two_column)
    layout = extract_baseline(two_column)
    layout.chunks = []
    html = render_comparison(baseline, layout)
    assert "No chunks." in html
