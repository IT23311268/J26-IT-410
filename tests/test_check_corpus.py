"""Tests for the corpus checker — scripts/check_corpus.py.

The checker is how "does this generalise?" stops being an opinion. It
has to earn that: a checker that cannot see the bugs it was written for
is worse than none, because a clean run would be taken as evidence.

So these tests reintroduce each bug found so far and assert the checker
notices, then assert it stays quiet on the fixed code.
"""

from collections import defaultdict
from pathlib import Path

import pytest

import ingestion.layout as layout
from scripts.check_corpus import check_page
from tests.test_charts import _build_table_and_chart_pdf


def _flags(pdf: Path) -> list[str]:
    regions = layout.extract_regions(pdf, "x")
    by_page: dict[int, list] = defaultdict(list)
    for region in regions:
        by_page[region.bbox.page].append(region)
    return [p for n in sorted(by_page) for p in check_page(n, by_page[n])]


@pytest.fixture
def page(tmp_path: Path) -> Path:
    return _build_table_and_chart_pdf(tmp_path / "page.pdf")


def test_a_correctly_parsed_page_raises_nothing(page: Path):
    assert _flags(page) == []


def test_it_catches_labels_left_outside_a_figure(page: Path, monkeypatch):
    """Reintroduce the bug: with no halo, the chart's ticks and axis
    title scatter into BODY regions again."""
    monkeypatch.setattr(layout, "FIGURE_HALO", 0.0)
    assert any("tiny BODY regions" in f for f in _flags(page))


def test_it_catches_a_table_swallowing_a_figure(page: Path, monkeypatch):
    """Reintroduce the other bug: let a chart's axes count as table
    rules and the table's box grows down over the chart."""
    monkeypatch.setattr(layout, "_rule_belongs_to_a_figure", lambda rule, figs: False)
    assert any("overlap" in f for f in _flags(page))


def test_it_reports_a_page_with_no_text_layer(tmp_path: Path):
    """Not a bug — the honest signal that a page needs Box 8."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "scan.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.rect(60, 60, 490, 700, fill=1)
    c.showPage()
    c.save()

    assert any("no-text-layer" in f for f in _flags(path))


def test_short_prose_away_from_a_figure_is_not_flagged(tmp_path: Path):
    """The scatter signature is short text *crowded around an artifact*.
    A page of short lines on its own must not raise it, or the checker
    cries wolf and stops being read."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "list.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.rect(72, 520, 450, 220)  # a figure, high on the page
    c.setFont("Helvetica", 10)
    for i in range(10):  # a short list far below it
        c.drawString(72, 400 - i * 16, f"Item {i}")
    c.showPage()
    c.save()

    assert not any("tiny BODY regions" in f for f in _flags(path))
