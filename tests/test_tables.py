"""Tests for table detection — Box 3.

Tables matter on their own account: they go to Member 4 for the IEEE
output. Before this, every cell came back as its own BODY region, so a
results table dissolved into a scatter of numbers in the retrieval
corpus.

A scholarly table is drawn in the booktabs style — horizontal rules
above the header, below it, and under the last row, with no vertical
lines at all. Those rules are what gives it away.
"""

from pathlib import Path

import pytest

from ingestion.layout import extract_regions
from schema.ingestion_schema_v1 import RegionType


def _rule(c, x0, x1, y, width=1.0):
    c.setLineWidth(width)
    c.line(x0, y, x1, y)


def _build_table_pdf(path: Path, n_rows: int = 3) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 10)
    c.drawString(72, 740, "Table 4 reports the comparison across both systems.")

    left, right = 72, 520
    _rule(c, left, right, 700)

    c.setFont("Helvetica-Bold", 9)
    c.drawString(80, 686, "Model")
    c.drawString(260, 686, "Accuracy")
    c.drawString(420, 686, "F1")
    _rule(c, left, right, 678, 0.5)

    c.setFont("Helvetica", 9)
    y = 664
    for i in range(n_rows):
        c.drawString(80, y, f"System {i}")
        c.drawString(260, y, f"{70 + i}.4")
        c.drawString(420, y, f"{68 + i}.2")
        y -= 16
    _rule(c, left, right, y + 8)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(72, y - 10, "Table 4: Accuracy and F1 across both systems.")
    c.showPage()
    c.save()
    return path


@pytest.fixture
def table_regions(tmp_path: Path):
    return extract_regions(_build_table_pdf(tmp_path / "table.pdf"), "tbl")


def test_a_ruled_table_is_detected(table_regions):
    tables = [r for r in table_regions if r.region_type == RegionType.TABLE]
    assert len(tables) == 1


def test_the_cells_do_not_become_separate_body_regions(table_regions):
    """The whole point: a results table must not dissolve into a
    scatter of numbers in the corpus."""
    body = [r for r in table_regions if r.region_type == RegionType.BODY]
    for region in body:
        assert "System 0" not in region.text
        assert "70.4" not in region.text


def test_the_cell_text_is_kept_on_the_table(table_regions):
    """Absorbed, not discarded — Member 4 needs the contents."""
    table = next(r for r in table_regions if r.region_type == RegionType.TABLE)
    assert "Model" in table.text
    assert "Accuracy" in table.text
    assert "System 0" in table.text


def test_the_table_caption_survives(table_regions):
    captions = [r for r in table_regions if r.region_type == RegionType.CAPTION]
    assert len(captions) == 1
    assert captions[0].text.startswith("Table 4:")


def test_the_sentence_mentioning_the_table_stays_prose(table_regions):
    """"Table 4 reports the comparison..." is a paragraph."""
    body = [r for r in table_regions if r.region_type == RegionType.BODY]
    assert any("reports the comparison" in r.text for r in body)


def test_the_table_is_read_before_its_caption(table_regions):
    table = next(r for r in table_regions if r.region_type == RegionType.TABLE)
    caption = next(r for r in table_regions if r.region_type == RegionType.CAPTION)
    assert table.order_index < caption.order_index


# --------------------------------------------------------------------------
# what must NOT be read as a table
# --------------------------------------------------------------------------


def test_a_single_rule_is_not_a_table(tmp_path: Path):
    """One horizontal line is a section divider or a running header."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "one_rule.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 10)
    c.drawString(72, 740, "A page with a single rule beneath its running head.")
    _rule(c, 72, 520, 730)
    c.drawString(72, 700, "Ordinary body text continues below the rule.")
    c.showPage()
    c.save()

    regions = extract_regions(path, "one")
    assert not [r for r in regions if r.region_type == RegionType.TABLE]


def test_rules_with_no_text_between_them_are_not_a_table(tmp_path: Path):
    """A page's header and footer rules enclose the whole page, but the
    requirement is text *inside* the rules' own span."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "empty_rules.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    _rule(c, 72, 520, 760)
    _rule(c, 72, 520, 755)
    c.setFont("Helvetica", 10)
    c.drawString(72, 400, "Body text, far below both rules.")
    c.showPage()
    c.save()

    regions = extract_regions(path, "empty")
    assert not [r for r in regions if r.region_type == RegionType.TABLE]


def test_a_ruled_table_does_not_also_become_a_figure(table_regions):
    """A table's rules are vector paths like any other, so the figure
    detector sees them too. Two things keep it off: three rules spaced a
    row apart do not merge into one drawing, and a merged one would be
    mostly covered by text. Without both, every table would come back
    twice — once as a table and once as a figure."""
    assert not [r for r in table_regions if r.region_type == RegionType.FIGURE]


def test_a_chart_is_still_a_figure_not_a_table(tmp_path: Path):
    """The two detectors must not fight over the same page."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "chart.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.rect(72, 400, 450, 300)
    c.setFont("Helvetica", 8)
    for i in range(6):
        c.drawString(90, 660 - i * 40, f"Series {i}")
    c.showPage()
    c.save()

    regions = extract_regions(path, "chart")
    kinds = {r.region_type for r in regions}
    assert RegionType.FIGURE in kinds
    assert RegionType.TABLE not in kinds
