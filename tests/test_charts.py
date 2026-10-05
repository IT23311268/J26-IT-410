"""Tests for a real chart's full extent — Box 3.

Found on page 10 of the GPT-4 technical report, which carries a ruled
table, its caption, and a bar chart with its own caption, all on one
page. Two things went wrong there and both are regressions guarded
here.

1. The table swallowed the chart. A chart's axes are wide hairlines,
   exactly like a table's booktabs rules, so the table's rule chain ran
   straight through the caption and down to the chart's baseline. One
   region came back covering two different artifacts — and tables and
   figures go to Member 4 as separate things.

2. The chart's tick labels stayed BODY. matplotlib draws ticks, axis
   titles and the chart title *outside* the plot frame, so they were
   not inside the figure and the containment test could not reach them.
   "gpt-3.5-base", "0-shot", "Model" each came back as prose.

The page is rebuilt here the way matplotlib actually emits one: the
spines as four separate line paths, the labels outside them.
"""

from pathlib import Path

import pytest

from ingestion.layout import extract_regions
from schema.ingestion_schema_v1 import RegionType

PLOT = (140.0, 320.0, 500.0, 560.0)  # x0, y0, x1, y1 in PDF coordinates


def _build_table_and_chart_pdf(path: Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    left, right = 72, 520

    # ---- a booktabs table, high on the page -------------------------
    c.setLineWidth(1.0)
    c.line(left, 720, right, 720)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(80, 706, "Model")
    c.drawString(260, 706, "Accuracy")
    c.drawString(420, 706, "F1")
    c.setLineWidth(0.5)
    c.line(left, 698, right, 698)

    c.setFont("Helvetica", 9)
    y = 684
    for i in range(3):
        c.drawString(80, y, f"System {i}")
        c.drawString(260, y, f"{70 + i}.4")
        c.drawString(420, y, f"{68 + i}.2")
        y -= 14
    c.setLineWidth(1.0)
    c.line(left, y + 6, right, y + 6)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(72, y - 10, "Table 4: Accuracy and F1 across both systems.")

    # ---- a bar chart below it ---------------------------------------
    px0, py0, px1, py1 = PLOT
    c.setLineWidth(0.8)
    # each spine its own path, as matplotlib emits them
    c.line(px0, py0, px1, py0)
    c.line(px0, py1, px1, py1)
    c.line(px0, py0, px0, py1)
    c.line(px1, py0, px1, py1)
    for i in range(5):
        c.rect(px0 + 10 + i * 70, py0, 54, 30 + i * 35, fill=1)

    c.setFont("Helvetica-Bold", 9)  # title, above the frame
    c.drawString(px0, py1 + 8, "Accuracy on adversarial questions")
    c.setFont("Helvetica", 7)  # y ticks, left of the frame
    for i in range(5):
        c.drawString(px0 - 26, py0 + i * 50, f"{i * 10}%")
    for i in range(5):  # x ticks, below the frame
        c.drawString(px0 + 14 + i * 70, py0 - 12, f"gpt-{i}")
    c.setFont("Helvetica", 8)
    c.drawString(300, py0 - 26, "Model")

    c.setFont("Helvetica", 9)
    c.drawString(72, 280, "Figure 7. Performance of GPT-4 on TruthfulQA.")
    c.showPage()
    c.save()
    return path


@pytest.fixture
def regions(tmp_path: Path):
    return extract_regions(_build_table_and_chart_pdf(tmp_path / "page.pdf"), "pg")


def _only(regions, kind):
    found = [r for r in regions if r.region_type == kind]
    assert len(found) == 1, f"expected one {kind.value}, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------
# 1. the table must stop at the table
# --------------------------------------------------------------------------


def test_the_table_does_not_swallow_the_chart(regions):
    """A chart's axes are hairlines too. The rule chain must not cross
    from the table into the figure below it."""
    table = _only(regions, RegionType.TABLE)
    chart_top_in_pdf_space = 842.0 - PLOT[3]  # page height - y1, flipped
    assert table.bbox.y1 < chart_top_in_pdf_space, (
        "the table region reaches into the chart: "
        f"table ends at {table.bbox.y1:.0f}, chart starts at "
        f"{chart_top_in_pdf_space:.0f}"
    )


def test_the_table_still_holds_its_own_cells(regions):
    table = _only(regions, RegionType.TABLE)
    assert "System 0" in table.text
    assert "70.4" in table.text


def test_the_chart_is_a_figure_in_its_own_right(regions):
    figures = [r for r in regions if r.region_type == RegionType.FIGURE]
    assert figures, "the chart came back as no figure at all"


# --------------------------------------------------------------------------
# 2. the chart's furniture belongs to the chart
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["gpt-0", "gpt-4", "Model", "40%"])
def test_axis_labels_do_not_become_prose(regions, label):
    """Ticks and axis titles sit outside the plot frame, but they are
    still part of the chart — not sentences for the retrieval corpus."""
    stray = [
        r
        for r in regions
        if r.region_type in (RegionType.BODY, RegionType.HEADING)
        and label in r.text
    ]
    assert not stray, f"{label!r} came back as {stray[0].region_type.value}"


def test_the_chart_title_is_not_a_heading(regions):
    """It is bold and short, so the heading rule would claim it — and
    Box 4 would open a section that does not exist."""
    headings = [r for r in regions if r.region_type == RegionType.HEADING]
    assert not any("adversarial" in r.text for r in headings)


def test_the_absorbed_labels_stay_on_the_figure(regions):
    """Absorbed, not discarded: Box 6 and Box 8 still need them."""
    figure_text = " ".join(
        r.text for r in regions if r.region_type == RegionType.FIGURE
    )
    assert "gpt-0" in figure_text
    assert "Model" in figure_text


# --------------------------------------------------------------------------
# 3. what must still survive
# --------------------------------------------------------------------------


def test_both_captions_survive(regions):
    captions = sorted(
        (r for r in regions if r.region_type == RegionType.CAPTION),
        key=lambda r: r.order_index,
    )
    assert [c.text[:8] for c in captions] == ["Table 4:", "Figure 7"]


# --------------------------------------------------------------------------
# 4. a table drawn as a box, not as booktabs rules
# --------------------------------------------------------------------------
#
# Table 4 of the GPT-4 report is boxed: a full border and a vertical
# divider between two cells of prose. Nothing in it is a filled shape.
#
# Treating a chart's axis lines as part of its drawing — which is what
# lets the tick labels be reached — made this table's border merge into
# a figure as well, and a figure's rules are excluded from the table
# detector, so the table disappeared entirely. Measured on the page:
# the table contributes 0 non-line shapes and the chart 8, while text
# coverage is 0.17 against 0.07 and cannot separate them.


def _build_boxed_table_pdf(path: Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    tx0, ty0, tx1, ty1 = 72, 560, 540, 760

    c.setLineWidth(0.8)
    c.line(tx0, ty0, tx1, ty0)
    c.line(tx0, ty1, tx1, ty1)
    c.line(tx0, ty0, tx0, ty1)
    c.line(tx1, ty0, tx1, ty1)
    c.line(306, ty0, 306, ty1)  # the divider between the two cells
    c.line(tx0, 735, tx1, 735)  # under the header

    c.setFont("Helvetica-Bold", 9)
    c.drawString(80, 742, "[GPT-4 answers correctly]")
    c.drawString(314, 742, "[GPT-4 answers incorrectly]")
    c.setFont("Helvetica", 9)
    c.drawString(80, 715, "Can you teach an old dog new tricks?")
    c.drawString(314, 715, "Son of an actor, this American guitarist")
    c.drawString(80, 690, "Yes, you can teach an old dog new tricks")
    c.drawString(314, 690, "Elvis Presley")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(130, 548, "Table 4: Example of GPT-4 giving correct responses.")

    # the chart, with adjacent bars and one wide row of tick labels
    px0, py0, px1, py1 = 150, 190, 530, 500
    c.setLineWidth(0.6)
    c.line(px0, py0, px1, py0)
    c.line(px0, py1, px1, py1)
    c.line(px0, py0, px0, py1)
    c.line(px1, py0, px1, py1)
    for i in range(8):
        c.rect(px0 + 12 + i * 46, py0, 38, 40 + i * 24, fill=1)
    c.setFont("Helvetica", 6)
    for i in range(8):
        c.drawString(px0 - 22, py0 + i * 38, f"{i * 10}%")
    # one wide block, the way PyMuPDF returns a chart's label row
    c.drawString(
        px0 - 4,
        py0 - 11,
        "Anthropic-LM 0-shot   Anthropic-LM RLHF   gpt-3.5-base 0-shot   "
        "gpt-3.5-base 5-shot   gpt-3.5-turbo RLHF   gpt-4 RLHF",
    )
    c.setFont("Helvetica", 9)
    c.drawString(72, 140, "Figure 7. Performance of GPT-4 on TruthfulQA.")
    c.showPage()
    c.save()
    return path


@pytest.fixture
def boxed(tmp_path: Path):
    return extract_regions(_build_boxed_table_pdf(tmp_path / "boxed.pdf"), "bx")


def test_a_boxed_table_is_a_table_not_a_figure(boxed):
    """A frame drawn in straight lines is furniture. A figure has
    artwork in it — a filled shape, a curve, a marker."""
    table = _only(boxed, RegionType.TABLE)
    assert "Elvis Presley" in table.text


def test_the_chart_on_the_same_page_is_still_a_figure(boxed):
    figure = _only(boxed, RegionType.FIGURE)
    assert figure.bbox.y0 > _only(boxed, RegionType.TABLE).bbox.y1


def test_a_wide_row_of_tick_labels_is_absorbed(boxed):
    """PyMuPDF returns a chart's x-axis labels as one block, and on a
    real chart that block runs past a hundred characters. Length alone
    would reject it; it is the type size that marks it as furniture."""
    stray = [
        r
        for r in boxed
        if r.region_type in (RegionType.BODY, RegionType.HEADING)
        and "gpt-3.5-turbo" in r.text
    ]
    assert not stray, f"the label row came back as {stray[0].region_type.value}"


def test_a_long_paragraph_at_body_size_is_not_absorbed(boxed):
    """The other side of that rule: the figure caption is long *and*
    set at body size, so nothing about it reads as chart furniture."""
    captions = [r.text for r in boxed if r.region_type == RegionType.CAPTION]
    assert any(t.startswith("Figure 7") for t in captions)


def test_the_figure_caption_is_not_absorbed_into_the_chart(regions):
    """It sits just below the axis labels, well within reach of the
    figure's halo, and it must stay a caption all the same."""
    caption = next(
        r
        for r in regions
        if r.region_type == RegionType.CAPTION and r.text.startswith("Figure 7")
    )
    assert "TruthfulQA" in caption.text
