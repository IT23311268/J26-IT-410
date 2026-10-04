"""Tests for Box 3 — layout region detection.

The headline test is `test_two_column_reading_order_is_recovered`. It is
the whole justification for this module: on a two-column page the text
must come out ALPHA, BETA, GAMMA, DELTA (down the left column, then down
the right) and not interleaved across the gutter.
"""

from pathlib import Path

import pytest

from ingestion.layout import (
    classify_block,
    columns_from_gutters,
    extract_regions,
    find_gutters,
)
from schema.ingestion_schema_v1 import RegionType


@pytest.fixture(scope="module")
def two_column_pdf(tmp_path_factory) -> Path:
    from scripts.generate_two_column_pdf import build_two_column_pdf

    path = tmp_path_factory.mktemp("pdfs") / "two_column.pdf"
    return build_two_column_pdf(path)


@pytest.fixture(scope="module")
def single_column_pdf(tmp_path_factory) -> Path:
    from scripts.generate_sample_pdf import build_sample_pdf

    path = tmp_path_factory.mktemp("pdfs") / "single.pdf"
    build_sample_pdf(path)
    return path


@pytest.fixture(scope="module")
def two_column_regions(two_column_pdf: Path):
    return extract_regions(two_column_pdf, "twocol")


def _region(regions, marker: str):
    """The region whose text starts with `marker`."""
    for region in regions:
        if region.text.lstrip().startswith(marker):
            return region
    raise AssertionError(f"no region starting with {marker!r}")


def _order_of(regions, marker: str) -> int:
    """Reading-order position of the region whose text starts with marker."""
    return _region(regions, marker).order_index


# --------------------------------------------------------------------------
# the point of the whole module
# --------------------------------------------------------------------------


def test_two_column_reading_order_is_recovered(two_column_regions):
    """Down the left column, then down the right — not across the gutter.

    Read naively the order would be ALPHA, GAMMA, BETA, DELTA, because
    GAMMA starts higher on the page than BETA.
    """
    alpha = _order_of(two_column_regions, "ALPHA")
    beta = _order_of(two_column_regions, "BETA")
    gamma = _order_of(two_column_regions, "GAMMA")
    delta = _order_of(two_column_regions, "DELTA")

    assert alpha < beta < gamma < delta


def test_naive_top_to_bottom_order_would_have_been_wrong(two_column_regions):
    """Guards the test above from passing for the wrong reason.

    If the page were laid out so that reading top-to-bottom happened to
    give the right answer, the previous test would prove nothing. So:
    confirm GAMMA really does start higher on the page than BETA.
    """
    assert _region(two_column_regions, "GAMMA").bbox.y0 < _region(
        two_column_regions, "BETA"
    ).bbox.y0


def test_columns_are_assigned(two_column_regions):
    assert _region(two_column_regions, "ALPHA").column_index == 0
    assert _region(two_column_regions, "BETA").column_index == 0
    assert _region(two_column_regions, "GAMMA").column_index == 1
    assert _region(two_column_regions, "DELTA").column_index == 1


def test_full_width_blocks_are_read_before_the_columns(two_column_regions):
    """A title and abstract printed across the page come first."""
    title = _order_of(two_column_regions, "Layout-Aware")
    abstract_body = _order_of(two_column_regions, "This synthetic paper")
    alpha = _order_of(two_column_regions, "ALPHA")

    assert title < abstract_body < alpha


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def test_headings_captions_and_figures_are_labelled(two_column_regions):
    types = {r.region_type for r in two_column_regions}
    assert RegionType.HEADING in types
    assert RegionType.BODY in types
    assert RegionType.CAPTION in types
    assert RegionType.FIGURE in types  # the vector-drawn box


def test_section_headings_are_headings(two_column_regions):
    for heading in ["1. Introduction", "2. Method", "3. Results", "4. Conclusion"]:
        region = next(r for r in two_column_regions if r.text.strip() == heading)
        assert region.region_type == RegionType.HEADING


def test_the_figure_is_read_just_before_its_caption(two_column_regions):
    figure = next(r for r in two_column_regions if r.region_type == RegionType.FIGURE)
    caption = next(r for r in two_column_regions if r.region_type == RegionType.CAPTION)
    assert figure.order_index < caption.order_index


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Figure 1: the pipeline", RegionType.CAPTION),
        ("Fig. 12 Overview", RegionType.CAPTION),
        ("Table 3 — results", RegionType.CAPTION),
        ("figure 2: lowercase prefix still counts", RegionType.CAPTION),
    ],
)
def test_caption_prefixes_are_recognised(text, expected):
    assert classify_block(text, 10.0, 10.0, is_image=False) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Table 4 reports the comparison across both systems.",
        "Figure 1 shows the overall pipeline used in this work.",
        "The figure shows our method applied to twenty languages.",
    ],
)
def test_prose_that_merely_mentions_a_figure_is_not_a_caption(text):
    """A sentence continuing in lower case is a paragraph. An earlier
    pattern stopped at the number and labelled these captions."""
    assert classify_block(text, 10.0, 10.0, is_image=False) == RegionType.BODY


def test_a_caption_wins_over_the_heading_rule():
    """Set large, a caption would otherwise look like a heading."""
    assert (
        classify_block("Figure 4: a big caption", 16.0, 10.0, is_image=False)
        == RegionType.CAPTION
    )


def _bold_block(spans: list[tuple[str, str]]) -> dict:
    return {
        "type": 0,
        "bbox": (0, 0, 200, 14),
        "lines": [{"spans": [{"font": f, "text": t, "size": 10.0} for f, t in spans]}],
    }


def test_a_heading_bold_at_body_size_is_still_a_heading():
    """Found on the RAG paper: "2.3  Generator: BART" is bold at exactly
    body size, and the size-only rule missed it.

    This one matters beyond tidiness — Box 4 builds the section tree out
    of HEADING regions, so a missed heading loses a whole section.
    """
    from ingestion.layout import block_bold_ratio

    block = _bold_block([("NimbusRomNo9L-Medi", "2.3  Generator: BART")])
    bold = block_bold_ratio(block)

    assert bold == 1.0
    assert (
        classify_block("2.3  Generator: BART", 10.0, 10.0, is_image=False, bold_ratio=bold)
        == RegionType.HEADING
    )


def test_a_paragraph_opening_with_a_bold_run_in_stays_body():
    """"**RAG-Sequence**  For RAG-Sequence, the likelihood..." is a
    paragraph, not a heading. Only a few percent of it is bold."""
    from ingestion.layout import block_bold_ratio

    prose = (
        " For RAG-Sequence, the likelihood does not break into a conventional "
        "per-token likelihood, hence we cannot solve it with a single beam search."
    )
    block = _bold_block(
        [("NimbusRomNo9L-Medi", "RAG-Sequence"), ("NimbusRomNo9L-Regu", prose)]
    )
    bold = block_bold_ratio(block)

    assert bold < 0.2
    assert (
        classify_block("RAG-Sequence" + prose, 10.0, 10.0, is_image=False, bold_ratio=bold)
        == RegionType.BODY
    )


def test_a_long_bold_passage_is_emphasis_not_a_heading():
    """Bold alone is not enough — a heading also has to be short."""
    assert (
        classify_block("bold warning text " * 12, 10.0, 10.0, is_image=False, bold_ratio=1.0)
        == RegionType.BODY
    )


@pytest.mark.parametrize(
    "font,expected",
    [
        ("NimbusRomNo9L-Medi", True),
        ("Helvetica-Bold", True),
        ("CMBX10", True),
        ("Arial-BoldMT", True),
        ("NimbusRomNo9L-Regu", False),
        ("CMR10", False),
        ("Times-Roman", False),
    ],
)
def test_bold_face_detection(font, expected):
    from ingestion.layout import is_bold_font

    assert is_bold_font(font) is expected


def test_large_but_long_text_is_body_not_heading():
    """Headings are short. An abstract set large is still body text."""
    long_text = "word " * 60
    assert classify_block(long_text, 14.0, 10.0, is_image=False) == RegionType.BODY


def test_body_sized_text_is_body():
    assert classify_block("ordinary prose", 10.0, 10.0, is_image=False) == RegionType.BODY


def test_an_image_block_is_a_figure():
    assert classify_block("", None, 10.0, is_image=True) == RegionType.FIGURE


# --------------------------------------------------------------------------
# single-column pages must not be mangled
# --------------------------------------------------------------------------


def test_single_column_page_reads_top_to_bottom(single_column_pdf: Path):
    regions = extract_regions(single_column_pdf, "single")

    assert regions, "no regions found at all"
    assert all(r.column_index == 0 for r in regions)

    # within a page, order follows the vertical position
    page0 = [r for r in regions if r.bbox.page == 0]
    tops = [r.bbox.y0 for r in sorted(page0, key=lambda r: r.order_index)]
    assert tops == sorted(tops)


def _build_highlighted_text_pdf(path: Path) -> Path:
    """A page whose inline code spans sit on a pale highlight.

    Those highlights are thin filled rectangles — vector artwork drawn
    *behind prose*, which is exactly what must not be mistaken for a
    figure.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 700
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "4.3  Jeopardy Question Generation")
    y -= 24

    c.setFont("Helvetica", 10)
    c.drawString(72, y, "We find evidence for this hypothesis by feeding the baseline:")
    y -= 18

    for _ in range(4):
        c.setFillColorRGB(0.92, 0.92, 0.92)
        c.rect(72, y - 3, 300, 12, stroke=0, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Courier", 9)
        c.drawString(74, y, "The Sun Also Rises is a novel by this author")
        y -= 14

    c.setFont("Helvetica", 10)
    c.drawString(72, y - 6, "This example shows how the two memories work together.")
    c.showPage()
    c.save()
    return path


def test_highlighted_code_spans_are_not_mistaken_for_a_figure(tmp_path: Path):
    """Regression: found on a real arXiv paper, not in any synthetic test.

    Inline code spans are set on a pale grey highlight. Each highlight is
    a thin rectangle, individually too small to count as a figure — but
    consecutive lines merge into a block large enough to pass, and a
    paragraph came back labelled FIGURE.
    """
    pdf = _build_highlighted_text_pdf(tmp_path / "highlighted.pdf")
    regions = extract_regions(pdf, "hl")

    figures = [r for r in regions if r.region_type == RegionType.FIGURE]
    assert figures == [], "a highlight behind prose was labelled as a figure"

    # and the prose itself is still there
    assert any("Sun Also Rises" in r.text for r in regions)


def test_text_coverage_ratio_measures_overlap():
    from ingestion.layout import text_coverage_ratio

    rect = [0.0, 0.0, 100.0, 100.0]  # area 10 000
    assert text_coverage_ratio(rect, []) == 0.0
    # one block covering the left half
    assert text_coverage_ratio(rect, [{"bbox": (0, 0, 50, 100)}]) == pytest.approx(0.5)
    # a block larger than the rect still caps at 1.0
    assert text_coverage_ratio(rect, [{"bbox": (-50, -50, 200, 200)}]) == 1.0


def test_a_real_figure_survives_the_text_coverage_check(two_column_regions):
    """The guard must not throw out genuine artwork."""
    assert any(r.region_type == RegionType.FIGURE for r in two_column_regions)


def test_a_scanned_page_degrades_to_one_figure(tmp_path: Path, two_column_pdf: Path):
    """A scan has no text layer at all — geometry has nothing to read.

    It must not crash, and it must not invent text regions. One figure
    covering the page is the honest answer, and it is the signal that
    this page needs the visual path (Box 8) rather than this one.
    """
    import pymupdf

    source = pymupdf.open(two_column_pdf)
    pixmap = source[0].get_pixmap(dpi=120)
    scanned = pymupdf.open()
    page = scanned.new_page(
        width=source[0].rect.width, height=source[0].rect.height
    )
    page.insert_image(page.rect, pixmap=pixmap)
    path = tmp_path / "scanned.pdf"
    scanned.save(path)
    scanned.close()
    source.close()

    regions = extract_regions(path, "scan")

    assert len(regions) == 1
    assert regions[0].region_type == RegionType.FIGURE
    assert regions[0].text == ""


def test_no_gutters_on_a_single_column_page(single_column_pdf: Path):
    import pymupdf

    doc = pymupdf.open(single_column_pdf)
    try:
        page_dict = doc[1].get_text("dict")
        gutters = find_gutters(page_dict["blocks"], page_dict["width"])
    finally:
        doc.close()
    assert gutters == []


def test_two_column_page_finds_exactly_one_gutter(two_column_pdf: Path):
    import pymupdf

    doc = pymupdf.open(two_column_pdf)
    try:
        page_dict = doc[0].get_text("dict")
        gutters = find_gutters(page_dict["blocks"], page_dict["width"])
    finally:
        doc.close()

    assert len(gutters) == 1
    start, end = gutters[0]

    # The detected strip runs from where the left column's longest line
    # ends to where the right column begins (x=321 in the generator).
    # Lines are ragged-right, so the strip starts a little before the
    # nominal 291pt column edge — that is correct, not drift.
    assert start >= 260
    assert end <= 325

    # What actually matters: the split lands between the two columns.
    midpoint = (start + end) / 2
    assert 268 < midpoint < 321


def test_columns_from_gutters_splits_at_the_midpoint():
    columns = columns_from_gutters([(290.0, 320.0)], 612.0)
    assert len(columns) == 2
    assert columns[0] == (0.0, 305.0)
    assert columns[1] == (305.0, 612.0)


def test_no_gutters_means_one_column():
    assert columns_from_gutters([], 612.0) == [(0.0, 612.0)]


# --------------------------------------------------------------------------
# contract shape
# --------------------------------------------------------------------------


def test_order_index_is_continuous_from_zero(two_column_regions):
    """Members 2/3/4 sort on order_index, so it must have no gaps."""
    assert [r.order_index for r in two_column_regions] == list(
        range(len(two_column_regions))
    )


def test_region_ids_are_unique_and_namespaced(two_column_regions):
    ids = [r.region_id for r in two_column_regions]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("twocol::region::") for i in ids)


def test_figures_carry_no_font_size(two_column_regions):
    """A figure has no single font size; its text, when it has any, is
    the labels it absorbed rather than prose it was set in."""
    for region in two_column_regions:
        if region.region_type == RegionType.FIGURE:
            assert region.font_size is None


def _build_chart_pdf(path: Path) -> Path:
    """A chart with twenty labelled rows inside its border, prose above
    it and a caption below — the shape that produced seventy stray BODY
    regions on one page of the GPT-4 report."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 10)
    c.drawString(72, 740, "Results are shown in Figure 3 below, across twenty languages.")
    c.rect(72, 300, 450, 400)

    c.setFont("Helvetica", 7)
    languages = [
        "Telugu", "Marathi", "Bengali", "Punjabi", "Nepali", "Swahili", "Urdu",
        "Thai", "Italian", "Spanish", "Polish", "Welsh", "Arabic", "Korean",
        "Russian", "German", "French", "English", "Dutch", "Latvian",
    ]
    for i, language in enumerate(languages):
        y = 680 - i * 19
        c.drawString(80, y, language)
        c.drawString(430, y, f"{60 + i}.{i}%")

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(72, 285, "Figure 3: GPT-4 3-shot accuracy on MMLU across languages.")
    c.showPage()
    c.save()
    return path


def test_chart_labels_do_not_become_separate_body_regions(tmp_path: Path):
    """Regression, found on the GPT-4 report.

    Every tick, bar label and legend entry inside a chart came back as
    its own BODY region. They would have become chunks — "Telugu",
    "25.0%" — in the corpus Members 2 and 3 retrieve from.
    """
    pdf = _build_chart_pdf(tmp_path / "chart.pdf")
    regions = extract_regions(pdf, "chart")

    body = [r for r in regions if r.region_type == RegionType.BODY]
    assert len(body) == 1, f"expected only the surrounding prose, got {len(body)}"
    assert "twenty languages" in body[0].text


def test_the_absorbed_labels_are_kept_on_the_figure(tmp_path: Path):
    """Absorbed, not discarded — Box 6 and Box 8 still need them."""
    pdf = _build_chart_pdf(tmp_path / "chart.pdf")
    regions = extract_regions(pdf, "chart")

    figure = next(r for r in regions if r.region_type == RegionType.FIGURE)
    assert "Telugu" in figure.text
    assert "Latvian" in figure.text


def test_a_caption_inside_a_figure_border_stays_a_caption(tmp_path: Path):
    pdf = _build_chart_pdf(tmp_path / "chart.pdf")
    regions = extract_regions(pdf, "chart")

    captions = [r for r in regions if r.region_type == RegionType.CAPTION]
    assert len(captions) == 1
    assert captions[0].text.startswith("Figure 3")


def test_containment_ratio():
    from ingestion.layout import containment_ratio

    outer = (0, 0, 100, 100)
    assert containment_ratio((10, 10, 20, 20), outer) == 1.0  # fully inside
    assert containment_ratio((200, 200, 210, 210), outer) == 0.0  # fully outside
    assert containment_ratio((90, 0, 110, 100), outer) == pytest.approx(0.5)  # half in
