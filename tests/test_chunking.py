"""Tests for section-aware chunking — Box 5.

This is the stage the proposal's claim rests on. The baseline cuts the
paper into fixed windows of characters; on a real paper that lands
mid-word — "…interleaves the column | s and shreds every sentence…" —
and it puts the end of the Method and the start of the Results in one
chunk, so nothing downstream can say which section a retrieved passage
came from.

A chunk out of this stage is a whole number of paragraphs, from exactly
one section, carrying the heading that opened it.
"""

from pathlib import Path

import pytest

from ingestion.chunking import (
    MIN_CHUNK_CHARS,
    TARGET_CHUNK_CHARS,
    chunk_regions,
)
from ingestion.layout import extract_regions
from ingestion.sections import assign_sections
from schema.ingestion_schema_v1 import RegionType, SectionType


def _pdf(path: Path, lines: list[tuple[str, bool]]) -> Path:
    """Build a PDF from (text, is_heading) pairs, paginating as needed."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 740
    for text, is_heading in lines:
        # Real papers leave space above a heading, and so must this: set
        # a heading one line below its paragraph and PyMuPDF folds the
        # two into a single block, which silently hides the heading from
        # Box 3 and made this fixture test nothing at all.
        if is_heading:
            y -= 14
        c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 13 if is_heading else 9)
        c.drawString(72, y, text)
        y -= 26 if is_heading else 15
        if y < 70:
            c.showPage()
            y = 740
    c.showPage()
    c.save()
    return path


def _chunks(pdf: Path, paper_id: str = "t"):
    return chunk_regions(assign_sections(extract_regions(pdf, paper_id)), paper_id)


@pytest.fixture
def paper(tmp_path: Path):
    return _chunks(
        _pdf(
            tmp_path / "paper.pdf",
            [
                ("Layout-Aware Ingestion of Scholarly PDFs", True),
                ("Abstract", True),
                ("We present a layout-aware ingestion engine for scholarly PDFs.", False),
                ("1. Introduction", True),
                ("Parsing a two-column paper naively shreds its sentences badly.", False),
                ("2. Method", True),
                ("Column boundaries come from the vertical strips no text touches.", False),
                ("3. Results", True),
                ("We beat the fixed-size baseline by a comfortable margin overall.", False),
            ],
        )
    )


# --------------------------------------------------------------------------
# the boundary that always cuts
# --------------------------------------------------------------------------


def test_a_chunk_belongs_to_exactly_one_section(paper):
    """The whole point. A chunk that is half Method and half Results
    cannot be attributed, and Member 2 verifies claims by attribution."""
    assert paper
    for chunk in paper:
        assert chunk.section != SectionType.UNKNOWN


def test_each_section_opens_its_own_chunk(paper):
    sections = [c.section for c in paper]
    assert sections == sorted(set(sections), key=sections.index), (
        f"a section is split across non-adjacent chunks: {sections}"
    )


def test_the_heading_leads_its_chunk(paper):
    """"2. Method" ahead of the paragraphs under it is a retrieval
    signal, and it costs four words."""
    method = next(c for c in paper if c.section == SectionType.METHOD)
    assert method.text.startswith("2. Method")


def test_the_raw_heading_rides_along(paper):
    method = next(c for c in paper if c.section == SectionType.METHOD)
    assert method.section_title_raw == "2. Method"


# --------------------------------------------------------------------------
# the boundary that cuts only when it has to
# --------------------------------------------------------------------------


@pytest.fixture
def long_section(tmp_path: Path):
    """One Method section far longer than a chunk should be."""
    lines = [("2. Method", True)]
    lines += [
        (f"Paragraph {i}. The retriever encodes each passage with a BERT tower.", False)
        for i in range(1, 46)
    ]
    return _chunks(_pdf(tmp_path / "long.pdf", lines), "l")


def test_a_long_section_is_split(long_section):
    assert len(long_section) > 1


def test_every_chunk_is_near_the_target_size(long_section):
    """Near, not under: a single paragraph longer than the target comes
    out whole rather than cut inside a sentence."""
    for chunk in long_section[:-1]:  # the last one is whatever is left
        assert len(chunk.text) <= TARGET_CHUNK_CHARS


def test_no_chunk_starts_or_ends_mid_paragraph(long_section):
    """The regression that matters. Every line of every chunk is a
    complete paragraph as it appeared on the page — this is exactly what
    the fixed-size baseline gets wrong."""
    for chunk in long_section:
        for line in chunk.text.split("\n"):
            if line.startswith("Paragraph"):
                assert line.endswith("tower."), f"cut mid-paragraph: {line!r}"


def test_the_paragraphs_run_in_order_with_none_lost(long_section):
    """Splitting must not drop or reorder anything."""
    seen = [
        int(line.split()[1].rstrip("."))
        for chunk in long_section
        for line in chunk.text.split("\n")
        if line.startswith("Paragraph")
    ]
    assert seen == list(range(1, 46))


def test_all_of_a_split_section_keeps_the_same_label(long_section):
    assert {c.section for c in long_section} == {SectionType.METHOD}


# --------------------------------------------------------------------------
# what is not prose
# --------------------------------------------------------------------------


def test_a_caption_is_its_own_chunk(tmp_path: Path):
    """Retrieving "Figure 1 shows the pipeline" should not drag a page
    of unrelated Method text with it."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "fig.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 740, "2. Method")
    c.setFont("Helvetica", 9)
    c.drawString(72, 700, "The pipeline is shown below and described in turn.")
    c.rect(72, 400, 400, 260)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(72, 380, "Figure 1: Pipeline overview.")
    c.showPage()
    c.save()

    chunks = _chunks(path, "fig")
    caption = [c for c in chunks if c.text.startswith("Figure 1:")]
    assert len(caption) == 1
    assert caption[0].text == "Figure 1: Pipeline overview."


def test_a_figure_does_not_become_a_chunk(tmp_path: Path):
    """A figure is an Artifact (Box 6), bound back to its chunks in Box
    7. Retrieving it as prose would return an empty passage."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "fig.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 740, "2. Method")
    c.setFont("Helvetica", 9)
    c.drawString(72, 700, "The pipeline is shown below and described in turn.")
    c.rect(72, 400, 400, 260)
    c.showPage()
    c.save()

    regions = assign_sections(extract_regions(path, "fig"))
    assert any(r.region_type == RegionType.FIGURE for r in regions)
    assert all(c.text.strip() for c in _chunks(path, "fig"))


# --------------------------------------------------------------------------
# the schema contract
# --------------------------------------------------------------------------


def test_ids_and_order_are_dense_and_in_order(paper):
    assert [c.order_index for c in paper] == list(range(len(paper)))
    assert [c.chunk_id for c in paper] == [
        f"t::chunk::{i}" for i in range(len(paper))
    ]


def test_pages_are_recorded(paper):
    for chunk in paper:
        assert chunk.page_start <= chunk.page_end


def test_a_single_page_chunk_carries_a_box(paper):
    """Member 4 highlights a retrieved passage on the page image."""
    one_page = [c for c in paper if c.page_start == c.page_end]
    assert one_page
    assert all(c.bbox is not None for c in one_page)


def test_a_chunk_spanning_a_page_break_carries_no_box(long_section):
    """There is no single rectangle for text on two pages, and
    inventing one would put a wrong highlight on a reader's screen."""
    for chunk in long_section:
        if chunk.page_start != chunk.page_end:
            assert chunk.bbox is None


def test_token_count_is_populated(paper):
    for chunk in paper:
        assert chunk.token_count and chunk.token_count > 0


# --------------------------------------------------------------------------
# runts
# --------------------------------------------------------------------------


def test_a_stray_short_paragraph_is_folded_into_its_neighbour(tmp_path: Path):
    """A one-line paragraph matches nothing on its own and dilutes the
    corpus. It stays attached to the text it belongs with."""
    chunks = _chunks(
        _pdf(
            tmp_path / "runt.pdf",
            [
                ("2. Method", True),
                ("The retriever encodes each passage with a BERT tower and is "
                 "fine-tuned end to end on the training split of the corpus.", False),
                ("Ibid.", False),
            ],
        ),
        "r",
    )
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in chunks)
    assert any("Ibid." in c.text for c in chunks)


def test_a_paper_with_no_headings_still_chunks(tmp_path: Path):
    """A scan, or a paper Box 3 could not read. It must not crash."""
    chunks = _chunks(
        _pdf(
            tmp_path / "flat.pdf",
            [("Plain prose with no headings anywhere in the document at all.", False)],
        ),
        "f",
    )
    assert len(chunks) == 1
    assert chunks[0].section == SectionType.TITLE
