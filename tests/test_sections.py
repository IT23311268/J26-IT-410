"""Tests for section identification — Box 4.

Box 3 labels a block HEADING. This stage reads the heading's words and
says *which* section it opens, so every chunk can be tagged with the
part of the paper it came from.

That tag is not decoration. The same sentence means different things in
different sections: "BM25 outperforms dense retrieval" in Related Work
is somebody else's finding being cited, and in Results it is this
paper's own. Member 2 verifies claims against these chunks and cannot
tell those apart without it.
"""

import pytest

from ingestion.sections import classify_section, strip_numbering
from schema.ingestion_schema_v1 import SectionType


# --------------------------------------------------------------------------
# the numbering comes off first
# --------------------------------------------------------------------------
#
# A heading's number says where it sits, never what it is. "2." could
# open Method in one paper and Related Work in another. It is stripped
# before anything is matched, and the original is kept on the chunk's
# `section_title_raw` so nothing is lost.


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("1. Introduction", "Introduction"),
        ("2 Method", "Method"),
        ("3.1 Results", "Results"),
        ("IV. EXPERIMENTS", "EXPERIMENTS"),          # IEEE roman numerals
        ("A. Dataset", "Dataset"),                   # IEEE lettered subsection
        ("Abstract", "Abstract"),                    # nothing to strip
        ("2.3 Generator: BART", "Generator: BART"),
    ],
)
def test_numbering_is_stripped(heading: str, expected: str):
    assert strip_numbering(heading) == expected


# --------------------------------------------------------------------------
# the ordinary IMRaD headings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("Abstract", SectionType.ABSTRACT),
        ("1. Introduction", SectionType.INTRODUCTION),
        ("2. Method", SectionType.METHOD),
        ("3. Results", SectionType.RESULTS),
        ("4. Conclusion", SectionType.CONCLUSION),
        ("References", SectionType.REFERENCES),
    ],
)
def test_the_plain_headings(heading: str, expected: SectionType):
    assert classify_section(heading) == expected


# --------------------------------------------------------------------------
# how papers actually write them
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("I. INTRODUCTION", SectionType.INTRODUCTION),      # IEEE, all caps
        ("2 Related Work", SectionType.RELATED_WORK),
        ("Background", SectionType.RELATED_WORK),
        ("Prior Work", SectionType.RELATED_WORK),
        ("3 Methodology", SectionType.METHOD),
        ("Our Approach", SectionType.METHOD),
        ("Model Architecture", SectionType.METHOD),
        ("4 Experiments", SectionType.RESULTS),
        ("Evaluation", SectionType.RESULTS),
        ("5. Discussion", SectionType.DISCUSSION),
        ("Bibliography", SectionType.REFERENCES),
        ("A Appendix", SectionType.APPENDIX),
        ("Supplementary Material", SectionType.APPENDIX),
    ],
)
def test_the_wordings_papers_really_use(heading: str, expected: SectionType):
    assert classify_section(heading) == expected


# --------------------------------------------------------------------------
# two section words in one heading
# --------------------------------------------------------------------------
#
# Combined headings are common and the order the keywords are tried in
# is what decides them. These four fix that order, so a later edit that
# reshuffles the list fails here instead of silently relabelling half a
# corpus.


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("Results and Discussion", SectionType.RESULTS),
        ("Discussion and Conclusion", SectionType.CONCLUSION),
        ("Conclusion and Future Work", SectionType.CONCLUSION),
        ("Experiments and Results", SectionType.RESULTS),
    ],
)
def test_combined_headings(heading: str, expected: SectionType):
    assert classify_section(heading) == expected


# --------------------------------------------------------------------------
# what must come back UNKNOWN
# --------------------------------------------------------------------------
#
# UNKNOWN is not a failure here — it is the answer that keeps a
# subsection inside the section it belongs to. "2.3 Generator: BART"
# opens no new section, so assign_sections leaves the paper in METHOD.
# A guess would end the Method section three paragraphs early.


@pytest.mark.parametrize(
    "heading",
    [
        "2.3 Generator: BART",
        "3.2 Retriever",
        "Acknowledgements",
        "",
        "   ",
    ],
)
def test_headings_that_open_no_section(heading: str):
    assert classify_section(heading) == SectionType.UNKNOWN


def test_a_paper_title_is_not_a_section():
    """The title is a HEADING to Box 3, and it must not be mistaken for
    one of the IMRaD sections just because a word happens to match."""
    assert classify_section("Layout-Aware Ingestion of Scholarly PDFs") == (
        SectionType.UNKNOWN
    )


# --------------------------------------------------------------------------
# walking the paper
# --------------------------------------------------------------------------
#
# classify_section reads one heading. assign_sections walks the whole
# document and carries the current section forward, which is where the
# UNKNOWN answer above earns its keep.


from pathlib import Path  # noqa: E402

from ingestion.layout import extract_regions  # noqa: E402
from ingestion.sections import assign_sections  # noqa: E402
from schema.ingestion_schema_v1 import RegionType  # noqa: E402


def _paper(path: Path, lines: list[tuple[str, bool]]) -> Path:
    """Build a PDF from (text, is_heading) pairs."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 740
    for text, is_heading in lines:
        c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 13 if is_heading else 10)
        c.drawString(72, y, text)
        y -= 30
    c.showPage()
    c.save()
    return path


@pytest.fixture
def walked(tmp_path: Path):
    pdf = _paper(
        tmp_path / "paper.pdf",
        [
            ("Layout-Aware Ingestion of Scholarly PDFs", True),
            ("Adheeshana Herath, SLIIT", False),
            ("Abstract", True),
            ("We present a layout-aware ingestion engine.", False),
            ("1. Introduction", True),
            ("Parsing a two-column paper naively shreds its sentences.", False),
            ("3. Method", True),
            ("We describe the retriever and the generator in turn.", False),
            ("3.1 Retriever: DPR", True),
            ("The retriever encodes the query with a BERT tower.", False),
            ("3.2 Generator: BART", True),
            ("The generator is a sequence-to-sequence transformer.", False),
            ("4. Results", True),
            ("We beat the baseline by 4.2 points.", False),
        ],
    )
    return assign_sections(extract_regions(pdf, "walk"))


def _section_of(regions, needle: str):
    return next(r for r in regions if needle in r.text).section


def test_the_title_block_is_the_title_section(walked):
    """Everything before the first recognised heading — the title line
    and the authors — belongs to TITLE, not to the abstract."""
    assert _section_of(walked, "Layout-Aware") == SectionType.TITLE
    assert _section_of(walked, "Adheeshana") == SectionType.TITLE


def test_body_text_inherits_its_heading(walked):
    assert _section_of(walked, "layout-aware ingestion engine") == SectionType.ABSTRACT
    assert _section_of(walked, "shreds its sentences") == SectionType.INTRODUCTION
    assert _section_of(walked, "4.2 points") == SectionType.RESULTS


def test_an_unrecognised_subheading_does_not_end_its_section(walked):
    """The regression this guards: '3.1 Retriever: DPR' and '3.2
    Generator: BART' classify as UNKNOWN. If UNKNOWN moved us, the rest
    of the Method would land in a section that does not exist."""
    assert _section_of(walked, "3.1 Retriever") == SectionType.METHOD
    assert _section_of(walked, "BERT tower") == SectionType.METHOD
    assert _section_of(walked, "3.2 Generator") == SectionType.METHOD
    assert _section_of(walked, "sequence-to-sequence") == SectionType.METHOD


def test_the_next_real_heading_does_move_us(walked):
    assert _section_of(walked, "4. Results") == SectionType.RESULTS


def test_the_raw_heading_is_kept_beside_the_label(walked):
    """A misclassification must lose nothing: the heading as printed
    stays on the region."""
    assert _section_of(walked, "BERT tower") == SectionType.METHOD
    region = next(r for r in walked if "BERT tower" in r.text)
    assert region.section_title_raw == "3. Method"


def test_every_region_is_tagged(walked):
    assert walked
    assert all(r.section is not None for r in walked)


def test_reading_order_is_untouched(walked):
    """Box 4 relabels; it must not reorder."""
    assert [r.order_index for r in walked] == sorted(r.order_index for r in walked)


def test_a_paper_with_no_headings_at_all(tmp_path: Path):
    """A scan, or a paper Box 3 could not read. Everything stays TITLE
    rather than crashing or inventing sections."""
    pdf = _paper(tmp_path / "flat.pdf", [("Plain prose with no headings at all.", False)])
    regions = assign_sections(extract_regions(pdf, "flat"))
    assert all(r.section == SectionType.TITLE for r in regions)


def test_figures_and_captions_are_tagged_too(walked):
    """Not only BODY: Box 6 crops figures and needs to know which
    section each one illustrates."""
    for region in walked:
        if region.region_type in (RegionType.FIGURE, RegionType.CAPTION):
            assert region.section != SectionType.UNKNOWN


# --------------------------------------------------------------------------
# the paper's own title
# --------------------------------------------------------------------------

from ingestion.sections import MAX_TITLE_CHARS, paper_title  # noqa: E402


def test_the_title_comes_off_the_page_not_the_filename(walked):
    assert paper_title(walked) == "Layout-Aware Ingestion of Scholarly PDFs"


def test_a_section_heading_is_never_mistaken_for_the_title(tmp_path: Path):
    """A paper whose front matter Box 3 did not mark as a heading must
    return nothing rather than handing back 'Abstract'."""
    pdf = _paper(
        tmp_path / "p.pdf",
        [
            ("Abstract", True),
            ("We present a layout-aware ingestion engine.", False),
            ("1. Introduction", True),
            ("Parsing a two-column paper naively shreds its sentences.", False),
        ],
    )
    assert paper_title(assign_sections(extract_regions(pdf, "p"))) == ""


def test_a_paragraph_is_not_a_title(tmp_path: Path):
    """A heading rule that fires on a long run of bold text would
    otherwise put a whole paragraph in the masthead."""
    pdf = _paper(
        tmp_path / "p.pdf",
        [("x" * (MAX_TITLE_CHARS + 40), True), ("Abstract", True), ("Body.", False)],
    )
    assert paper_title(assign_sections(extract_regions(pdf, "p"))) == ""


def test_no_regions_means_no_title():
    assert paper_title([]) == ""


def test_the_arxiv_stamp_is_not_the_title():
    """arXiv prints its own identifier down the left margin of page
    one, rotated a quarter turn. It sits above the real title and it
    is set large enough to pass the heading rule, so every paper
    fetched from arXiv came back titled after its own catalogue
    number. Measured on arXiv:2610.12376 the stamp's box is 20 points
    wide and 335 tall; the title below it is 389 by 46."""
    from schema.ingestion_schema_v1 import BoundingBox, LayoutRegion

    def heading(text, x0, y0, x1, y1):
        return LayoutRegion(
            region_id=f"r{y0}",
            paper_id="p",
            region_type=RegionType.HEADING,
            bbox=BoundingBox(page=0, x0=x0, y0=y0, x1=x1, y1=y1),
            text=text,
            order_index=int(y0),
            section=SectionType.TITLE,
        )

    regions = [
        heading("arXiv:2610.12376v1  [cs.CL]  8 Oct 2026", 24.3, 253.2, 44.3, 588.7),
        heading("Latent Core Tokenizer: Compress, but Meaningfully", 103, 96, 492, 142),
    ]
    assert paper_title(regions) == "Latent Core Tokenizer: Compress, but Meaningfully"


def test_an_ordinary_title_is_not_mistaken_for_a_stamp():
    """The rule is `taller than wide`. A short title set over three
    lines is still wider than it is tall."""
    from schema.ingestion_schema_v1 import BoundingBox, LayoutRegion

    region = LayoutRegion(
        region_id="r1",
        paper_id="p",
        region_type=RegionType.HEADING,
        bbox=BoundingBox(page=0, x0=120, y0=90, x1=460, y1=170),
        text="Attention Is All You Need",
        order_index=0,
        section=SectionType.TITLE,
    )
    assert paper_title([region]) == "Attention Is All You Need"


def _title_line(text, y0, y1, size, kind=RegionType.HEADING, section=SectionType.TITLE):
    from schema.ingestion_schema_v1 import BoundingBox, LayoutRegion

    return LayoutRegion(
        region_id=f"r{y0}",
        paper_id="p",
        region_type=kind,
        bbox=BoundingBox(page=0, x0=100, y0=y0, x1=500, y1=y1),
        text=text,
        order_index=int(y0),
        font_size=size,
        section=section,
    )


def test_a_title_set_over_two_lines_comes_back_whole():
    """Any title longer than about forty characters is broken across
    lines at title size, and each printed line is its own region. The
    page still says one title.

    The numbers are measured off a rendering of this layout: both
    lines 17.0 pt, the second beginning 0.4 pt above where the first
    ends, and the author line 10.5 pt and 20.5 pt clear."""
    regions = [
        _title_line("Retrieval-Augmented Generation for", 71.8, 95.2, 17.0),
        _title_line("Knowledge-Intensive NLP Tasks", 94.8, 118.2, 17.0),
        _title_line("Patrick Lewis, Ethan Perez", 138.7, 153.1, 10.5,
                    kind=RegionType.BODY),
    ]
    assert paper_title(regions) == (
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    )


def test_a_second_line_in_different_type_is_not_part_of_the_title():
    """A subtitle, a venue line or a running head set under the title
    is a different block, and the size says so."""
    regions = [
        _title_line("Latent Core Tokenizer", 90.0, 112.0, 17.0),
        _title_line("Proceedings of the 41st Conference", 113.0, 127.0, 9.0),
    ]
    assert paper_title(regions) == "Latent Core Tokenizer"


def test_a_banner_set_in_title_type_is_not_dragged_into_the_title():
    """Same size, but a third of the page away. Consecutive lines of
    one block sit against each other; this one does not."""
    regions = [
        _title_line("Journal of Document Engineering", 40.0, 62.0, 17.0),
        _title_line("Layout-Aware Ingestion", 300.0, 322.0, 17.0),
    ]
    assert paper_title(regions) == "Journal of Document Engineering"


def test_a_thesis_title_page_still_yields_a_title():
    """A thesis title page has no type hierarchy at all: the title,
    "A Thesis", the degree and the year are every one of them 12 pt.
    Box 3 marks nothing on it as a heading, so the size rule has
    nothing to find and the page used to fall back to showing the
    filename. Capitals are what is left.

    Geometry measured off a rendering of a Guelph MSc title page: the
    two title lines 19.0 pt apart (1.58 line heights) and "A Thesis"
    101.0 pt below them (8.42)."""
    regions = [
        _title_line("A PRELIMINARY STUDY TOWARDS AUTOMATIC LINGUISTIC",
                    192.4, 208.4, 12.0, kind=RegionType.BODY),
        _title_line("INDEXING OF IMAGES BASED ON SPATIAL RELATIONS",
                    227.4, 243.4, 12.0, kind=RegionType.BODY),
        _title_line("A Thesis", 344.4, 360.4, 12.0, kind=RegionType.BODY),
        _title_line("XIN WANG", 611.4, 627.4, 12.0, kind=RegionType.BODY),
    ]
    assert paper_title(regions) == (
        "A PRELIMINARY STUDY TOWARDS AUTOMATIC LINGUISTIC "
        "INDEXING OF IMAGES BASED ON SPATIAL RELATIONS"
    )


def test_capitals_never_overrule_a_heading():
    """The fallback runs only when the front matter has no heading at
    all. A paper with a shouted banner above its title still gets its
    title, because the banner is not the heading."""
    regions = [
        _title_line("PROCEEDINGS OF THE 41ST CONFERENCE", 40.0, 52.0, 9.0,
                    kind=RegionType.BODY),
        _title_line("Layout-Aware Ingestion of Scholarly PDFs",
                    90.0, 112.0, 17.0),
    ]
    assert paper_title(regions) == "Layout-Aware Ingestion of Scholarly PDFs"


def test_a_front_matter_with_neither_heading_nor_capitals_gives_nothing():
    """Better the filename than a guess."""
    regions = [
        _title_line("a note on the margins", 90.0, 102.0, 11.0,
                    kind=RegionType.BODY),
    ]
    assert paper_title(regions) == ""
