"""
docx_renderer.py - stage 4: write the IEEE-formatted Word document.

Input : a StructuredPaper that has been through stage 3 (it knows its IEEE
        headings and numbers).
Output: a .docx file laid out the way IEEE requires.

The document is built in two parts, because that is how IEEE pages work:

    PART 1 - one column across the full page
             the title, the authors, the affiliations

    PART 2 - two columns
             abstract, index terms, every section, references

Word calls these parts "sections" (not to be confused with the paper's
sections). Part 2 starts with a "continuous" section break, which means the
column change happens without starting a new page.

All sizes and measurements come from ieee_style.py - this file only decides
WHAT goes on the page, never HOW BIG it is.
"""

from __future__ import annotations

import os
import re
from typing import List

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor

from ..models import Element, Section, StructuredPaper
from ..structure.ieee_mapper import to_roman
from . import ieee_style as S

PLACEHOLDER_COLOUR = RGBColor(0x80, 0x80, 0x80)

# Word refuses invisible control characters and crashes if given one.
# The readers already remove them; this is a second safety net.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]")


def _add(paragraph, text: str):
    """Add text to a paragraph, minus any characters Word cannot store."""
    return paragraph.add_run(CONTROL_CHARS.sub("", text))

CAPTION_PARTS = re.compile(r"^(fig\.?|figure|table)\s*([0-9]+|[IVXL]+)\s*[.:\-–—]?\s*(.*)$", re.I)
EQUATION_NUMBER = re.compile(r"^(.*?)\s*(\(\s*\d+\s*\))\s*$")
ALREADY_NUMBERED_REF = re.compile(r"^\s*(\[\d+\]|\d{1,3}\.)\s*")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _set_font(run, size, bold=False, italic=False, small_caps=False) -> None:
    """Apply IEEE font settings to one run of text.

    Word stores the font name in several places (one per alphabet). If we set
    only the main one, some text silently falls back to Calibri, so we set
    the East-Asian slot too.
    """
    run.font.name = S.FONT
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    run.font.small_caps = small_caps
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        r_fonts.set(qn(attr), S.FONT)


def _paragraph(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0,
               first_line=None, keep_with_next=False):
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = align
    fmt.space_before = before
    fmt.space_after = after
    fmt.line_spacing = 1.0
    if first_line is not None:
        fmt.first_line_indent = first_line
    fmt.keep_with_next = keep_with_next
    return p


def _set_columns(section, count: int, gap) -> None:
    """Word has no python-docx setting for columns, so we write the XML tag
    <w:cols w:num="2" w:space="360"/> ourselves. 360 twips = 0.25 inch."""
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(int(gap.twips)))


def _page_setup(section) -> None:
    section.page_width = S.PAGE_WIDTH
    section.page_height = S.PAGE_HEIGHT
    section.top_margin = S.MARGIN_TOP
    section.bottom_margin = S.MARGIN_BOTTOM
    section.left_margin = S.MARGIN_LEFT
    section.right_margin = S.MARGIN_RIGHT


def _placeholder(doc, text: str) -> None:
    """A grey, bracketed note marking something a later step will fill in.
    Better an honest gap than invented content."""
    p = _paragraph(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=S.CAPTION_SPACE_BEFORE)
    run = _add(p, f"[{text}]")
    _set_font(run, S.CAPTION_SIZE, italic=True)
    run.font.color.rgb = PLACEHOLDER_COLOUR


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class IEEEDocxRenderer:
    def __init__(self, paper: StructuredPaper):
        self.paper = paper
        self.doc = Document()
        self.notes: List[str] = []       # what the renderer had to work around

    # -- public -------------------------------------------------------------

    def render(self, output_path: str) -> List[str]:
        self._setup_default_style()
        _page_setup(self.doc.sections[0])
        _set_columns(self.doc.sections[0], 1, S.COLUMN_GAP)

        self._title_block()

        two_col = self.doc.add_section(WD_SECTION.CONTINUOUS)
        _page_setup(two_col)
        _set_columns(two_col, S.COLUMN_COUNT, S.COLUMN_GAP)

        self._abstract_and_index_terms()
        for section in self.paper.sections:
            self._section(section)
        self._references()

        # IEEE wants the two columns on the last page to end at the same
        # height. Word does this automatically when the two-column part is
        # closed by one more continuous section break.
        closing = self.doc.add_section(WD_SECTION.CONTINUOUS)
        _page_setup(closing)
        _set_columns(closing, S.COLUMN_COUNT, S.COLUMN_GAP)

        self.doc.save(output_path)
        return self.notes

    # -- document defaults --------------------------------------------------

    def _setup_default_style(self) -> None:
        normal = self.doc.styles["Normal"]
        normal.font.name = S.FONT
        normal.font.size = S.BODY_SIZE
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), S.FONT)
        normal.paragraph_format.space_after = 0
        normal.paragraph_format.line_spacing = 1.0
        # python-docx starts every document with one empty paragraph; reuse
        # nothing from it - we remove it so the title is the first line.
        body = self.doc.element.body
        for p in list(body.iterchildren(qn("w:p"))):
            body.remove(p)

    # -- part 1: title, authors ---------------------------------------------

    def _title_block(self) -> None:
        p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER, after=S.TITLE_SPACE_AFTER)
        title = self.paper.title or "Untitled Manuscript"
        if not self.paper.title:
            self.notes.append("No title found - used 'Untitled Manuscript'.")
        _set_font(_add(p, title), S.TITLE_SIZE)

        if self.paper.authors:
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER)
            _set_font(_add(p, ", ".join(self.paper.authors)), S.AUTHOR_SIZE)
        else:
            self.notes.append("No authors found.")

        for i, affiliation in enumerate(self.paper.affiliations):
            last = i == len(self.paper.affiliations) - 1
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           after=S.AUTHOR_BLOCK_SPACE_AFTER if last else 0)
            _set_font(_add(p, affiliation), S.AFFILIATION_SIZE, italic=True)

        if not self.paper.affiliations:
            self.doc.paragraphs[-1].paragraph_format.space_after = S.AUTHOR_BLOCK_SPACE_AFTER

    # -- abstract and index terms -------------------------------------------

    def _abstract_and_index_terms(self) -> None:
        # IEEE: 9pt bold, the label in bold italic, justified.
        p = _paragraph(self.doc, after=S.HEADING2_SPACE_AFTER)
        _set_font(_add(p, S.ABSTRACT_LABEL), S.ABSTRACT_SIZE, bold=True, italic=True)
        if self.paper.abstract:
            _set_font(_add(p, self.paper.abstract), S.ABSTRACT_SIZE, bold=True)
        else:
            run = _add(p, "[Abstract missing - will be generated in Step 5.]")
            _set_font(run, S.ABSTRACT_SIZE, bold=True)
            run.font.color.rgb = PLACEHOLDER_COLOUR
            self.notes.append("Abstract missing - placeholder inserted.")

        p = _paragraph(self.doc, after=S.HEADING1_SPACE_BEFORE)
        _set_font(_add(p, S.INDEX_TERMS_LABEL), S.ABSTRACT_SIZE, bold=True, italic=True)
        if self.paper.keywords:
            # IEEE: index terms in lower case except proper nouns and acronyms.
            # We keep the author's casing - changing "CT" to "ct" would be wrong.
            terms = ", ".join(self.paper.keywords)
            _set_font(_add(p, terms), S.ABSTRACT_SIZE, bold=True)
        else:
            run = _add(p, "[Index terms missing - will be generated in Step 5.]")
            _set_font(run, S.ABSTRACT_SIZE, bold=True)
            run.font.color.rgb = PLACEHOLDER_COLOUR
            self.notes.append("Index terms missing - placeholder inserted.")

    # -- sections -----------------------------------------------------------

    def _heading(self, section: Section) -> None:
        if section.level == 1:
            # "I. Introduction" in small caps, centred.
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           before=S.HEADING1_SPACE_BEFORE, after=S.HEADING1_SPACE_AFTER,
                           keep_with_next=True)
            label = f"{section.ieee_number}. {section.ieee_heading}" if section.ieee_number \
                else section.ieee_heading
            _set_font(_add(p, label), S.HEADING_SIZE, small_caps=True)
        elif section.level == 2:
            # "A. Dataset" in italics, left-aligned.
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.LEFT,
                           before=S.HEADING2_SPACE_BEFORE, after=S.HEADING2_SPACE_AFTER,
                           keep_with_next=True)
            _set_font(_add(p, f"{section.ieee_number}. {section.ieee_heading}"),
                      S.HEADING_SIZE, italic=True)
        else:
            # "1) Name:" in italics, indented like a paragraph.
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.LEFT,
                           first_line=S.BODY_FIRST_LINE_INDENT, keep_with_next=True)
            _set_font(_add(p, f"{section.ieee_number} {section.ieee_heading}:"),
                      S.HEADING_SIZE, italic=True)

    def _section(self, section: Section) -> None:
        self._heading(section)
        previous = ""
        for element in section.elements:
            if element.kind == "paragraph":
                # "where alpha was fixed..." after an equation continues the
                # same sentence, so IEEE does not indent it.
                continues = previous == "equation" and element.text[:1].islower()
                self._body_paragraph(element.text, indent=not continues)
            elif element.kind == "equation":
                self._equation(element.text)
            elif element.kind == "code":
                self._code(element.text)
            elif element.kind == "math_region":
                # A display equation the PDF stored as scattered symbols. It
                # cannot be rebuilt from text, so it will be copied from the
                # source PDF as an image in Step 6. Never print the scraps.
                number = re.search(r"\((\d{1,3})\)", element.text)
                label = f"Equation ({number.group(1)})" if number else "Display equation"
                _placeholder(self.doc, f"{label} - copied from the source PDF in Step 6")
            elif element.kind == "caption":
                self._caption(element)
            previous = element.kind

    def _code(self, text: str) -> None:
        """A code or prompt listing: typewriter font, small, line for line."""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.LEFT,
                           before=S.CAPTION_SPACE_BEFORE if i == 0 else 0,
                           after=S.CAPTION_SPACE_BEFORE if i == len(lines) - 1 else 0)
            run = _add(p, line)
            _set_font(run, S.CODE_SIZE)
            run.font.name = S.CODE_FONT
            r_fonts = run._element.get_or_add_rPr().find(qn("w:rFonts"))
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                r_fonts.set(qn(attr), S.CODE_FONT)

    def _body_paragraph(self, text: str, indent: bool = True) -> None:
        p = _paragraph(self.doc, first_line=S.BODY_FIRST_LINE_INDENT if indent else 0)
        _set_font(_add(p, text), S.BODY_SIZE)

    def _equation(self, text: str) -> None:
        """Equation centred in the column, its number pushed to the right edge.

        Done with two tab stops: a centre tab in the middle of the column and
        a right tab at its edge. This is how the IEEE Word template does it.
        """
        match = EQUATION_NUMBER.match(text)
        body, number = (match.group(1).strip(), match.group(2).replace(" ", "")) \
            if match else (text.strip(), "")

        p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.LEFT,
                       before=S.CAPTION_SPACE_BEFORE, after=S.CAPTION_SPACE_BEFORE)
        stops = p.paragraph_format.tab_stops
        stops.add_tab_stop(S.COLUMN_WIDTH // 2, WD_TAB_ALIGNMENT.CENTER)
        stops.add_tab_stop(S.COLUMN_WIDTH, WD_TAB_ALIGNMENT.RIGHT)
        _set_font(_add(p, "\t" + body), S.BODY_SIZE, italic=True)
        if number:
            _set_font(_add(p, "\t" + number), S.BODY_SIZE)

    def _caption(self, element: Element) -> None:
        """Figures:  'Fig. 1. Caption text.'       - BELOW the figure
           Tables:   'TABLE I' / 'CAPTION TEXT'    - ABOVE the table, small caps
        """
        match = CAPTION_PARTS.match(element.text)
        if not match:
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           before=S.CAPTION_SPACE_BEFORE, after=S.CAPTION_SPACE_AFTER)
            _set_font(_add(p, element.text), S.CAPTION_SIZE)
            return

        kind, number, rest = match.group(1).lower(), match.group(2), match.group(3).strip()
        rest = rest.rstrip(".")

        if kind.startswith("fig"):
            if element.image and os.path.exists(element.image):
                # A real image came with the content (e.g. from Member 1's figure
                # repository): insert it at the full width of the column.
                p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                               before=S.CAPTION_SPACE_BEFORE, keep_with_next=True)
                p.add_run().add_picture(element.image, width=S.COLUMN_WIDTH)
            else:
                _placeholder(self.doc, f"Figure {number} image - placed in Step 6")
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           before=S.CAPTION_SPACE_BEFORE, after=S.CAPTION_SPACE_AFTER)
            _set_font(_add(p, f"{S.FIGURE_LABEL} {number}. "), S.CAPTION_SIZE)
            _set_font(_add(p, f"{rest}."), S.CAPTION_SIZE)
        else:
            roman = number if not number.isdigit() else to_roman(int(number))
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           before=S.CAPTION_SPACE_BEFORE, keep_with_next=True)
            _set_font(_add(p, f"{S.TABLE_LABEL} {roman}"), S.CAPTION_SIZE)
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER, keep_with_next=True)
            _set_font(_add(p, rest), S.CAPTION_SIZE, small_caps=True)
            _placeholder(self.doc, f"Table {roman} content - placed in Step 6")
            self.doc.paragraphs[-1].paragraph_format.space_after = S.CAPTION_SPACE_AFTER

    # -- references ---------------------------------------------------------

    def _references(self) -> None:
        p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                       before=S.HEADING1_SPACE_BEFORE, after=S.HEADING1_SPACE_AFTER,
                       keep_with_next=True)
        _set_font(_add(p, S.REFERENCES_HEADING), S.HEADING_SIZE, small_caps=True)

        if not self.paper.references:
            _placeholder(self.doc, "No references found in the source document")
            self.notes.append("No references found.")
            return

        records = self.paper.reference_records
        for i, ref in enumerate(self.paper.references, start=1):
            # Hanging indent: "[1]" sits in the margin, the wrapped lines line up
            # with the start of the text - the IEEE reference-list look.
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
            fmt = p.paragraph_format
            fmt.left_indent = S.REFERENCE_HANGING_INDENT
            fmt.first_line_indent = -S.REFERENCE_HANGING_INDENT
            fmt.tab_stops.add_tab_stop(S.REFERENCE_HANGING_INDENT)
            _set_font(_add(p, f"[{i}]\t"), S.REFERENCE_SIZE)

            if i <= len(records) and records[i - 1].get("segments"):
                # Step 5 has split the reference into pieces; the journal or
                # conference name is printed in italics, as IEEE requires.
                for text, italic in records[i - 1]["segments"]:
                    _set_font(_add(p, text), S.REFERENCE_SIZE, italic=italic)
            else:
                _set_font(_add(p, ALREADY_NUMBERED_REF.sub("", ref, count=1)), S.REFERENCE_SIZE)

        if not records:
            self.notes.append(
                "References are numbered [1], [2]... but still in the author's original "
                "style (Step 5 was skipped)."
            )


def render_docx(paper: StructuredPaper, output_path: str) -> List[str]:
    """Write the paper as an IEEE-formatted .docx. Returns notes about any gaps."""
    return IEEEDocxRenderer(paper).render(output_path)
