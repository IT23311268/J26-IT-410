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
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Emu, RGBColor

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

CAPTION_PARTS = re.compile(
    r"^(fig\.?|figure|table|algorithm)\s*([0-9]+|[IVXL]+)[a-z]?\s*[.:\-–—|]?\s*(.*)$", re.I)
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
        self._run_in = None              # a level-3/4 heading waiting for its paragraph

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
        for section in self._ordered_sections():
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
            shown = self._author_block()
        else:
            shown = set()
            self.notes.append("No authors found.")

        # affiliations not printed under any author (none linked, or e-mail lines)
        rest = [a for i, a in enumerate(self.paper.affiliations) if i not in shown]
        for i, affiliation in enumerate(rest):
            last = i == len(rest) - 1
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           after=S.AUTHOR_BLOCK_SPACE_AFTER if last else 0)
            _set_font(_add(p, affiliation), S.AFFILIATION_SIZE, italic=True)
        if self.paper.authors and not rest:
            spacer = _paragraph(self.doc, after=0)
            spacer.paragraph_format.space_after = S.AUTHOR_BLOCK_SPACE_AFTER

        if not self.paper.affiliations:
            self.doc.paragraphs[-1].paragraph_format.space_after = S.AUTHOR_BLOCK_SPACE_AFTER

    def _author_block(self) -> set:
        """TTL-06: the IEEE author block - every author in a column of their own,
        their affiliation(s) in italics underneath, up to three authors per row.
        Returns the indexes of the affiliations it printed."""
        authors = self.paper.authors
        links = self.paper.author_affiliations
        if len(links) != len(authors):
            links = [[] for _ in authors]
        per_row = min(S.AUTHORS_PER_ROW, len(authors))
        rows = (len(authors) + per_row - 1) // per_row
        table = self.doc.add_table(rows=rows, cols=per_row)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        width = S.PAGE_WIDTH - S.MARGIN_LEFT - S.MARGIN_RIGHT
        shown: set = set()
        for i, name in enumerate(authors):
            cell = table.cell(i // per_row, i % per_row)
            cell.width = width // per_row
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4) if i >= per_row else 0
            _set_font(_add(p, name), S.AUTHOR_SIZE)
            for index in links[i]:
                if 0 <= index < len(self.paper.affiliations):
                    q = cell.add_paragraph()
                    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    _set_font(_add(q, self.paper.affiliations[index]), S.AFFILIATION_SIZE,
                              italic=True)
                    shown.add(index)
        # an uneven last row: centre it by merging the empty cells away is not
        # possible in Word, so the empty cells simply stay blank
        return shown

    # -- abstract and index terms -------------------------------------------

    def _abstract_and_index_terms(self) -> None:
        # IEEE: 9pt bold, the label in bold italic, justified.
        p = _paragraph(self.doc, after=S.HEADING2_SPACE_AFTER)
        _set_font(_add(p, S.ABSTRACT_LABEL), S.ABSTRACT_SIZE, bold=True, italic=True)
        if self.paper.abstract:
            _set_font(_add(p, self.paper.abstract), S.ABSTRACT_SIZE, bold=True)
        else:
            run = _add(p, "[Abstract missing - the author should add a 150-250 word abstract.]")
            _set_font(run, S.ABSTRACT_SIZE, bold=True)
            run.font.color.rgb = PLACEHOLDER_COLOUR
            self.notes.append("Abstract missing - placeholder inserted.")

        p = _paragraph(self.doc, after=S.HEADING1_SPACE_BEFORE)
        _set_font(_add(p, S.INDEX_TERMS_LABEL), S.ABSTRACT_SIZE, bold=True, italic=True)
        if self.paper.keywords:
            # ABS-07/ABS-08: alphabetical, lower case except acronyms and proper
            # nouns - and only the first word of the list is capitalised.
            terms = ", ".join(self.paper.keywords)
            terms = terms[:1].upper() + terms[1:]
            _set_font(_add(p, terms), S.ABSTRACT_SIZE, bold=True)
        else:
            run = _add(p, "[Index terms missing - the author should add 3-5 keywords.]")
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
            # HDG-03 / HDG-04: "1) Name:" (level 3) or "a) Name:" (level 4) in
            # italics, RUN INTO the first paragraph of the section. Level 4 is
            # indented further. If the section starts with no paragraph, the
            # heading stands on its own line.
            label = f"{section.ieee_number} {section.ieee_heading}:"
            first = next((e for e in section.elements), None)
            if first is not None and first.kind == "paragraph":
                self._run_in = (label, section.level)
                return
            indent = S.BODY_FIRST_LINE_INDENT * (2 if section.level >= 4 else 1)
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.LEFT,
                           first_line=indent, keep_with_next=True)
            _set_font(_add(p, label), S.HEADING_SIZE, italic=True)

    def _ordered_sections(self) -> List[Section]:
        """ACK-02: the main sections first, then appendices, then the
        Acknowledgment (References follow). Subsections stay with their parent."""
        groups: List[List[Section]] = []
        for s in self.paper.sections:
            if s.level == 1 or not groups:
                groups.append([s])
            else:
                groups[-1].append(s)
        rank = {"appendix": 1, "acknowledgment": 2}
        groups.sort(key=lambda g: rank.get(g[0].ieee_slot, 0))     # stable: keeps author order
        return [s for g in groups for s in g]

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
                # cannot be rebuilt from text, so Step 6 copied it from the
                # source PDF as a picture. Never print the scraps.
                if element.image and os.path.exists(element.image):
                    p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                                   before=S.CAPTION_SPACE_BEFORE, after=S.CAPTION_SPACE_BEFORE)
                    self._picture(p, element.image)
                else:
                    number = re.search(r"\((\d{1,3})\)", element.text)
                    label = f"Equation ({number.group(1)})" if number else "Display equation"
                    _placeholder(self.doc, f"{label} - could not be recovered from the source")
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
        run_in, self._run_in = self._run_in, None
        if run_in:                                   # HDG-03 / HDG-04 run-in heading
            label, level = run_in
            p = _paragraph(self.doc, first_line=S.BODY_FIRST_LINE_INDENT * (2 if level >= 4 else 1))
            _set_font(_add(p, label + " "), S.HEADING_SIZE, italic=True)
            _set_font(_add(p, text), S.BODY_SIZE)
            return
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
        has_image = bool(element.image) and os.path.exists(element.image)

        if kind.startswith("fig"):
            # IEEE: the figure first, its caption BELOW it
            if has_image:
                p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                               before=S.CAPTION_SPACE_BEFORE, keep_with_next=True)
                self._picture(p, element.image)
            else:
                _placeholder(self.doc, f"Figure {number} image - not found in the source")
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           before=S.CAPTION_SPACE_BEFORE, after=S.CAPTION_SPACE_AFTER)
            _set_font(_add(p, f"{S.FIGURE_LABEL} {number}. "), S.CAPTION_SIZE)
            _set_font(_add(p, f"{rest}."), S.CAPTION_SIZE)
        elif kind == "algorithm":
            # caption line, then the algorithm box exactly as in the source
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.LEFT,
                           before=S.CAPTION_SPACE_BEFORE, keep_with_next=True)
            _set_font(_add(p, f"{S.ALGORITHM_LABEL} {number}. "), S.CAPTION_SIZE, bold=True)
            _set_font(_add(p, rest), S.CAPTION_SIZE)
            if has_image:
                p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                               after=S.CAPTION_SPACE_AFTER)
                self._picture(p, element.image)
            else:
                _placeholder(self.doc, f"Algorithm {number} - not found in the source")
        else:
            # IEEE: "TABLE II" and its title in small caps ABOVE the table
            roman = number if not number.isdigit() else to_roman(int(number))
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                           before=S.CAPTION_SPACE_BEFORE, keep_with_next=True)
            _set_font(_add(p, f"{S.TABLE_LABEL} {roman}"), S.CAPTION_SIZE)
            p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER, keep_with_next=True)
            _set_font(_add(p, rest), S.CAPTION_SIZE, small_caps=True)
            if has_image:
                p = _paragraph(self.doc, align=WD_ALIGN_PARAGRAPH.CENTER,
                               before=S.CAPTION_SPACE_BEFORE)
                self._picture(p, element.image)
            else:
                _placeholder(self.doc, f"Table {roman} content - not found in the source")
            self.doc.paragraphs[-1].paragraph_format.space_after = S.CAPTION_SPACE_AFTER

    def _picture(self, paragraph, path: str) -> None:
        """Insert an image at its natural size, but never wider than a column.

        Images cropped from the PDF are rendered at 200 dpi, so their natural
        width in inches is pixels / 200. A small equation stays small; a big
        figure is shrunk to fit the 3.5-inch column.
        """
        try:
            import pymupdf
            natural = Emu(int(pymupdf.Pixmap(path).width / 200 * 914400))
            width = min(natural, S.COLUMN_WIDTH)
        except Exception:
            width = S.COLUMN_WIDTH
        paragraph.add_run().add_picture(path, width=width)

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
