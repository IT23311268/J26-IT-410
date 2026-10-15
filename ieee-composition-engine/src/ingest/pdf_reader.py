"""
pdf_reader.py - stage 1a: turn a PDF into a list of TextBlock objects.

A PDF does not store paragraphs or headings. It stores thousands of tiny
instructions that mean "draw these glyphs at this position in this font".
PyMuPDF groups those glyphs back into spans, lines and blocks for us, and -
crucially - tells us the font size and style of each one. Those are the clues
the classifier will use in stage 2.

We work at LINE level here. Merging lines back into paragraphs needs to know
which lines are headings and which are body text, and we don't know that yet,
so that job belongs to stage 2.
"""

from __future__ import annotations

import re

import pymupdf

from ..models import ParsedDocument, TextBlock

# PyMuPDF packs font styles into a single integer called "flags".
# Each style is one bit. We test the bits we care about.
FLAG_ITALIC = 1 << 1   # value 2
FLAG_BOLD = 1 << 4     # value 16


def _span_is_bold(span: dict) -> bool:
    """A span is bold if PyMuPDF says so, or if the font name admits it."""
    if span.get("flags", 0) & FLAG_BOLD:
        return True
    name = span.get("font", "").lower()
    return "bold" in name or "black" in name or "heavy" in name


def _span_is_italic(span: dict) -> bool:
    if span.get("flags", 0) & FLAG_ITALIC:
        return True
    name = span.get("font", "").lower()
    return "italic" in name or "oblique" in name


# Characters Word cannot store: invisible control codes some PDFs contain
# (everything below the space character except tab and newline).
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]")

# A gap wider than this fraction of the font size counts as a space.
# Letters inside a word sit edge to edge (gap about 0); LaTeX squeezes the
# gap between words down to roughly 0.1 of the font size, so 0.07 sits
# safely between the two.
SPACE_GAP_RATIO = 0.07


def _is_horizontal(line: dict) -> bool:
    """Normal text runs left to right: direction (1, 0). Sideways text - like
    the arXiv identifier printed down the left margin - is skipped."""
    dx, dy = line.get("dir", (1.0, 0.0))
    return dx > 0.99 and abs(dy) < 0.01


def _rebuild_text(spans: list) -> str:
    """Rebuild a line's text letter by letter, adding the spaces the PDF left out.

    Many PDFs made with LaTeX store no space characters at all - just a
    small visual gap between words. PyMuPDF only notices gaps of about 1.5pt
    or more, so tightly justified LaTeX lines come out as
    "Actionforecastingfitsthe". Looking at each letter's position fixes that.
    """
    out = []
    prev = None                                  # (right edge, font size, char)
    for span in spans:
        size = float(span.get("size", 10.0)) or 10.0
        for ch in span.get("chars", []):
            c = ch.get("c", "")
            x0, _, x1, _ = ch["bbox"]
            if prev is not None and c != " " and prev[2] != " ":
                gap = x0 - prev[0]
                if gap > SPACE_GAP_RATIO * max(size, prev[1]):
                    out.append(" ")
            out.append(c)
            prev = (x1, size, c)
    text = CONTROL_CHARS.sub("", "".join(out))
    return " ".join(text.split())                 # squash runs of whitespace


def _line_to_block(line: dict, page_number: int) -> TextBlock | None:
    """Collapse one line's spans into a single TextBlock.

    A line can be split into several spans when the styling changes mid-line,
    for example "Figure 1. The system architecture" where only "Figure 1." is
    bold. We join the text back together and describe the line by its
    DOMINANT span - the one carrying the most characters - because that is
    what a human eye reads the line as.
    """
    if not _is_horizontal(line):
        return None

    spans = [s for s in line.get("spans", [])
             if "".join(c.get("c", "") for c in s.get("chars", [])).strip()]
    if not spans:
        return None

    text = _rebuild_text(spans)
    if not text:
        return None

    dominant = max(spans, key=lambda s: len(s.get("chars", [])))
    x0, y0, x1, y1 = line["bbox"]

    return TextBlock(
        text=text,
        page=page_number,
        font_size=round(float(dominant.get("size", 0.0)), 1),
        font_name=dominant.get("font", ""),
        bold=_span_is_bold(dominant),
        italic=_span_is_italic(dominant),
        x0=round(x0, 1),
        y0=round(y0, 1),
        x1=round(x1, 1),
        y1=round(y1, 1),
        max_font_size=round(max(float(sp.get("size", 0.0)) for sp in spans), 1),
    )


# A line that is ONLY a section number: "1", "3.2", "A", "A.1"
SECTION_NUMBER_ONLY = re.compile(r"^([0-9]{1,2}(\.[0-9]{1,2})*|[A-Z](\.[0-9]{1,2})*)\.?$")


def _merge_split_section_numbers(lines: list) -> list:
    """Re-attach section numbers that the PDF stored as separate lines.

    LaTeX templates such as ICLR and NeurIPS typeset "1  INTRODUCTION" with a
    wide gap, and PyMuPDF returns the "1" as a line of its own. On its own it
    looks exactly like a page number and gets thrown away, and the heading
    loses its number. Here we glue it back on: a number-only line, followed on
    the SAME baseline (same height on the page) by text just to its right.
    """
    used = set()
    for i, num in enumerate(lines):
        if i in used or not SECTION_NUMBER_ONLY.match(num.text):
            continue
        for j, other in enumerate(lines):
            if j == i or j in used or other.page != num.page:
                continue
            same_baseline = abs(other.y1 - num.y1) < 3.0
            just_right = 0 <= other.x0 - num.x1 < 40
            if same_baseline and just_right and not SECTION_NUMBER_ONLY.match(other.text):
                other.text = f"{num.text.rstrip('.')} {other.text}"
                other.x0 = num.x0
                other.max_font_size = max(other.max_font_size, num.max_font_size)
                used.add(i)
                break
    return [line for k, line in enumerate(lines) if k not in used]


FULL_WIDTH, LEFT, RIGHT = "full", "left", "right"


def _column_of(bbox: tuple, page_width: float) -> str:
    """Decide whether a text block sits in the left column, the right column,
    or stretches across both (a title, a wide figure caption, etc.)."""
    x0, _, x1, _ = bbox
    middle = page_width / 2
    gutter = page_width * 0.02          # a small tolerance around the middle
    if x1 <= middle + gutter:
        return LEFT
    if x0 >= middle - gutter:
        return RIGHT
    return FULL_WIDTH


def _reading_order(blocks: list, page_width: float) -> list:
    """Put a page's text blocks into the order a human would read them.

    Sorting purely top-to-bottom breaks two-column papers: the first paragraph
    of the left column and the first paragraph of the right column sit at the
    same height, so they would be read one after the other and the text would
    come out interleaved.

    Instead we cut the page into horizontal "bands" wherever a full-width block
    appears (the title, a wide figure). Inside each band we read the whole left
    column top-to-bottom, then the whole right column. A single-column page has
    no left/right blocks at all, so it simply comes out top-to-bottom.
    """
    tagged = [(_column_of(b["bbox"], page_width), b) for b in blocks]
    tagged.sort(key=lambda item: item[1]["bbox"][1])          # by top edge

    ordered: list = []
    band_left: list = []
    band_right: list = []

    def flush_band() -> None:
        ordered.extend(sorted(band_left, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(band_right, key=lambda b: b["bbox"][1]))
        band_left.clear()
        band_right.clear()

    for column, block in tagged:
        if column == FULL_WIDTH:
            flush_band()
            block["_column"] = 0
            ordered.append(block)
        elif column == LEFT:
            block["_column"] = 0
            band_left.append(block)
        else:
            block["_column"] = 1
            band_right.append(block)

    flush_band()
    return ordered


def read_pdf(path: str) -> ParsedDocument:
    """Open a PDF and return every text line in it, with layout clues attached."""
    doc = pymupdf.open(path)
    parsed = ParsedDocument(
        source_path=path,
        source_format="pdf",
        page_count=doc.page_count,
    )

    for page_index, page in enumerate(doc, start=1):
        # "rawdict" gives every letter with its exact position (see _rebuild_text)
        page_dict = page.get_text("rawdict")

        # block["type"] == 0 means text. Type 1 is an image, which we will
        # handle on day 5 when we place figures.
        text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

        page_lines = []
        for block in _reading_order(text_blocks, page.rect.width):
            for line in block.get("lines", []):
                text_block = _line_to_block(line, page_index)
                if text_block is not None:
                    text_block.column = block["_column"]
                    page_lines.append(text_block)
        parsed.blocks.extend(_merge_split_section_numbers(page_lines))

    doc.close()
    parsed.measure_body_font()
    return parsed
