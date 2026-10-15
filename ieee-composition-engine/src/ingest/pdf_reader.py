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


def _line_to_block(line: dict, page_number: int) -> TextBlock | None:
    """Collapse one line's spans into a single TextBlock.

    A line can be split into several spans when the styling changes mid-line,
    for example "Figure 1. The system architecture" where only "Figure 1." is
    bold. We join the text back together and describe the line by its
    DOMINANT span - the one carrying the most characters - because that is
    what a human eye reads the line as.
    """
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return None

    text = "".join(s["text"] for s in spans)
    text = " ".join(text.split())          # squash runs of whitespace
    if not text:
        return None

    dominant = max(spans, key=lambda s: len(s.get("text", "")))
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
    )


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
        page_dict = page.get_text("dict")

        # block["type"] == 0 means text. Type 1 is an image, which we will
        # handle on day 5 when we place figures.
        text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

        for block in _reading_order(text_blocks, page.rect.width):
            for line in block.get("lines", []):
                text_block = _line_to_block(line, page_index)
                if text_block is not None:
                    text_block.column = block["_column"]
                    parsed.blocks.append(text_block)

    doc.close()
    parsed.measure_body_font()
    return parsed
