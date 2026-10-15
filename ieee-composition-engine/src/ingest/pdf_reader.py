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


def read_pdf(path: str) -> ParsedDocument:
    """Open a PDF and return every text line in it, with layout clues attached."""
    doc = pymupdf.open(path)
    parsed = ParsedDocument(
        source_path=path,
        source_format="pdf",
        page_count=doc.page_count,
    )

    for page_index, page in enumerate(doc, start=1):
        # sort=True asks PyMuPDF to return blocks in natural reading order
        # (top to bottom, left column before right column) instead of the
        # arbitrary order they happen to sit in inside the file.
        page_dict = page.get_text("dict", sort=True)

        for block in page_dict.get("blocks", []):
            # block["type"] == 0 means text. Type 1 is an image, which we
            # will handle on day 5 when we place figures.
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text_block = _line_to_block(line, page_index)
                if text_block is not None:
                    parsed.blocks.append(text_block)

    doc.close()
    parsed.measure_body_font()
    return parsed
