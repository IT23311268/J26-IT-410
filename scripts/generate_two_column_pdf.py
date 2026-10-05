"""
Build a synthetic two-column paper for testing layout detection.

A real arXiv PDF would do, but a generated one is better for tests: we
know exactly where every block was placed, so a test can assert the
*correct* reading order instead of eyeballing the result.

The layout mirrors a real IMRaD paper:

    +-------------------------------------------+
    |            TITLE (full width)             |
    |          Abstract (full width)            |
    +---------------------+---------------------+
    | 1. Introduction     | 3. Results          |
    | Alpha text...       | Gamma text...       |
    |                     |                     |
    | 2. Method           | 4. Conclusion       |
    | Beta text...        | Delta text...       |
    +---------------------+---------------------+

Read naively, top line by top line, the text comes out
"Alpha Gamma Alpha Gamma..." — shredded. Read correctly it is
"Alpha, Beta, Gamma, Delta". That is exactly what the test checks.

    python scripts/generate_two_column_pdf.py data/raw/two_column.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = letter  # 612 x 792 points
MARGIN = 54.0
GUTTER = 30.0
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
LEFT_X = MARGIN
RIGHT_X = MARGIN + COL_W + GUTTER

BODY_SIZE = 9.0
HEADING_SIZE = 12.0
TITLE_SIZE = 18.0


def _wrap(text: str, width_chars: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_para(c: canvas.Canvas, x: float, y: float, text: str, size: float) -> float:
    """Draw a wrapped paragraph; return the y position below it."""
    c.setFont("Helvetica", size)
    for line in _wrap(text, int(COL_W / (size * 0.5))):
        c.drawString(x, y, line)
        y -= size * 1.35
    return y


def _draw_heading(c: canvas.Canvas, x: float, y: float, text: str) -> float:
    c.setFont("Helvetica-Bold", HEADING_SIZE)
    c.drawString(x, y, text)
    return y - HEADING_SIZE * 1.6


def build_two_column_pdf(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)

    # --- full-width masthead -------------------------------------------
    c.setFont("Helvetica-Bold", TITLE_SIZE)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 72, "Layout-Aware Ingestion of Scholarly PDFs")

    y = PAGE_H - 110
    c.setFont("Helvetica-Bold", HEADING_SIZE)
    c.drawString(MARGIN, y, "Abstract")
    y -= 18
    c.setFont("Helvetica", BODY_SIZE)
    abstract = (
        "This synthetic paper exists to test column detection. The abstract runs "
        "the full width of the page, above the two columns, so it must be read "
        "before either of them."
    )
    for line in _wrap(abstract, 110):
        c.drawString(MARGIN, y, line)
        y -= BODY_SIZE * 1.35

    # --- two columns ----------------------------------------------------
    col_top = y - 24

    ly = _draw_heading(c, LEFT_X, col_top, "1. Introduction")
    ly = _draw_para(
        c, LEFT_X, ly,
        "ALPHA. Parsing a two-column paper line by line interleaves the columns "
        "and shreds every sentence. This paragraph sits in the left column and "
        "must be read before anything in the right column.",
        BODY_SIZE,
    )

    ly = _draw_heading(c, LEFT_X, ly - 14, "2. Method")
    _draw_para(
        c, LEFT_X, ly,
        "BETA. Column boundaries are found from the vertical strips of the page "
        "that no text block touches. The gutter between these two columns is "
        "thirty points wide.",
        BODY_SIZE,
    )

    ry = _draw_heading(c, RIGHT_X, col_top, "3. Results")
    ry = _draw_para(
        c, RIGHT_X, ry,
        "GAMMA. This paragraph is in the right column. In correct reading order "
        "it comes after both left-column paragraphs, even though it starts higher "
        "on the page than the Method section.",
        BODY_SIZE,
    )

    ry = _draw_heading(c, RIGHT_X, ry - 14, "4. Conclusion")
    ry = _draw_para(
        c, RIGHT_X, ry,
        "DELTA. Geometry alone is enough to recover reading order on a standard "
        "scholarly layout, with no model and no GPU.",
        BODY_SIZE,
    )

    # a figure with its caption, bottom of the right column
    fig_h = 90
    fig_y = ry - 30 - fig_h
    c.rect(RIGHT_X, fig_y, COL_W, fig_h)
    c.setFont("Helvetica-Oblique", BODY_SIZE)
    c.drawString(RIGHT_X, fig_y - 14, "Figure 1: A placeholder figure with its caption.")

    c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/two_column.pdf")
    print(f"wrote {build_two_column_pdf(out)}")
