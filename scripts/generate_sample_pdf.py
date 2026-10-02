"""
Generate a small synthetic academic-paper-shaped PDF for local testing,
so tests and manual /ingest calls don't depend on downloading real papers.

Usage:
    python scripts/generate_sample_pdf.py [output_path]

Once you have a GPU sorted and start pulling real arXiv PDFs into
data/raw/, use those for actual evaluation — this generated PDF is only
a structural stand-in (title, abstract, numbered sections, a figure box)
for exercising the pipeline end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def build_sample_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER

    # --- Page 1: title + abstract ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 100, "A Demonstration Paper for the J26-IT-410 Ingestion Pipeline")
    c.setFont("Helvetica", 11)
    c.drawString(72, height - 130, "Jane Researcher, John Example")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 170, "Abstract")
    c.setFont("Helvetica", 10)
    abstract = (
        "This synthetic document exercises the baseline PyMuPDF extraction pipeline. "
        "It contains a title, an abstract, numbered sections, and a labeled figure box, "
        "so that chunking and page-tracking logic can be exercised without a real corpus."
    )
    _wrap_text(c, abstract, 72, height - 190, max_width=460, font="Helvetica", size=10)
    c.showPage()

    # --- Page 2: sections + a "figure" ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 80, "1. Introduction")
    c.setFont("Helvetica", 10)
    intro = (
        "Document ingestion for research synthesis must preserve structure. "
        "This page simulates a Method section followed by a figure placeholder."
    )
    _wrap_text(c, intro, 72, height - 100, max_width=460, font="Helvetica", size=10)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 220, "2. Method")
    c.setFont("Helvetica", 10)
    _wrap_text(
        c,
        "Figure 1 shows the overall pipeline. Text is rendered here to give the "
        "baseline extractor real page content to chunk.",
        72,
        height - 240,
        max_width=460,
        font="Helvetica",
        size=10,
    )

    # a simple box standing in for a figure
    c.rect(100, height - 470, 380, 190, stroke=1, fill=0)
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(290, height - 480, "Figure 1: Pipeline overview (placeholder box).")
    c.showPage()

    # --- Page 3: results/conclusion ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 80, "3. Results")
    c.setFont("Helvetica", 10)
    _wrap_text(
        c,
        "Results would be reported here. This synthetic paper exists only to give the "
        "ingestion engine multiple pages and sections to split into chunks.",
        72,
        height - 100,
        max_width=460,
        font="Helvetica",
        size=10,
    )
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 200, "4. Conclusion")
    c.setFont("Helvetica", 10)
    _wrap_text(c, "This concludes the demonstration document.", 72, height - 220, max_width=460, font="Helvetica", size=10)
    c.showPage()

    c.save()


def _wrap_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, font: str, size: int) -> None:
    """Extremely simple word-wrap for reportlab (no external deps)."""
    c.setFont(font, size)
    words = text.split()
    line = ""
    line_height = size + 3
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= max_width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = word
    if line:
        c.drawString(x, y, line)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/sample_paper.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    build_sample_pdf(out)
    print(f"wrote {out}")
