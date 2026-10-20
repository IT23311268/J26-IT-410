"""
export.py - Step 6b: cut figures, tables, algorithms and broken equations
out of the source PDF as images.

Each picture is rendered at 200 dpi from exactly the region regions.py found,
so it looks the same as in the original paper - nothing is redrawn or
re-typeset. The images are saved next to the output:

    data/output/<paper>_assets/figure_1.png
    data/output/<paper>_assets/table_2.png
    data/output/<paper>_assets/equation_3.png

and the renderer places them in the Word document.

Why images for equations? A PDF stores a display equation as dozens of
separately positioned symbols. No text extraction can rebuild "Σ" with its
limits and a fraction under it - but a picture of it is exact.
"""

from __future__ import annotations

import os
import re
from typing import Dict

from ..models import StructuredPaper

DPI = 200
EQUATION_PADDING = 4


def _slug(text: str) -> str:
    match = re.match(r"^(fig\.?|figure|table|algorithm)\s*([0-9]+|[IVXL]+)", text, re.I)
    if not match:
        return "visual"
    word = match.group(1).lower()
    word = "figure" if word.startswith("fig") else word
    return f"{word}_{match.group(2)}"


def export_visuals(paper: StructuredPaper, pdf_path: str, assets_dir: str) -> Dict[str, int]:
    stats = {"figures": 0, "tables": 0, "algorithms": 0, "equations": 0}
    if not pdf_path.lower().endswith(".pdf") or not os.path.exists(pdf_path):
        return stats
    import pymupdf

    pdf = pymupdf.open(pdf_path)
    os.makedirs(assets_dir, exist_ok=True)
    equation_no = 0
    used = set()

    for section in paper.sections:
        for element in section.elements:
            box, name = None, None
            if element.kind == "caption" and element.region:
                box = element.region
                name = _slug(element.text)
            elif element.kind == "math_region" and element.bbox:
                equation_no += 1
                b = element.bbox
                # wide sideways, tight top and bottom - so the lines above and
                # below the equation do not creep into the picture
                box = [b[0] - EQUATION_PADDING, b[1] - 1, b[2] + EQUATION_PADDING, b[3] + 1]
                name = f"equation_{equation_no}"
            if not box or not (1 <= element.page <= pdf.page_count):
                continue

            page = pdf[element.page - 1]
            clip = pymupdf.Rect(*box) & page.rect
            if clip.is_empty or clip.width < 5 or clip.height < 5:
                continue
            while name in used:                       # e.g. two captions both "Table 4"
                name += "_b"
            used.add(name)

            path = os.path.join(assets_dir, f"{name}.png")
            page.get_pixmap(clip=clip, dpi=DPI).save(path)
            element.image = path
            key = "equations" if name.startswith("equation") else \
                name.split("_")[0] + "s" if name.split("_")[0] + "s" in stats else "figures"
            stats[key] += 1
    pdf.close()
    return stats
