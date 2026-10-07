"""
Page rasterisation — pipeline Box 2.

Turns each PDF page into a PNG bitmap on disk and records where it went,
at what size, and at what resolution.

Why this exists as its own stage
--------------------------------
These stages need pixels, not PDF drawing operators:

  Box 6  artifact extraction     — figures are cropped out of the page image
  Box 8  ColPali visual retrieval — the model embeds page images directly
  the /layout overlay            — draws Box 3's regions over the page

If each of those rendered its own copy we would pay the render cost
several times and, worse, risk slightly different coordinate spaces. So
the page is rendered once, here, and `PageImage.scale` is the single
place the points-to-pixels conversion is defined.

Note what is *not* on that list. **Box 3 does not read these images.**
Layout detection works on the PDF's own text and vector data through
PyMuPDF — font names, block boxes, drawing paths — none of which
survives rasterisation. The image is how a human *sees* what Box 3
decided, not how Box 3 decides it. That is also why a scanned page
yields one figure and no text: the pixels are all there is, and this
stage cannot read them. Box 8 is where that changes.

This module deliberately knows nothing about `api/` — it is handed an
output directory and returns data. That keeps ingestion runnable from a
plain script with no web server involved.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf  # fitz

from schema.ingestion_schema_v1 import PageImage

# 200 dpi is the working compromise: high enough that 8pt caption text and
# thin table rules survive for layout detection, low enough that a 20-page
# paper stays well under ~100 MB of PNGs. Raise it per-call if a specific
# paper's figures come out too coarse.
DEFAULT_DPI = 200

PAGES_SUBDIR = "pages"


def page_image_dir(out_root: Path, paper_id: str) -> Path:
    """Where this paper's page PNGs live. One folder per paper keeps the
    processed/ directory navigable once there are 15 papers in it."""
    return Path(out_root) / paper_id / PAGES_SUBDIR


def rasterise_pdf(
    pdf_path: Path,
    paper_id: str,
    out_root: Path,
    dpi: int = DEFAULT_DPI,
) -> list[PageImage]:
    """Render every page of `pdf_path` to a PNG under `out_root`.

    Args:
        pdf_path:  the source PDF.
        paper_id:  stable ID from `make_paper_id()` — becomes the folder name.
        out_root:  the processed-data root (the API passes storage.PROCESSED_DIR).
        dpi:       render resolution. See DEFAULT_DPI above.

    Returns:
        One PageImage per page, in page order. `image_path` on each is
        relative to `out_root`, never absolute — absolute paths would break
        the moment Members 2/3/4 read the JSON on a different machine.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    out_root = Path(out_root)
    target_dir = page_image_dir(out_root, paper_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        pages: list[PageImage] = []
        for page_index, page in enumerate(doc):
            pixmap = page.get_pixmap(dpi=dpi)
            filename = f"page_{page_index:04d}.png"
            pixmap.save(target_dir / filename)

            pages.append(
                PageImage(
                    page_index=page_index,
                    image_path=f"{paper_id}/{PAGES_SUBDIR}/{filename}",
                    width_px=pixmap.width,
                    height_px=pixmap.height,
                    dpi=dpi,
                )
            )
        return pages
    finally:
        doc.close()


def resolve_image_path(out_root: Path, page: PageImage) -> Path:
    """Turn a PageImage's stored relative path back into a real file path.

    Downstream code should go through this rather than joining by hand, so
    that if the storage layout ever changes there is one place to fix.
    """
    return Path(out_root) / page.image_path
