"""Tests for caption-driven artifact labelling — Box 3.

Everything else in `layout.py` infers what an artifact is from how it
was drawn: rules mean a table, artwork means a figure. That is
guesswork next to what the author wrote. "Table 4:" printed under a box
settles it.

The division of labour this pins down:

    geometry decides WHERE an artifact is — a caption cannot give you
    a bounding box;
    the caption decides WHAT IT IS CALLED — and overrules the drawing
    when they disagree.

It also reaches a case geometry cannot: a table pasted into a paper as
a screenshot has no rules to read and is a figure by every structural
test there is. Its caption still says Table, and Member 4 needs it
filed as one.
"""

from pathlib import Path

import pytest

from ingestion.layout import extract_regions
from schema.ingestion_schema_v1 import RegionType


def _png(path: Path, size=(360, 200)) -> Path:
    """A flat raster image, as a screenshotted table would arrive."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for i in range(4):  # some rows, so it reads as a picture of a table
        draw.line([(10, 30 + i * 40), (size[0] - 10, 30 + i * 40)], fill="black")
    image.save(path)
    return path


def _pdf_with_image(path: Path, caption: str) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    png = _png(path.with_suffix(".png"))
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawImage(str(png), 110, 500, width=360, height=200)
    c.setFont("Helvetica", 9)
    c.drawString(110, 488, caption)
    c.showPage()
    c.save()
    return path


def test_a_table_pasted_in_as_an_image_is_labelled_a_table(tmp_path: Path):
    """No rules, no booktabs, nothing structural to read — a raster
    image is a figure by every geometric test. The caption is the only
    thing on the page that knows better."""
    pdf = _pdf_with_image(tmp_path / "shot.pdf", "Table 2: Results on the held-out set.")
    regions = extract_regions(pdf, "shot")

    kinds = [r.region_type for r in regions]
    assert RegionType.TABLE in kinds
    assert RegionType.FIGURE not in kinds


def test_the_same_image_under_a_figure_caption_is_a_figure(tmp_path: Path):
    """The control. Only the caption differs, and it decides."""
    pdf = _pdf_with_image(tmp_path / "fig.pdf", "Figure 2: Accuracy over time.")
    regions = extract_regions(pdf, "fig")

    kinds = [r.region_type for r in regions]
    assert RegionType.FIGURE in kinds
    assert RegionType.TABLE not in kinds


def test_an_uncaptioned_artifact_keeps_what_the_geometry_said(tmp_path: Path):
    """The caption corrects a label; it does not replace the detector.
    A chart with no caption must still come back a figure."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "bare.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(5):
        c.rect(120 + i * 70, 400, 50, 40 + i * 40, fill=1)
    c.setFont("Helvetica", 10)
    c.drawString(72, 300, "Ordinary prose, nowhere near the chart above.")
    c.showPage()
    c.save()

    regions = extract_regions(path, "bare")
    assert any(r.region_type == RegionType.FIGURE for r in regions)


def test_a_distant_caption_does_not_claim_an_artifact(tmp_path: Path):
    """A caption 24 pt away or more belongs to something else. Measured
    on the pages this was built from, a caption sits within 8.5 pt of
    its own artifact and 33 pt or more from any other."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "far.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(5):
        c.rect(120 + i * 70, 560, 50, 40 + i * 30, fill=1)
    c.setFont("Helvetica", 9)
    # far below the chart — it belongs to something on the next page
    c.drawString(110, 300, "Table 9: An unrelated table overleaf.")
    c.showPage()
    c.save()

    regions = extract_regions(path, "far")
    assert any(r.region_type == RegionType.FIGURE for r in regions)
    assert not any(r.region_type == RegionType.TABLE for r in regions)


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("Table 1: Results.", RegionType.TABLE),
        ("TABLE 1. Results.", RegionType.TABLE),
        ("Tbl. 3: Results.", RegionType.TABLE),
        ("Figure 1: Overview.", RegionType.FIGURE),
        ("Fig. 4 Overview", RegionType.FIGURE),
    ],
)
def test_caption_wording(tmp_path: Path, caption: str, expected: RegionType):
    pdf = _pdf_with_image(tmp_path / "c.pdf", caption)
    kinds = [r.region_type for r in extract_regions(pdf, "c")]
    assert expected in kinds
