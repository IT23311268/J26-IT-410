"""
Layout overlay — draws Box 3's output on top of the page image.

Region detection is invisible otherwise: the JSON holds a list of
numbers, and "the reading order is correct now" is not something a panel
can check from a terminal. Drawn on the page, it is obvious at a glance —
boxes around what was found, numbered in the order the text will be read.

The numbering is the part worth looking at on a two-column page. If the
boxes run 1,2,3 down the left column and only then move to the right,
reading order was reconstructed. If they alternate across the gutter, it
was not.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from schema.ingestion_schema_v1 import LayoutRegion, RegionType

# Distinct hues rather than a gradient: these are categories, not
# magnitudes. Readable on a white page at a glance from the back of a room.
REGION_COLOURS: dict[RegionType, tuple[int, int, int]] = {
    RegionType.HEADING: (24, 86, 196),  # blue
    RegionType.BODY: (31, 122, 70),  # green
    RegionType.CAPTION: (191, 103, 7),  # amber
    RegionType.FIGURE: (124, 58, 173),  # violet
    RegionType.TABLE: (173, 38, 45),  # red
}

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    """A real font if the machine has one, else PIL's bitmap fallback.

    Never raises: a missing font must not take down the endpoint, it just
    makes the labels smaller.
    """
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_overlay(
    page_image_path: Path,
    regions: list[LayoutRegion],
    scale: float,
) -> bytes:
    """Draw `regions` over the page image and return it as PNG bytes.

    Args:
        page_image_path: the rendered page PNG from Box 2.
        regions: the regions on *this* page, already in reading order.
        scale: pixels per PDF point for this page — `PageImage.scale`.
            Region boxes are in PDF points and the image is in pixels, so
            every coordinate is multiplied by this. Getting it wrong
            misplaces every box, which is precisely why the conversion
            lives in one place on the schema.
    """
    image = Image.open(page_image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    label_size = max(14, int(image.width / 55))
    font = _load_font(label_size)
    line_width = max(2, int(image.width / 500))

    for position, region in enumerate(regions, start=1):
        colour = REGION_COLOURS.get(region.region_type, (90, 90, 90))
        x0 = region.bbox.x0 * scale
        y0 = region.bbox.y0 * scale
        x1 = region.bbox.x1 * scale
        y1 = region.bbox.y1 * scale

        # a wash of colour so overlapping boxes stay distinguishable
        draw.rectangle([x0, y0, x1, y1], fill=(*colour, 26), outline=colour, width=line_width)

        # reading-order badge, pinned to the top-left corner of the box
        badge = str(position)
        text_box = draw.textbbox((0, 0), badge, font=font)
        pad = max(3, label_size // 4)
        bw = text_box[2] - text_box[0] + pad * 2
        bh = text_box[3] - text_box[1] + pad * 2
        bx = max(x0, 0)
        by = max(y0 - bh, 0)
        draw.rectangle([bx, by, bx + bw, by + bh], fill=colour)
        draw.text((bx + pad - text_box[0], by + pad - text_box[1]), badge, fill="white", font=font)

    _draw_legend(draw, image.width, regions, font, line_width)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    image_width: int,
    regions: list[LayoutRegion],
    font: ImageFont.ImageFont,
    line_width: int,
) -> None:
    """A key in the top-right, listing only the types actually present."""
    present = []
    for region in regions:
        if region.region_type not in present:
            present.append(region.region_type)
    if not present:
        return

    swatch = font.size if hasattr(font, "size") else 14
    pad = swatch // 2
    rows = [(t, f"{t.value}  ({sum(1 for r in regions if r.region_type == t)})") for t in present]
    text_w = max(draw.textbbox((0, 0), label, font=font)[2] for _, label in rows)
    box_w = swatch + pad * 3 + text_w
    box_h = pad * 2 + len(rows) * (swatch + pad) - pad

    x = image_width - box_w - swatch
    y = swatch

    draw.rectangle([x, y, x + box_w, y + box_h], fill=(255, 255, 255, 235), outline=(140, 140, 140), width=line_width)

    cy = y + pad
    for region_type, label in rows:
        colour = REGION_COLOURS.get(region_type, (90, 90, 90))
        draw.rectangle([x + pad, cy, x + pad + swatch, cy + swatch], fill=colour)
        draw.text((x + pad * 2 + swatch, cy), label, fill=(20, 20, 20), font=font)
        cy += swatch + pad
