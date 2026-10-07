"""
Artifact extraction — pipeline Box 6.

Box 3 found the figures, tables and equations on each page and gave
each one a bounding box in PDF points. Box 2 rendered every page to a
PNG. This box is the join between them: it cuts each artifact out of
its page image and writes it as a file of its own, so Member 3 can put
a picture in the knowledge graph and Member 4 can place one in the IEEE
output without re-opening the PDF.

Nothing here re-reads the PDF. The crop is `region.bbox` multiplied by
`page.scale`, which is the one place points become pixels — see
`PageImage.scale`. Getting that factor backwards misplaces every crop
in the paper, which is why it lives in the schema and not here.

The crop margin
---------------
A bbox cut exactly on its own edge shaves the outermost stroke, so the
crop is padded. How much was measured rather than guessed, on a page
carrying a ruled table and a bar chart:

    ink running outside the bbox      0.7 pt
    gap to the nearest other region   1.9 pt

Those two numbers are the whole decision. Padding under 0.7 pt clips
the artwork; padding over 1.9 pt drags the caption into the picture.
Since no single constant is safe on every page, each edge is padded by
`CROP_PADDING` *or half the distance to whatever is next to it*,
whichever is smaller — generous on an artifact with white space around
it, tight on one with a caption pressed against it.

Captions
--------
`Artifact.caption` is filled by the same nearest-first pairing Box 3
uses to tell a figure from a table, so the two stages always agree on
which caption belongs to which artifact. It is re-run here on
`LayoutRegion`s rather than shared, because Box 3 works on raw PyMuPDF
blocks and the pairing is six lines; the constants are imported so the
*numbers* stay defined in exactly one place.

`linked_chunk_ids` stays empty. Binding an artifact to the chunks that
discuss it is Box 7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ingestion.layout import CAPTION_X_OVERLAP
from schema.ingestion_schema_v1 import (
    Artifact,
    ArtifactType,
    BoundingBox,
    LayoutRegion,
    PageImage,
    RegionType,
)

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

ARTIFACTS_SUBDIR = "artifacts"

#: Points of white space to keep around a crop, before the
#: nearest-neighbour clamp below cuts it back. Six is comfortable on a
#: figure standing alone; the clamp is what makes it safe on a crowded
#: page.
CROP_PADDING = 6.0

#: A region smaller than this on either side is a rule, a bullet or a
#: stray mark rather than an artifact worth a file of its own.
MIN_ARTIFACT_SIDE = 24.0

#: Which region types become artifacts, and what they are called.
_ARTIFACT_TYPES: dict[RegionType, ArtifactType] = {
    RegionType.FIGURE: ArtifactType.FIGURE,
    RegionType.TABLE: ArtifactType.TABLE,
    RegionType.EQUATION: ArtifactType.EQUATION,
}

#: Filename stem per type: fig_01.png, tbl_01.png, eq_01.png.
_FILE_STEMS: dict[ArtifactType, str] = {
    ArtifactType.FIGURE: "fig",
    ArtifactType.TABLE: "tbl",
    ArtifactType.EQUATION: "eq",
}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def _x_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    """Shared width as a fraction of the narrower box's width.

    A caption sits under its figure, so they overlap horizontally. One
    in the next column does not.
    """
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrower = min(a.x1 - a.x0, b.x1 - b.x0)
    return overlap / narrower if narrower > 0 else 0.0


def _gap(a: BoundingBox, b: BoundingBox) -> float:
    """How far apart two boxes are, in points. Zero if they touch or
    overlap. The larger of the two axes, so a block that is beside the
    figure *and* above it is measured by the longer reach."""
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0.0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0.0)
    return max(dx, dy)


def padded_box(
    target: BoundingBox,
    neighbours: Iterable[BoundingBox],
    page_width: float,
    page_height: float,
    padding: float = CROP_PADDING,
) -> tuple[float, float, float, float]:
    """`target` grown by `padding` on each side, but never more than
    halfway to whatever is next to it, and never off the page.

    Each edge is handled on its own: a figure with a caption pressed
    against its bottom still gets full padding on the other three
    sides. The half is deliberate — it leaves the same clearance on
    both sides of the gap, so the crop stops short of the neighbour's
    ink instead of ending exactly on it.
    """
    room = {"left": padding, "right": padding, "top": padding, "bottom": padding}

    for other in neighbours:
        if other is target:
            continue
        # Horizontal neighbours have to share some of the target's
        # height to be in the way at all, and vice versa.
        vertically_aligned = min(target.y1, other.y1) > max(target.y0, other.y0)
        horizontally_aligned = min(target.x1, other.x1) > max(target.x0, other.x0)

        if vertically_aligned:
            if other.x1 <= target.x0:
                room["left"] = min(room["left"], (target.x0 - other.x1) / 2)
            elif other.x0 >= target.x1:
                room["right"] = min(room["right"], (other.x0 - target.x1) / 2)
        if horizontally_aligned:
            if other.y1 <= target.y0:
                room["top"] = min(room["top"], (target.y0 - other.y1) / 2)
            elif other.y0 >= target.y1:
                room["bottom"] = min(room["bottom"], (other.y0 - target.y1) / 2)

    return (
        max(0.0, target.x0 - max(0.0, room["left"])),
        max(0.0, target.y0 - max(0.0, room["top"])),
        min(page_width, target.x1 + max(0.0, room["right"])),
        min(page_height, target.y1 + max(0.0, room["bottom"])),
    )


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------


def pair_captions(regions: list[LayoutRegion]) -> dict[str, str]:
    """Map each artifact region's id to its caption text.

    Nearest first, one caption to one artifact: the closest pair in the
    whole page is settled before the second closest is considered, so a
    caption between two figures goes to the one it is actually under
    rather than to whichever was reached first.
    """
    artifacts = [r for r in regions if r.region_type in _ARTIFACT_TYPES]
    captions = [r for r in regions if r.region_type == RegionType.CAPTION]

    pairs = sorted(
        (
            (_gap(a.bbox, c.bbox), a.region_id, c.region_id, c.text)
            for a in artifacts
            for c in captions
            if a.bbox.page == c.bbox.page
            and _x_overlap_ratio(a.bbox, c.bbox) >= CAPTION_X_OVERLAP
        ),
        key=lambda p: (p[0], p[1], p[2]),
    )

    matched: dict[str, str] = {}
    used: set[str] = set()
    for _, artifact_id, caption_id, text in pairs:
        if artifact_id in matched or caption_id in used:
            continue
        matched[artifact_id] = text.strip()
        used.add(caption_id)
    return matched


# --------------------------------------------------------------------------
# the box
# --------------------------------------------------------------------------


def artifact_dir(out_root: Path, paper_id: str) -> Path:
    return Path(out_root) / paper_id / ARTIFACTS_SUBDIR


def extract_artifacts(
    regions: list[LayoutRegion],
    pages: list[PageImage],
    paper_id: str,
    out_root: Path,
) -> list[Artifact]:
    """Crop every figure, table and equation out of its page image.

    Returns the `Artifact` objects in reading order and writes one PNG
    each under `{out_root}/{paper_id}/artifacts/`.

    A run without page images (`rasterise=false`) still returns the
    artifacts, with `image_path` left None — the boxes and captions are
    useful on their own, and a missing picture must not cost Member 3
    the record that the figure exists.
    """
    from PIL import Image  # local: keeps the schema importable without Pillow

    by_index = {p.page_index: p for p in pages}
    captions = pair_captions(regions)

    targets = [
        r
        for r in regions
        if r.region_type in _ARTIFACT_TYPES
        and (r.bbox.x1 - r.bbox.x0) >= MIN_ARTIFACT_SIDE
        and (r.bbox.y1 - r.bbox.y0) >= MIN_ARTIFACT_SIDE
    ]
    targets.sort(key=lambda r: r.order_index)

    out_dir = artifact_dir(out_root, paper_id)
    if targets and any(r.bbox.page in by_index for r in targets):
        out_dir.mkdir(parents=True, exist_ok=True)

    # One open image per page, not per artifact: a page with six
    # subfigures would otherwise decode the same PNG six times.
    opened: dict[int, "Image.Image"] = {}
    counters: dict[ArtifactType, int] = {}
    artifacts: list[Artifact] = []

    try:
        for n, region in enumerate(targets):
            kind = _ARTIFACT_TYPES[region.region_type]
            counters[kind] = counters.get(kind, 0) + 1

            image_path: Optional[str] = None
            page = by_index.get(region.bbox.page)
            if page is not None:
                source = Path(out_root) / page.image_path
                if source.exists():
                    if region.bbox.page not in opened:
                        opened[region.bbox.page] = Image.open(source).convert("RGB")
                    sheet = opened[region.bbox.page]

                    neighbours = [
                        r.bbox
                        for r in regions
                        if r.bbox.page == region.bbox.page
                        and r.region_id != region.region_id
                    ]
                    x0, y0, x1, y1 = padded_box(
                        region.bbox,
                        neighbours,
                        page_width=page.width_px / page.scale,
                        page_height=page.height_px / page.scale,
                    )

                    s = page.scale
                    crop = sheet.crop(
                        (
                            max(0, int(x0 * s)),
                            max(0, int(y0 * s)),
                            min(sheet.width, int(round(x1 * s))),
                            min(sheet.height, int(round(y1 * s))),
                        )
                    )
                    name = f"{_FILE_STEMS[kind]}_{counters[kind]:02d}.png"
                    crop.save(out_dir / name)
                    image_path = f"{paper_id}/{ARTIFACTS_SUBDIR}/{name}"

            artifacts.append(
                Artifact(
                    artifact_id=f"{paper_id}::artifact::{n}",
                    paper_id=paper_id,
                    artifact_type=kind,
                    caption=captions.get(region.region_id, ""),
                    bbox=region.bbox,
                    image_path=image_path,
                    linked_chunk_ids=[],  # Box 7
                )
            )
    finally:
        for sheet in opened.values():
            sheet.close()

    return artifacts
