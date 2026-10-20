"""
regions.py - Step 6a: find the area of the page each figure, table and
algorithm occupies.

A PDF has no "figure" object. A chart is just hundreds of drawn lines and
little text labels; a table is text cells plus a few ruling lines. What a
PDF DOES have is the caption ("Figure 3: ..."), and figures follow strict
typesetting conventions around their captions. We use those conventions:

  1. WHICH SIDE   figures sit ABOVE their caption, tables and algorithms
                  BELOW it. Look there first; if nothing is found, look on
                  the other side (some papers do it the other way round).
  2. HOW WIDE     as wide as the caption. Side-by-side figures each have
                  their own caption, so each caption claims only its own
                  column of graphics - never a neighbour's.
  3. HOW FAR      up to the nearest line of ordinary PROSE (a sentence of
                  body text). Prose marks the edge of the figure.
  4. WHAT'S IN IT every drawing, image and small text label between the
                  caption and that edge.

Text found inside a figure (axis labels, a screenshot of a terminal) is
marked as junk, because it belongs to the picture, not the paper's text.
The picture itself is cropped out as an image in export.py.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..models import BlockType, ParsedDocument, TextBlock
from ..structure.classifier import CAPTION

Box = Tuple[float, float, float, float]

PROSE_TYPES = {BlockType.TITLE, BlockType.HEADING, BlockType.SUBHEADING,
               BlockType.ABSTRACT, BlockType.ABSTRACT_HEADING, BlockType.REFERENCE,
               BlockType.AUTHOR, BlockType.AFFILIATION, BlockType.KEYWORDS}
MAX_GAP_TO_CAPTION = 90      # a figure further than this from its caption is not its figure
CONTIGUITY = 14              # pieces of one visual sit closer together than this (points)
MIN_HEIGHT = 24              # nothing shorter than this is a real figure or table
PADDING = 3


def _union(boxes: List[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _center(box: Box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box(block: TextBlock) -> Box:
    return (block.x0, block.y0, block.x1, block.y1)


class _Page:
    """Everything on one page that a figure could be made of."""

    def __init__(self, pdf_page, blocks: List[TextBlock]):
        area = pdf_page.rect.width * pdf_page.rect.height
        self.width = pdf_page.rect.width
        self.height = pdf_page.rect.height
        self.graphics: List[Box] = []
        for drawing in pdf_page.get_drawings():
            r = drawing.get("rect")
            if r is None or r.is_empty and r.width == 0 and r.height == 0:
                continue
            if r.width * r.height > 0.8 * area:          # a full-page background, not a figure
                continue
            self.graphics.append((r.x0, r.y0, r.x1, r.y1))
        for image in pdf_page.get_image_info():
            x0, y0, x1, y1 = image["bbox"]
            self.graphics.append((x0, y0, x1, y1))
        self.blocks = blocks


def _is_prose(block: TextBlock, body_size: float, span_width: float, kind: str = "") -> bool:
    """A line of ordinary running text - the edge of any figure.

    Inside an algorithm box, steps such as "Require: shared backbone, group
    size G, horizon K" are long-ish too, so there the bar is higher, and a
    numbered step ("3: ...") is never prose.
    """
    if block.block_type in PROSE_TYPES:
        return True
    if block.block_type is BlockType.BODY:
        wide = (block.x1 - block.x0) >= 0.55 * span_width
        min_words = 14 if kind == "algorithm" else 8
        if kind == "algorithm" and re.match(r"^\d+:\s", block.text):
            return False
        return block.word_count >= min_words and block.font_size >= body_size - 0.6 and wide
    return False


def _find_region(page: _Page, caption: Box, caption_lines: List[TextBlock],
                 kind: str, direction: str, x_range: Tuple[float, float],
                 blocked: List[Tuple[float, float]], body_size: float,
                 side_by_side: bool = False) -> Optional[Box]:
    """Look in one direction ('up' or 'down') from the caption for its visual."""
    x0, x1 = x_range
    span = x1 - x0
    # running headers and page numbers live in the top and bottom margins
    top_margin, bottom_margin = page.height * 0.06, page.height * 0.94

    def in_columns(box: Box) -> bool:
        cx, cy = _center(box)
        if not (x0 <= cx <= x1) or not (top_margin <= cy <= bottom_margin):
            return False
        return not any(b0 <= cx <= b1 for b0, b1 in blocked)

    def overlaps(box: Box) -> bool:
        # a line only counts as an edge if it really runs across these columns;
        # text wrapping round a narrow table only touches them by a few points
        shared = min(box[2], x1) - max(box[0], x0)
        return shared > min(40, 0.3 * (box[2] - box[0]))

    # 1. the edge: the nearest prose line - or ANY other caption - that
    #    overlaps these columns. A figure never runs past another caption.
    edge = 0.0 if direction == "up" else page.height
    for b in page.blocks:
        if b in caption_lines or b.block_type is BlockType.JUNK or not overlaps(_box(b)):
            continue
        is_edge = _is_prose(b, body_size, span, kind) or b.block_type is BlockType.CAPTION
        if not is_edge:
            continue
        if direction == "up" and b.y1 <= caption[1] + 1:
            edge = max(edge, b.y1)
        elif direction == "down" and b.y0 >= caption[3] - 1:
            edge = min(edge, b.y0)

    top, bottom = (edge, caption[1]) if direction == "up" else (caption[3], edge)

    # 2. everything between the caption and that edge
    members: List[Box] = []
    graphic_count = 0
    for g in page.graphics:
        cy = _center(g)[1]
        if top - 1 <= cy <= bottom + 1 and in_columns(g):
            members.append(g)
            graphic_count += 1
    text_members = [b for b in page.blocks
                    if b not in caption_lines and b.block_type not in PROSE_TYPES
                    and not (b.block_type is BlockType.CAPTION and CAPTION.match(b.text))
                    and top - 1 <= _center(_box(b))[1] <= bottom + 1
                    and in_columns(_box(b)) and not _is_prose(b, body_size, span, kind)]
    members.extend(_box(b) for b in text_members)

    # side-by-side visuals: never let a wide background shape pull a
    # neighbour's area in - trim every piece to this visual's columns
    if side_by_side:
        members = [(max(m[0], x0), m[1], min(m[2], x1), m[3]) for m in members
                   if min(m[2], x1) > max(m[0], x0)]

    # a figure needs drawings; a table can be text cells with only a rule or two
    if graphic_count == 0 and (kind == "figure" or len(text_members) < 2):
        return None
    if not members:
        return None

    # Grow outwards from the caption, only through pieces that sit close
    # together. A table above a figure on the same page is separated from
    # it by a band of white space - growth stops there.
    members.sort(key=lambda m: -m[3] if direction == "up" else m[1])
    near = caption[1] if direction == "up" else caption[3]
    first = members[0]
    first_gap = near - first[3] if direction == "up" else first[1] - near
    if first_gap > MAX_GAP_TO_CAPTION:
        return None
    region = first
    changed = True
    while changed:
        changed = False
        for m in members:
            if m[0] >= region[0] and m[1] >= region[1] and m[2] <= region[2] and m[3] <= region[3]:
                continue
            gap = region[1] - m[3] if direction == "up" else m[1] - region[3]
            vertical_touch = m[3] >= region[1] - CONTIGUITY and m[1] <= region[3] + CONTIGUITY
            if vertical_touch or gap <= CONTIGUITY:
                region = _union([region, m])
                changed = True
    if direction == "up":
        region = (region[0], region[1], region[2], min(region[3], caption[1]))
    else:
        region = (region[0], max(region[1], caption[3]), region[2], region[3])
    gap = caption[1] - region[3] if direction == "up" else region[1] - caption[3]
    if gap > MAX_GAP_TO_CAPTION:
        return None
    if (region[2] - region[0]) * (region[3] - region[1]) < 300:
        return None
    if region[3] - region[1] < MIN_HEIGHT:          # a sliver of text, not a figure or table
        return None
    return region


def _whitespace_cut(page: "_Page", upper: Box, lower: Box) -> Optional[float]:
    """Find the y-position of the widest empty horizontal strip in the band
    two regions share. Returns None if there is no clear gap."""
    x0, x1 = max(upper[0], lower[0]), min(upper[2], lower[2])
    top, bottom = max(upper[1], lower[1]), min(upper[3], lower[3])
    if bottom - top < 4 or x1 - x0 < 4:
        return None
    spans = []
    for g in page.graphics:
        if min(g[2], x1) > max(g[0], x0) and g[3] > top and g[1] < bottom:
            spans.append((max(g[1], top), min(g[3], bottom)))
    for b in page.blocks:
        if min(b.x1, x1) > max(b.x0, x0) and b.y1 > top and b.y0 < bottom:
            spans.append((max(b.y0, top), min(b.y1, bottom)))
    if not spans:
        return None
    spans.sort()
    best_gap, cut = 0.0, None
    reach = spans[0][1]
    for start, end in spans[1:]:
        if start - reach > best_gap:
            best_gap, cut = start - reach, (start + reach) / 2
        reach = max(reach, end)
    return cut if best_gap >= 3 else None


def detect_regions(doc: ParsedDocument) -> Dict[str, int]:
    """Give every caption in a PDF the page area of its figure/table/algorithm.

    Sets block.region on caption blocks and marks text inside a figure as JUNK.
    Returns counts for the report.
    """
    stats = {"captions": 0, "found": 0, "text_lines_inside": 0}
    if doc.source_format != "pdf":
        return stats
    try:
        import pymupdf
        pdf = pymupdf.open(doc.source_path)
    except Exception:                       # file moved, not a real PDF, ...
        return stats

    by_page: Dict[int, List[TextBlock]] = {}
    for b in doc.blocks:
        by_page.setdefault(b.page, []).append(b)

    for page_no, blocks in by_page.items():
        starts = [b for b in blocks if b.block_type is BlockType.CAPTION and CAPTION.match(b.text)]
        if not starts:
            continue
        page = _Page(pdf[page_no - 1], blocks)

        def caption_lines(cap: TextBlock) -> List[TextBlock]:
            """The caption's first line plus its continuation lines."""
            lines = [cap]
            for b in blocks[blocks.index(cap) + 1:]:
                if b.block_type is BlockType.CAPTION and not CAPTION.match(b.text):
                    lines.append(b)
                elif b.block_type is not BlockType.JUNK:
                    break
            return lines

        extent = {id(c): _union([_box(b) for b in caption_lines(c)]) for c in starts}

        found: List[tuple] = []        # (caption, lines, region, conventional side?)
        for cap in starts:
            stats["captions"] += 1
            lines = caption_lines(cap)
            caption = extent[id(cap)]
            word = CAPTION.match(cap.text).group(1).lower()
            kind = "figure" if word.startswith("fig") else word

            # neighbours on the same row (side-by-side figures) own their own columns
            siblings = [extent[id(s)] for s in starts
                        if s is not cap and abs(s.y0 - cap.y0) < 40]
            blocked = [(e[0] - 2, e[2] + 2) for e in siblings]
            x_range = (caption[0] - 4, caption[2] + 4)
            if siblings:
                # Side-by-side visuals: each caption owns the space nearest to
                # it - the boundary lies halfway between neighbouring caption
                # centres. Works for left-aligned, centred and justified
                # captions alike, even a short caption under a wide chart.
                centre = (caption[0] + caption[2]) / 2
                centres = [(e[0] + e[2]) / 2 for e in siblings]
                left_c = [c for c in centres if c < centre]
                right_c = [c for c in centres if c > centre]
                x_range = ((centre + max(left_c)) / 2 if left_c else 0,
                           (centre + min(right_c)) / 2 if right_c else page.width)
                blocked = []
            if not siblings:
                # a short caption centred under a wide figure: allow the full column/page
                full = caption[2] - caption[0] > 0.45 * page.width
                if not full:
                    middle = page.width / 2
                    x_range = (0, middle) if caption[2] <= middle + 10 else \
                        (middle, page.width) if caption[0] >= middle - 10 else (0, page.width)
                else:
                    x_range = (0, page.width)

            order = ["up", "down"] if kind == "figure" else ["down", "up"]
            region, conventional = None, False
            for direction in order:
                region = _find_region(page, caption, lines, kind, direction, x_range,
                                      blocked, doc.body_font_size,
                                      side_by_side=bool(siblings))
                if region:
                    conventional = direction == order[0]
                    break
            if region:
                found.append((cap, lines, region, conventional, direction))

        # Two captions claiming the same area: the one whose visual sits on
        # the conventional side (figure above, table below) keeps it.
        keep = []
        for item in found:
            clash = None
            for other in keep:
                a, b = item[2], other[2]
                w = min(a[2], b[2]) - max(a[0], b[0])
                h = min(a[3], b[3]) - max(a[1], b[1])
                smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if w > 0 and h > 0 and w * h > 0.4 * smaller:
                    clash = other
                    break
            if clash is None:
                keep.append(item)
            elif item[3] and clash[3] and item[4] != clash[4]:
                # A table (caption above, content below) and a figure (content
                # above, caption below) share one band: split it at the widest
                # strip of white space - where a person would draw the line.
                upper, lower = (item, clash) if item[4] == "down" else (clash, item)
                cut = _whitespace_cut(page, upper[2], lower[2])
                if cut:
                    a, b = upper[2], lower[2]
                    keep[keep.index(clash)] = clash
                    new_upper = (a[0], a[1], a[2], min(a[3], cut))
                    new_lower = (b[0], max(b[1], cut), b[2], b[3])
                    keep = [k for k in keep if k is not clash]
                    keep.append((upper[0], upper[1], new_upper, True, upper[4]))
                    keep.append((lower[0], lower[1], new_lower, True, lower[4]))
            elif item[3] and not clash[3]:
                keep[keep.index(clash)] = item

        for cap, lines, region, _, direction in keep:
            region = (max(0, region[0] - PADDING), max(0, region[1] - PADDING),
                      min(page.width, region[2] + PADDING), min(page.height, region[3] + PADDING))
            # the padding must not reach into the caption itself
            caption = _union([_box(b) for b in lines])
            if direction == "up":
                region = (region[0], region[1], region[2], min(region[3], caption[1] - 1))
            else:
                region = (region[0], max(region[1], caption[3] + 1), region[2], region[3])
            cap.region = [round(v, 1) for v in region]
            stats["found"] += 1

            # text inside the picture is part of the picture
            for b in blocks:
                # captions and headings are never swallowed by a picture
                if b in lines or b.block_type in (BlockType.HEADING, BlockType.SUBHEADING,
                                                  BlockType.CAPTION):
                    continue
                cx, cy = _center(_box(b))
                if region[0] <= cx <= region[2] and region[1] <= cy <= region[3] \
                        and b.block_type is not BlockType.JUNK:
                    b.block_type = BlockType.JUNK
                    stats["text_lines_inside"] += 1
    pdf.close()
    return stats
