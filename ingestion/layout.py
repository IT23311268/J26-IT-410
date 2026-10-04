"""
Layout region detection — pipeline Box 3.

Takes a PDF page and works out what is on it and in what order it is
meant to be read.

The problem this solves
-----------------------
A two-column paper read naively comes out interleaved: first line of the
left column, first line of the right column, back to the left. The
sentences are shredded, and every chunk built from that text is noise.
That is the failure the proposal's "real-world problem" slide describes,
and it is what this module fixes.

How it works, without a GPU
---------------------------
PyMuPDF hands us every block on the page with its bounding box and font
size. That geometry is enough:

  1. Group the page's blocks and measure the body font size.
  2. Find the *gutters* — vertical strips of the page that no block
     touches. Those strips are the column separators, so the columns are
     whatever lies between them.
  3. Full-width blocks (a title, a wide figure) cut the page into bands.
     Inside a band, read column by column, each top to bottom.
  4. Classify each block from its font size and text shape.

No model, no GPU, no training data. The ColPali vision parser in Box 8 is
the upgrade for pages this geometry cannot read (heavily designed layouts,
scans with no text layer) — not a replacement for it.

What this does NOT do yet
-------------------------
TABLE is in the RegionType enum but is never emitted here. Telling a
table apart from a column of short text lines needs ruling-line detection,
which belongs with the artifact extraction work in Box 6. Emitting a
guess now would put wrong labels into the contract that Members 2/3/4
read, which is worse than emitting none.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pymupdf  # fitz

from schema.ingestion_schema_v1 import BoundingBox, LayoutRegion, RegionType

# A vertical strip counts as a column gutter once it is this wide. Real
# two-column papers leave 4-6% of the page width; noise gaps between
# words are far narrower.
MIN_GUTTER_RATIO = 0.035

# Gutters are found in two passes, because of a chicken-and-egg problem:
# a gutter is a strip no *column* block touches, but we cannot tell which
# blocks belong to a column until we know where the columns are.
#
# Demanding an empty strip in one pass was the first attempt and it failed
# on the very first test paper — a centred title crosses the gutter, so
# the strip was never empty and the page read as one column.
#
# Pass 1 uses a loose threshold to get a rough idea of where the columns
# are; that is enough to spot the blocks that span them (title, abstract).
# Pass 2 removes those and re-measures, where a real gutter now stands out
# as very nearly empty.
GUTTER_ROUGH_RATIO = 0.50
GUTTER_STRICT_RATIO = 0.08

# Text this much larger than the page's body size reads as a heading.
HEADING_SIZE_RATIO = 1.12

# Headings are short. A large-font block longer than this is display text
# (a pull quote, an abstract set large), not a section heading.
MAX_HEADING_CHARS = 120

# Not every heading is set larger. "2.3  Generator: BART" in the RAG
# paper is bold at exactly body size, and size alone missed it — which
# matters more than it sounds, because Box 4 builds the section tree out
# of the HEADING regions, so a missed heading loses a whole section.
#
# A block counts as bold when this much of it is set in a bold face.
# The bar is high on purpose: a paragraph opening with a bold run-in
# ("**RAG-Sequence**  For RAG-Sequence, ...") is a few percent bold and
# must stay BODY.
BOLD_RATIO_FOR_HEADING = 0.6

BOLD_FONT_MARKERS = ("bold", "-bd", "medi", "black", "heavy", "semib", "cmbx")

# "Figure 3:", "Fig. 2.", "Table 1 —" and friends, at the very start of
# the block. Captions are bound to their figure in Box 6, so mislabelling
# one as BODY would put caption text into the prose chunks.
CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table|tbl\.?)\s*\d+", re.IGNORECASE)

# Vector artwork smaller than this in either direction is furniture — a
# rule under a heading, a box around a word, a table's ruling lines — not
# a figure.
MIN_FIGURE_SIDE = 36.0

# Vector paths this close together belong to the same drawing. A chart is
# drawn as dozens of separate paths (axes, ticks, each bar), and each one
# on its own is noise; merged, they are one figure.
FIGURE_MERGE_GAP = 12.0

# Typesetters switch to a separate font for mathematics, and the font
# name is the giveaway. Measured on arXiv:2005.11401 (RAG), whose body
# face is NimbusRomNo9L while its equations use CMMI (math italic), CMSY
# (symbols) and CMEX (the extensible glyphs — big sigmas and brackets).
#
# Only faces that are *exclusively* mathematical are listed. CMR and CMBX
# appear inside equations too, but they are also the body face of any
# paper typeset wholly in Computer Modern, so matching them would label
# every paragraph of an older paper as an equation.
MATH_FONT_MARKERS = (
    "cmmi",     # Computer Modern math italic
    "cmsy",     # Computer Modern math symbols
    "cmex",     # Computer Modern extensible (large operators, brackets)
    "msam",     # AMS symbols A
    "msbm",     # AMS symbols B (blackboard bold)
    "rsfs",     # Ralph Smith's formal script
    "lmmath",   # Latin Modern Math
    "stix",     # STIX Two Math
    "xits",     # XITS Math
    "mathjax",  # MathJax web fonts, in HTML-to-PDF conversions
    "euclid",   # Euclid Math, used by Word's equation editor
    "cambria math",
)

# Faces that appear *inside* equations but are not proof of one: the
# upright digits, parentheses and operator names of a formula
# ("BERT", "max", "exp") are set in these, and so is the body text of
# any paper typeset wholly in Computer Modern.
#
# They count as maths only in a block that already contains unambiguous
# maths AND is short enough to be a display equation — see
# block_math_ratio. Without that rule, "d(z) = BERT(z)" came back as
# BODY because only its variables were strong maths; with it applied
# unconditionally, every paragraph of an older paper became an equation.
WEAK_MATH_FONT_MARKERS = ("cmr", "cmbx", "cmti", "cmss")

# A block longer than this is prose, whatever faces it mixes in. Display
# equations are short; paragraphs are not.
MAX_EQUATION_CHARS = 200

# A block is an equation when at least this share of its characters are
# set in a maths face.
#
# The threshold matters because *inline* maths is everywhere: a body
# paragraph in the RAG paper came back with 170 ordinary characters and 7
# in CMMI — a ratio of 0.04 — and must stay BODY. A display equation is
# almost entirely maths. Nothing real sits near the middle, so half is a
# safe place to draw the line.
MATH_RATIO_FOR_EQUATION = 0.5

# A display equation arrives shattered: the RAG paper's first equation
# came back as ten separate blocks — the sigma, each subscript, each
# variable. Fragments within this distance are re-joined into one region.
EQUATION_MERGE_GAP = 8.0

# A candidate figure with more than this fraction of its area covered by
# text is not a figure — it is decoration sitting behind prose.
#
# Found on a real arXiv paper: inline code spans are set with a pale grey
# highlight, each one a thin filled rectangle. Individually they are too
# small to count, but consecutive lines merge into a block large enough
# to pass, and a paragraph came back labelled FIGURE. Measured, that
# false positive covers 0.71 of its area with text while a real figure
# covers 0.00, so the two separate cleanly. A chart's axis labels take up
# only a small part of its area, well under this bar.
MAX_TEXT_COVERAGE = 0.5


def _block_text(block: dict) -> str:
    """Flatten a PyMuPDF text block into a single string."""
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
        parts.append("\n")
    return "".join(parts).strip()


def is_math_font(font_name: str) -> bool:
    """Is this font face used only for mathematics?"""
    lowered = (font_name or "").lower()
    return any(marker in lowered for marker in MATH_FONT_MARKERS)


def is_bold_font(font_name: str) -> bool:
    lowered = (font_name or "").lower()
    return any(marker in lowered for marker in BOLD_FONT_MARKERS)


def block_bold_ratio(block: dict) -> float:
    """The share of this block's characters set in a bold face."""
    bold_chars = 0
    total_chars = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if not text:
                continue
            total_chars += len(text)
            if is_bold_font(span.get("font", "")):
                bold_chars += len(text)
    if total_chars == 0:
        return 0.0
    return bold_chars / total_chars


def is_weak_math_font(font_name: str) -> bool:
    """A face that appears inside equations but also sets ordinary text."""
    lowered = (font_name or "").lower()
    return any(marker in lowered for marker in WEAK_MATH_FONT_MARKERS)


def block_math_ratio(block: dict) -> float:
    """The share of this block's characters that belong to mathematics.

    Near 1.0 for a display equation, near 0.0 for prose, and a few
    percent for a paragraph containing inline maths.

    A short block that already contains unambiguous maths is a display
    equation, so its upright digits and operator names count toward the
    formula as well. In a long block the same faces are prose and do not.
    """
    strong_chars = 0
    weak_chars = 0
    total_chars = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if not text:
                continue
            total_chars += len(text)
            font = span.get("font", "")
            if is_math_font(font):
                strong_chars += len(text)
            elif is_weak_math_font(font):
                weak_chars += len(text)

    if total_chars == 0:
        return 0.0
    if strong_chars > 0 and total_chars <= MAX_EQUATION_CHARS:
        return (strong_chars + weak_chars) / total_chars
    return strong_chars / total_chars


def _block_font_size(block: dict) -> float | None:
    """The block's dominant font size.

    Weighted by character count, not a plain average: a heading that ends
    with a small-font footnote marker should still read as a heading.
    """
    weighted: list[tuple[float, int]] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "")
            if text.strip():
                weighted.append((float(span.get("size", 0.0)), len(text)))
    if not weighted:
        return None
    # the size covering the most characters
    totals: dict[float, int] = {}
    for size, count in weighted:
        totals[round(size, 1)] = totals.get(round(size, 1), 0) + count
    return max(totals.items(), key=lambda kv: kv[1])[0]


def x_coverage(blocks: list[dict], page_width: float) -> list[float]:
    """How much text height sits at each horizontal position on the page.

    One bin per point of page width. A block adds its own height to every
    bin it spans. The result is a profile with tall plateaus under the
    columns and a deep notch at the gutter.
    """
    width = max(int(page_width), 1)
    coverage = [0.0] * width
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        height = max(y1 - y0, 0.0)
        start = max(int(x0), 0)
        end = min(int(x1) + 1, width)
        for i in range(start, end):
            coverage[i] += height
    return coverage


def _low_coverage_runs(
    blocks: list[dict], page_width: float, ratio: float
) -> list[tuple[float, float]]:
    """Runs of the coverage profile that sit below `ratio` of its peak.

    Runs touching either page edge are margins, not gutters, so they are
    dropped; only an interior run separates two columns.
    """
    coverage = x_coverage(blocks, page_width)
    if not coverage:
        return []

    peak = max(coverage)
    if peak <= 0:
        return []

    threshold = peak * ratio
    min_gutter = page_width * MIN_GUTTER_RATIO

    runs: list[tuple[float, float]] = []
    run_start: int | None = None

    for i, value in enumerate(coverage):
        if value <= threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                # a run starting at index 0 is the left margin
                if run_start > 0 and (i - run_start) >= min_gutter:
                    runs.append((float(run_start), float(i)))
                run_start = None
    # a run still open at the end reaches the right edge: the right margin

    return runs


def _crosses_columns(block: dict, columns: list[tuple[float, float]]) -> bool:
    x0, _, x1, _ = block["bbox"]
    return _column_of(x0 + 1.0, columns) != _column_of(max(x1 - 1.0, x0), columns)


def find_gutters(blocks: list[dict], page_width: float) -> list[tuple[float, float]]:
    """Find this page's column gutters. See the two-pass note at the top.

    A single-column page falls out correctly without a special case: the
    loose first pass may pick up a dip in the text, but then nearly every
    block "spans" those phantom columns, pass 2 is left with almost no
    coverage, the whole width reads as one run touching both edges, and
    that run is discarded as margin. No gutters, one column.
    """
    rough = _low_coverage_runs(blocks, page_width, GUTTER_ROUGH_RATIO)
    if not rough:
        return []

    candidate_columns = columns_from_gutters(rough, page_width)
    within_columns = [b for b in blocks if not _crosses_columns(b, candidate_columns)]

    return _low_coverage_runs(within_columns, page_width, GUTTER_STRICT_RATIO)


def columns_from_gutters(
    gutters: list[tuple[float, float]], page_width: float
) -> list[tuple[float, float]]:
    """Turn gutters into the column x-ranges that lie between them."""
    edges = [0.0]
    for start, end in gutters:
        edges.append((start + end) / 2.0)
    edges.append(page_width)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _column_of(x_centre: float, columns: list[tuple[float, float]]) -> int:
    for i, (left, right) in enumerate(columns):
        if left <= x_centre < right:
            return i
    return len(columns) - 1  # past the right edge: last column


def classify_block(
    text: str,
    font_size: float | None,
    body_size: float,
    is_image: bool,
    math_ratio: float = 0.0,
    bold_ratio: float = 0.0,
) -> RegionType:
    """Label one block.

    Order matters. The caption test runs before the heading test, because
    a caption set in bold at body size would otherwise fall through to
    BODY and one set large would read as a heading. The equation test
    runs after the caption test, so that "Figure 2: where x = y" stays a
    caption.
    """
    if is_image:
        return RegionType.FIGURE
    if CAPTION_RE.match(text):
        return RegionType.CAPTION
    if math_ratio >= MATH_RATIO_FOR_EQUATION:
        return RegionType.EQUATION
    # A heading is short, and then either larger than the body text or
    # set bold. Either route alone is not enough: a long bold passage is
    # emphasis, and a short large one could be a pull quote.
    if len(text) <= MAX_HEADING_CHARS and text.strip():
        is_larger = font_size is not None and font_size >= body_size * HEADING_SIZE_RATIO
        is_bold = bold_ratio >= BOLD_RATIO_FOR_HEADING
        if is_larger or is_bold:
            return RegionType.HEADING
    return RegionType.BODY


def merge_equation_fragments(blocks: list[dict]) -> list[dict]:
    """Re-join the pieces of a display equation into one block.

    A typeset equation does not arrive as one block. The RAG paper's
    first equation came back as ten: the summation sign, its subscript,
    each variable, each bracket. Ten regions for one formula is wrong on
    its own terms, and it also breaks Box 5 — a chunk would start in the
    middle of a formula.

    Equation blocks close to one another are merged; everything else is
    returned untouched. The gap is small, so two display equations on
    separate lines stay separate, and fragments in different columns
    cannot reach each other across a gutter.
    """
    equations: list[dict] = []
    others: list[dict] = []
    for block in blocks:
        if block.get("type") == 0 and block_math_ratio(block) >= MATH_RATIO_FOR_EQUATION:
            equations.append(block)
        else:
            others.append(block)

    if len(equations) < 2:
        return blocks

    # groups[i] = (bbox as a mutable list, the fragments in that group)
    groups: list[tuple[list[float], list[dict]]] = []
    for block in equations:
        x0, y0, x1, y1 = block["bbox"]
        placed = False
        for bbox, members in groups:
            if (
                x0 <= bbox[2] + EQUATION_MERGE_GAP
                and bbox[0] <= x1 + EQUATION_MERGE_GAP
                and y0 <= bbox[3] + EQUATION_MERGE_GAP
                and bbox[1] <= y1 + EQUATION_MERGE_GAP
            ):
                bbox[0], bbox[1] = min(bbox[0], x0), min(bbox[1], y0)
                bbox[2], bbox[3] = max(bbox[2], x1), max(bbox[3], y1)
                members.append(block)
                placed = True
                break
        if not placed:
            groups.append(([x0, y0, x1, y1], [block]))

    # One pass can leave two groups that have since grown into contact,
    # so keep going until a pass changes nothing.
    changed = True
    while changed:
        changed = False
        merged_groups: list[tuple[list[float], list[dict]]] = []
        for bbox, members in groups:
            for other_bbox, other_members in merged_groups:
                if (
                    bbox[0] <= other_bbox[2] + EQUATION_MERGE_GAP
                    and other_bbox[0] <= bbox[2] + EQUATION_MERGE_GAP
                    and bbox[1] <= other_bbox[3] + EQUATION_MERGE_GAP
                    and other_bbox[1] <= bbox[3] + EQUATION_MERGE_GAP
                ):
                    other_bbox[0] = min(other_bbox[0], bbox[0])
                    other_bbox[1] = min(other_bbox[1], bbox[1])
                    other_bbox[2] = max(other_bbox[2], bbox[2])
                    other_bbox[3] = max(other_bbox[3], bbox[3])
                    other_members.extend(members)
                    changed = True
                    break
            else:
                merged_groups.append((list(bbox), list(members)))
        groups = merged_groups

    for bbox, members in groups:
        # fragments read left to right along each line of the formula
        members.sort(key=lambda b: (round(b["bbox"][1] / 4), b["bbox"][0]))
        lines: list[dict] = []
        for member in members:
            lines.extend(member.get("lines", []))
        others.append({"type": 0, "bbox": tuple(bbox), "lines": lines})

    return others


def text_coverage_ratio(rect: list[float], text_blocks: list[dict]) -> float:
    """How much of `rect` is covered by text blocks, as a fraction.

    Text blocks on a page do not overlap each other in practice, so
    summing the intersections is a fair estimate of the covered area.
    """
    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    if area <= 0:
        return 1.0

    covered = 0.0
    for block in text_blocks:
        bx0, by0, bx1, by1 = block["bbox"]
        ix0, iy0 = max(rect[0], bx0), max(rect[1], by0)
        ix1, iy1 = min(rect[2], bx1), min(rect[3], by1)
        if ix1 > ix0 and iy1 > iy0:
            covered += (ix1 - ix0) * (iy1 - iy0)
    return min(covered / area, 1.0)


def vector_figure_blocks(page, text_blocks: list[dict] | None = None) -> list[dict]:
    """Find figures drawn as vector artwork, as pseudo-blocks.

    PyMuPDF's text dictionary reports *embedded raster images* as blocks,
    but most charts in a research paper are not images — matplotlib,
    TikZ and Illustrator all emit vector paths, which never appear there.
    Detecting only raster images would miss the majority of real figures.

    So the drawing list is read separately, paths close together are
    merged into one drawing, and anything large enough comes back shaped
    like an image block so the rest of the pipeline treats it as one.

    `text_blocks` are the page's text blocks. A merged candidate sitting
    mostly under text is discarded — see MAX_TEXT_COVERAGE.

    Known limitation: a ruled table merges into a single box and is
    labelled FIGURE, because telling the two apart needs the ruling-line
    analysis that belongs with Box 6.
    """
    text_blocks = text_blocks or []
    rects: list[list[float]] = []
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if r.is_empty or r.is_infinite:
            continue
        rects.append([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])

    if not rects:
        return []

    # Merge repeatedly until nothing else touches: one pass is not enough,
    # since merging two paths can bring a third within range.
    merged = True
    while merged:
        merged = False
        output: list[list[float]] = []
        for rect in rects:
            for existing in output:
                if (
                    rect[0] <= existing[2] + FIGURE_MERGE_GAP
                    and existing[0] <= rect[2] + FIGURE_MERGE_GAP
                    and rect[1] <= existing[3] + FIGURE_MERGE_GAP
                    and existing[1] <= rect[3] + FIGURE_MERGE_GAP
                ):
                    existing[0] = min(existing[0], rect[0])
                    existing[1] = min(existing[1], rect[1])
                    existing[2] = max(existing[2], rect[2])
                    existing[3] = max(existing[3], rect[3])
                    merged = True
                    break
            else:
                output.append(list(rect))
        rects = output

    return [
        {"type": 1, "bbox": tuple(r), "lines": []}
        for r in rects
        if (r[2] - r[0]) >= MIN_FIGURE_SIDE
        and (r[3] - r[1]) >= MIN_FIGURE_SIDE
        and text_coverage_ratio(r, text_blocks) <= MAX_TEXT_COVERAGE
    ]


def _page_body_size(blocks: list[dict]) -> float:
    """The page's body font size: the median size across its text,
    weighted by how much text is set at each size.

    Median rather than mean, so a single huge title cannot drag the
    baseline up and make every real heading look like body text.
    """
    sizes: list[float] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    # one entry per character, so long paragraphs dominate
                    sizes.extend([float(span.get("size", 0.0))] * len(text.strip()))
    if not sizes:
        return 10.0  # a sane default; nothing on the page to measure
    return statistics.median(sizes)


def order_page_blocks(
    blocks: list[dict], page_width: float, columns: list[tuple[float, float]]
) -> list[tuple[dict, int]]:
    """Put one page's blocks into reading order.

    Returns (block, column_index) pairs. A block wider than
    SPANNING_WIDTH_RATIO of the page spans the columns and acts as a
    horizontal divider: everything above it is read before it, everything
    below after. Inside each band the columns are read left to right,
    each one top to bottom.
    """
    if len(columns) <= 1:
        # single column: plain top-to-bottom
        return [(b, 0) for b in sorted(blocks, key=lambda b: b["bbox"][1])]

    def is_spanning(block: dict) -> bool:
        """Does this block cross a column boundary?

        Measured against the detected columns rather than a width
        threshold: a narrow centred title spans the columns just as a
        full-width abstract does, and only the boundaries know that.
        """
        x0, _, x1, _ = block["bbox"]
        return _column_of(x0 + 1.0, columns) != _column_of(max(x1 - 1.0, x0), columns)

    by_y = sorted(blocks, key=lambda b: b["bbox"][1])

    ordered: list[tuple[dict, int]] = []
    band: list[dict] = []

    def flush_band() -> None:
        """Emit one band's blocks: column by column, top to bottom."""
        if not band:
            return
        band.sort(
            key=lambda b: (
                _column_of((b["bbox"][0] + b["bbox"][2]) / 2.0, columns),
                b["bbox"][1],
            )
        )
        for b in band:
            ordered.append((b, _column_of((b["bbox"][0] + b["bbox"][2]) / 2.0, columns)))
        band.clear()

    for block in by_y:
        if is_spanning(block):
            flush_band()
            ordered.append((block, 0))
        else:
            band.append(block)
    flush_band()

    return ordered


def extract_regions(pdf_path: Path, paper_id: str) -> list[LayoutRegion]:
    """Detect the layout regions of every page, in reading order.

    `order_index` runs continuously across the whole document, so sorting
    the returned list by it gives the text in the order a human reads it —
    which is the thing the baseline extractor gets wrong on a two-column
    paper.
    """
    doc = pymupdf.open(pdf_path)
    try:
        regions: list[LayoutRegion] = []
        order = 0

        for page_index, page in enumerate(doc):
            page_dict = page.get_text("dict")
            page_width = float(page_dict.get("width") or page.rect.width)
            text_blocks = [
                b for b in page_dict.get("blocks", []) if b.get("type") == 0 and _block_text(b)
            ]
            blocks = [
                b
                for b in page_dict.get("blocks", [])
                if b.get("type") == 1 or _block_text(b)
            ]
            blocks.extend(vector_figure_blocks(page, text_blocks))
            if not blocks:
                continue

            blocks = merge_equation_fragments(blocks)
            body_size = _page_body_size(blocks)
            gutters = find_gutters(blocks, page_width)
            columns = columns_from_gutters(gutters, page_width)

            for block, column_index in order_page_blocks(blocks, page_width, columns):
                is_image = block.get("type") == 1
                text = "" if is_image else _block_text(block)
                font_size = None if is_image else _block_font_size(block)
                math_ratio = 0.0 if is_image else block_math_ratio(block)
                bold_ratio = 0.0 if is_image else block_bold_ratio(block)
                x0, y0, x1, y1 = block["bbox"]

                regions.append(
                    LayoutRegion(
                        region_id=f"{paper_id}::region::{order}",
                        paper_id=paper_id,
                        region_type=classify_block(
                            text, font_size, body_size, is_image, math_ratio, bold_ratio
                        ),
                        bbox=BoundingBox(page=page_index, x0=x0, y0=y0, x1=x1, y1=y1),
                        text=text,
                        column_index=column_index,
                        order_index=order,
                        font_size=font_size,
                    )
                )
                order += 1

        return regions
    finally:
        doc.close()
