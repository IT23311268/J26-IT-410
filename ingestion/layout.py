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

Artifacts
---------
Figures, tables and equations are each found from a different signal:

  figures    vector artwork, merged from the paths that make it up
  tables     horizontal rules in the booktabs style
  equations  the font a typesetter switches to for mathematics

Each absorbs the text that sits inside it, so a chart's axis labels and
a table's cells stay with the thing they belong to instead of scattering
into the prose chunks that Members 2 and 3 retrieve from. A figure is
first grown to its real extent, because a chart's ticks and axis titles
are printed just *outside* its plot frame.

The two detectors have to be told apart explicitly: a chart's axes are
wide hairlines, exactly like a table's booktabs rules, so a rule inside
a figure is read as that figure's axis and never as a table.

Known limits
------------
A table drawn with no ruling lines, or with vertical ones only, is
missed and comes back as body text. A page with no text layer at all —
a scan — yields a single figure covering the page. Both are the honest
place for the vision model in Box 8 to take over.
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
# "Figure 3:", "Fig. 2.", "Table 1 —", "Figure 4 Overview".
#
# What follows the number matters. A caption continues with a separator
# or a capitalised title; a sentence of prose continues in lower case —
# "Table 4 reports the comparison across both systems" is a paragraph,
# and an earlier version of this pattern labelled it a caption.
# The (?-i:[A-Z]) turns case-insensitivity off for that one test. With
# it on, "[A-Z]" also matches lower case, and "Table 4 reports..." was
# still read as a caption.
CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|tbl\.?)\s*\d+\s*(?:[:.–—-]|(?-i:[A-Z])|$)",
    re.IGNORECASE,
)

# Which kind of artifact a caption announces.
#
# This is the most authoritative signal on the page, and it is the
# author's own: geometry has to *infer* whether a box of ruled lines is
# a table or a chart, but "Table 4:" printed underneath it simply says
# so. Geometry still has to find where the artifact is — a caption
# cannot give you a bounding box — so the division of labour is:
# geometry decides where, the caption decides what it is called.
#
# A caption claims the artifact nearest to it. Measured across the
# pages this was built on, a caption sits 1.9 to 8.5 pt from its own
# artifact and 33 pt or more from any other, so the pairing is not
# close to ambiguous.
TABLE_CAPTION_RE = re.compile(r"^\s*(table|tbl\.?)\s*\d+", re.IGNORECASE)

CAPTION_REACH = 24.0

# A caption must sit under or over its artifact, not beside it in the
# next column. True pairs overlap by 0.78 of the caption's width or
# more.
CAPTION_X_OVERLAP = 0.5

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

# A scholarly table is drawn as horizontal rules with nothing between
# them — the booktabs style: one rule above the header, one below it,
# one under the last row, and no vertical lines at all.
#
# Those rules are how a table is found. They are far too thin to be
# figures, so the figure detector discards them; read on their own they
# are the clearest signal a page has. This matters because tables go to
# Member 4, and without it every cell came back as its own BODY region —
# the same corpus pollution as a chart's axis labels.
MIN_TABLE_RULE_WIDTH_RATIO = 0.25   # of page width
MAX_TABLE_RULE_THICKNESS = 3.0      # points; a rule is a hairline

# Rules belong to the same table when they line up horizontally and sit
# within this far apart vertically. A long table's body can be deep, so
# the span is generous; two tables stacked on one page further apart
# than this are read as two, which is the common case.
MAX_TABLE_RULE_SPAN = 350.0

# A table needs at least this many rules. One rule on its own is a
# section divider or a running header, not a table.
MIN_TABLE_RULES = 2

# A text block this far inside a figure belongs to the figure.
#
# A chart carries its own text: axis ticks, bar labels, a legend, the
# language names down the side. On one page of the GPT-4 report that was
# seventy separate blocks inside a single chart, and each came back as
# its own BODY region. They would have become seventy chunks — "Telugu",
# "25.0%", "Marathi" — polluting the corpus Members 2 and 3 retrieve
# from. Absorbed into the figure, they stay available on its `text`
# field without being mistaken for prose.
#
# Captions are exempt: one printed inside a figure's border is still a
# caption, and Box 6 needs it to label the artifact.
FIGURE_ABSORB_RATIO = 0.7

# A chart's labels do not all sit inside it. matplotlib draws the tick
# labels, the axis titles and the chart title *outside* the plot frame,
# so containment alone cannot reach them and "gpt-3.5-base", "0-shot"
# and "Model" each came back as a BODY region of their own — the same
# corpus pollution the containment rule was written to stop.
#
# A figure is therefore grown to its real extent first: text that all
# but touches its edge is part of the graphic. Measured on the page of
# the GPT-4 report where this was found, the furniture sits 4.5 to
# 11.6 pt from the frame while the nearest thing that is *not* part of
# the chart — its caption — is 30.3 pt away. Nothing real sits in
# between, so 12 pt is a safe place to draw the line.
FIGURE_HALO = 12.0

# Growth only takes in furniture, never prose. Two things mark it, and
# either is enough:
#
#   it is set smaller than the page's body text — a chart's ticks and
#   axis titles always are, 6 pt against a 9 pt body on the page this
#   was measured on;
#
#   or it is a few words long.
#
# Length alone was the first attempt and it was not enough: PyMuPDF
# returns a chart's whole row of x-axis labels as *one* block, and on a
# real chart that block runs past a hundred characters —
# "Anthropic-LM 0-shot Anthropic-LM RLHF gpt-3.5-base 0-shot ..." — so
# the length test threw it out and the row came back as a paragraph.
#
# Size alone is not enough either: "Model" is an axis title set at body
# size on some charts. Together they cover both, and a paragraph of
# prose — long *and* at body size — fails both.
MAX_FIGURE_LABEL_CHARS = 60
SMALLER_THAN_BODY = 0.95

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

# A figure has to contain artwork — something that is not a straight
# line. A box, a frame, a divider and a table's rules are all furniture,
# and a cluster made only of those is not a picture of anything.
#
# This is what tells a boxed table from a chart, and the two cannot be
# told apart any other way I could find. Measured on page 10 of the
# GPT-4 report, which carries both: the table contributes 6 straight
# lines and *no* other shape, the chart 4 lines and 8 filled bars. Text
# coverage, the guard that was already there, reads 0.17 against 0.07 —
# a table is mostly white space between its columns, so it cannot
# separate them.
#
# It matters because the two detectors feed each other: a chart's axes
# have to be read as part of its drawing for the tick labels outside the
# frame to be reachable, and a rule inside a figure is then excluded
# from the table detector. Without this rule a boxed table became a
# figure and so lost its own rules, and the table vanished.
def _is_straight_line(box: list[float]) -> bool:
    """A stroke with no thickness in one direction: a rule or an axis."""
    return min(box[2] - box[0], box[3] - box[1]) <= MAX_TABLE_RULE_THICKNESS



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
    region_kind: str | None = None,
) -> RegionType:
    """Label one block.

    Order matters. The caption test runs before the heading test, because
    a caption set in bold at body size would otherwise fall through to
    BODY and one set large would read as a heading. The equation test
    runs after the caption test, so that "Figure 2: where x = y" stays a
    caption.
    """
    if is_image:
        return RegionType.TABLE if region_kind == "table" else RegionType.FIGURE
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


def _rule_belongs_to_a_figure(
    rule: tuple[float, float, float], figures: list[dict]
) -> bool:
    """Is this hairline the axis of a chart rather than a table's rule?

    A chart's axes are wide, perfectly horizontal and hairline-thin —
    indistinguishable from a booktabs rule read on their own. What tells
    them apart is company: an axis sits inside the drawing its own chart
    is made of, and a table's rule does not.

    `containment_ratio` cannot answer this, because a rule has zero area.
    """
    y, x0, x1 = rule
    width = x1 - x0
    for figure in figures:
        fx0, fy0, fx1, fy1 = figure["bbox"]
        if not (fy0 <= y <= fy1):
            continue
        overlap = min(x1, fx1) - max(x0, fx0)
        if width > 0 and overlap / width >= 0.9:
            return True
    return False


def table_blocks(
    page,
    page_width: float,
    text_blocks: list[dict],
    figures: list[dict] | None = None,
) -> list[dict]:
    """Find tables from their horizontal rules, as pseudo-blocks.

    Returns blocks shaped like images so the rest of the pipeline treats
    them as artifacts, each marked `region_kind="table"`.

    A group of rules only counts as a table when there is text between
    them — otherwise a page's header and footer rules would qualify.

    `figures` are the vector figures already found on this page. Their
    axes are hairlines too, and left in the pool they chain onto a real
    table's rules and drag the region down over the chart — see
    `_rule_belongs_to_a_figure`.
    """
    rules: list[tuple[float, float, float]] = []  # (y, x0, x1)
    min_width = page_width * MIN_TABLE_RULE_WIDTH_RATIO

    for drawing in page.get_drawings():
        rect = drawing["rect"]
        width = float(rect.x1 - rect.x0)
        height = float(rect.y1 - rect.y0)
        if width >= min_width and height <= MAX_TABLE_RULE_THICKNESS:
            rules.append((float(rect.y0), float(rect.x0), float(rect.x1)))

    rules = [r for r in rules if not _rule_belongs_to_a_figure(r, figures or [])]

    if len(rules) < MIN_TABLE_RULES:
        return []

    rules.sort()
    groups: list[list[tuple[float, float, float]]] = [[rules[0]]]
    for rule in rules[1:]:
        last = groups[-1][-1]
        overlap = min(rule[2], last[2]) - max(rule[1], last[1])
        shorter = min(rule[2] - rule[1], last[2] - last[1])
        aligned = shorter > 0 and overlap / shorter > 0.5
        if aligned and (rule[0] - last[0]) <= MAX_TABLE_RULE_SPAN:
            groups[-1].append(rule)
        else:
            groups.append([rule])

    tables: list[dict] = []
    for group in groups:
        if len(group) < MIN_TABLE_RULES:
            continue
        x0 = min(r[1] for r in group)
        x1 = max(r[2] for r in group)
        y0 = min(r[0] for r in group)
        y1 = max(r[0] for r in group)
        bbox = (x0, y0, x1, y1)

        has_content = any(
            containment_ratio(b["bbox"], bbox) >= FIGURE_ABSORB_RATIO for b in text_blocks
        )
        if has_content:
            tables.append({"type": 1, "bbox": bbox, "lines": [], "region_kind": "table"})

    return tables


def containment_ratio(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """How much of `inner`'s area lies inside `outer`."""
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if area <= 0:
        return 0.0
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / area


def _gap(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """How far `inner` lies outside `outer`, in points. Zero if it
    overlaps. The larger of the two axes, so a block that is beside the
    figure *and* above it is measured by the longer reach."""
    dx = max(outer[0] - inner[2], inner[0] - outer[2], 0.0)
    dy = max(outer[1] - inner[3], inner[1] - outer[3], 0.0)
    return max(dx, dy)


def grow_figures_to_their_labels(blocks: list[dict]) -> None:
    """Stretch each figure to take in the text printed against its edge.

    A chart's tick labels and axis titles are drawn outside the plot
    frame, so the figure PyMuPDF reports stops short of the graphic a
    reader sees. Growing it first is what lets the containment rule
    below reach them.

    Grows repeatedly: the x-axis labels bring the frame down far enough
    that the axis title beneath *them* comes within reach on the next
    pass. Captions are never taken in, whatever their distance.

    Mutates the figure blocks in place.
    """
    figures = [b for b in blocks if b.get("type") == 1]
    # Measured here rather than taken from the caller: the pipeline's
    # body size is computed *after* absorption, on purpose, so that a
    # chart's small labels cannot drag it down.
    body_size = _page_body_size(blocks)

    def is_furniture(block: dict) -> bool:
        text = _block_text(block)
        if not text or CAPTION_RE.match(text):
            return False
        size = _block_font_size(block)
        if size is not None and size < body_size * SMALLER_THAN_BODY:
            return True
        return len(text) <= MAX_FIGURE_LABEL_CHARS

    candidates = [b for b in blocks if b.get("type") == 0 and is_furniture(b)]
    if not figures or not candidates:
        return

    growing = True
    while growing:
        growing = False
        for figure in figures:
            box = list(figure["bbox"])
            for block in candidates:
                bbox = block["bbox"]
                if containment_ratio(bbox, box) >= FIGURE_ABSORB_RATIO:
                    continue  # already inside; nothing to stretch to
                if _gap(bbox, box) > FIGURE_HALO:
                    continue
                box[0], box[1] = min(box[0], bbox[0]), min(box[1], bbox[1])
                box[2], box[3] = max(box[2], bbox[2]), max(box[3], bbox[3])
                growing = True
            figure["bbox"] = tuple(box)


def absorb_text_into_figures(blocks: list[dict]) -> list[dict]:
    """Fold a figure's own labels into the figure.

    Returns the blocks with the absorbed ones removed; each figure gains
    an `absorbed_text` entry holding what it swallowed, so the chart's
    labels are still available to Box 6 and Box 8 without appearing as
    prose.
    """
    figures = [b for b in blocks if b.get("type") == 1]
    if not figures:
        return blocks

    grow_figures_to_their_labels(blocks)

    kept: list[dict] = []
    for block in blocks:
        if block.get("type") == 1:
            kept.append(block)
            continue

        text = _block_text(block)
        if CAPTION_RE.match(text):
            kept.append(block)  # a caption stays a caption, wherever it sits
            continue

        host = next(
            (
                figure
                for figure in figures
                if containment_ratio(block["bbox"], figure["bbox"]) >= FIGURE_ABSORB_RATIO
            ),
            None,
        )
        if host is None:
            kept.append(block)
        elif text:
            host.setdefault("absorbed_text", []).append(text)

    return kept


def label_artifacts_from_captions(blocks: list[dict]) -> None:
    """Let each caption say what the thing beside it is.

    Everything else in this module infers an artifact's kind from how it
    is drawn, which is guesswork next to what the author wrote. "Table
    4:" printed under a box settles it, and it settles cases the
    drawing cannot: a table shipped as a screenshot has no rules to
    read, and comes back a figure until its caption is consulted.

    Captions are paired to artifacts nearest-first, each caption and
    each artifact used once. An artifact with no caption keeps whatever
    the geometry decided — this corrects a label, it does not replace
    the detector.

    Mutates the artifact blocks in place.
    """
    artifacts = [b for b in blocks if b.get("type") == 1]
    captions = [
        b for b in blocks if b.get("type") == 0 and CAPTION_RE.match(_block_text(b))
    ]
    if not artifacts or not captions:
        return

    pairs: list[tuple[float, int, int]] = []
    for ci, caption in enumerate(captions):
        cx0, cy0, cx1, cy1 = caption["bbox"]
        width = max(cx1 - cx0, 1.0)
        for ai, artifact in enumerate(artifacts):
            ax0, ay0, ax1, ay1 = artifact["bbox"]
            if (min(cx1, ax1) - max(cx0, ax0)) / width < CAPTION_X_OVERLAP:
                continue
            gap = max(ay0 - cy1, cy0 - ay1, 0.0)
            if gap <= CAPTION_REACH:
                pairs.append((gap, ci, ai))

    used_captions: set[int] = set()
    used_artifacts: set[int] = set()
    for _, ci, ai in sorted(pairs):
        if ci in used_captions or ai in used_artifacts:
            continue
        used_captions.add(ci)
        used_artifacts.add(ai)
        text = _block_text(captions[ci])
        artifacts[ai]["region_kind"] = (
            "table" if TABLE_CAPTION_RE.match(text) else "figure"
        )


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
    mostly under text is discarded — see MAX_TEXT_COVERAGE. That check
    is also what keeps a ruled table from coming back as a figure as
    well as a table: a table is dense with text, a chart is not.
    """
    text_blocks = text_blocks or []
    rects: list[list[float]] = []
    artwork: list[list[float]] = []
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if r.is_infinite:
            continue
        box = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
        # A rule or an axis has zero area, so `is_empty` is true for it,
        # but it is still part of the drawing. Skipping those left a
        # chart's bounding box at the edge of its *bars* instead of its
        # axes, which is both wrong on its own terms and the reason the
        # tick labels outside the frame could not be reached.
        rects.append(box)
        if not _is_straight_line(box):
            artwork.append(box)

    if not rects or not artwork:
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
        and any(containment_ratio(a, r) >= 0.9 for a in artwork)
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
            figures = vector_figure_blocks(page, text_blocks)
            blocks.extend(figures)
            blocks.extend(table_blocks(page, page_width, text_blocks, figures))
            if not blocks:
                continue

            blocks = merge_equation_fragments(blocks)
            blocks = absorb_text_into_figures(blocks)
            # last word on figure-vs-table, after the boxes are final
            label_artifacts_from_captions(blocks)
            body_size = _page_body_size(blocks)
            gutters = find_gutters(blocks, page_width)
            columns = columns_from_gutters(gutters, page_width)

            for block, column_index in order_page_blocks(blocks, page_width, columns):
                is_image = block.get("type") == 1
                if is_image:
                    # a figure's own labels, kept but not mistaken for prose
                    text = "\n".join(block.get("absorbed_text", []))
                else:
                    text = _block_text(block)
                font_size = None if is_image else _block_font_size(block)
                math_ratio = 0.0 if is_image else block_math_ratio(block)
                bold_ratio = 0.0 if is_image else block_bold_ratio(block)
                x0, y0, x1, y1 = block["bbox"]

                regions.append(
                    LayoutRegion(
                        region_id=f"{paper_id}::region::{order}",
                        paper_id=paper_id,
                        region_type=classify_block(
                            text,
                            font_size,
                            body_size,
                            is_image,
                            math_ratio,
                            bold_ratio,
                            block.get("region_kind"),
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
