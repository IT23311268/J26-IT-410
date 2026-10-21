"""
ieee_style.py - the IEEE layout values, in the units Word needs.

This file holds NO rules of its own. Every value is read from the IEEE
rulebook (src/rules/ieee_rules.py) and converted into Word's units
(inches, points). The renderer and the compliance checker both import from
here, so they always use the rulebook's values.

    use_paper("a4")     switches every page value to A4 (rule LAY-01)
"""

from docx.shared import Inches, Pt

from ..rules import ieee_rules as R

# --- Page and columns (LAY-01 .. LAY-04) - filled in by use_paper() ----------
PAPER = R.PAPER_SIZES[R.DEFAULT_PAPER]
PAGE_WIDTH = PAGE_HEIGHT = MARGIN_TOP = MARGIN_BOTTOM = MARGIN_LEFT = MARGIN_RIGHT = None
COLUMN_WIDTH = COLUMN_GAP = BODY_FIRST_LINE_INDENT = None
COLUMN_COUNT = R.COLUMN_COUNT


def use_paper(size: str = R.DEFAULT_PAPER) -> None:
    """Set every page value from the rulebook's paper size ("letter" or "a4")."""
    global PAPER, PAGE_WIDTH, PAGE_HEIGHT, MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT
    global COLUMN_WIDTH, COLUMN_GAP, BODY_FIRST_LINE_INDENT
    key = (size or R.DEFAULT_PAPER).lower()
    if key not in R.PAPER_SIZES:
        raise ValueError(f"Unknown paper size '{size}'. Use: {', '.join(R.PAPER_SIZES)}")
    PAPER = R.PAPER_SIZES[key]
    PAGE_WIDTH, PAGE_HEIGHT = Inches(PAPER.width_in), Inches(PAPER.height_in)
    MARGIN_TOP, MARGIN_BOTTOM = Inches(PAPER.margin_top_in), Inches(PAPER.margin_bottom_in)
    MARGIN_LEFT = MARGIN_RIGHT = Inches(PAPER.margin_side_in)
    COLUMN_WIDTH, COLUMN_GAP = Inches(PAPER.column_width_in), Inches(PAPER.column_gap_in)
    BODY_FIRST_LINE_INDENT = Inches(PAPER.first_line_indent_in)


use_paper(R.DEFAULT_PAPER)

# --- Font (TYP-01) ------------------------------------------------------------
FONT = R.FONT
CODE_FONT = R.CODE_FONT

# --- Sizes (TYP-02 .. TYP-09) -------------------------------------------------
TITLE_SIZE = Pt(R.SIZE_PT["title"])
AUTHOR_SIZE = Pt(R.SIZE_PT["author"])
AFFILIATION_SIZE = Pt(R.SIZE_PT["affiliation"])
ABSTRACT_SIZE = Pt(R.SIZE_PT["abstract"])
BODY_SIZE = Pt(R.SIZE_PT["body"])
HEADING_SIZE = Pt(R.SIZE_PT["heading"])
CAPTION_SIZE = Pt(R.SIZE_PT["caption"])
REFERENCE_SIZE = Pt(R.SIZE_PT["reference"])
CODE_SIZE = Pt(R.SIZE_PT["code"])

# --- Paragraph spacing --------------------------------------------------------
HEADING1_SPACE_BEFORE = Pt(R.SPACING_PT["heading1_before"])
HEADING1_SPACE_AFTER = Pt(R.SPACING_PT["heading1_after"])
HEADING2_SPACE_BEFORE = Pt(R.SPACING_PT["heading2_before"])
HEADING2_SPACE_AFTER = Pt(R.SPACING_PT["heading2_after"])
CAPTION_SPACE_BEFORE = Pt(R.SPACING_PT["caption_before"])
CAPTION_SPACE_AFTER = Pt(R.SPACING_PT["caption_after"])
TITLE_SPACE_AFTER = Pt(R.SPACING_PT["title_after"])
AUTHOR_BLOCK_SPACE_AFTER = Pt(R.SPACING_PT["author_block_after"])
REFERENCE_HANGING_INDENT = Inches(R.REFERENCE_HANGING_INDENT_IN)
AUTHORS_PER_ROW = R.AUTHORS_PER_ROW                 # TTL-06

# --- Fixed wording (ABS-01, ABS-06, FIG-02, TAB-02) ---------------------------
ABSTRACT_LABEL = R.LABELS["abstract"]
INDEX_TERMS_LABEL = R.LABELS["index_terms"]
REFERENCES_HEADING = R.LABELS["references"]
FIGURE_LABEL = R.LABELS["figure"]
CAPTION_GAP = R.LABELS["caption_gap"]
TABLE_LABEL = R.LABELS["table"]
ALGORITHM_LABEL = R.LABELS["algorithm"]
