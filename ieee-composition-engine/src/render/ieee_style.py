"""
ieee_style.py - every IEEE layout measurement, in one place.

These values come from the official IEEE conference paper template for
US Letter paper (the "conference-template-letter.docx" that IEEE publishes
on its Author Center). Keeping them all in one file means:

  * you can cite this file in your report as "the IEEE specification the
    engine enforces"
  * if your target venue uses A4 or a journal layout, you change numbers
    here and nothing else
  * stage 6 (the compliance checker) reads these SAME numbers, so the
    renderer and the checker can never disagree about what "correct" means
"""

from docx.shared import Inches, Pt

# --- Page ------------------------------------------------------------------
PAGE_WIDTH = Inches(8.5)            # US Letter
PAGE_HEIGHT = Inches(11)
MARGIN_TOP = Inches(0.75)
MARGIN_BOTTOM = Inches(1.0)
MARGIN_LEFT = Inches(0.625)
MARGIN_RIGHT = Inches(0.625)

# --- Columns ---------------------------------------------------------------
# 0.625 + 3.5 + 0.25 + 3.5 + 0.625 = 8.5 inches: the page adds up exactly.
COLUMN_COUNT = 2
COLUMN_WIDTH = Inches(3.5)
COLUMN_GAP = Inches(0.25)

# --- Font ------------------------------------------------------------------
FONT = "Times New Roman"
CODE_FONT = "Courier New"

# --- Sizes (points) --------------------------------------------------------
TITLE_SIZE = Pt(24)
AUTHOR_SIZE = Pt(11)
AFFILIATION_SIZE = Pt(10)
ABSTRACT_SIZE = Pt(9)               # abstract and index terms: 9pt bold
BODY_SIZE = Pt(10)
HEADING_SIZE = Pt(10)               # headings are NOT bigger - they use small caps / italics
CAPTION_SIZE = Pt(8)
REFERENCE_SIZE = Pt(8)
CODE_SIZE = Pt(7)                   # code and prompt listings

# --- Paragraph spacing -----------------------------------------------------
BODY_FIRST_LINE_INDENT = Inches(0.2)
HEADING1_SPACE_BEFORE = Pt(8)
HEADING1_SPACE_AFTER = Pt(4)
HEADING2_SPACE_BEFORE = Pt(6)
HEADING2_SPACE_AFTER = Pt(3)
CAPTION_SPACE_BEFORE = Pt(4)
CAPTION_SPACE_AFTER = Pt(8)
REFERENCE_HANGING_INDENT = Inches(0.25)
TITLE_SPACE_AFTER = Pt(12)
AUTHOR_BLOCK_SPACE_AFTER = Pt(18)

# --- Fixed wording IEEE uses -----------------------------------------------
ABSTRACT_LABEL = "Abstract—"          # "Abstract—"  (em dash, no space)
INDEX_TERMS_LABEL = "Index Terms—"    # "Index Terms—"
REFERENCES_HEADING = "References"
FIGURE_LABEL = "Fig."                      # IEEE abbreviates in captions
TABLE_LABEL = "TABLE"                      # IEEE capitalises and uses Roman numerals
