"""
classifier.py - stage 2a: give every line a label.

Stage 1 handed us lines with clues (size, bold, position) but every label set
to UNKNOWN. This file replaces UNKNOWN with a real label using RULES - the same
reasoning a person uses when glancing at a printed page:

    the biggest text on page 1 is the title
    text that appears on every page is a running header -> junk
    a line that starts with "Figure 3." is a caption
    a short, bold, larger-than-normal line is a heading
    after the word "References", everything is a reference

The rules run as a series of PASSES over the whole document. Each pass looks
for one kind of thing and only touches lines that are still UNKNOWN, so a
later rule can never overwrite an earlier, more confident decision.

Why rules and not an AI model?
  * they are fast, free and work offline
  * every decision can be explained - you can point at the exact rule
  * they give the same answer every time, which makes testing possible
The LLM arrives on day 4, where it is actually needed: rewriting text.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ..models import BlockType, ParsedDocument, TextBlock

# ---------------------------------------------------------------------------
# Patterns. Each one is a regular expression: a tiny search language.
#   ^      start of the line          $     end of the line
#   \d+    one or more digits         \s+   one or more spaces
#   ( )    a group we can pull out    ?     "optional"
#   re.I   ignore upper/lower case
# ---------------------------------------------------------------------------

# "Figure 1.", "Fig. 2:", "TABLE III", "Table 4 -"
CAPTION = re.compile(r"^(fig\.?|figure|table)\s*([0-9]+|[IVXL]+)\s*[.:\-–—]?", re.I)

# "Abstract", "ABSTRACT", "Abstract—This paper ...", "Abstract: We ..."
ABSTRACT_START = re.compile(r"^abstract\b\s*[—–\-:.]?\s*", re.I)

# "Keywords:", "Key words -", "Index Terms—"
KEYWORDS_START = re.compile(r"^(key\s*words?|index\s+terms)\s*[—–\-:.]?\s*", re.I)

# The heading that opens the reference list
REFERENCES_HEADING = re.compile(
    r"^(\d+\.?\s*|[IVXL]+\.\s*)?(references|bibliography|works\s+cited|literature\s+cited)\s*$",
    re.I,
)

# An equation line usually ends with its number in brackets: "... (1)"
EQUATION_NUMBER = re.compile(r"\(\s*\d+\s*\)\s*$")
MATH_SIGNS = re.compile(r"[=<>≤≥∑∫√±×÷∝≈]")

# Page numbers: "3", "- 3 -", "Page 3", "3 of 12"
PAGE_NUMBER = re.compile(r"^(page\s*)?[-–]?\s*\d{1,4}\s*[-–]?(\s*of\s*\d+)?$", re.I)

# Section numbering styles authors use
NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")     # "3", "3.1", "3.1.2"
ROMAN = re.compile(r"^([IVXL]+)\.\s+(\S.*)$")                # "IV. RESULTS"
LETTERED = re.compile(r"^([A-H])\.\s+(\S.*)$")               # "B. Dataset"

AFFILIATION_WORDS = re.compile(
    r"(universit|department|dept\.|institute|college|faculty|school of|laborator|"
    r"centre|center|hospital|@|e-?mail|sri lanka|\bltd\b|\binc\b|corporation)",
    re.I,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and strip digits so 'Page 1' and 'Page 2' count as the same."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def split_heading_number(text: str) -> tuple:
    """'3.1 Dataset' -> ('3.1', 'Dataset', 2).  'Results' -> ('', 'Results', 0).

    The third value is the level implied by the numbering, or 0 when the
    heading has no number and its level must be guessed from font size.
    """
    match = NUMBERED.match(text)
    if match:
        number = match.group(1)
        return number, match.group(2).strip(), number.count(".") + 1
    match = ROMAN.match(text)
    if match:
        return match.group(1), match.group(2).strip(), 1
    match = LETTERED.match(text)
    if match:
        return match.group(1), match.group(2).strip(), 2
    return "", text.strip(), 0


WORD_HEADING_STYLE = re.compile(r"^heading\s*(\d)", re.I)


def word_style_level(block: TextBlock) -> int:
    """'Heading 2' -> 2. Anything else -> 0 (meaning: Word gave us no hint)."""
    match = WORD_HEADING_STYLE.match(block.style_name or "")
    return int(match.group(1)) if match else 0


def _looks_like_heading(block: TextBlock, body_size: float) -> bool:
    """Could this line be a section heading, judging by how it looks?

    A heading has to be SHORT and has to STAND OUT - bigger than normal text,
    or bold, or a real Word heading style. It also must not look like the
    middle of a sentence.
    """
    text = block.text
    if block.word_count > 14 or len(text) > 110:
        return False
    if text[0].islower():
        return False                       # sentences continue in lower case
    if CAPTION.match(text):
        return False
    if text.endswith((",", ";")):
        return False

    has_number = bool(NUMBERED.match(text) or ROMAN.match(text))
    # A plain sentence that happens to be short and ends with a full stop is
    # NOT a heading - unless it is numbered, like "1. Background".
    if text.endswith(".") and not has_number:
        return False

    bigger = block.font_size >= body_size + 0.5
    bold_normal_size = block.bold and block.font_size >= body_size - 0.2
    word_heading_style = block.style_name.lower().startswith("heading")
    shouting = block.is_upper and block.word_count <= 6 and has_number

    return bigger or bold_normal_size or word_heading_style or shouting


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

class BlockClassifier:
    """Labels every TextBlock in a ParsedDocument, one pass at a time."""

    def __init__(self, doc: ParsedDocument):
        self.doc = doc
        self.blocks: List[TextBlock] = doc.blocks
        self.body_size = doc.body_font_size or 10.0
        self.notes: List[str] = []

    # -- public ---------------------------------------------------------

    def run(self) -> ParsedDocument:
        for block in self.blocks:
            block.block_type = BlockType.UNKNOWN   # always start from a clean slate

        self._pass_junk()
        self._pass_title()
        abstract_index = self._pass_abstract_and_keywords()
        self._pass_front_matter(abstract_index)
        self._pass_references()
        self._pass_captions_and_equations()
        self._pass_headings()
        self._pass_body()
        return self.doc

    # -- passes ---------------------------------------------------------

    def _set(self, block: TextBlock, label: BlockType) -> None:
        if block.block_type is BlockType.UNKNOWN:
            block.block_type = label

    def _pass_junk(self) -> None:
        """Running headers, footers and page numbers.

        The giveaway for a running header is repetition: the same short text
        appearing on most pages. Real content never repeats like that.
        """
        if self.doc.page_count >= 2:
            pages_containing: Dict[str, set] = defaultdict(set)
            for b in self.blocks:
                if b.word_count <= 12:
                    pages_containing[_normalise(b.text)].add(b.page)

            threshold = max(2, self.doc.page_count // 2)
            repeated = {t for t, pages in pages_containing.items() if len(pages) >= threshold}
            for b in self.blocks:
                if _normalise(b.text) in repeated:
                    self._set(b, BlockType.JUNK)

        for b in self.blocks:
            if PAGE_NUMBER.match(b.text):
                self._set(b, BlockType.JUNK)

    def _pass_title(self) -> None:
        """The title is the largest text near the top of page 1.

        A long title wraps onto two or three lines, so we take every
        consecutive line at that same size, not just the first one.
        """
        candidates = [
            (i, b) for i, b in enumerate(self.blocks)
            if b.page == 1 and b.block_type is BlockType.UNKNOWN
        ][:25]                                        # only look near the top
        if not candidates:
            return

        # A Word document with a real "Title" style tells us directly.
        for i, b in candidates:
            if b.style_name.lower() == "title":
                self._set(b, BlockType.TITLE)
                return

        largest = max(b.font_size for _, b in candidates)
        if largest < self.body_size + 1.0:
            self.notes.append("No title found: nothing on page 1 is clearly larger than body text.")
            return

        started = False
        for i, b in candidates:
            if abs(b.font_size - largest) < 0.3:
                self._set(b, BlockType.TITLE)
                started = True
            elif started:
                break

    def _pass_abstract_and_keywords(self) -> Optional[int]:
        """Find the abstract and the keywords line. Returns where the abstract starts."""
        abstract_index = None

        for i, b in enumerate(self.blocks):
            if b.block_type is not BlockType.UNKNOWN or b.page > 2:
                continue
            match = ABSTRACT_START.match(b.text)
            if not match:
                continue
            rest = b.text[match.end():].strip()
            if rest:
                # "Abstract—This paper presents..." : heading and text on one line
                self._set(b, BlockType.ABSTRACT)
            else:
                self._set(b, BlockType.ABSTRACT_HEADING)
            abstract_index = i
            break

        if abstract_index is None:
            self.notes.append("No abstract found - one will be generated on day 4.")
        else:
            # Everything after the abstract heading is abstract text, until we
            # hit the keywords line or something that looks like a heading.
            for b in self.blocks[abstract_index + 1:]:
                if b.block_type is BlockType.JUNK:
                    continue
                if b.block_type is not BlockType.UNKNOWN:
                    break
                if KEYWORDS_START.match(b.text) or _looks_like_heading(b, self.body_size):
                    break
                self._set(b, BlockType.ABSTRACT)

        # Keywords may wrap onto a second line written in the same style.
        for i, b in enumerate(self.blocks):
            if b.block_type is BlockType.UNKNOWN and b.page <= 2 and KEYWORDS_START.match(b.text):
                self._set(b, BlockType.KEYWORDS)
                for nxt in self.blocks[i + 1:]:
                    same_style = nxt.italic == b.italic and abs(nxt.font_size - b.font_size) < 0.3
                    if (nxt.block_type is BlockType.UNKNOWN and same_style
                            and not _looks_like_heading(nxt, self.body_size)):
                        self._set(nxt, BlockType.KEYWORDS)
                    else:
                        break
                break

        return abstract_index

    def _pass_front_matter(self, abstract_index: Optional[int]) -> None:
        """Lines between the title and the abstract are authors and affiliations."""
        title_indexes = [i for i, b in enumerate(self.blocks) if b.block_type is BlockType.TITLE]
        if not title_indexes:
            return

        start = title_indexes[-1] + 1
        if abstract_index is not None:
            end = abstract_index
        else:
            # No abstract: front matter ends at the first heading-looking line.
            end = start
            while end < len(self.blocks) and end < start + 10:
                if _looks_like_heading(self.blocks[end], self.body_size):
                    break
                end += 1

        for b in self.blocks[start:end]:
            if b.block_type is not BlockType.UNKNOWN:
                continue
            if b.italic or AFFILIATION_WORDS.search(b.text):
                self._set(b, BlockType.AFFILIATION)
            else:
                self._set(b, BlockType.AUTHOR)

    def _pass_references(self) -> None:
        """After the 'References' heading, every line is part of a reference,
        unless another clearly larger heading starts (an appendix, say)."""
        for i, b in enumerate(self.blocks):
            if b.block_type is BlockType.UNKNOWN and REFERENCES_HEADING.match(b.text):
                self._set(b, BlockType.HEADING)
                heading_size = b.font_size
                for nxt in self.blocks[i + 1:]:
                    if nxt.block_type is not BlockType.UNKNOWN:
                        continue
                    if (_looks_like_heading(nxt, self.body_size)
                            and nxt.font_size >= heading_size - 0.2 and nxt.bold):
                        break
                    self._set(nxt, BlockType.REFERENCE)
                return
        self.notes.append("No reference list found.")

    def _pass_captions_and_equations(self) -> None:
        for b in self.blocks:
            if b.block_type is not BlockType.UNKNOWN:
                continue
            if CAPTION.match(b.text) and b.word_count >= 2:
                self._set(b, BlockType.CAPTION)
            elif (EQUATION_NUMBER.search(b.text) and MATH_SIGNS.search(b.text)
                  and b.word_count <= 25):
                self._set(b, BlockType.EQUATION)

    def _pass_headings(self) -> None:
        """Find headings, then decide which are sections and which are subsections.

        Numbered headings tell us their level directly: "3" is level 1,
        "3.1" is level 2. Un-numbered headings get their level from font size:
        the biggest heading size in the paper is level 1, the next one down is
        level 2, and so on.
        """
        found = [b for b in self.blocks
                 if b.block_type is BlockType.UNKNOWN and _looks_like_heading(b, self.body_size)]

        for b in found:
            self._set(b, BlockType.HEADING)

        all_headings = [b for b in self.blocks if b.block_type is BlockType.HEADING]
        sizes = sorted({round(b.font_size, 1) for b in all_headings}, reverse=True)

        for b in all_headings:
            _, _, numbered_level = split_heading_number(b.text)
            style_level = word_style_level(b)
            if style_level:
                # A real Word heading style is the most reliable hint of all.
                if style_level >= 2:
                    b.block_type = BlockType.SUBHEADING
            elif numbered_level >= 2:
                b.block_type = BlockType.SUBHEADING
            elif numbered_level == 0 and len(sizes) > 1 and round(b.font_size, 1) != sizes[0]:
                if not REFERENCES_HEADING.match(b.text):
                    b.block_type = BlockType.SUBHEADING

    def _pass_body(self) -> None:
        """Whatever is left is ordinary body text."""
        for b in self.blocks:
            self._set(b, BlockType.BODY)


def classify(doc: ParsedDocument) -> tuple:
    """Label every line in the document. Returns (doc, notes)."""
    classifier = BlockClassifier(doc)
    classifier.run()
    return doc, classifier.notes


def label_counts(doc: ParsedDocument) -> Dict[str, int]:
    counter = Counter(b.block_type.value for b in doc.blocks)
    return dict(sorted(counter.items(), key=lambda kv: -kv[1]))
