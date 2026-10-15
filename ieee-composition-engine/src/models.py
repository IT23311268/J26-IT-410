"""
models.py - the shared data shapes used by every stage of the pipeline.

Think of this file as the "vocabulary" of the project. Stage 1 produces these
objects, stage 2 fills in their labels, stage 5 reads them to build the Word
file. Because every stage speaks the same vocabulary, you can rewrite any one
stage without touching the others.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class BlockType(str, Enum):
    """What a piece of text IS inside a paper.

    Stage 1 sets everything to UNKNOWN. Stage 2 (the classifier) replaces
    UNKNOWN with a real label. Nothing else in the project invents new labels.
    """

    UNKNOWN = "unknown"
    TITLE = "title"
    AUTHOR = "author"
    AFFILIATION = "affiliation"
    ABSTRACT_HEADING = "abstract_heading"
    ABSTRACT = "abstract"
    KEYWORDS = "keywords"
    HEADING = "heading"
    SUBHEADING = "subheading"
    BODY = "body"
    CAPTION = "caption"
    REFERENCE = "reference"
    EQUATION = "equation"
    CODE = "code"
    JUNK = "junk"


@dataclass
class TextBlock:
    """One line of text from the source document, plus the clues around it.

    The clues are the whole point. "Introduction" written in 14pt bold is a
    heading; the same word in 10pt regular is just a word in a sentence. If we
    kept only the text we would lose the ability to tell them apart.
    """

    text: str
    page: int                 # 1-based page number
    font_size: float          # in points, rounded to 1 decimal
    font_name: str            # e.g. "TimesNewRoman-Bold"
    bold: bool
    italic: bool
    x0: float                 # left edge of the line on the page
    y0: float                 # top edge  (smaller y = higher up the page)
    x1: float                 # right edge
    y1: float                 # bottom edge
    block_type: BlockType = BlockType.UNKNOWN
    style_name: str = ""      # Word paragraph style, e.g. "Heading 1" (docx only)
    column: int = 0           # 0 = left column or full width, 1 = right column
    max_font_size: float = 0.0  # largest letter on the line (small-caps headings
                                # use a big first letter: "I" 12pt + "NTRODUCTION" 9.6pt)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_upper(self) -> bool:
        """True if the line is written in CAPITALS (a common heading style)."""
        letters = [c for c in self.text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["block_type"] = self.block_type.value
        return d


@dataclass
class ParsedDocument:
    """Everything stage 1 managed to pull out of one source file."""

    source_path: str
    source_format: str                   # "pdf" or "docx"
    page_count: int
    blocks: List[TextBlock] = field(default_factory=list)

    # Filled in by _measure_body_font(). The most common font size in the
    # document is almost always the body text size, which gives us a baseline:
    # anything noticeably larger is probably a heading or the title.
    body_font_size: float = 0.0

    @property
    def plain_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    def blocks_on_page(self, page: int) -> List[TextBlock]:
        return [b for b in self.blocks if b.page == page]

    def measure_body_font(self) -> float:
        """Find the document's normal text size by counting characters.

        We weight by characters rather than by lines so that a document with
        many short headings doesn't fool us into thinking headings are normal.
        """
        counter: Counter = Counter()
        for b in self.blocks:
            counter[round(b.font_size, 1)] += len(b.text)
        if counter:
            self.body_font_size = counter.most_common(1)[0][0]
        return self.body_font_size

    def to_json(self, path: str) -> None:
        payload = {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "page_count": self.page_count,
            "body_font_size": self.body_font_size,
            "block_count": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stage 2 output: the paper rebuilt as a real document structure.
# ---------------------------------------------------------------------------

@dataclass
class Element:
    """One piece of content inside a section, kept in its original order.

    kind is "paragraph", "caption", "equation", "math_region" or "code". Order matters: in Step 6 we
    move figures and tables next to the paragraph that talks about them, and
    to do that we need to know where everything sat in the first place.
    """

    kind: str
    text: str
    page: int = 1


@dataclass
class Section:
    """One heading and everything underneath it, up to the next heading.

    Sections are stored flat, one after another, with a level number:
    level 1 = "3. Our Approach", level 2 = "3.1 Dataset". A flat list is much
    easier to work with than a tree, and the renderer only needs the level to
    decide how to print the heading.
    """

    heading: str              # heading text WITHOUT its number, e.g. "Our Approach"
    number: str               # the author's original number, e.g. "3" or "3.1"
    level: int                # 1 = section, 2 = subsection, 3 = sub-subsection
    elements: List[Element] = field(default_factory=list)
    in_appendix: bool = False     # True for sections that come after the references

    # --- filled in by stage 3 (the IEEE mapper) ---------------------------
    ieee_slot: str = ""           # which IEEE part this is, e.g. "methodology"
    ieee_heading: str = ""        # the heading IEEE will print, e.g. "Methodology"
    ieee_number: str = ""         # IEEE numbering: "III" for sections, "B" for subsections
    mapping_confidence: float = 0.0   # 0.0 to 1.0 - how sure the mapper is
    mapping_reason: str = ""      # plain-English explanation of the decision

    @property
    def paragraphs(self) -> List[str]:
        return [e.text for e in self.elements if e.kind == "paragraph"]

    @property
    def word_count(self) -> int:
        return sum(len(e.text.split()) for e in self.elements if e.kind == "paragraph")


@dataclass
class StructuredPaper:
    """The whole paper after stage 2: every part in its own labelled slot."""

    source_path: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    # How many lines got each label - useful for checking the classifier
    # and for the "before" half of your evaluation.
    label_counts: dict = field(default_factory=dict)

    # Filled in by stage 3: IEEE sections this paper does not have.
    missing_sections: List[str] = field(default_factory=list)

    # Things the classifier noticed but could not fix, e.g. "no abstract found".
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "StructuredPaper":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sections = []
        for s in data.pop("sections", []):
            s["elements"] = [Element(**e) for e in s.get("elements", [])]
            sections.append(Section(**s))
        return cls(sections=sections, **data)
