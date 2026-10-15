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
    ABSTRACT = "abstract"
    KEYWORDS = "keywords"
    HEADING = "heading"
    SUBHEADING = "subheading"
    BODY = "body"
    CAPTION = "caption"
    REFERENCE = "reference"
    EQUATION = "equation"
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
