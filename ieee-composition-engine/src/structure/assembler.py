"""
assembler.py - stage 2b: rebuild the paper from labelled lines.

After the classifier, we know what every line IS, but the paper is still in
pieces: a six-line paragraph is six separate lines, a long title is two lines,
and nothing says which paragraphs belong under which heading.

This file glues the pieces back together:

    lines   -> paragraphs   (using the gaps between lines)
    lines   -> references   (one entry per reference, not per line)
    labels  -> sections     (every paragraph filed under its heading)

The result is a StructuredPaper: title, authors, abstract, keywords, a list of
sections, and a list of references - exactly the slots an IEEE paper has.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

from ..models import (BlockType, Element, ParsedDocument, Section,
                      StructuredPaper, TextBlock)
from .classifier import (ABSTRACT_START, KEYWORDS_START, split_heading_number,
                         word_style_level)

# A new reference entry usually starts like one of these:
#   [12] ...              12. ...             Smith, J. ...        Smith J, ...
REFERENCE_START = re.compile(
    r"^(\[\d+\]|\d{1,3}\.\s|[A-Z][A-Za-z'’\-]+,\s+[A-Z]\.|[A-Z][A-Za-z'’\-]+\s+[A-Z]{1,3},)"
)


# ---------------------------------------------------------------------------
# Joining lines
# ---------------------------------------------------------------------------

def join_lines(lines: List[str]) -> str:
    """Join wrapped lines into one string, repairing words split by a hyphen.

    "convolu-" + "tional" -> "convolutional"
    "low-" + "Dose"       -> "low-Dose"   (capital letter: a real hyphen, keep it)
    """
    text = ""
    for line in lines:
        line = line.strip()
        if not text:
            text = line
        elif text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = text + " " + line
    return " ".join(text.split())


class _LayoutStats:
    """Measurements of a document's layout that the paragraph rules need.

    line_pitch : the normal distance from one line's top to the next line's
                 top inside a paragraph. A bigger jump than this means a gap,
                 and a gap means a new paragraph.
    right_edge : how far right a full line reaches, per page and per column.
                 A line that stops well short of it is the last line of a
                 paragraph.
    """

    def __init__(self, blocks: List[TextBlock]):
        pitches = []
        for prev, cur in zip(blocks, blocks[1:]):
            if (prev.block_type is BlockType.BODY and cur.block_type is BlockType.BODY
                    and prev.page == cur.page and prev.column == cur.column):
                pitch = cur.y0 - prev.y0
                if 0 < pitch < 40:
                    pitches.append(pitch)
        self.line_pitch = statistics.median(pitches) if pitches else 0.0

        self.right_edge: Dict[Tuple[int, int], float] = defaultdict(float)
        self.left_edge: Dict[Tuple[int, int], float] = defaultdict(lambda: 1e9)
        for b in blocks:
            if b.block_type in (BlockType.BODY, BlockType.ABSTRACT, BlockType.REFERENCE):
                key = (b.page, b.column)
                self.right_edge[key] = max(self.right_edge[key], b.x1)
                self.left_edge[key] = min(self.left_edge[key], b.x0)

    def is_short_line(self, b: TextBlock) -> bool:
        key = (b.page, b.column)
        width = self.right_edge[key] - self.left_edge[key]
        if width <= 0:
            return False
        return b.x1 < self.right_edge[key] - 0.12 * width

    def starts_new_paragraph(self, prev: TextBlock, cur: TextBlock) -> bool:
        """Does `cur` begin a new paragraph, or continue the one `prev` is in?"""
        if prev.x1 == 0 and cur.x1 == 0:
            return True                     # Word file: each block already is a paragraph

        # The previous line stopped short AND ended a sentence -> paragraph over.
        if self.is_short_line(prev) and prev.text.rstrip().endswith((".", "?", "!", ":")):
            return True

        if prev.page == cur.page and prev.column == cur.column and self.line_pitch:
            if cur.y0 - prev.y0 > self.line_pitch * 1.35:
                return True                 # a visible vertical gap

        # First-line indent: the line starts noticeably to the right.
        key = (cur.page, cur.column)
        if cur.x0 > self.left_edge[key] + 8 and prev.page == cur.page:
            return True

        return False


def _group_into_paragraphs(blocks: List[TextBlock], stats: _LayoutStats) -> List[List[TextBlock]]:
    paragraphs: List[List[TextBlock]] = []
    for b in blocks:
        if paragraphs and not stats.starts_new_paragraph(paragraphs[-1][-1], b):
            paragraphs[-1].append(b)
        else:
            paragraphs.append([b])
    return paragraphs


def _group_into_references(blocks: List[TextBlock], stats: _LayoutStats) -> List[str]:
    """One string per reference. A new entry starts when the line looks like
    the beginning of a reference, or after a visible gap."""
    entries: List[List[TextBlock]] = []
    for b in blocks:
        new_entry = not entries
        if entries:
            prev = entries[-1][-1]
            if REFERENCE_START.match(b.text):
                new_entry = True
            elif (prev.page == b.page and stats.line_pitch
                  and b.y0 - prev.y0 > stats.line_pitch * 1.25 and prev.x1 != 0):
                new_entry = True
            elif prev.x1 == 0:                # Word file: one paragraph per entry
                new_entry = True
        if new_entry:
            entries.append([b])
        else:
            entries[-1].append(b)
    return [join_lines([b.text for b in entry]) for entry in entries]


# ---------------------------------------------------------------------------
# Front matter helpers
# ---------------------------------------------------------------------------

def _split_authors(lines: List[str]) -> List[str]:
    """Pull individual names out of the author lines.

    'K. M. Perera, S. Fernando and A. Jayasuriya'  -> three names
    'Haofei Yu1 ∗'  'Yixing Chen2 ∗'  (one per line) -> 'Haofei Yu', 'Yixing Chen'

    Superscript affiliation numbers and footnote stars come out of the PDF as
    ordinary characters glued to the name, so they are cut off here.
    """
    names: List[str] = []
    for line in lines:
        line = re.sub(r"\s+and\s+|\s*&\s*", ", ", line)
        # a name followed by markers such as "1 ∗," also separates two names
        line = re.sub(r"(?<=[a-z])[\d]+[\s∗*†‡§,]*", ", ", line)
        for part in line.split(","):
            part = re.sub(r"[\s\d∗*†‡§]+$", "", part).strip()
            if part:
                names.append(part)
    return names


def _split_keywords(text: str) -> List[str]:
    text = KEYWORDS_START.sub("", text, count=1)
    parts = re.split(r"[;,·•]", text)
    return [p.strip().rstrip(".") for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assemble(doc: ParsedDocument, label_counts: dict, notes: List[str]) -> StructuredPaper:
    """Turn a classified ParsedDocument into a StructuredPaper."""
    paper = StructuredPaper(source_path=doc.source_path, label_counts=label_counts)
    paper.warnings.extend(notes)

    blocks = [b for b in doc.blocks if b.block_type is not BlockType.JUNK]
    stats = _LayoutStats(blocks)

    def of_type(*types: BlockType) -> List[TextBlock]:
        return [b for b in blocks if b.block_type in types]

    # --- front matter --------------------------------------------------
    paper.title = join_lines([b.text for b in of_type(BlockType.TITLE)])
    paper.authors = _split_authors([b.text for b in of_type(BlockType.AUTHOR)])
    paper.affiliations = [b.text for b in of_type(BlockType.AFFILIATION)]

    abstract_text = join_lines([b.text for b in of_type(BlockType.ABSTRACT)])
    paper.abstract = ABSTRACT_START.sub("", abstract_text, count=1).strip()

    paper.keywords = _split_keywords(join_lines([b.text for b in of_type(BlockType.KEYWORDS)]))
    paper.references = _group_into_references(of_type(BlockType.REFERENCE), stats)

    # --- body: walk the lines in order, filing content under headings --
    current: Section | None = None
    pending_body: List[TextBlock] = []
    after_references = False        # sections after the reference list are appendices

    def flush_body() -> None:
        """Turn the body lines collected so far into paragraphs."""
        nonlocal current
        if not pending_body:
            return
        if current is None:
            # Text before the first heading (common in drafts). Keep it rather
            # than silently losing it; stage 3 decides where it belongs.
            current = Section(heading="", number="", level=1)
            paper.sections.append(current)
        for para in _group_into_paragraphs(pending_body, stats):
            current.elements.append(
                Element(kind="paragraph", text=join_lines([b.text for b in para]), page=para[0].page)
            )
        pending_body.clear()

    for b in blocks:
        kind = b.block_type
        if kind in (BlockType.HEADING, BlockType.SUBHEADING):
            flush_body()
            number, heading, numbered_level = split_heading_number(b.text)
            if heading.lower().strip() in ("references", "bibliography", "works cited",
                                           "literature cited"):
                current = None          # the reference list is stored separately
                after_references = True
                continue
            level = (numbered_level or word_style_level(b)
                     or (2 if kind is BlockType.SUBHEADING else 1))
            current = Section(heading=heading, number=number, level=level,
                              in_appendix=after_references
                              or heading.lower().startswith("appendi"))
            paper.sections.append(current)
        elif kind is BlockType.BODY:
            pending_body.append(b)
        elif kind is BlockType.CODE:
            flush_body()
            if current is None:
                current = Section(heading="", number="", level=1)
                paper.sections.append(current)
            last = current.elements[-1] if current.elements else None
            if last is not None and last.kind == "code":
                last.text = f"{last.text}\n{b.text}"       # one listing, line by line
            else:
                current.elements.append(Element(kind="code", text=b.text, page=b.page))
        elif kind in (BlockType.CAPTION, BlockType.EQUATION):
            flush_body()
            if current is None:
                current = Section(heading="", number="", level=1)
                paper.sections.append(current)
            last = current.elements[-1] if current.elements else None
            if kind is BlockType.EQUATION and last is not None \
                    and last.kind in ("equation", "math_region"):
                # Several equation lines in a row are scraps of ONE display
                # equation that the PDF stored in pieces. Keep them together.
                last.kind = "math_region"
                last.text = f"{last.text} {b.text.strip()}"
            else:
                current.elements.append(
                    Element(kind=kind.value, text=b.text.strip(), page=b.page)
                )
    flush_body()

    # --- sanity checks -------------------------------------------------
    if not paper.title:
        paper.warnings.append("Title is empty.")
    if not paper.authors:
        paper.warnings.append("No authors found.")
    if not paper.keywords:
        paper.warnings.append("No keywords found - index terms will be generated in Step 5.")
    if not [s for s in paper.sections if s.heading]:
        paper.warnings.append("No section headings found - the paper may be one block of text.")

    return paper
