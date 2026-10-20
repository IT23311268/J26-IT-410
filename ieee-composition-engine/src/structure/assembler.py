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
from .classifier import (ABSTRACT_START, CAPTION, KEYWORDS_START, split_heading_number,
                         word_style_level)

# A new reference entry usually starts like one of these:
#   [12] ...              12. ...             Smith, J. ...        Smith J, ...
REFERENCE_START = re.compile(
    r"^(\[\d+\]|\d{1,3}\.\s|[A-Z][A-Za-z'’\-]+,\s+[A-Z]\.|[A-Z][A-Za-z'’\-]+\s+[A-Z]{1,3},)"
)


# ---------------------------------------------------------------------------
# Positions (Step 6 needs to know where captions and equations were)
# ---------------------------------------------------------------------------

def _bbox_of(block: TextBlock) -> List[float]:
    return [block.x0, block.y0, block.x1, block.y1] if block.x1 else []


def _grow_bbox(element: Element, block: TextBlock) -> None:
    """Stretch an element's box to cover one more line (same page only)."""
    box = _bbox_of(block)
    if not box or block.page != element.page:
        return
    if not element.bbox:
        element.bbox = box
    else:
        element.bbox = [min(element.bbox[0], box[0]), min(element.bbox[1], box[1]),
                        max(element.bbox[2], box[2]), max(element.bbox[3], box[3])]


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
    """One string per reference.

    A new entry starts when:
      * the line looks like the start of a reference ("[3]", "Smith, J."), or
      * there is a bigger vertical gap than usual INSIDE the reference list
        (lines within one entry sit ~11pt apart, entries ~20pt apart), or
      * the list continues at the top of a new page or column and the
        previous line finished with a full stop.
    The gap is measured against the reference list's OWN line spacing, not
    the body text's: references are usually set in smaller type.
    """
    pitches = [c.y0 - p.y0 for p, c in zip(blocks, blocks[1:])
               if p.page == c.page and p.column == c.column and 0 < c.y0 - p.y0 < 40]
    ref_pitch = statistics.median(pitches) if pitches else stats.line_pitch

    entries: List[List[TextBlock]] = []
    for b in blocks:
        new_entry = not entries
        if entries:
            prev = entries[-1][-1]
            same_column = prev.page == b.page and prev.column == b.column
            jump = b.y0 - prev.y0
            if prev.x1 == 0:                              # Word file: one paragraph per entry
                new_entry = True
            elif REFERENCE_START.match(b.text):
                new_entry = True
            elif same_column and ref_pitch and jump > ref_pitch * 1.4:
                new_entry = True                          # a visible gap
            elif (not same_column or jump < -ref_pitch * 2) and prev.text.rstrip().endswith("."):
                new_entry = True                          # new page/column, previous entry ended
        if new_entry:
            entries.append([b])
        else:
            entries[-1].append(b)
    return [join_lines([b.text for b in entry]) for entry in entries]


def _rejoin_split_paragraphs(section: Section) -> None:
    """Join the two halves of a paragraph that a float cut in two.

    In a PDF, a figure, table or algorithm box can interrupt a paragraph:
    "...the forecaster is trained on successful on-policy" [Algorithm 1]
    "trajectories (Sec. 5.2), and the forecasting loss is added..."
    The first half has no full stop, the second starts in lower case. They
    are joined back into one paragraph, and the float moves after it.
    (A paragraph ending in ':' followed by an equation is left alone - that
    is a deliberate break.)
    """
    elements = section.elements
    i = 0
    while i < len(elements):
        first = elements[i]
        if first.kind != "paragraph" or first.text.rstrip().endswith((".", "?", "!", ":")):
            i += 1
            continue
        j = i + 1
        while j < len(elements) and elements[j].kind in ("caption", "code"):
            j += 1                                   # skip over the interrupting float(s)
        if j < len(elements) and j > i + 1 and elements[j].kind == "paragraph" \
                and elements[j].text[:1].islower():
            second = elements[j]
            first.text = join_lines([first.text, second.text])
            floats = elements[i + 1:j]
            elements[i + 1:j + 1] = floats          # second half gone, floats now follow
            continue                                # the joined paragraph may continue again
        i += 1


# ---------------------------------------------------------------------------
# Front matter helpers
# ---------------------------------------------------------------------------

MARK = r"(?:\d{1,2}|[∗*†‡§¶])"
NOTE_WORDS = re.compile(r"equal contribution|corresponding|work done|intern|contact", re.I)


def parse_authors(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Pull individual names, and each name's affiliation marks, out of the author lines.

    'Shangyin Tan1, Ben Zhou1,3'                  -> names ['Shangyin Tan', 'Ben Zhou'],
                                                     marks [['1'], ['1', '3']]
    'K. M. Perera, S. Fernando and A. Jayasuriya' -> three names, no marks
    'Haofei Yu1 ∗'  'Yixing Chen2 ∗'  (one per line)

    Superscript numbers come out of the PDF as ordinary digits glued to the
    name; they are the link to the affiliation list. Footnote symbols
    (∗ equal contribution, † ...) are dropped. A name split over two lines
    ('..., Carol' / 'White1, ...') is joined back together.
    """
    # a list that wraps in the middle of a name: the line ends in a name, not a separator
    joined: List[str] = []
    for line in lines:
        line = line.strip()
        if joined and "," in joined[-1] and re.search(r"[A-Za-z]$", joined[-1]):
            joined[-1] += " " + line
        else:
            joined.append(line)

    names: List[str] = []
    marks: List[List[str]] = []
    for line in joined:
        line = re.sub(r"\s+and\s+|\s*&\s*", ", ", line)
        for m in re.finditer(r"(\d{1,2})|([∗*†‡§¶])|(,)|([^\d,∗*†‡§¶]+)", line):
            digit, symbol, comma, text = m.groups()
            if text is not None:
                text = text.strip()
                if re.search(r"[A-Za-z]", text):
                    names.append(text)
                    marks.append([])
            elif digit is not None and names and digit not in marks[-1]:
                marks[-1].append(digit)
    return names, marks


def parse_affiliations(lines: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """Split the affiliation lines into clean names and a mark -> index map.

    '1UC Berkeley 2Stanford' / '3Bespoke Labs' -> ['UC Berkeley', 'Stanford', 'Bespoke Labs'],
                                                  {'1': 0, '2': 1, '3': 2}
    Notes such as '∗Equal contribution' are not affiliations and are dropped.
    """
    names: List[str] = []
    by_mark: Dict[str, int] = {}
    for line in lines:
        pieces = re.split(rf"(?:^|\s)({MARK})\s*(?=[A-Z{{(])", line.strip())
        lead = pieces[0].strip()
        if lead and not NOTE_WORDS.search(lead):
            names.append(lead)
        for mark, text in zip(pieces[1::2], pieces[2::2]):
            text = text.strip().rstrip(",;")
            if not text or NOTE_WORDS.search(text):
                continue
            if mark not in by_mark:
                by_mark[mark] = len(names)
                names.append(text)
    return names, by_mark


def link_authors(names: List[str], marks: List[List[str]], affiliations: List[str],
                 by_mark: Dict[str, int]) -> List[List[int]]:
    """For each author, the indexes of their affiliations (rule TTL-06)."""
    if any(marks) and by_mark:
        return [[by_mark[m] for m in ms if m in by_mark] for ms in marks]
    if len(affiliations) == 1:                     # everyone shares the one affiliation
        return [[0] for _ in names]
    return []


def _split_authors(lines: List[str]) -> List[str]:
    return parse_authors(lines)[0]


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
    paper.authors, marks = parse_authors([b.text for b in of_type(BlockType.AUTHOR)])
    paper.affiliations, by_mark = parse_affiliations(
        [b.text for b in of_type(BlockType.AFFILIATION)])
    paper.author_affiliations = link_authors(paper.authors, marks, paper.affiliations, by_mark)

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
            if kind is BlockType.CAPTION and last is not None and last.kind == "caption" \
                    and not CAPTION.match(b.text):
                # the next line of a caption that wraps over several lines
                last.text = join_lines([last.text, b.text.strip()])
                _grow_bbox(last, b)
            elif kind is BlockType.EQUATION and last is not None \
                    and last.kind in ("equation", "math_region"):
                # Several equation lines in a row are scraps of ONE display
                # equation that the PDF stored in pieces. Keep them together.
                last.kind = "math_region"
                last.text = f"{last.text} {b.text.strip()}"
                _grow_bbox(last, b)
            else:
                current.elements.append(
                    Element(kind=kind.value, text=b.text.strip(), page=b.page,
                            bbox=_bbox_of(b), region=list(b.region))
                )
    flush_body()

    # --- a front-page figure before the first heading ------------------
    # It has no section of its own: give it to the first real section, where
    # contextual placement (Step 6) puts it after the paragraph that mentions it.
    if len(paper.sections) > 1 and not paper.sections[0].heading \
            and not any(e.kind == "paragraph" for e in paper.sections[0].elements):
        orphan = paper.sections.pop(0)
        paper.sections[0].elements = orphan.elements + paper.sections[0].elements

    # --- rejoin paragraphs a figure split in two ------------------------
    for section in paper.sections:
        _rejoin_split_paragraphs(section)

    # --- sanity checks -------------------------------------------------
    if not paper.title:
        paper.warnings.append("Title is empty.")
    if not paper.authors:
        paper.warnings.append("No authors found.")
    if not paper.keywords:
        paper.warnings.append("No keywords found - index terms are suggested during composition.")
    if not [s for s in paper.sections if s.heading]:
        paper.warnings.append("No section headings found - the paper may be one block of text.")

    return paper
