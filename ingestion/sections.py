"""
Section identification — pipeline Box 4.

Box 3 labelled each block HEADING / BODY / FIGURE / … This stage reads
the HEADING blocks, works out which section of the paper each one opens,
and tags every region with the section it sits in.

Why the section matters
-----------------------
The same sentence means different things in different sections. "BM25
outperforms dense retrieval on short queries" in Related Work is
somebody else's finding being cited; in Results it is this paper's own.
Member 2 verifies claims against this text and cannot tell those apart
without the label — which is most of the point of grounding
verification. Box 5 needs it too: a chunk must not straddle a section
boundary, or half of it is Method and half is Results.

No model, no GPU — the heading's own words, and the order they are
tried in.
"""

from __future__ import annotations

import itertools

import re

from schema.ingestion_schema_v1 import LayoutRegion, RegionType, SectionType

# The wordings papers really use. A paper almost never prints the IMRaD
# word itself: "Method" is as likely to be "Our Approach", "Model
# Architecture" or "Experimental Setup", and the results section is
# often "Evaluation" or "Experiments".
#
# THE ORDER OF THIS LIST IS LOAD-BEARING. It decides a heading that
# names two sections at once — "Results and Discussion", "Discussion and
# Conclusion" — and those are common. DISCUSSION sits last on purpose:
# it is the one that loses every combination, because a heading pairing
# it with Results or Conclusion is opening a section whose substance is
# the other one. tests/test_sections.py pins all four, so a later edit
# that reshuffles this list fails there instead of silently relabelling
# half a corpus.
SECTION_KEYWORDS: list[tuple[SectionType, tuple[str, ...]]] = [
    (SectionType.ABSTRACT, ("abstract",)),
    (SectionType.INTRODUCTION, ("introduction", "motivation")),
    (SectionType.RELATED_WORK, ("related work", "background", "prior work", "literature")),
    (SectionType.METHOD, (
        "experimental setup", "methodology", "method", "approach", "model",
        "architecture", "implementation", "proposed", "system",
    )),
    (SectionType.RESULTS, ("result", "evaluation", "experiment", "finding", "analysis")),
    (SectionType.CONCLUSION, ("conclusion", "concluding", "summary")),
    (SectionType.REFERENCES, ("reference", "bibliography")),
    (SectionType.APPENDIX, ("appendix", "supplementary")),
    (SectionType.DISCUSSION, ("discussion", "limitation", "future work")),
]

# A heading's number says where it sits, never what it is: "2." opens
# Method in one paper and Related Work in another. It comes off before
# anything is matched. Covers "3.", "2.1", IEEE roman numerals ("IV.")
# and IEEE lettered subsections ("A.").
NUMBERING_RE = re.compile(r"^\s*([0-9]+(\.[0-9]+)*|[IVXLC]+|[A-Z])[.)]?\s+")

# Words a heading opens with before its real first word. "Our Approach"
# and "The Proposed Model" are method sections; the keyword simply is
# not the first thing printed.
LEADING_FILLER_RE = re.compile(r"^(our|the|a|an)\s+", re.IGNORECASE)

# A heading naming two sections at once. Split on this and the keyword
# order above decides which one wins.
COMBINED_RE = re.compile(r"\s+(and|&)\s+", re.IGNORECASE)


def strip_numbering(heading_text: str) -> str:
    """Remove a leading section number: '2.1 Method' -> 'Method'.

    The original is kept on `LayoutRegion.section_title_raw`, so nothing
    is lost when the classification is wrong.
    """
    return NUMBERING_RE.sub("", heading_text).strip()


def classify_section(heading_text: str) -> SectionType:
    """Work out which section a heading opens.

    Two rules, in this order.

    A heading that names ONE section must *start* with the keyword once
    its number and any leading "our"/"the" are stripped. That is what
    separates a real heading from a line that merely contains the word:
    "Method" opens a method section, "Limitations of Prior Methods" does
    not — it is a subsection discussing other people's work, and an
    earlier version of this function labelled it METHOD.

    A heading that names TWO sections — joined by "and" or "&" — is
    decided by SECTION_KEYWORDS order instead, because neither keyword
    can be at the start.

    Returns UNKNOWN when nothing matches, and that is a real answer, not
    a failure: see `assign_sections`. A wrong section is worse than no
    section, because it hands Member 2 a chunk under the wrong
    provenance.
    """
    text = strip_numbering(heading_text)
    text = LEADING_FILLER_RE.sub("", text).strip().lower()

    if not text:
        return SectionType.UNKNOWN

    if COMBINED_RE.search(text):
        for section, keywords in SECTION_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return section
        return SectionType.UNKNOWN

    for section, keywords in SECTION_KEYWORDS:
        if any(text.startswith(keyword) for keyword in keywords):
            return section

    return SectionType.UNKNOWN


def assign_sections(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Tag every region with the section it belongs to.

    Returns the same regions, in the same order — already reading order
    from Box 3 — with `section` and `section_title_raw` filled in. Box
    3's own output is not modified, so the two stages stay separable:

        regions = extract_regions(pdf, paper_id)   # Box 3
        regions = assign_sections(regions)         # Box 4

    The rule is the one a reader uses without thinking: a heading opens
    a section, and everything after it belongs to that section until the
    next heading opens a new one.

    Two details that are not obvious:

    A heading we cannot classify does NOT start a new section. "2.3
    Generator: BART" is a subsection of the Method, and letting it move
    us would drop the rest of the Method into a section that does not
    exist. Only a recognised heading moves us — which is what makes
    UNKNOWN the useful answer rather than a gap.

    Everything before the first recognised heading is TITLE: the title,
    the authors, the affiliations. That is where they belong, and it
    keeps them out of the abstract.
    """
    out: list[LayoutRegion] = []

    current = SectionType.TITLE
    current_heading = ""

    for region in regions:
        if region.region_type == RegionType.HEADING:
            found = classify_section(region.text)
            if found != SectionType.UNKNOWN:
                current = found
                current_heading = region.text.strip()

        out.append(
            region.model_copy(
                update={"section": current, "section_title_raw": current_heading}
            )
        )

    return out


#: A title longer than this is a run-on paragraph the heading rule
#: caught, not a title. Measured against arXiv: the longest titles run
#: to about 150 characters, and body text starts well above that.
MAX_TITLE_CHARS = 200

#: How far apart two font sizes may be and still count as the same
#: type. Lines of one title are set identically; this is slack for a
#: PDF that reports 16.99 and 17.0 for the same run, not room for a
#: subtitle. The gap it has to stay under in practice is title to
#: authors — 17.0 pt against 10.5 on the two-line case above.
SAME_TYPE_PT = 0.6

#: How much air may sit between two lines of one block, in line
#: heights. Measured on a thesis title page, whose lines are set
#: 1.58 line heights apart while the next block down is 8.42 away.
MAX_LINE_GAP = 2.0


def _is_sideways(region: LayoutRegion) -> bool:
    """Is this text printed rotated, running up the page?

    arXiv prints its own stamp down the left margin of page one —
    `arXiv:2005.11401v4 [cs.CL] 12 Apr 2021` — rotated a quarter turn.
    It sits above the real title, it is set large enough to pass the
    heading rule, and it is not the paper's title; it is a label arXiv
    added, like a catalogue sticker on a library book. Every paper
    fetched from arXiv carries one.

    A line of text is wider than it is tall. Turn it on its side and
    that reverses, which is what makes this measurable rather than a
    guess. Measured on arXiv:2610.12376: the stamp's box is 20 points
    wide and 335 tall — seventeen times taller than wide — while the
    title below it is 389 by 46.
    """
    box = region.bbox
    width, height = box.x1 - box.x0, box.y1 - box.y0
    return width > 0 and height > width


def _is_shouted(region: LayoutRegion) -> bool:
    """Is this line set in capitals?

    The signal of last resort. A thesis title page has no type
    hierarchy at all — title, "A Thesis", the degree and the year are
    every one of them 12 pt Times — so Box 3 marks nothing on it as a
    heading and there is no largest type to find. The title is still
    the only thing on the page shouted in capitals.

    Four letters at least, so "of" and "by" and a copyright mark
    cannot open a title block.
    """
    text = " ".join(region.text.split())
    return sum(ch.isalpha() for ch in text) >= 4 and text == text.upper()


def _join_block(regions: list[LayoutRegion], is_line) -> str:
    """Join the first run of lines `is_line` accepts into one string.

    A title longer than about forty characters does not fit one line
    at title size, so it is broken across two or three — "Retrieval-
    Augmented Generation for / Knowledge-Intensive NLP Tasks". Each
    printed line is its own region, so taking the first returns half a
    title. Two measurements say where the block ends, and on the two
    documents this was built against they do the work in turn:

    * the type stops matching. On a paper the title is 17.0 pt and the
      author line under it 10.5, which separates them outright.
    * the lines stop being adjacent. On a thesis title page everything
      is 12 pt, so size says nothing — but the two title lines sit
      1.58 line heights apart and the next block down the page is
      8.42 away. Two line heights splits those with room either side.
    """
    lines: list[str] = []
    size: float | None = None
    bottom = 0.0

    for region in regions:
        text = " ".join(region.text.split())
        if not text or not is_line(region):
            if lines:
                break  # the block ended at the line above
            continue

        if not lines:
            if len(text) > MAX_TITLE_CHARS:
                continue  # too long to be a title; keep looking
            lines, size, bottom = [text], region.font_size, region.bbox.y1
            continue

        if size is None or region.font_size is None:
            break  # nothing to compare the type against
        if abs(region.font_size - size) > SAME_TYPE_PT:
            break  # different type: a subtitle, or the authors
        if region.bbox.y0 - bottom > MAX_LINE_GAP * size:
            break  # too much air between them to be one block

        joined = " ".join(lines + [text])
        if len(joined) > MAX_TITLE_CHARS:
            break
        lines.append(text)
        bottom = region.bbox.y1

    return " ".join(lines)


def paper_title(regions: list[LayoutRegion]) -> str:
    """The paper's own title, taken from the page rather than the
    filename.

    `PaperMeta.title` is Optional because the flat baseline has no way
    to find one — a PDF's embedded metadata title is wrong or missing
    more often than not, and flat text cannot tell a title from the
    first line of the abstract. Once Box 3 has measured the fonts and
    Box 4 has marked the front matter, it is the first heading before
    any recognised section begins: the largest type on page one, above
    the abstract. `_join_block` keeps the rest of that heading's lines.

    When Box 3 marked no heading at all in the front matter, the page
    has no type hierarchy to read — a thesis title page sets every
    line in the same 12 pt. Capitals are what is left, and on such a
    page the title is the only thing in them. That fallback runs only
    when there is no heading to be had, so it cannot overrule one.

    Returns "" when there is nothing defensible, which leaves
    `PaperMeta.title` None rather than guessing.
    """
    front = [
        r
        for r in itertools.takewhile(
            lambda r: r.section is SectionType.TITLE, regions
        )
        if not _is_sideways(r)
    ]

    def is_heading(region: LayoutRegion) -> bool:
        return region.region_type is RegionType.HEADING

    if any(is_heading(r) for r in front):
        return _join_block(front, is_heading)
    return _join_block(front, _is_shouted)
