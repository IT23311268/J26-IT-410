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
