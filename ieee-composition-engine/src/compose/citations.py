"""
citations.py - turn author-year citations into IEEE numbers, and renumber.

Most papers you will feed in cite like this (author-year style):
    "...more than 300 slices (Sung et al., 2021)."
    "Setio et al. (2016) proposed a multi view network..."
    "(Zheng et al., 2024; Deng et al., 2023)"

IEEE cites with numbers in square brackets:
    "...more than 300 slices [1]."
    "Setio et al. [2] proposed a multi view network..."
    "[3], [4]"

And IEEE numbers references IN THE ORDER THEY ARE FIRST CITED - not
alphabetically. So the reference list itself must be reordered.

How a citation is matched to a reference: by the FIRST AUTHOR'S SURNAME plus
the YEAR. "Sung et al., 2021" matches the reference whose first author is
"Sung" and whose year is 2021.

If a citation matches no reference, or matches two (two different "Wang,
2025" papers), it is LEFT UNCHANGED and reported - never guessed. A wrong
citation number is worse than an unconverted one.

This is all plain Python. No AI.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..models import StructuredPaper
from .references import Record, strip_accents, surname_key

YEAR = r"(?:19|20)\d{2}[a-z]?"

# "Setio et al. (2016)"   "Liu and van der Schaar (2025)"   "Buehler (2024)"
NARRATIVE = re.compile(
    rf"(?P<names>(?:[A-Z][\w'’\-]+\s+){{0,3}}?[A-Z][\w'’\-]+"
    rf"(?:\s+et\s+al\.?|\s+(?:and|&)\s+(?:[a-z]+\s+){{0,2}}[A-Z][\w'’\-]+)?)"
    rf"\s+\((?P<year>{YEAR})\)")

# "(Sung et al., 2021)"   "(Zheng et al., 2024; Deng et al., 2023)"   "(see Choudhury, 2025)"
PAREN = re.compile(rf"\(([^()]*?\b{YEAR}\b[^()]*?)\)")
PAREN_PART = re.compile(
    rf"^(?:see\s+|e\.g\.,?\s+|cf\.\s+|i\.e\.,?\s+)?(?P<names>.+?),?\s+(?P<year>{YEAR})$")

# Existing numeric citations: "[3]", "[3, 5]", "[3]-[5]"
NUMERIC = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def _first_surname(names: str) -> str:
    """'Sung et al.' -> 'sung'   'Liu & van der Schaar' -> 'liu'."""
    first = re.split(r"\s+et\s+al\.?|\s*&\s*|\s+and\s+", names.strip())[0]
    return surname_key(first)


class CitationIndex:
    """Looks up a reference number from a (surname, year) pair."""

    def __init__(self, records: List[Record]):
        self.records = records
        self.by_key: Dict[Tuple[str, str], List[int]] = {}
        for i, record in enumerate(records):
            authors = record.get("authors") or []
            if not authors:
                continue
            key = (surname_key(authors[0].get("family", "")), str(record.get("year", "")).lower())
            self.by_key.setdefault(key, []).append(i)

    def find(self, surname: str, year: str, names: str = "") -> Optional[int]:
        year = year.lower()
        hits = self.by_key.get((surname, year))
        if hits is None and year[-1:].isalpha():          # "2024a" cited, "2024" listed
            hits = self.by_key.get((surname, year[:-1]))
        if hits is None and not year[-1:].isalpha():       # "2026" cited, "2026a" listed
            candidates = [v for (s, y), v in self.by_key.items()
                          if s == surname and y[:4] == year and len(y) == 5]
            hits = [i for group in candidates for i in group] if len(candidates) == 1 else None
        if hits and len(hits) > 1 and names:
            hits = self._disambiguate(hits, names)
        if hits and len(hits) == 1:
            return hits[0]
        return None                                        # missing or ambiguous

    def _disambiguate(self, hits: List[int], names: str) -> List[int]:
        """Two references share a surname and year - use the citation's own
        shape to pick: "Liu et al." means 3+ authors, "Liu & van der Schaar"
        means exactly two (and names the second), "Liu" alone means one."""
        def authors_of(i: int) -> list:
            return self.records[i].get("authors") or []

        if re.search(r"et\s+al", names):
            return [i for i in hits if len(authors_of(i)) >= 3]
        pair = re.split(r"\s*&\s*|\s+and\s+", names.strip())
        if len(pair) == 2:
            second = surname_key(pair[1])
            return [i for i in hits if len(authors_of(i)) == 2
                    and surname_key(authors_of(i)[1].get("family", "")) == second]
        return [i for i in hits if len(authors_of(i)) == 1]


def convert_citations(paper: StructuredPaper, records: List[Record]) -> Tuple[List[int], dict]:
    """Rewrite every paragraph's citations as [n] and work out the new order.

    Returns (new_order, stats):
      new_order : old reference positions, in IEEE (first-cited) order
      stats     : counts for the report
    """
    from ..rubric.evidence import anchor_of, record, where_of

    index = CitationIndex(records)
    first_seen: List[int] = []                 # old indexes in order of first citation
    pending: List[tuple] = []                  # (rules, section, element, before, after-with-markers)
    current: dict = {}                         # the section/element being converted
    stats = {"converted": 0, "unresolved": 0, "unresolved_examples": [],
             "already_numeric": 0}

    def number_for(old: int) -> str:
        # A temporary marker. Final numbers are handed out afterwards, by
        # reading the finished text from start to end - so numbering follows
        # READING order, not the order in which the patterns were applied.
        return f"\u0000{old}\u0000"

    def narrative(match: re.Match) -> str:
        old = index.find(_first_surname(match.group("names")), match.group("year"),
                         match.group("names"))
        if old is None:
            # Only report it if it clearly IS a citation ("X et al. (2020)");
            # "Screening (2021)" is just a word followed by a year.
            if re.search(r"et\s+al|\s(and|&)\s", match.group("names")):
                _unresolved(match.group(0))
            return match.group(0)
        stats["converted"] += 1
        after = f"{match.group('names')} {number_for(old)}"
        pending.append((["CIT-01"], current["s"], current["e"], match.group(0), after))
        return after

    def parenthetical(match: re.Match) -> str:
        parts = [p.strip() for p in match.group(1).split(";")]
        found = []
        for part in parts:
            m = PAREN_PART.match(part)
            if not m:
                return match.group(0)          # not a citation, e.g. "(see Table 2, 2021 data)"
            old = index.find(_first_surname(m.group("names")), m.group("year"), m.group("names"))
            if old is None:
                _unresolved(match.group(0))
                return match.group(0)          # all-or-nothing: never half-convert
            found.append(old)
        stats["converted"] += len(found)
        after = ", ".join(number_for(o) for o in found)
        rules = ["CIT-01", "CIT-03"] if len(found) > 1 else ["CIT-01"]
        pending.append((rules, current["s"], current["e"], match.group(0), after))
        return after

    def numeric(match: re.Match) -> str:
        # Already IEEE-style: map "[3]" to the 3rd reference in the original list.
        stats["already_numeric"] += 1
        out = []
        for piece in re.split(r"\s*,\s*", match.group(1)):
            span = re.split(r"\s*[–-]\s*", piece)
            if len(span) == 2 and span[0].isdigit() and span[1].isdigit():
                numbers = range(int(span[0]), int(span[1]) + 1)
            else:
                numbers = [int(span[0])] if span[0].isdigit() else []
            for n in numbers:
                if 1 <= n <= len(records):
                    out.append(number_for(n - 1))
        if not out:
            return match.group(0)
        after = ", ".join(out)
        rules = ["CIT-02", "CIT-03"] if re.search(r"[–-]", match.group(1)) else ["CIT-02"]
        pending.append((rules, current["s"], current["e"], match.group(0), after))
        return after

    def _unresolved(text: str) -> None:
        stats["unresolved"] += 1
        if len(stats["unresolved_examples"]) < 5:
            stats["unresolved_examples"].append(text)

    # Only treat "[3]" as a citation if the references were numbered to begin
    # with. In an author-year paper, "[0, 1]" is an interval in the maths.
    numbered_list = bool(records) and sum(
        1 for r in records if re.match(r"^\s*(\[\d+\]|\d{1,3}\.)\s", r.get("raw", ""))
    ) >= len(records) * 0.8

    for section in paper.sections:
        for element in section.elements:
            if element.kind != "paragraph":
                continue
            current["s"], current["e"] = section, element
            text = NUMERIC.sub(numeric, element.text) if numbered_list else element.text
            text = NARRATIVE.sub(narrative, text)
            text = PAREN.sub(parenthetical, text)
            element.text = text

    # Walk every paragraph in reading order and note each reference the first
    # time it is cited.
    for section in paper.sections:
        for element in section.elements:
            if element.kind == "paragraph":
                for found in re.finditer(r"\u0000(\d+)\u0000", element.text):
                    old = int(found.group(1))
                    if old not in first_seen:
                        first_seen.append(old)

    # IEEE order: cited references first (in order of first citation), then
    # any reference never cited, in its original position.
    new_order = first_seen + [i for i in range(len(records)) if i not in first_seen]
    new_number = {old: new + 1 for new, old in enumerate(new_order)}

    def final(match: re.Match) -> str:
        return f"[{new_number[int(match.group(1))]}]"

    for section in paper.sections:
        for element in section.elements:
            if element.kind == "paragraph":
                element.text = re.sub(r"\u0000(\d+)\u0000", final, element.text)

    # receipts, now that the final numbers are known
    for rules, section, element, before, after in pending:
        after = re.sub(r"\u0000(\d+)\u0000", final, after)
        if after != before:
            record(paper, rules, where_of(paper, section, element), before, after,
                   anchor=anchor_of(element))
    moved = [(old + 1, new) for old, new in sorted(new_number.items()) if old + 1 != new]
    if moved:
        examples = ", ".join(f"[{a}] → [{b}]" for a, b in moved[:6])
        record(paper, "CIT-02", "Reference list", "author's order",
               f"order of first citation ({examples}{'…' if len(moved) > 6 else ''})",
               anchor="references")

    stats["uncited"] = len(records) - len(first_seen)
    return new_order, stats
