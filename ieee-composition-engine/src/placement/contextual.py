"""
contextual.py - Step 6c: place every figure, table and algorithm next to the
text that discusses it.  (The novelty in this component's TAF:
"preserving contextual relationships among text, figures, tables,
equations and citations".)

THE PROBLEM
  In a typeset PDF, figures FLOAT: LaTeX pushes each one to the top of
  whichever page has room. When the paper is rebuilt in reading order, a
  figure often lands a page or more away from the sentence that refers to
  it - sometimes even BEFORE that sentence. The IEEE guideline is the
  opposite: a figure or table appears AFTER its first mention, as close to
  it as possible.

HOW EACH VISUAL IS PLACED (first rule that applies wins)
  1. EXPLICIT MENTION   the first paragraph that says "Fig. 3" / "Table II"
                        / "Algorithm 1" -> the visual goes right after it.
  2. SEMANTIC MATCH     nothing mentions it -> compare the caption's words
                        with every paragraph (TF-IDF cosine similarity) and
                        place it after the closest one, IF the match is
                        strong enough (>= MIN_SIMILARITY).
  3. KEEP               otherwise leave it where it was. An honest "not
                        sure" beats a confident wrong guess.

THE MEASUREMENT (for the evaluation chapter)
  For every visual, the distance - in paragraphs - from its first mention
  is recorded BEFORE and AFTER placement. A visual sitting directly after
  its mention has distance 0; one placed before its mention counts as a
  violation. The summary gives the mean distance and the share of visuals
  that follow their first mention, before vs. after.

Plain Python, deterministic, no AI: the same paper always gives the same
placement, and every decision is explained in the report.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..models import Element, StructuredPaper

MIN_SIMILARITY = 0.12

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}
STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or that the
their this to was were with which we our than then these those can each both also via using
used show shows results figure fig table algorithm""".split())

LIST_SEP = r"\s*(?:,|and|&|–|-|to)\s*"
FIG_REF = re.compile(rf"\b(?:Figs?\.|Figures?)\s*((?:\d+[a-z]?(?:{LIST_SEP})?)+)")
TABLE_REF = re.compile(rf"\bTables?\s+((?:(?:[IVXL]+|\d+)\b(?:{LIST_SEP})?)+)")
ALG_REF = re.compile(r"\b(?:Alg\.|Algorithm)\s*(\d+)")
CAPTION_LABEL = re.compile(r"^(fig\.?|figure|table|algorithm)\s*([0-9]+|[IVXL]+)", re.I)


# ---------------------------------------------------------------------------
# Numbers and labels
# ---------------------------------------------------------------------------

def roman_to_int(text: str) -> int:
    total, prev = 0, 0
    for ch in reversed(text.upper()):
        value = ROMAN.get(ch, 0)
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total


def _to_int(token: str) -> Optional[int]:
    token = re.sub(r"[a-z]$", "", token.strip())
    if token.isdigit():
        return int(token)
    if token and all(c in ROMAN for c in token.upper()):
        return roman_to_int(token)
    return None


def _numbers(listing: str) -> List[int]:
    """'3, 4 and 6' -> [3, 4, 6]    '3–5' -> [3, 4, 5]    'II and III' -> [2, 3]"""
    out: List[int] = []
    for part in re.split(r"\s*(?:,|and|&)\s*", listing):
        span = re.split(r"\s*(?:–|-|to)\s*", part)
        if len(span) == 2:
            a, b = _to_int(span[0]), _to_int(span[1])
            if a and b and 0 < b - a < 20:
                out.extend(range(a, b + 1))
                continue
        n = _to_int(span[0])
        if n:
            out.append(n)
    return out


def caption_label(element: Element) -> Optional[Tuple[str, int]]:
    """'Figure 3: ...' -> ('figure', 3)   'TABLE II' -> ('table', 2)"""
    match = CAPTION_LABEL.match(element.text)
    if not match:
        return None
    word = match.group(1).lower()
    kind = "figure" if word.startswith("fig") else word
    number = _to_int(match.group(2))
    return (kind, number) if number else None


def mentions(text: str) -> List[Tuple[str, int]]:
    """Every figure / table / algorithm a paragraph refers to, in order."""
    found: List[Tuple[int, str, int]] = []
    for m in FIG_REF.finditer(text):
        found.extend((m.start(), "figure", n) for n in _numbers(m.group(1)))
    for m in TABLE_REF.finditer(text):
        found.extend((m.start(), "table", n) for n in _numbers(m.group(1)))
    for m in ALG_REF.finditer(text):
        found.append((m.start(), "algorithm", int(m.group(1))))
    return [(kind, n) for _, kind, n in sorted(found)]


def display_label(kind: str, number: int) -> str:
    from ..structure.ieee_mapper import to_roman
    from ..rules.ieee_rules import LABELS          # FIG-03 / TAB-04 wording from the rulebook
    return {"figure": f"{LABELS['figure']} {number}", "table": f"Table {to_roman(number)}",
            "algorithm": f"Algorithm {number}"}[kind]


# ---------------------------------------------------------------------------
# Semantic similarity (TF-IDF cosine), for visuals no paragraph mentions
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower()) if w not in STOPWORDS]


class _TfIdf:
    def __init__(self, documents: List[str]):
        self.docs = [Counter(_tokens(d)) for d in documents]
        n = len(self.docs)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vectors = [self._vector(d) for d in self.docs]

    def _vector(self, counts: Counter) -> Dict[str, float]:
        vec = {w: c * self.idf.get(w, 1.0) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def best_match(self, query: str) -> Tuple[int, float]:
        q = self._vector(Counter(_tokens(query)))
        scores = [sum(q.get(w, 0) * v for w, v in vec.items()) for vec in self.vectors]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else -1
        return best, (scores[best] if best >= 0 else 0.0)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _flatten(paper: StructuredPaper) -> List[Tuple[int, Element]]:
    """Every element in reading order, with the index of its section."""
    return [(si, e) for si, s in enumerate(paper.sections) for e in s.elements]


def _continues(previous: Element, element: Element) -> bool:
    """Is `element` still part of the passage that `previous` belongs to?

    An equation right after a paragraph ("...background voxels:" + equation)
    continues it, and so does the text after an equation that carries on
    the same sentence in lower case ("where alpha was fixed..."). A figure
    must never be dropped into the middle of such a passage.
    """
    if element.kind in ("equation", "math_region"):
        return True
    return (element.kind == "paragraph" and element.text[:1].islower()
            and previous.kind in ("equation", "math_region"))


def _distance(order: List[Tuple[int, Element]], visual: Element, mention: Element) -> int:
    """Paragraphs between a mention and its visual (0 = right after it).
    Negative when the visual comes BEFORE its first mention. Text that only
    continues the mentioning passage (see _continues) does not count."""
    elements = [e for _, e in order]
    ids = [id(e) for e in elements]
    vi, mi = ids.index(id(visual)), ids.index(id(mention))
    lo, hi = sorted((vi, mi))
    between = 0
    for k in range(lo + 1, hi):
        e = elements[k]
        if e.kind == "paragraph" and not _continues(elements[k - 1], e):
            between += 1
    return between if vi > mi else -(between + 1)


def place_visuals(paper: StructuredPaper) -> List[dict]:
    """Move every visual to just after the paragraph that first discusses it.

    Returns the placement report and stores it in paper.placement_report,
    plus a before/after summary in paper.placement_summary.
    """
    before = _flatten(paper)
    paragraphs = [e for _, e in before if e.kind == "paragraph"]
    visuals = [(si, e) for si, e in before if e.kind == "caption" and caption_label(e)]
    if not visuals:
        paper.placement_report, paper.placement_summary = [], {}
        return []

    # first paragraph that mentions each (kind, number)
    first_mention: Dict[Tuple[str, int], Element] = {}
    for p in paragraphs:
        for key in mentions(p.text):
            first_mention.setdefault(key, p)

    tfidf = _TfIdf([p.text for p in paragraphs]) if paragraphs else None
    section_of = {id(e): si for si, e in before}
    targets: Dict[int, List[Element]] = {}        # id(paragraph) -> visuals to put after it
    report: List[dict] = []

    for si, visual in visuals:
        kind, number = caption_label(visual)
        label = display_label(kind, number)
        entry = {"label": label, "from": paper.sections[si].ieee_heading or paper.sections[si].heading,
                 "method": "", "to": "", "distance_before": None, "distance_after": None,
                 "similarity": None}
        mention = first_mention.get((kind, number))
        target = None
        if mention is not None:
            entry["distance_before"] = _distance(before, visual, mention)
            if entry["distance_before"] == 0:
                entry["method"] = "already right after its first mention"
            else:
                target = mention
                entry["method"] = "after its first mention"
        elif tfidf is not None:
            index, score = tfidf.best_match(visual.text)
            entry["similarity"] = round(score, 2)
            if score >= MIN_SIMILARITY:
                target = paragraphs[index]
                entry["method"] = f"semantic match (similarity {score:.2f})"
            else:
                entry["method"] = f"kept in place (not mentioned; best similarity {score:.2f})"
        else:
            entry["method"] = "kept in place (no paragraphs to match)"
        if target is not None:
            targets.setdefault(id(target), []).append(visual)
        report.append((visual, mention, entry))

    # rebuild every section: remove moved visuals, insert them after their target
    moved = {id(v) for group in targets.values() for v in group}
    def ordered(group: List[Element]) -> List[Element]:
        return sorted(group, key=lambda v: (caption_label(v)[0] != "figure", caption_label(v)[1]))

    for section in paper.sections:
        kept = [e for e in section.elements if id(e) not in moved]
        rebuilt: List[Element] = []
        pending: List[Element] = []
        for i, e in enumerate(kept):
            rebuilt.append(e)
            if id(e) in targets:
                pending.extend(ordered(targets[id(e)]))
            # insert only once the mentioning passage is complete
            nxt = kept[i + 1] if i + 1 < len(kept) else None
            if pending and (nxt is None or not _continues(e, nxt)):
                rebuilt.extend(pending)
                pending = []
        section.elements = rebuilt

    after = _flatten(paper)
    section_after = {id(e): si for si, e in after}
    final = []
    for visual, mention, entry in report:
        si = section_after.get(id(visual), section_of[id(visual)])
        entry["to"] = paper.sections[si].ieee_heading or paper.sections[si].heading
        if mention is not None:
            entry["distance_after"] = _distance(after, visual, mention)
        final.append(entry)

    paper.placement_report = final
    paper.placement_summary = summarise(final)
    return final


def summarise(report: List[dict]) -> dict:
    """Before/after numbers for the evaluation."""
    measured = [r for r in report if r["distance_before"] is not None]

    def stats(key: str) -> dict:
        values = [r[key] for r in measured]
        if not values:
            return {"mean_distance": None, "after_first_mention": None, "adjacent": None}
        return {
            "mean_distance": round(sum(abs(v) for v in values) / len(values), 2),
            "after_first_mention": round(100 * sum(1 for v in values if v >= 0) / len(values)),
            "adjacent": round(100 * sum(1 for v in values if v == 0) / len(values)),
        }

    return {
        "visuals": len(report),
        "mentioned": len(measured),
        "moved": sum(1 for r in report if r["method"].startswith(("after its", "semantic"))),
        "semantic": sum(1 for r in report if r["method"].startswith("semantic")),
        "kept": sum(1 for r in report if r["method"].startswith("kept")),
        "before": stats("distance_before"),
        "after": stats("distance_after"),
    }
