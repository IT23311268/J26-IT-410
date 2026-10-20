"""
ieee_mapper.py - stage 3: match the author's sections onto the IEEE skeleton.

Authors name their sections however they like: "Background", "Our Approach",
"Experiments and Findings", "Final Remarks". An IEEE research paper follows a
recognised skeleton instead:

    I.   INTRODUCTION
    II.  RELATED WORK
    III. METHODOLOGY
    IV.  RESULTS
    V.   DISCUSSION
    VI.  CONCLUSION
         ACKNOWLEDGMENT        (not numbered)
         REFERENCES            (not numbered)

This file decides which IEEE slot each top-level section belongs to, gives it
the IEEE heading and numbering, and reports which slots are missing.

How the decision is made - three kinds of evidence, added up as a score:

    1. THE HEADING  (strongest)  "Previous Studies" contains "previous studies",
                                 a known name for Related Work.
    2. THE CONTENT  (supporting) Results sections are full of numbers and
                                 words like "accuracy" and "outperform";
                                 introductions say "in this paper".
    3. THE POSITION (tie-breaker) The first section is usually the
                                 introduction; the last one, the conclusion.

The slot with the highest score wins. If no slot scores high enough, the
section is kept under its own name rather than forced into the wrong box -
a wrong guess is worse than an honest "I don't know".

Every decision stores a plain-English reason, so you can explain any mapping
in your viva by pointing at exactly what the engine saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..models import Section, StructuredPaper
from ..rules import ieee_rules as R

# ---------------------------------------------------------------------------
# The IEEE skeleton
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    key: str            # internal name
    heading: str        # what IEEE prints
    order: int          # where it sits in a standard IEEE paper
    numbered: bool      # does it get a Roman numeral?


# The skeleton comes from the IEEE rulebook (rules HDG-05, HDG-06, HDG-07,
# ACK-01): names, order, and which sections get a Roman numeral.
# "statement" holds ethics / reproducibility statements, which keep their own
# name, unnumbered.
SLOTS: Dict[str, Slot] = {key: Slot(key, heading, order, numbered)
                          for key, (heading, order, numbered) in R.SECTIONS.items()}

OTHER = "other"     # a real section that fits none of the slots

# Slots a research paper is expected to have. Missing ones are reported.
REQUIRED = list(R.REQUIRED_SECTIONS)


# ---------------------------------------------------------------------------
# Evidence 1: words that appear in headings.  (phrase, weight)
# A phrase can point at more than one slot - "background" is sometimes the
# introduction and sometimes the literature review - so it gets a lower
# weight in each and the position or content evidence settles it.
# ---------------------------------------------------------------------------

HEADING_CUES: Dict[str, List[Tuple[str, float]]] = {
    "introduction": [
        ("introduction", 1.0), ("intro", 0.9), ("overview", 0.6),
        ("motivation", 0.7), ("problem statement", 0.7), ("background", 0.5),
    ],
    "related_work": [
        ("related work", 1.0), ("related works", 1.0), ("literature review", 1.0),
        ("literature survey", 1.0), ("review of literature", 1.0),
        ("previous studies", 1.0), ("previous work", 1.0), ("prior work", 1.0),
        ("state of the art", 0.9), ("existing", 0.6), ("background", 0.5),
        ("literature", 0.8),
    ],
    "methodology": [
        ("methodology", 1.0), ("methods", 1.0), ("method", 0.9),
        ("materials and methods", 1.0), ("approach", 0.9), ("proposed", 0.8),
        ("system design", 0.9), ("architecture", 0.7), ("framework", 0.7),
        ("design", 0.6), ("implementation", 0.6), ("model", 0.5), ("algorithm", 0.6),
    ],
    "experiments": [
        ("experimental setup", 1.0), ("experiment setup", 1.0),
        ("evaluation setup", 1.0), ("setup", 0.7), ("implementation details", 0.8),
        ("experimental design", 0.9), ("experimental settings", 1.0),
        ("experimental setting", 1.0), ("settings", 0.6),
    ],
    "results": [
        ("results", 1.0), ("result", 0.9), ("findings", 0.9), ("experiments", 0.8),
        ("experiment", 0.7), ("evaluation", 0.8), ("performance", 0.7),
        ("analysis", 0.5), ("outcomes", 0.8), ("ablation", 0.8), ("ablations", 0.8),
    ],
    "discussion": [
        ("discussion", 1.0), ("limitations", 0.8), ("threats to validity", 0.9),
        ("implications", 0.7), ("discussions", 1.0),
    ],
    "conclusion": [
        ("conclusion", 1.0), ("conclusions", 1.0), ("concluding", 1.0),
        ("final remarks", 1.0), ("closing remarks", 1.0), ("summary", 0.7),
        ("future work", 0.8), ("future directions", 0.8),
    ],
    "acknowledgment": [
        ("acknowledgment", 1.0), ("acknowledgement", 1.0),
        ("acknowledgments", 1.0), ("acknowledgements", 1.0),
    ],
    "statement": [
        ("statement", 1.0), ("ethics", 0.9), ("ethical", 0.8), ("broader impact", 1.0),
        ("reproducibility", 0.9), ("ai use", 0.9), ("data availability", 1.0),
        ("conflict of interest", 1.0), ("author contributions", 1.0), ("funding", 0.8),
    ],
    "appendix": [("appendix", 1.0), ("appendices", 1.0), ("supplementary", 0.8)],
}

# ---------------------------------------------------------------------------
# Evidence 2: tell-tale words inside the section's text.
# Each hit adds a little; the total is capped so content can SUPPORT a
# heading match but never overrule a clear one on its own.
# ---------------------------------------------------------------------------

CONTENT_CUES: Dict[str, List[str]] = {
    "introduction": [r"\bin this (paper|work|study)\b", r"\bwe propose\b",
                     r"\bcontributions?\b", r"\bmotivat", r"\bthe rest of this paper\b",
                     r"\bremains? (one of|a)\b", r"\bchalleng"],
    "related_work": [r"\bet al\.?", r"\(\d{4}\)", r"\[\d+\]", r"\bproposed\b",
                     r"\bprevious(ly)?\b", r"\bextended\b", r"\breported\b"],
    "methodology": [r"\bwe (used|use|adopt|employ)\b", r"\barchitecture\b",
                    r"\balgorithm\b", r"\bdataset\b", r"\bpre-?process", r"\bloss\b",
                    r"\blayer", r"\btrain(ed|ing)?\b"],
    "experiments": [r"\bhyper-?parameter", r"\blearning rate\b", r"\bbatch size\b",
                    r"\bgpu\b", r"\boptimi[sz]er\b", r"\bepochs?\b"],
    "results": [r"\d+(\.\d+)?\s*(%|percent)", r"\baccuracy\b", r"\bsensitivity\b",
                r"\bprecision\b", r"\brecall\b", r"\boutperform", r"\bcompared with\b",
                r"\bablation\b", r"\btable\s+\d"],
    "discussion": [r"\bthis suggests\b", r"\bone (possible )?explanation\b",
                   r"\binterpret", r"\bimplication"],
    "conclusion": [r"\bin conclusion\b", r"\bwe (have )?presented\b", r"\bfuture work\b",
                   r"\blimitation", r"\bin summary\b", r"\bwill (examine|explore|investigate)\b"],
    "acknowledgment": [r"\bthank", r"\bfund(ed|ing)\b", r"\bgrant\b"],
    "statement": [],
    "appendix": [],
}

TITLE_WORDS_BONUS = 0.6     # heading repeats the paper's own title -> proposed method
STOPWORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on",
             "or", "the", "to", "via", "with", "using", "towards", "toward"}
CONTENT_HIT = 0.12          # score added per distinct cue found
CONTENT_CAP = 0.5           # most that content alone can add
POSITION_BONUS = 0.4        # first -> introduction, last -> conclusion
MIN_SCORE = 0.55            # below this, keep the author's own heading


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalise_heading(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


def _section_text(sections: List[Section], index: int) -> str:
    """A level-1 section's own text PLUS its subsections' text, because
    '3. Our Approach' often has no text of its own - it all lives in 3.1, 3.2."""
    parts = [e.text for e in sections[index].elements]
    for s in sections[index + 1:]:
        if s.level <= sections[index].level:
            break
        parts.extend(e.text for e in s.elements)
    return " ".join(parts)


def _content_words(text: str) -> set:
    return {w for w in _normalise_heading(text).split() if w not in STOPWORDS and len(w) > 2}


def score_section(heading: str, text: str, is_first: bool, is_last: bool,
                  title: str = "") -> Dict[str, Tuple[float, List[str]]]:
    """Score one section against every IEEE slot.

    Returns {slot: (score, [reasons])}. Kept as a separate function so the
    tests - and you - can call it on any heading to see the reasoning.
    """
    norm = _normalise_heading(heading)
    lowered = text.lower()
    results: Dict[str, Tuple[float, List[str]]] = {}

    for slot in SLOTS:
        score = 0.0
        reasons: List[str] = []

        # 1. heading: take the single best matching phrase
        best_phrase, best_weight = "", 0.0
        for phrase, weight in HEADING_CUES.get(slot, []):
            if re.search(rf"\b{re.escape(phrase)}\b", norm) and weight > best_weight:
                best_phrase, best_weight = phrase, weight
        if best_weight:
            score += best_weight
            reasons.append(f'heading contains "{best_phrase}"')

        # 2. content
        hits = [p for p in CONTENT_CUES.get(slot, []) if re.search(p, lowered)]
        if hits:
            content_score = min(CONTENT_CAP, CONTENT_HIT * len(hits))
            score += content_score
            reasons.append(f"{len(hits)} content clue(s)")

        # 4. the paper's own title: a section named after the method the paper
        #    proposes ("COPE: Co-Evolving Acting and Forecasting" in a paper
        #    titled "Co-Evolving Acting and Forecasting ...") is the methodology.
        if slot == "methodology" and title:
            shared = _content_words(heading) & _content_words(title)
            if len(shared) >= 2:
                score += TITLE_WORDS_BONUS
                reasons.append("heading repeats the paper's title (the proposed method)")

        # 3. position
        if is_first and slot == "introduction":
            score += POSITION_BONUS
            reasons.append("it is the first section")
        if is_last and slot == "conclusion":
            score += POSITION_BONUS
            reasons.append("it is the last section")

        results[slot] = (round(score, 3), reasons)
    return results


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------

def to_roman(n: int) -> str:
    numerals = [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out


def to_letter(n: int) -> str:
    return chr(ord("A") + n - 1) if 1 <= n <= 26 else str(n)


# Acronyms that must stay in capitals when an ALL-CAPS heading is re-cased.
ACRONYMS = R.KEEP_UPPER


def _smart_title(text: str) -> str:
    """'OUR APPROACH' or 'our approach' -> 'Our Approach', keeping short words
    small and known acronyms in capitals ('AI USE STATEMENT' -> 'AI Use Statement')."""
    small = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
    words = text.lower().split()
    out = []
    for i, w in enumerate(words):
        core = w.strip("():,;")
        if core in ACRONYMS:
            out.append(w.replace(core, core.upper()))
        elif i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


# ---------------------------------------------------------------------------
# The mapper
# ---------------------------------------------------------------------------

def map_to_ieee(paper: StructuredPaper) -> StructuredPaper:
    """Fill in ieee_slot / ieee_heading / ieee_number on every section,
    and record any required IEEE sections that are missing."""
    sections = paper.sections
    top_level = [i for i, s in enumerate(sections) if s.level == 1]
    # Only real headings compete for "last section"; an acknowledgment or
    # appendix at the end shouldn't steal the conclusion bonus.
    back_matter = re.compile(r"acknowledg|appendi|statement|ethic|reproducib|broader impact", re.I)
    content_top = [i for i in top_level
                   if not sections[i].in_appendix and not back_matter.search(sections[i].heading)]

    used_slots: set = set()

    for position, i in enumerate(top_level):
        section = sections[i]

        # Everything after the reference list is an appendix.
        if section.in_appendix:
            section.ieee_slot = "appendix"
            section.mapping_confidence = 0.95
            section.mapping_reason = "it comes after the reference list"
            continue

        # Text that came before the first heading has no heading to judge by.
        if not section.heading:
            section.ieee_slot = "introduction"
            section.mapping_confidence = 0.5
            section.mapping_reason = "untitled text at the start of the paper"
            continue

        is_first = position == 0
        is_last = bool(content_top) and i == content_top[-1] and len(content_top) > 1
        scores = score_section(section.heading, _section_text(sections, i), is_first, is_last,
                               title=paper.title)

        best_slot, (best_score, reasons) = max(scores.items(), key=lambda kv: kv[1][0])
        runner_up = sorted((v[0] for v in scores.values()), reverse=True)[1]

        if best_score < MIN_SCORE:
            section.ieee_slot = OTHER
            section.mapping_confidence = round(best_score, 2)
            section.mapping_reason = (
                f"no IEEE slot scored high enough (best: {best_slot}, {best_score:.2f}); "
                "kept the author's heading"
            )
        else:
            section.ieee_slot = best_slot
            # Confidence = how strong the winner is and how far it is ahead of
            # the runner-up, squeezed into 0..1. Capped at 0.99: a rule-based
            # guess should never claim to be certain.
            margin = best_score - runner_up
            section.mapping_confidence = round(min(0.99, 0.5 + margin / 2 + best_score / 4), 2)
            section.mapping_reason = "; ".join(reasons)

    # --- headings and numbering -----------------------------------------
    roman = 0
    letter = 0
    sub_number = 0
    sub_letter = 0
    # HDG-08: one appendix is just "Appendix"; several are "Appendix A", "Appendix B"
    appendix_count = sum(1 for s in sections if s.level == 1 and s.ieee_slot == "appendix")
    for section in sections:
        if section.level == 1:
            letter = 0
            slot = SLOTS.get(section.ieee_slot)
            if slot and slot.key == "appendix":
                # IEEE: "Appendix A" + its title, not numbered with the sections
                name = _smart_title(section.heading) if section.heading else ""
                label = f"Appendix {section.number}" if section.number and appendix_count > 1 \
                    else "Appendix"
                section.ieee_heading = f"{label}: {name}" if name and name.lower() != "appendix" \
                    else label
            elif slot and slot.key == "statement":
                section.ieee_heading = _smart_title(section.heading)
            elif slot and slot.key not in used_slots:
                section.ieee_heading = slot.heading
                used_slots.add(slot.key)
            elif section.heading:
                # A second section in the same slot (e.g. "Experiments" AND
                # "Results") or an "other" section keeps its own name.
                section.ieee_heading = _smart_title(section.heading)
            else:
                section.ieee_heading = "Introduction"
                used_slots.add("introduction")

            numbered = slot.numbered if slot else True
            if numbered:
                roman += 1
                section.ieee_number = to_roman(roman)
            else:
                section.ieee_number = ""
        elif section.level == 2:
            letter += 1
            sub_number = 0
            section.ieee_slot = "subsection"
            section.ieee_heading = _smart_title(section.heading)
            section.ieee_number = to_letter(letter)
        elif section.level == 3:
            sub_number += 1                       # HDG-03: "1) Name:"
            sub_letter = 0
            section.ieee_slot = "subsection"
            section.ieee_heading = _smart_title(section.heading)
            section.ieee_number = f"{sub_number})"
        else:
            sub_letter += 1                       # HDG-04: "a) Name:"
            section.ieee_slot = "subsection"
            section.ieee_heading = _smart_title(section.heading)
            section.ieee_number = f"{to_letter(sub_letter).lower()})"

    # --- what's missing, and is anything out of order? --------------------
    found = {s.ieee_slot for s in sections if s.level == 1}
    paper.missing_sections = [SLOTS[k].heading for k in REQUIRED if k not in found]
    for name in paper.missing_sections:
        paper.warnings.append(f'IEEE section "{name}" is missing from this paper.')

    # Only the FIRST section in each slot counts for ordering: a second results
    # section (e.g. "Ablation Studies" after the discussion) is not a mistake.
    seen_for_order: set = set()
    orders = []
    for s in sections:
        if s.level == 1 and s.ieee_slot in SLOTS and s.ieee_slot != "statement" \
                and s.ieee_slot not in seen_for_order:
            seen_for_order.add(s.ieee_slot)
            orders.append(SLOTS[s.ieee_slot].order)
    if orders != sorted(orders):
        paper.warnings.append(
            "Sections are not in the usual IEEE order. The author's order has been kept; "
            "check whether it should be rearranged."
        )

    return paper
