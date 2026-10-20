"""
ieee_text.py - IEEE house-style rules for running text. Plain Python, no AI.

Every rule here is in the IEEE rulebook (src/rules/ieee_rules.py), and every
word list (contractions, units, abbreviations) is read from there:

  FIG-03/FIG-04  "Figure 1" -> "Fig. 1", "Figure 3a" -> "Fig. 3(a)", "Figures" -> "Figs."
  TAB-04         "Table 2"  -> "Table II"
  EQN-02         "Eq. 1"    -> "(1)"; "Equation (1)" only at the start of a sentence
  HDG-09         "Section 3.1" / "Subsection 3.1" -> "Section III-A" (the IEEE number)
  CIT-04         "...the result. [3] The next"  -> "...the result [3]. The next"
  CIT-05         "in reference [3]" -> "in [3]"
  NUM-01         ".25"      -> "0.25"
  NUM-02         "10mm"     -> "10 mm"
  NUM-03         "2 %"      -> "2%"
  NUM-04         "40-50 mm" -> "40–50 mm",  "20-30%" -> "20%–30%"
  STY-01         "don't"    -> "do not"
  STY-03         two spaces after a full stop -> one

Each rule is a find-and-replace, so its effect is predictable and testable.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ..models import StructuredPaper
from ..rules import ieee_rules as R
from ..structure.ieee_mapper import to_roman

CONTRACTIONS = R.CONTRACTIONS

# --- figures, tables, equations ---------------------------------------------
FIGURE = re.compile(r"\b(Figures|Figs|Figure|Fig)\.?\s+(\d+)(?:\s?\(([a-z])\)|([a-z])(?![a-zA-Z]))?")
TABLES_PAIR = re.compile(r"\bTables\s+(\d+)\s+and\s+(\d+)\b")
TABLE = re.compile(r"\bTable\s+(\d+)\b")
EQUATION = re.compile(r"(?P<lead>^|[.!?]\s+)?\b(?:Equation|Eq\.?|Eqn\.?)\s*\(?(?P<n>\d+)\)?", re.I)

# --- cross-references to sections ---------------------------------------------
SECTION_REF = re.compile(
    r"\b(?P<word>Sections|Section|Secs\.|Sec\.|Subsections|Subsection|§)\s*"
    r"(?P<a>\d+(?:\.\d+)*)(?:\s+(?P<join>and|to)\s+(?P<b>\d+(?:\.\d+)*))?")

# --- citations ----------------------------------------------------------------
CITE_GROUP = r"\[\d+\](?:,\s*\[\d+\])*"
CITE_AFTER_STOP = re.compile(rf"(?P<word>[\w.]+)\.\s+(?P<cite>{CITE_GROUP})(?=\s+[A-Z]|\s*$)")
REFERENCE_WORD = re.compile(rf"\b(?:[Rr]eferences?|[Rr]efs?\.)\s+(?P<cite>{CITE_GROUP})")

# --- numbers and units --------------------------------------------------------
_UNIT = "|".join(re.escape(u) for u in sorted(R.UNITS, key=len, reverse=True))
NUM = r"\d+(?:\.\d+)?"
PERCENT_RANGE = re.compile(rf"(?<![\w.])({NUM})\s*[-–]\s*({NUM})\s*%")
PERCENT_SPACE = re.compile(r"(\d)\s+%")
UNIT_RANGE = re.compile(rf"(?<![\w.])({NUM})\s*-\s*({NUM})\s*({_UNIT})(?![A-Za-z0-9])")
UNIT_ATTACHED = re.compile(rf"(?<![\w.])({NUM})({_UNIT})(?![A-Za-z0-9])")
UNIT_SPACED = re.compile(rf"(?<![\w.])({NUM}) ({_UNIT})(?![A-Za-z0-9])")
BARE_DECIMAL = re.compile(r"(?<![\w.\d/])\.(\d)")
LEADING_ZERO = re.compile(r"(?<![\w.])0\.\d")
PERCENT_OK = re.compile(r"\d%")
DOUBLE_SPACE = re.compile(r" {2,}")

COUNT_KEYS = ("figure", "table", "equation", "section_ref", "citation", "number", "contraction",
              "spacing")


# ---------------------------------------------------------------------------
# Section numbers: the author's "3.1" -> IEEE "III-A"
# ---------------------------------------------------------------------------

def section_number_map(paper: StructuredPaper) -> Dict[str, str]:
    """Map every author section number to the IEEE one (rule HDG-09)."""
    out: Dict[str, str] = {}
    parents = {1: "", 2: "", 3: ""}
    for s in paper.sections:
        if s.level == 1:
            parents[1] = s.ieee_number or ""
            ieee = parents[1]
        elif s.level == 2:
            parents[2] = f"{parents[1]}-{s.ieee_number}" if parents[1] and s.ieee_number else ""
            ieee = parents[2]
        else:
            n = (s.ieee_number or "").rstrip(")")
            ieee = f"{parents[2]}{n}" if parents[2] and n else ""
        if s.number and ieee:
            out[s.number.rstrip(".")] = ieee
    return out


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

def apply_number_rules(text: str) -> Tuple[str, int]:
    """NUM-01 .. NUM-04. Returns (text, number of changes)."""
    changes = 0

    def count(pattern: re.Pattern, repl, s: str) -> str:
        nonlocal changes
        s, n = pattern.subn(repl, s)
        changes += n
        return s

    text = count(PERCENT_RANGE, r"\1%–\2%", text)        # NUM-04 (before NUM-03)
    text = count(PERCENT_SPACE, r"\1%", text)            # NUM-03
    text = count(UNIT_RANGE, r"\1–\2 \3", text)          # NUM-04
    text = count(UNIT_ATTACHED, r"\1 \2", text)          # NUM-02
    text = count(BARE_DECIMAL, r"0.\1", text)            # NUM-01
    return text, changes


def number_style_counts(text: str) -> Tuple[int, int]:
    """(correct, incorrect) number/unit forms in a text - used by the compliance check."""
    bad = (len(BARE_DECIMAL.findall(text)) + len(UNIT_ATTACHED.findall(text))
           + len(PERCENT_SPACE.findall(text)) + len(UNIT_RANGE.findall(text)))
    good = (len(LEADING_ZERO.findall(text)) + len(UNIT_SPACED.findall(text))
            + len(PERCENT_OK.findall(text)))
    return good, bad


def apply_text_rules(text: str, sections: Optional[Dict[str, str]] = None
                     ) -> Tuple[str, Dict[str, int]]:
    """Apply the IEEE wording rules to one paragraph. Returns (text, counts)."""
    counts = {k: 0 for k in COUNT_KEYS}

    def fig(m: re.Match) -> str:                         # FIG-03, FIG-04
        counts["figure"] += 1
        word = "Figs." if m.group(1) in ("Figures", "Figs") else R.LABELS["figure"]
        part = m.group(3) or m.group(4)
        return f"{word} {m.group(2)}" + (f"({part})" if part else "")

    def tables(m: re.Match) -> str:                      # TAB-04
        counts["table"] += 1
        return f"Tables {to_roman(int(m.group(1)))} and {to_roman(int(m.group(2)))}"

    def table(m: re.Match) -> str:
        counts["table"] += 1
        return f"Table {to_roman(int(m.group(1)))}"

    def eq(m: re.Match) -> str:                          # EQN-02
        counts["equation"] += 1
        lead = m.group("lead")
        if lead is not None:                             # start of a sentence
            return f"{lead}Equation ({m.group('n')})"
        return f"({m.group('n')})"

    def section(m: re.Match) -> str:                     # HDG-09
        lookup = sections or {}
        plural = m.group("word") in ("Sections", "Secs.", "Subsections") or bool(m.group("b"))
        a = lookup.get(m.group("a"), m.group("a"))
        out = f"{R.LABELS['section']}{'s' if plural else ''} {a}"
        if m.group("b"):
            out += f" {m.group('join')} {lookup.get(m.group('b'), m.group('b'))}"
        if out != m.group(0):
            counts["section_ref"] += 1
        return out

    def cite_after_stop(m: re.Match) -> str:             # CIT-04
        word = m.group("word")
        if word.lower().rstrip(".") in R.ABBREVIATIONS_BEFORE_CITATION or len(word) == 1:
            return m.group(0)                            # "et al. [3]" stays
        counts["citation"] += 1
        return f"{word} {m.group('cite')}."

    def reference_word(m: re.Match) -> str:              # CIT-05
        counts["citation"] += 1
        return m.group("cite")

    text = FIGURE.sub(fig, text)
    text = TABLES_PAIR.sub(tables, text)
    text = TABLE.sub(table, text)
    text = EQUATION.sub(eq, text)
    text = SECTION_REF.sub(section, text)
    text = REFERENCE_WORD.sub(reference_word, text)
    text = CITE_AFTER_STOP.sub(cite_after_stop, text)
    text, counts["number"] = apply_number_rules(text)

    def contraction(m: re.Match) -> str:                 # STY-01
        word = m.group(0)
        full = CONTRACTIONS[word.lower().replace("’", "'")]
        counts["contraction"] += 1
        return full[0].upper() + full[1:] if word[0].isupper() else full

    pattern = r"\b(" + "|".join(re.escape(c).replace("'", "['’]") for c in CONTRACTIONS) + r")\b"
    text = re.sub(pattern, contraction, text, flags=re.I)

    text, counts["spacing"] = DOUBLE_SPACE.subn(" ", text)   # STY-03
    return text, counts


def apply_to_paper(paper: StructuredPaper) -> Dict[str, int]:
    totals = {k: 0 for k in COUNT_KEYS}
    sections = section_number_map(paper)
    for section in paper.sections:
        for element in section.elements:
            if element.kind == "paragraph":
                element.text, counts = apply_text_rules(element.text, sections)
                for k, v in counts.items():
                    totals[k] += v
    return totals
