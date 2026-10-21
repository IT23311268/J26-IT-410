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
  FIG-11         caption text: capital first letter, no leading "A", "An", "The"
  FIG-12         the rules above also apply to the text INSIDE captions

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

def apply_number_rules(text: str, log=None) -> Tuple[str, int]:
    """NUM-01 .. NUM-04. Returns (text, number of changes)."""
    changes = 0

    def count(pattern: re.Pattern, template: str, rule: str, s: str) -> str:
        def repl(m: re.Match) -> str:
            nonlocal changes
            changes += 1
            new = m.expand(template)
            if log:
                log([rule], m.group(0), new)
            return new
        return pattern.sub(repl, s)

    text = count(PERCENT_RANGE, r"\1%–\2%", "NUM-04", text)     # before NUM-03
    text = count(PERCENT_SPACE, r"\1%", "NUM-03", text)
    text = count(UNIT_RANGE, r"\1–\2 \3", "NUM-04", text)
    text = count(UNIT_ATTACHED, r"\1 \2", "NUM-02", text)
    text = count(BARE_DECIMAL, r"0.\1", "NUM-01", text)
    return text, changes


def number_style_counts(text: str) -> Tuple[int, int]:
    """(correct, incorrect) number/unit forms in a text - used by the compliance check."""
    bad = (len(BARE_DECIMAL.findall(text)) + len(UNIT_ATTACHED.findall(text))
           + len(PERCENT_SPACE.findall(text)) + len(UNIT_RANGE.findall(text)))
    good = (len(LEADING_ZERO.findall(text)) + len(UNIT_SPACED.findall(text))
            + len(PERCENT_OK.findall(text)))
    return good, bad


def apply_text_rules(text: str, sections: Optional[Dict[str, str]] = None, log=None
                     ) -> Tuple[str, Dict[str, int]]:
    """Apply the IEEE wording rules to one paragraph. Returns (text, counts).

    `log(rules, before, after)`, if given, is called for every single change -
    that is how the rubric knows exactly where each rule was applied."""
    counts = {k: 0 for k in COUNT_KEYS}

    def note(rules, before: str, after: str) -> None:
        if log and before != after:
            log(rules, before, after)

    def fig(m: re.Match) -> str:                         # FIG-03, FIG-04
        counts["figure"] += 1
        word = "Figs." if m.group(1) in ("Figures", "Figs") else R.LABELS["figure"]
        part = m.group(3) or m.group(4)
        new = f"{word} {m.group(2)}" + (f"({part})" if part else "")
        note(["FIG-03", "FIG-04"] if part else ["FIG-03"], m.group(0), new)
        return new

    def tables(m: re.Match) -> str:                      # TAB-04
        counts["table"] += 1
        new = f"Tables {to_roman(int(m.group(1)))} and {to_roman(int(m.group(2)))}"
        note(["TAB-04"], m.group(0), new)
        return new

    def table(m: re.Match) -> str:
        counts["table"] += 1
        new = f"Table {to_roman(int(m.group(1)))}"
        note(["TAB-04"], m.group(0), new)
        return new

    def eq(m: re.Match) -> str:                          # EQN-02
        counts["equation"] += 1
        lead = m.group("lead")
        if lead is not None:                             # start of a sentence
            new = f"{lead}Equation ({m.group('n')})"
        else:
            new = f"({m.group('n')})"
        note(["EQN-02"], m.group(0).strip(), new.strip())
        return new

    def section(m: re.Match) -> str:                     # HDG-09
        lookup = sections or {}
        plural = m.group("word") in ("Sections", "Secs.", "Subsections") or bool(m.group("b"))
        a = lookup.get(m.group("a"), m.group("a"))
        out = f"{R.LABELS['section']}{'s' if plural else ''} {a}"
        if m.group("b"):
            out += f" {m.group('join')} {lookup.get(m.group('b'), m.group('b'))}"
        if out != m.group(0):
            counts["section_ref"] += 1
            note(["HDG-09"], m.group(0), out)
        return out

    def cite_after_stop(m: re.Match) -> str:             # CIT-04
        word = m.group("word")
        if word.lower().rstrip(".") in R.ABBREVIATIONS_BEFORE_CITATION or len(word) == 1:
            return m.group(0)                            # "et al. [3]" stays
        counts["citation"] += 1
        new = f"{word} {m.group('cite')}."
        note(["CIT-04"], m.group(0), new)
        return new

    def reference_word(m: re.Match) -> str:              # CIT-05
        counts["citation"] += 1
        note(["CIT-05"], m.group(0), m.group("cite"))
        return m.group("cite")

    text = FIGURE.sub(fig, text)
    text = TABLES_PAIR.sub(tables, text)
    text = TABLE.sub(table, text)
    text = EQUATION.sub(eq, text)
    text = SECTION_REF.sub(section, text)
    text = REFERENCE_WORD.sub(reference_word, text)
    text = CITE_AFTER_STOP.sub(cite_after_stop, text)
    text, counts["number"] = apply_number_rules(text, log=note)

    def contraction(m: re.Match) -> str:                 # STY-01
        word = m.group(0)
        full = CONTRACTIONS[word.lower().replace("’", "'")]
        counts["contraction"] += 1
        new = full[0].upper() + full[1:] if word[0].isupper() else full
        note(["STY-01"], word, new)
        return new

    pattern = r"\b(" + "|".join(re.escape(c).replace("'", "['’]") for c in CONTRACTIONS) + r")\b"
    text = re.sub(pattern, contraction, text, flags=re.I)

    text, counts["spacing"] = DOUBLE_SPACE.subn(" ", text)   # STY-03
    if counts["spacing"]:
        note(["STY-03"], "two spaces", f"one space ({counts['spacing']}×)")
    return text, counts


CAPTION_LABEL_PART = re.compile(
    r"^(\s*(?:fig\.?|figure|table|algorithm)\s*(?:[0-9]+|[IVXLC]+)[a-z]?\s*[.:\-–—|]?\s*)(.*)$",
    re.I | re.S)
LEADING_ARTICLE = re.compile(
    r"^(?:" + "|".join(R.CAPTION_AVOID_FIRST) + r")\s+(?=[A-Za-z])", re.I)


def apply_caption_rules(text: str, sections: Optional[Dict[str, str]] = None, log=None
                        ) -> Tuple[str, Dict[str, int]]:
    """FIG-11 / FIG-12 on one caption. The label ("Figure 3:") is left for the
    renderer, which writes it as "Fig. 3." / "TABLE III"; only the text after
    it is changed."""
    match = CAPTION_LABEL_PART.match(text)
    if not match:
        return text, {k: 0 for k in COUNT_KEYS} | {"caption": 0}
    label, body = match.groups()
    inner = (lambda rules, b, a: log(rules + ["FIG-12"], b, a)) if log else None
    body, counts = apply_text_rules(body, sections, log=inner)   # FIG-12
    counts["caption"] = 0
    new = LEADING_ARTICLE.sub("", body, count=1)                  # FIG-11
    first_word = re.match(r"\S+", new)
    # capitalise an ordinary word only - never a symbol or variable ("τ", "k-means", "x")
    if first_word and re.fullmatch(r"[a-z]{2,}[,.;:]?", first_word.group(0)):
        new = new[:1].upper() + new[1:]
    if new != body:
        counts["caption"] = 1
        if log:
            log(["FIG-11"], short_words(body), short_words(new))
    return label + new, counts


def short_words(text: str, n: int = 6) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")


def apply_to_paper(paper: StructuredPaper) -> Dict[str, int]:
    from ..rubric.evidence import anchor_of, record, where_of

    totals = {k: 0 for k in COUNT_KEYS}
    totals["caption"] = 0
    sections = section_number_map(paper)
    for section in paper.sections:
        for element in section.elements:
            if element.kind not in ("paragraph", "caption"):
                continue
            place, anchor = where_of(paper, section, element), anchor_of(element)

            def log(rules, before, after, place=place, anchor=anchor):
                record(paper, rules, place, before, after, anchor=anchor)

            if element.kind == "paragraph":
                element.text, counts = apply_text_rules(element.text, sections, log=log)
            else:
                element.text, counts = apply_caption_rules(element.text, sections, log=log)
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
    return totals
