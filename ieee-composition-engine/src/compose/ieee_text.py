"""
ieee_text.py - IEEE house-style rules for running text. Plain Python, no AI.

These follow the IEEE Editorial Style Manual:

  * Figures are cited as "Fig. 1" - abbreviated, even at the start of a sentence.
  * Tables are numbered with Roman numerals: "Table I", "Table II".
  * Equations are cited by number in brackets: "(1)", not "Eq. 1" / "Equation 1"
    - except at the start of a sentence, where IEEE writes "Equation (1)".
  * Contractions are not used in formal writing: "don't" -> "do not".

Each rule is a simple find-and-replace, so its effect is fully predictable
and can be checked by a test.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from ..models import StructuredPaper
from ..structure.ieee_mapper import to_roman

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot",
    "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "it's": "it is", "that's": "that is", "there's": "there is",
    "we're": "we are", "they're": "they are", "we've": "we have", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "hasn't": "has not",
    "haven't": "have not", "let's": "let us",
}

FIGURE = re.compile(r"\b(?:Figure|Figures|Fig)\.?\s+(\d+)", re.I)
TABLE = re.compile(r"\bTable\s+(\d+)\b")
EQUATION = re.compile(r"(?P<lead>^|[.!?]\s+)?\b(?:Equation|Eq\.?|Eqn\.?)\s*\(?(?P<n>\d+)\)?", re.I)


def apply_text_rules(text: str) -> Tuple[str, Dict[str, int]]:
    """Apply the IEEE wording rules to one paragraph. Returns (text, counts)."""
    counts = {"figure": 0, "table": 0, "equation": 0, "contraction": 0}

    def fig(m: re.Match) -> str:
        counts["figure"] += 1
        return f"Fig. {m.group(1)}"

    def table(m: re.Match) -> str:
        counts["table"] += 1
        return f"Table {to_roman(int(m.group(1)))}"

    def eq(m: re.Match) -> str:
        counts["equation"] += 1
        lead = m.group("lead")
        if lead is not None:                          # start of a sentence
            return f"{lead}Equation ({m.group('n')})"
        return f"({m.group('n')})"

    text = FIGURE.sub(fig, text)
    text = TABLE.sub(table, text)
    text = EQUATION.sub(eq, text)

    def contraction(m: re.Match) -> str:
        word = m.group(0)
        full = CONTRACTIONS[word.lower().replace("’", "'")]
        counts["contraction"] += 1
        return full[0].upper() + full[1:] if word[0].isupper() else full

    pattern = r"\b(" + "|".join(re.escape(c).replace("'", "['’]") for c in CONTRACTIONS) + r")\b"
    text = re.sub(pattern, contraction, text, flags=re.I)
    return text, counts


def apply_to_paper(paper: StructuredPaper) -> Dict[str, int]:
    totals = {"figure": 0, "table": 0, "equation": 0, "contraction": 0}
    for section in paper.sections:
        for element in section.elements:
            if element.kind == "paragraph":
                element.text, counts = apply_text_rules(element.text)
                for k, v in counts.items():
                    totals[k] += v
    return totals
