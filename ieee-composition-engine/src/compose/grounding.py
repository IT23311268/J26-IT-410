"""
grounding.py - checks that AI-written text is supported by the paper.

A language model can "hallucinate": write a convincing number or claim that
is not in the source. For a research manuscript that is unacceptable, so
every piece of text Gemini writes is checked here BEFORE it is accepted.

The checks are deliberately simple and strict - easy to explain, impossible
to argue with:

  numbers_supported()   every number in the new text appears in the source
  same_numbers()        a rewrite keeps exactly the same numbers
  same_citations()      a rewrite keeps exactly the same [n] citations
  words_supported()     an index term's words actually appear in the paper

If a check fails, the AI text is thrown away and the original is kept (or a
placeholder is shown). The report says how many were rejected and why.
"""

from __future__ import annotations

import re
from collections import Counter

NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:%|\b)")
CITATION = re.compile(r"\[\d+\]")


def numbers_in(text: str) -> Counter:
    """All numbers in a text, normalised so '1,000' and '1000' count as one."""
    found = [m.group(0).rstrip("%").replace(",", "") for m in NUMBER.finditer(text)]
    return Counter(n for n in found if n)


def numbers_supported(new_text: str, source_text: str) -> tuple:
    """Every number written by the AI must exist in the source.
    Returns (ok, the unsupported numbers)."""
    source = set(numbers_in(source_text))
    invented = sorted(set(numbers_in(new_text)) - source)
    return not invented, invented


def same_numbers(new_text: str, old_text: str) -> bool:
    return numbers_in(new_text) == numbers_in(old_text)


def same_citations(new_text: str, old_text: str) -> bool:
    return Counter(CITATION.findall(new_text)) == Counter(CITATION.findall(old_text))


def words_supported(phrase: str, source_text: str) -> bool:
    """Every real word (3+ letters) of a phrase must appear in the source."""
    source = source_text.lower()
    words = re.findall(r"[a-z][a-z\-]{2,}", phrase.lower())
    return bool(words) and all(w in source for w in words)


def length_ratio_ok(new_text: str, old_text: str, low: float = 0.6, high: float = 1.3) -> bool:
    old_len = max(1, len(old_text.split()))
    return low <= len(new_text.split()) / old_len <= high
