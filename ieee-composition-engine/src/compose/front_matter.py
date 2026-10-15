"""
front_matter.py - IEEE index terms and abstract.

INDEX TERMS (IEEE's name for keywords)
  IEEE rules: listed in ALPHABETICAL order, lower case except acronyms and
  proper nouns, separated by commas.
    * the paper has keywords  -> tidy them to IEEE rules (plain Python)
    * the paper has none      -> Gemini suggests 4-6, each checked against
                                 the paper's text; without Gemini, the most
                                 frequent key phrases are extracted by rules

ABSTRACT
  IEEE asks for a single paragraph of 150-250 words.
    * present, 150-250 words  -> left exactly as the author wrote it
    * present, over 250 words -> Gemini shortens it; the result must not
                                 contain any number the original did not
    * present, under 150      -> reported only; we never pad it with filler
    * missing                 -> Gemini drafts one from the introduction and
                                 conclusion; every number must come from
                                 those sections, or it is rejected

Everything Gemini writes is recorded in paper.ai_log, so it is always clear
which words came from the author and which from the AI.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from ..models import StructuredPaper
from . import grounding
from .llm import BaseLLM, LLMUnavailable

ABSTRACT_MIN, ABSTRACT_MAX = 150, 250

STOPWORDS = set("""a an and are as at be been being but by can could did do does for from had has
have how in into is it its may might more most no not of on or our over such than that the their
them then there these they this those through to under up via was we were what when where which
while who will with within without would using use used based new paper approach method results per vs
study propose proposed show shows work also each one two three first however both different""".split())


# ---------------------------------------------------------------------------
# Index terms
# ---------------------------------------------------------------------------

def ieee_case(term: str) -> str:
    """Lower-case a term, except words that are acronyms ('CT', 'GraphSAGE', 'GPT-5')."""
    out = []
    for word in term.split():
        keep = sum(1 for c in word if c.isupper()) >= 2 or any(c.isdigit() for c in word)
        out.append(word if keep else word.lower())
    return " ".join(out).strip(" .;")


def tidy_terms(terms: List[str]) -> List[str]:
    seen, out = set(), []
    for term in terms:
        term = ieee_case(term)
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return sorted(out, key=str.lower)


def extract_terms(paper: StructuredPaper, limit: int = 5) -> List[str]:
    """Rule-based key-phrase extraction (no AI).

    Candidate phrases (1-3 words, no stopwords) come from the title and
    abstract; each is scored by how often it appears in the whole paper,
    with a bonus for appearing in the title.
    """
    full_text = " ".join([paper.abstract] + [e.text for s in paper.sections
                                             for e in s.elements]).lower()
    # An ALL-CAPS title would make every word look like an acronym ("ACTING"),
    # so it is lower-cased first. Links are removed ("github.com" -> "com").
    title = paper.title.lower() if paper.title.isupper() else paper.title
    source = re.sub(r"https?://\S+", " ", f"{title}. {paper.abstract}")
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", source)
    candidates: Counter = Counter()
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if any(w.lower() in STOPWORDS for w in gram) or any(len(w) < 3 for w in gram):
                continue
            phrase = " ".join(gram)
            # a single word is only a good index term if it is an acronym
            # ("CNN") or a long, specific word; "step" or "action" are too vague
            if n == 1 and sum(c.isupper() for c in phrase) < 2:
                continue                       # single words: acronyms only ("CNN", "GRPO")
            count = len(re.findall(rf"\b{re.escape(phrase.lower())}\b", full_text))
            if count < 2:
                continue                       # a key phrase is one the paper keeps using
            bonus = 2.0 if phrase.lower() in paper.title.lower() else 1.0
            weight = 0.4 if n == 1 else n ** 1.5     # real phrases beat lone acronyms
            candidates[phrase] = max(candidates[phrase], count * weight * bonus)

    chosen: List[str] = []
    for phrase, _ in candidates.most_common():
        low = phrase.lower()
        if any(low in c.lower() or c.lower() in low for c in chosen):
            continue                                  # avoid "nodule" next to "nodule detection"
        chosen.append(phrase)
        if len(chosen) == limit:
            break
    return tidy_terms(chosen)


TERMS_PROMPT = """Suggest IEEE index terms (keywords) for this research paper.
Rules:
- 4 to 6 terms, each 1-3 words.
- Every term must be a phrase that literally appears in the text below.
- Lower case except acronyms and proper nouns.
- Return ONLY JSON: {{"terms": ["...", "..."]}}

Title: {title}

Abstract: {abstract}

Section headings: {headings}
"""


def make_index_terms(paper: StructuredPaper, llm: BaseLLM) -> List[str]:
    """Fill paper.keywords with IEEE index terms. Returns report lines."""
    report: List[str] = []
    if paper.keywords:
        before = list(paper.keywords)
        paper.keywords = tidy_terms(paper.keywords)
        if paper.keywords != before:
            report.append("Index terms: author's keywords put into IEEE form "
                          "(alphabetical, lower case except acronyms).")
        return report

    full_text = " ".join([paper.title, paper.abstract] +
                         [e.text for s in paper.sections for e in s.elements])
    if llm.available:
        try:
            headings = ", ".join(s.heading for s in paper.sections if s.level == 1)
            answer = llm.ask_json("index_terms", TERMS_PROMPT.format(
                title=paper.title, abstract=paper.abstract[:3000], headings=headings))
            terms = answer.get("terms", []) if isinstance(answer, dict) else []
            grounded = [t for t in terms if isinstance(t, str)
                        and grounding.words_supported(t, full_text)]
            rejected = len(terms) - len(grounded)
            if len(grounded) >= 3:
                paper.keywords = tidy_terms(grounded)
                note = f"Index terms: {len(paper.keywords)} suggested by Gemini"
                if rejected:
                    note += f" ({rejected} rejected - their words are not in the paper)"
                report.append(note + ".")
                paper.ai_log.append(f"Index terms generated by {llm.name}: "
                                    + ", ".join(paper.keywords))
                return report
            report.append("Index terms: Gemini's suggestions failed the grounding check; "
                          "used rule-based extraction.")
        except LLMUnavailable as error:
            report.append(f"Index terms: Gemini unavailable ({error}); used rule-based extraction.")

    paper.keywords = extract_terms(paper)
    if paper.keywords:
        report.append(f"Index terms: {len(paper.keywords)} key phrases extracted by rules.")
        paper.ai_log.append("Index terms extracted automatically (rules, no AI): "
                            + ", ".join(paper.keywords))
    return report


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------

SHORTEN_PROMPT = """Shorten this research paper abstract to at most {limit} words.
Rules:
- Use ONLY information already in the abstract. Add nothing.
- Keep every key result and number that you keep exactly as written.
- One paragraph, formal academic English, no citations.
- Return ONLY JSON: {{"abstract": "..."}}

Abstract:
{abstract}
"""

DRAFT_PROMPT = """Write an abstract for this research paper using ONLY the text provided.
Rules:
- {low} to {high} words, one paragraph, formal academic English.
- State the problem, the proposed approach, and the main results.
- Use ONLY facts and numbers that appear in the text below. Invent nothing.
- No citations, no references to figures or tables.
- Return ONLY JSON: {{"abstract": "..."}}

Title: {title}

{sections}
"""


def _word_count(text: str) -> int:
    return len(text.split())


def _with_subsections(paper: StructuredPaper, index: int) -> str:
    """A section's paragraphs PLUS those of its subsections. ("3. Our Approach"
    often has no text of its own - it all lives in 3.1 and 3.2.)"""
    top = paper.sections[index]
    texts = list(top.paragraphs)
    for s in paper.sections[index + 1:]:
        if s.level <= top.level:
            break
        texts.extend(s.paragraphs)
    return " ".join(texts)


def _whole_paper(paper: StructuredPaper) -> str:
    return " ".join([paper.title, paper.abstract] +
                    [e.text for s in paper.sections for e in s.elements])


def _source_for_draft(paper: StructuredPaper, limit: int = 6000) -> str:
    """The introduction, methodology, results and conclusion text - what an
    abstract summarises."""
    wanted = ("introduction", "methodology", "results", "conclusion")
    parts = []
    for slot in wanted:
        for i, s in enumerate(paper.sections):
            if s.ieee_slot == slot and s.level == 1:
                parts.append(f"{s.ieee_heading}:\n" + _with_subsections(paper, i)[:limit // 4])
                break
    if not parts:                                       # mapper found nothing: use the start
        parts = [" ".join(e.text for e in s.elements if e.kind == "paragraph")
                 for s in paper.sections[:2]]
    return "\n\n".join(parts)[:limit]


def make_abstract(paper: StructuredPaper, llm: BaseLLM) -> List[str]:
    report: List[str] = []
    words = _word_count(paper.abstract)

    if paper.abstract and ABSTRACT_MIN <= words <= ABSTRACT_MAX:
        return [f"Abstract: {words} words - within IEEE's {ABSTRACT_MIN}-{ABSTRACT_MAX}; unchanged."]

    if paper.abstract and words < ABSTRACT_MIN:
        return [f"Abstract: {words} words - shorter than IEEE's recommended {ABSTRACT_MIN}. "
                "Left unchanged (the engine never pads with filler)."]

    if paper.abstract and words > ABSTRACT_MAX:
        if not llm.available:
            return [f"Abstract: {words} words - over IEEE's {ABSTRACT_MAX}-word limit. "
                    "Gemini is needed to shorten it; left unchanged."]
        try:
            answer = llm.ask_json("shorten_abstract", SHORTEN_PROMPT.format(
                limit=ABSTRACT_MAX, abstract=paper.abstract))
            new = (answer or {}).get("abstract", "").strip() if isinstance(answer, dict) else ""
            ok, invented = grounding.numbers_supported(new, paper.abstract)
            if new and ok and _word_count(new) <= ABSTRACT_MAX:
                report.append(f"Abstract: shortened by Gemini from {words} to "
                              f"{_word_count(new)} words (all numbers checked against the original).")
                paper.ai_log.append(f"Abstract shortened by {llm.name}; original was {words} words.")
                paper.abstract = new
            else:
                why = f"it introduced numbers {invented}" if invented else "it was still too long"
                report.append(f"Abstract: Gemini's shortened version was rejected ({why}); "
                              "original kept.")
        except LLMUnavailable as error:
            report.append(f"Abstract: Gemini unavailable ({error}); original kept.")
        return report

    # no abstract at all
    if not llm.available:
        return ["Abstract: missing. Gemini is needed to draft one; a placeholder is shown."]
    source = _source_for_draft(paper)
    if len(source.split()) < 80:
        return ["Abstract: missing, and the paper has too little text to draft one from."]
    try:
        answer = llm.ask_json("draft_abstract", DRAFT_PROMPT.format(
            low=ABSTRACT_MIN, high=ABSTRACT_MAX, title=paper.title, sections=source))
        new = (answer or {}).get("abstract", "").strip() if isinstance(answer, dict) else ""
        # every number must appear somewhere in the paper itself
        ok, invented = grounding.numbers_supported(new, _whole_paper(paper))
        has_citation = bool(re.search(r"\[\d+\]|et al\.", new))
        if new and ok and not has_citation and _word_count(new) <= ABSTRACT_MAX + 20:
            paper.abstract = new
            report.append(f"Abstract: drafted by Gemini ({_word_count(new)} words) from the "
                          "introduction and conclusion; every number checked against them.")
            paper.ai_log.append(f"Abstract DRAFTED by {llm.name} - the author should review it.")
        else:
            why = (f"it contained numbers not in the paper: {invented}" if invented
                   else "it contained citations" if has_citation else "it was empty or too long")
            report.append(f"Abstract: Gemini's draft was rejected ({why}); placeholder shown.")
    except LLMUnavailable as error:
        report.append(f"Abstract: Gemini unavailable ({error}); placeholder shown.")
    return report
