"""
polish.py - optional: Gemini rewrites paragraphs in formal IEEE register.

Switched on with:   python main.py paper.pdf --polish

This is "template-constrained generation": Gemini may improve the WORDING of
a paragraph, but a rewrite is only accepted if it keeps:

    * exactly the same numbers             (no changed or invented results)
    * exactly the same [n] citations       (no lost or invented sources)
    * a similar length (60%-130%)          (nothing dropped, nothing padded)

Anything that fails is thrown away and the author's original paragraph is
kept. The report gives the counts, so you can quote an acceptance rate.

It is OFF by default because it costs one Gemini request per section, and
the free tier allows only a limited number of requests per minute.
"""

from __future__ import annotations

from typing import List

from ..models import StructuredPaper
from . import grounding
from .llm import BaseLLM, LLMUnavailable

MIN_WORDS = 25          # very short paragraphs are left alone

POLISH_PROMPT = """Rewrite each paragraph in formal academic English suitable for an IEEE paper.
Rules - follow them exactly:
- Keep the meaning. Do not add facts, claims or examples. Do not remove sentences.
- Keep EVERY number, symbol, and citation such as [3] exactly as written.
- Keep technical terms and names unchanged.
- No contractions. Prefer clear, direct sentences.
- Return ONLY JSON: {{"paragraphs": ["...", "..."]}} with exactly {count} items, same order.

Paragraphs:
{paragraphs}
"""


def polish_paper(paper: StructuredPaper, llm: BaseLLM, max_sections: int = 40) -> List[str]:
    if not llm.available:
        return ["Polish: skipped - Gemini is not available."]

    accepted = rejected = 0
    reasons = {"numbers": 0, "citations": 0, "length": 0, "shape": 0}
    sections_done = 0

    for section in paper.sections:
        if sections_done >= max_sections:
            break
        targets = [e for e in section.elements
                   if e.kind == "paragraph" and len(e.text.split()) >= MIN_WORDS]
        if not targets:
            continue
        sections_done += 1
        listing = "\n\n".join(f"[P{i + 1}] {e.text}" for i, e in enumerate(targets))
        try:
            answer = llm.ask_json("polish", POLISH_PROMPT.format(
                count=len(targets), paragraphs=listing))
        except LLMUnavailable as error:
            return [f"Polish: stopped - Gemini unavailable ({error}). "
                    f"{accepted} paragraph(s) polished before that."]

        new_texts = answer.get("paragraphs") if isinstance(answer, dict) else None
        if not isinstance(new_texts, list) or len(new_texts) != len(targets):
            rejected += len(targets)
            reasons["shape"] += len(targets)
            continue

        for element, new in zip(targets, new_texts):
            new = str(new).strip()
            new = new.split("] ", 1)[1] if new.startswith("[P") and "] " in new else new
            if not grounding.same_numbers(new, element.text):
                rejected += 1
                reasons["numbers"] += 1
            elif not grounding.same_citations(new, element.text):
                rejected += 1
                reasons["citations"] += 1
            elif not grounding.length_ratio_ok(new, element.text):
                rejected += 1
                reasons["length"] += 1
            else:
                element.text = new
                accepted += 1

    total = accepted + rejected
    if total == 0:
        return ["Polish: no paragraphs long enough to polish."]
    paper.ai_log.append(f"{accepted} of {total} paragraphs reworded by {llm.name} "
                        "(numbers and citations verified unchanged).")
    detail = ", ".join(f"{v} {k}" for k, v in reasons.items() if v)
    line = f"Polish: {accepted}/{total} paragraphs accepted ({100 * accepted // total}%)."
    if rejected:
        line += f" Rejected and kept original: {detail}."
    return [line]
