"""
The compose package - stage 5 of the pipeline.

compose() runs these jobs, in this order:

  1. references   split each reference into parts (Gemini, or rules), then
                  write it in IEEE style (always plain Python)
  2. citations    "(Sung et al., 2021)" -> "[1]", and renumber the reference
                  list in order of first citation (plain Python)
  3. text rules   "Figure 1" -> "Fig. 1", "Table 2" -> "Table II",
                  "don't" -> "do not" (plain Python)
  4. index terms  tidy the author's keywords, or suggest them (Gemini, checked)
  5. abstract     shorten if too long, draft if missing (Gemini, checked)
  6. polish       optional wording polish (Gemini, checked) - only with --polish

The order matters: citations become [n] BEFORE polishing, so the polish
check can confirm no [n] was lost.
"""

from __future__ import annotations

from typing import List

from ..models import StructuredPaper
from .citations import convert_citations
from .front_matter import make_abstract, make_index_terms
from .ieee_text import apply_to_paper
from .llm import BaseLLM, FakeLLM, LLMUnavailable, NoLLM, make_llm
from .polish import polish_paper
from .references import format_ieee, parse_references, segments_to_text


def compose(paper: StructuredPaper, llm: BaseLLM, polish: bool = False) -> StructuredPaper:
    report: List[str] = []

    # 1-2. references and citations ----------------------------------------
    if paper.references:
        records, notes = parse_references(paper.references, llm)
        report.extend(notes)
        by_ai = sum(1 for r in records if r.get("source") == "gemini")
        report.append(f"References: {len(records)} parsed "
                      f"({by_ai} by Gemini, {len(records) - by_ai} by rules) and "
                      "written in IEEE style.")

        order, stats = convert_citations(paper, records)
        records = [records[i] for i in order]
        for record in records:
            record["segments"] = format_ieee(record)
        paper.reference_records = records
        paper.references = [segments_to_text(r["segments"]) for r in records]

        line = f"Citations: {stats['converted']} converted to IEEE [n] form"
        if stats["already_numeric"]:
            line += f", {stats['already_numeric']} numeric citation(s) renumbered"
        report.append(line + "; references renumbered in order of first citation.")
        if stats["unresolved"]:
            examples = "; ".join(stats["unresolved_examples"])
            report.append(f"Citations: {stats['unresolved']} could not be matched to exactly one "
                          f"reference and were left as written (e.g. {examples}).")
        if stats["uncited"]:
            report.append(f"References: {stats['uncited']} never cited in the text - "
                          "kept at the end of the list.")
    else:
        report.append("References: none found, nothing to convert.")

    # 3. IEEE wording rules ----------------------------------------------------
    counts = apply_to_paper(paper)
    changed = {k: v for k, v in counts.items() if v}
    if changed:
        parts = {"figure": "figure mentions -> 'Fig. n'", "table": "table mentions -> Roman numerals",
                 "equation": "equation mentions -> '(n)'", "contraction": "contractions expanded"}
        report.append("IEEE wording: " + ", ".join(f"{v} {parts[k]}" for k, v in changed.items()) + ".")

    # 4-5. index terms and abstract -------------------------------------------
    report.extend(make_index_terms(paper, llm))
    report.extend(make_abstract(paper, llm))

    # 6. optional polish -------------------------------------------------------
    if polish:
        report.extend(polish_paper(paper, llm))

    paper.compose_report = report
    return paper


__all__ = ["compose", "make_llm", "BaseLLM", "FakeLLM", "NoLLM", "LLMUnavailable"]
