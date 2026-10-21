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
from .front_matter import apply_front_matter_rules, make_abstract, make_index_terms
from .ieee_text import apply_to_paper
from .llm import BaseLLM, FakeLLM, LLMUnavailable, NoLLM, make_llm
from .polish import polish_paper
from .references import (abbreviate_venue, format_ieee, parse_references, reference_type,
                         segments_to_text)


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
        _record_reference_evidence(paper)

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
    # Every rule applied here is in the IEEE rulebook (src/rules/ieee_rules.py).
    counts = apply_to_paper(paper)
    changed = {k: v for k, v in counts.items() if v}
    if changed:
        parts = {"figure": "figure mentions -> 'Fig. n' (FIG-03/04)",
                 "table": "table mentions -> Roman numerals (TAB-04)",
                 "equation": "equation mentions -> '(n)' (EQN-02)",
                 "section_ref": "section references -> IEEE numbers (HDG-09)",
                 "citation": "citations moved inside the punctuation or 'reference' removed "
                             "(CIT-04/05)",
                 "number": "numbers and units corrected (NUM-01..04)",
                 "contraction": "contractions expanded (STY-01)",
                 "spacing": "double spaces removed (STY-03)",
                 "caption": "captions reworded to start without 'A', 'An', 'The' (FIG-11)"}
        report.append("IEEE wording: " + ", ".join(f"{v} {parts[k]}" for k, v in changed.items()) + ".")

    # 4-5. title, authors, index terms and abstract ---------------------------
    report.extend(apply_front_matter_rules(paper))
    report.extend(make_index_terms(paper, llm))
    report.extend(make_abstract(paper, llm))

    # 6. optional polish -------------------------------------------------------
    if polish:
        report.extend(polish_paper(paper, llm))

    paper.compose_report = report
    return paper


def _record_reference_evidence(paper: StructuredPaper) -> None:
    """One receipt per reference, naming every reference rule it follows."""
    from ..rubric.evidence import record as receipt
    from ..rules.ieee_rules import MAX_AUTHORS_BEFORE_ET_AL

    for n, (rec, text) in enumerate(zip(paper.reference_records, paper.references), start=1):
        rules = ["REF-02", "REF-04", "REF-11"]
        kind = reference_type(rec)
        venue = rec.get("venue", "")
        if venue and kind != "arxiv":
            rules.append("REF-05")
        if kind == "journal":
            rules.append("REF-08")
        if len(rec.get("authors", [])) > MAX_AUTHORS_BEFORE_ET_AL:
            rules.append("REF-03")
        if kind in ("journal", "conference") and abbreviate_venue(venue) != venue:
            rules.append("REF-06")
        if kind == "conference":
            rules.append("REF-07")
        if kind == "arxiv":
            rules.append("REF-09")
        if not str(rec.get("year", "")).strip():
            rules.append("REF-10")
        if rec.get("doi") or (rec.get("url") and kind in ("web", "other")):
            rules.append("REF-12")
        receipt(paper, rules, f"Reference [{n}]", rec.get("raw", ""), text,
                anchor=f"reference:{n}")


__all__ = ["compose", "make_llm", "BaseLLM", "FakeLLM", "NoLLM", "LLMUnavailable"]
