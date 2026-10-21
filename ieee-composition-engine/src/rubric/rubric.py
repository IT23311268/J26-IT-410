"""
rubric.py - a rubric-based IEEE evaluation of one paper.

One rubric line for every rule in the IEEE rulebook (src/rules/ieee_rules.py):

    applies    does this rule apply to THIS paper? (no tables -> table rules do not)
    state      what the engine found or did:
                 applied   the engine changed the paper to follow the rule
                 met       nothing needed changing - the paper already follows it
                 partly    measured: only partly followed (the reason says why)
                 not_met   measured: not followed, and a program cannot fix it safely
                 review    the author's judgement - a person must check it
                 na        does not apply to this paper
    evidence   the receipts: exactly where the rule was applied, before -> after
    suggested  whether the tick box starts ticked (applied or met)

A reviewer ticks the rules they confirm. The rubric score is

    ticked rules / rules that apply to this paper

shown next to the engine's automatic compliance score.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ..rules import ieee_rules as R
from .evidence import evidence_for

STATE_TEXT = {
    "applied": "Applied by the engine",
    "met": "Already followed",
    "partly": "Partly followed",
    "not_met": "Not followed",
    "review": "Check yourself",
    "na": "Does not apply",
}


@dataclass
class RubricItem:
    id: str
    group: str
    rule: str
    value: str
    status: str                  # rulebook status: enforced / checked / author
    source: str
    source_url: str
    applies: bool
    state: str
    state_text: str
    reason: str = ""
    evidence: List[dict] = field(default_factory=list)
    suggested: bool = False
    check: str = ""
    focus: str = ""              # where to look in the manuscript when there are no receipts


# ---------------------------------------------------------------------------
# What this paper contains - decides which rules apply
# ---------------------------------------------------------------------------

def paper_features(paper, is_package: bool = False) -> Dict[str, bool]:
    elements = [e for s in paper.sections for e in s.elements]
    captions = [e.text for e in elements if e.kind == "caption"]
    records = paper.reference_records or []
    from ..compose.references import reference_type

    kinds = [reference_type(r) for r in records]
    return {
        "authors": bool(paper.authors),
        "abstract": bool(paper.abstract),
        "keywords": bool(paper.keywords),
        "figures": any(re.match(r"(?i)fig", c) for c in captions),
        "tables": any(re.match(r"(?i)table", c) for c in captions),
        "visuals": bool(captions),
        "equations": any(e.kind in ("equation", "math_region") for e in elements),
        "references": bool(paper.references),
        "citations": bool(paper.references) and any(
            re.search(r"\[\d+\]", e.text) for e in elements if e.kind == "paragraph"),
        "level2": any(s.level == 2 for s in paper.sections),
        "level3": any(s.level == 3 for s in paper.sections),
        "level4": any(s.level >= 4 for s in paper.sections),
        "appendix": any(s.ieee_slot == "appendix" for s in paper.sections),
        "ack": any(s.ieee_slot == "acknowledgment" for s in paper.sections),
        "back_matter": any(s.ieee_slot in ("appendix", "acknowledgment") for s in paper.sections),
        "many_authors_ref": any(len(r.get("authors", [])) > R.MAX_AUTHORS_BEFORE_ET_AL
                                for r in records),
        "journal_or_conf": any(k in ("journal", "conference") for k in kinds),
        "conference": "conference" in kinds,
        "journal": "journal" in kinds,
        "arxiv": "arxiv" in kinds,
        "online": any(r.get("doi") or r.get("url") for r in records),
        "pdf": not is_package,
    }


# which feature a rule needs before it applies (rules not listed always apply)
NEEDS: Dict[str, str] = {
    "TTL-03": "authors", "TTL-05": "authors", "TTL-06": "authors", "TTL-07": "authors",
    "ABS-01": "abstract", "ABS-02": "abstract", "ABS-03": "abstract", "ABS-04": "abstract",
    "ABS-05": "abstract", "ABS-06": "keywords", "ABS-07": "keywords", "ABS-08": "keywords",
    "HDG-02": "level2", "HDG-03": "level3", "HDG-04": "level4", "HDG-08": "appendix",
    "ACK-01": "ack", "ACK-02": "back_matter", "ACK-03": "ack",
    "REF-03": "many_authors_ref", "REF-06": "journal_or_conf", "REF-07": "conference",
    "REF-08": "journal", "REF-09": "arxiv", "REF-12": "online",
    "FIG-05": "visuals", "FIG-06": "visuals",
}
GROUP_NEEDS = {"Figures": "figures", "Tables": "tables", "Equations": "equations",
               "Citations": "citations", "References": "references"}


def _feature_for(rule: R.Rule) -> Optional[str]:
    return NEEDS.get(rule.id) or GROUP_NEEDS.get(rule.group)


# ---------------------------------------------------------------------------
# Building the rubric
# ---------------------------------------------------------------------------

def build_rubric(paper, compliance=None, is_package: bool = False) -> List[RubricItem]:
    features = paper_features(paper, is_package)
    checks = {c.key: c for c in (compliance.checks if compliance else [])}
    items: List[RubricItem] = []

    for rule in R.RULES:
        name, url = R.SOURCES[rule.source]
        source = f"{name} - {rule.section}" if rule.section else name
        evidence = evidence_for(paper, rule.id)
        need = _feature_for(rule)
        applies = need is None or features.get(need, False) or bool(evidence)
        check = checks.get(rule.check) if rule.check else None
        measured = check.after if check is not None else None

        if not applies:
            state, reason = "na", f"this paper has no {_feature_name(need)}"
        elif rule.status == R.AUTHOR:
            state, reason = "review", "needs a person to check it"
        elif measured is not None and measured < 1:
            state = "partly" if measured > 0 or evidence else "not_met"
            reason = check.after_reason
        elif evidence:
            state, reason = "applied", ""          # the places are listed under "Where"
        elif rule.status == R.CHECKED and measured is None:
            state, reason = "review", "could not be measured for this paper"
        else:
            state = "met"
            reason = check.after_reason if check is not None else \
                "nothing needed changing"

        # a measured rule with no receipts still shows what was measured, before -> after
        if check is not None and not evidence and state != "na":
            evidence = [{"rules": [rule.id], "where": CHECK_PLACE.get(rule.check, "Whole document"),
                         "before": check.before_reason or "", "after": check.after_reason or "",
                         "note": "measured by the compliance check",
                         "anchor": CHECK_ANCHOR.get(rule.check, "document")}]
        # whole-document receipts: show what the source had, from the same check
        elif check is not None and check.before_reason:
            evidence = [dict(e, before=check.before_reason) if not e["before"] else e
                        for e in evidence]

        items.append(RubricItem(
            id=rule.id, group=rule.group, rule=rule.rule, value=rule.value, status=rule.status,
            source=source, source_url=url, applies=applies, state=state,
            state_text=STATE_TEXT[state], reason=reason, evidence=evidence,
            suggested=state in ("applied", "met"), check=rule.check, focus=focus_of(rule)))
    return items


# where in the manuscript a measured rule lives (for the highlight)
CHECK_ANCHOR = {
    "page_size": "document", "margins": "document", "columns": "document", "font": "document",
    "body_size": "all:paragraph", "title_size": "front:title", "title_case": "front:title",
    "author_block": "front:authors", "abstract_label": "front:abstract",
    "abstract_length": "front:abstract", "abstract_clean": "front:abstract",
    "index_terms": "front:index_terms", "heading_numbering": "all:heading",
    "heading_names": "all:heading", "citation_style": "all:citation",
    "reference_style": "references", "reference_order": "references",
    "venue_abbreviation": "references", "caption_style": "all:caption",
    "visual_mentions": "all:mention", "visual_order": "all:mention", "placement": "all:visual",
    "number_style": "all:number", "acronyms": "all:acronym",
}

# where to look for a rule that has no receipts (already followed, or for a person to check)
FOCUS = {
    "LAY-09": "document", "TTL-03": "front:authors", "TTL-04": "front:title",
    "TTL-05": "front:authors", "TTL-07": "front:authors", "ABS-05": "front:abstract",
    "HDG-06": "references", "FIG-07": "all:figure", "FIG-08": "all:figure",
    "FIG-09": "all:figure", "FIG-10": "all:caption", "FIG-11": "all:caption",
    "FIG-12": "all:caption", "TAB-05": "all:table", "TAB-06": "all:table",
    "EQN-03": "all:equation", "EQN-04": "all:equation", "CIT-04": "all:citation",
    "CIT-05": "all:citation", "CIT-06": "all:citation", "REF-10": "references",
    "NUM-05": "all:paragraph", "NUM-06": "all:number", "NUM-07": "all:paragraph",
    "STY-01": "all:paragraph", "STY-03": "all:paragraph", "STY-04": "all:paragraph",
    "STY-05": "all:paragraph", "ACK-01": "slot:acknowledgment", "ACK-02": "slot:appendix",
    "ACK-03": "slot:acknowledgment",
}
GROUP_FOCUS = {
    "Page layout": "document", "Typography": "document", "Title and authors": "front:title",
    "Abstract and index terms": "front:abstract", "Headings": "all:heading",
    "Figures": "all:figure", "Tables": "all:table", "Equations": "all:equation",
    "Citations": "all:citation", "References": "references", "Numbers and units": "all:number",
    "Writing style": "all:paragraph", "Back matter": "slot:acknowledgment",
}


def focus_of(rule) -> str:
    return FOCUS.get(rule.id) or (CHECK_ANCHOR.get(rule.check) if rule.check else None) \
        or GROUP_FOCUS.get(rule.group, "document")


# where a measured rule lives in the paper, for its receipt
CHECK_PLACE = {
    "page_size": "Whole document", "margins": "Whole document", "columns": "Whole document",
    "font": "Whole document", "body_size": "Body paragraphs", "title_size": "Title",
    "title_case": "Title", "author_block": "Author block", "abstract_label": "Abstract",
    "abstract_length": "Abstract", "abstract_clean": "Abstract", "index_terms": "Index Terms",
    "heading_numbering": "Section headings", "heading_names": "Section headings",
    "citation_style": "All citations in the text", "reference_style": "Reference list",
    "reference_order": "Reference list", "venue_abbreviation": "Reference list",
    "caption_style": "All captions", "visual_mentions": "All figure and table mentions",
    "visual_order": "All figure and table mentions", "placement": "All figures and tables",
    "number_style": "All paragraphs", "acronyms": "Abstract and all paragraphs",
}


def _feature_name(feature: Optional[str]) -> str:
    return {"level2": "subsections", "level3": "level-3 headings", "level4": "level-4 headings",
            "ack": "acknowledgment", "back_matter": "appendix or acknowledgment",
            "many_authors_ref": "reference with more than six authors",
            "journal_or_conf": "journal or conference references", "online": "DOIs or links",
            "arxiv": "arXiv references", "visuals": "figures or tables",
            "keywords": "index terms"}.get(feature or "", feature or "")


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

def rubric_score(items: List[RubricItem], ticks: Optional[Dict[str, bool]] = None
                 ) -> Tuple[int, int, Optional[float]]:
    """(ticked, applicable, percent). Without ticks, the suggested ticks are used."""
    ticks = ticks or {}
    applicable = [i for i in items if i.applies]
    ticked = sum(1 for i in applicable if ticks.get(i.id, i.suggested))
    pct = round(100 * ticked / len(applicable), 1) if applicable else None
    return ticked, len(applicable), pct


def summary(items: List[RubricItem]) -> Dict[str, int]:
    out = {k: 0 for k in STATE_TEXT}
    for i in items:
        out[i.state] += 1
    return out


def save_rubric(items: List[RubricItem], path: str, ticks: Optional[Dict[str, bool]] = None) -> None:
    ticked, applicable, pct = rubric_score(items, ticks)
    data = {"score": {"ticked": ticked, "applicable": applicable, "percent": pct},
            "summary": summary(items),
            "items": [asdict(i) | {"ticked": (ticks or {}).get(i.id, i.suggested)} for i in items]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
