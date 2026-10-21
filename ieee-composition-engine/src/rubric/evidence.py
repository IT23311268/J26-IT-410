"""
evidence.py - a "receipt" for every IEEE rule the engine applies.

Every time a stage changes the paper to follow a rule, it calls record():

    record(paper, ["FIG-03"], where="Section III-A, paragraph 2",
           before="Figure 1", after="Fig. 1", anchor=anchor_of(element))

The receipts are kept in paper.evidence. The rubric (rubric.py) shows them
next to each rule, and the annotated Word file turns them into margin
comments at the exact place they were applied.

    rules    the rulebook IDs this change follows (src/rules/ieee_rules.py)
    where    a human-readable place: "Section II, paragraph 3", "Fig. 2 caption",
             "Reference [7]", "Title", "Whole document"
    before   the text as the author wrote it (shortened)
    after    the text after the rule was applied (shortened)
    anchor   where the annotated Word file puts the comment
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Optional, Union

_ids = itertools.count(1)
SNIPPET = 140


def short(text: str, limit: int = SNIPPET) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def anchor_of(item) -> str:
    """A stable id for an element or section, so its receipt can find it later."""
    if not getattr(item, "uid", ""):
        item.uid = f"u{next(_ids)}"
    kind = "section" if hasattr(item, "level") else "element"
    return f"{kind}:{item.uid}"


def record(paper, rules: Union[str, Iterable[str]], where: str, before: str = "",
           after: str = "", note: str = "", anchor: str = "document", **extra) -> None:
    """extra: 'related' (a second anchor, e.g. the paragraph that mentions a moved
    figure) and 'related_text' (the text to highlight there)."""
    rules = [rules] if isinstance(rules, str) else list(rules)
    paper.evidence.append({"rules": rules, "where": where, "before": short(before),
                           "after": short(after), "note": note, "anchor": anchor, **extra})


# ---------------------------------------------------------------------------
# Readable places
# ---------------------------------------------------------------------------

def section_label(paper, section) -> str:
    """'Section III-A (Dataset)', 'Acknowledgment', 'Opening text'."""
    if not section.heading and not section.ieee_heading:
        return "Opening text"
    parent = ""
    for s in paper.sections:
        if s.level == 1:
            parent = s.ieee_number or ""
        if s is section:
            break
    if section.level == 1:
        number = section.ieee_number
    elif section.level == 2 and parent and section.ieee_number:
        number = f"{parent}-{section.ieee_number}"
    else:
        number = ""
    name = section.ieee_heading or section.heading
    return f"Section {number} ({name})" if number else name


def where_of(paper, section, element=None) -> str:
    place = section_label(paper, section)
    if element is None:
        return f"{place}, heading"
    if element.kind == "caption":
        from ..placement.contextual import caption_label, display_label
        label = caption_label(element)
        return f"{display_label(*label)} caption" if label else f"{place}, caption"
    count = 0
    for e in section.elements:
        if e.kind == element.kind:
            count += 1
        if e is element:
            break
    kind = "paragraph" if element.kind == "paragraph" else element.kind
    return f"{place}, {kind} {count}"


def evidence_for(paper, rule_id: str) -> List[dict]:
    return [e for e in getattr(paper, "evidence", []) if rule_id in e["rules"]]


def comments_by_anchor(paper) -> dict:
    """anchor -> list of 'RULE: before -> after' lines, for the annotated Word file."""
    out: dict = {}
    for e in getattr(paper, "evidence", []):
        text = ", ".join(e["rules"]) + ": "
        if e["before"] or e["after"]:
            text += f"“{e['before']}” → “{e['after']}”" if e["before"] else e["after"]
        else:
            text += e["note"]
        if e["note"] and (e["before"] or e["after"]):
            text += f" ({e['note']})"
        anchor = e.get("anchor") or "document"
        if anchor.startswith(("all:", "slot:")):         # whole-paper rules: one comment at the top
            anchor = "document"
        out.setdefault(anchor, []).append(text)
    return out


def first(items: Optional[list]):
    return items[0] if items else None
