"""
highlights.py - which exact words of the manuscript to highlight for a rule.

Used by the rubric review in both front ends (the React page through
server.py, and the Streamlit page). No Streamlit or web code in here.
"""

from __future__ import annotations

from typing import List

from ..render import ieee_style as S

MAX_MARKS = 400            # highlights per rule (a rule with 53 references marks all 53)

# which part of a reference a reference rule is about
SEGMENT = {"REF-02": "authors", "REF-03": "authors", "REF-04": "title", "REF-05": "venue",
           "REF-06": "venue", "REF-07": "venue", "REF-08": "tail", "REF-09": "tail",
           "REF-10": "tail", "REF-11": "tail", "REF-12": "tail"}
# rules whose receipt describes the whole place, so the whole place is outlined
WHOLE_PLACE = {"ABS-02", "ABS-03", "ABS-04", "ABS-07", "ABS-08", "TTL-06", "FIG-05",
               "CIT-02", "REF-01", "ACK-02", "HDG-06"}


def mark_text(rule_id: str, evidence: dict) -> str:
    """The exact words to highlight in the manuscript for this receipt."""
    after = evidence.get("after", "")
    if evidence.get("note") == "measured by the compliance check" or rule_id in WHOLE_PLACE:
        return ""
    if rule_id == "ABS-01":
        return S.ABSTRACT_LABEL
    if rule_id == "ABS-06":
        return S.INDEX_TERMS_LABEL
    if rule_id.startswith("TAB-"):
        return after.split(" / ")[0]                 # "TABLE II"
    if rule_id == "EQN-01":
        return after.split(" ")[0]                   # "(6)"
    if rule_id.startswith(("REF-", "LAY-", "TYP-")):
        return ""
    return after


def highlights_for(item, place: int) -> List[dict]:
    if not item.evidence:
        return [{"anchor": item.focus or "document", "text": "", "current": True}]
    out = []
    for k, e in enumerate(item.evidence[:MAX_MARKS]):
        out.append({"anchor": e.get("anchor") or "document", "text": mark_text(item.id, e),
                    "segment": SEGMENT.get(item.id, ""), "current": k == place,
                    "related": e.get("related"), "related_text": e.get("related_text", "")})
    return out
