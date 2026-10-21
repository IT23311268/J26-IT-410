"""
rubric_tab.py - the "IEEE rubric" tab: a guided, rule-by-rule review.

Purpose: let the user SEE that the engine followed IEEE, and confirm it.

For every IEEE rule that applies to the paper, one at a time:
    left    the rule, what the engine did, and HOW it changed the text
            (before -> after) for the place being shown ("Place 3 of 19",
            with buttons to step through every place)
    right   the finished IEEE manuscript, with that rule's places
            highlighted and the current one scrolled into view
    then    "Yes, applied correctly" / "No" - and on to the next rule

At the end: the IEEE rubric-based evaluation score,
    rules confirmed "Yes" / rules that apply to this paper,
next to the engine's automatic compliance score, with the filled rubric
(Excel) and the Word file with every rule as a margin comment.
"""

from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components

from ..render import ieee_style as S
from ..rubric.export import rubric_workbook
from ..rubric.rubric import rubric_score
from .preview import build_preview, preview_images
from .views import esc

PREVIEW_HEIGHT = 860
MAX_MARKS = 400            # highlights per rule (a rule with 53 references marks all 53)

# which part of a reference a reference rule is about
SEGMENT = {"REF-02": "authors", "REF-03": "authors", "REF-04": "title", "REF-05": "venue",
           "REF-06": "venue", "REF-07": "venue", "REF-08": "tail", "REF-09": "tail",
           "REF-10": "tail", "REF-11": "tail", "REF-12": "tail"}
# rules whose receipt describes the whole place, so the whole place is outlined
WHOLE_PLACE = {"ABS-02", "ABS-03", "ABS-04", "ABS-07", "ABS-08", "TTL-06", "FIG-05",
               "CIT-02", "REF-01", "ACK-02", "HDG-06"}


# ---------------------------------------------------------------------------
# Highlights for one rule
# ---------------------------------------------------------------------------

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


def change_html(e: dict) -> str:
    if e.get("before") and e.get("after"):
        return (f'<div class="chg"><span class="chg-label">Before</span>'
                f'<span class="ev-before">{esc(e["before"])}</span></div>'
                f'<div class="chg"><span class="chg-label">After</span>'
                f'<span class="ev-after">{esc(e["after"])}</span></div>')
    text = e.get("after") or e.get("before") or ""
    return f'<div class="chg"><span class="chg-label">Result</span><span class="ev-after">{esc(text)}</span></div>'


# ---------------------------------------------------------------------------
# State: one review per run
# ---------------------------------------------------------------------------

def _keys(run_id: int):
    return f"rub_dec_{run_id}", f"rub_idx_{run_id}", f"rub_place_{run_id}"


def _decide(run_id: int, rule_id: str, value: bool, total: int) -> None:
    dec, idx, place = _keys(run_id)
    st.session_state[dec][rule_id] = value
    st.session_state[idx] = min(st.session_state[idx] + 1, total)
    st.session_state[place] = 0


def _move(run_id: int, delta: int, total: int) -> None:
    _, idx, place = _keys(run_id)
    st.session_state[idx] = max(0, min(st.session_state[idx] + delta, total))
    st.session_state[place] = 0


def _place(run_id: int, delta: int, count: int) -> None:
    _, _, place = _keys(run_id)
    st.session_state[place] = (st.session_state[place] + delta) % max(1, count)


def _jump(run_id: int, ids: List[str]) -> None:
    _, idx, place = _keys(run_id)
    choice = st.session_state[f"rub_jump_{run_id}"]
    st.session_state[idx] = ids.index(choice.split(" ")[1])
    st.session_state[place] = 0


def _accept_rest(run_id: int, items) -> None:
    """Yes for every unreviewed rule the engine applied or found already followed.
    Rules for a person to check stay "Not reviewed" - nobody looked at them."""
    dec, idx, place = _keys(run_id)
    for i in items:
        if i.suggested:
            st.session_state[dec].setdefault(i.id, True)
    st.session_state[idx] = len(items)


def _restart(run_id: int) -> None:
    _, idx, place = _keys(run_id)
    st.session_state[idx] = 0
    st.session_state[place] = 0


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------

def render_rubric_tab(result, run_id: int) -> None:
    items = [i for i in result.rubric if i.applies]
    if not items:
        st.info("The rubric is built after the paper is composed.")
        return
    dec_key, idx_key, place_key = _keys(run_id)
    st.session_state.setdefault(dec_key, {})
    st.session_state.setdefault(idx_key, 0)
    st.session_state.setdefault(place_key, 0)
    images_key = f"rub_images_{run_id}"
    if images_key not in st.session_state:
        st.session_state[images_key] = preview_images(result.paper)

    decisions: Dict[str, bool] = st.session_state[dec_key]
    total = len(items)
    idx = st.session_state[idx_key]

    if idx >= total:
        _results(result, items, decisions, run_id)
        return

    item = items[idx]
    reviewed = len(decisions)
    yes = sum(1 for v in decisions.values() if v)
    st.progress(reviewed / total, text=f"Rule {idx + 1} of {total}  ·  {reviewed} reviewed, "
                                         f"{yes} confirmed")

    left, right = st.columns([0.4, 0.6], gap="medium")
    with left:
        _rule_card(item, run_id, total, decisions, items)
    with right:
        place = st.session_state[place_key] if item.evidence else 0
        e = item.evidence[place] if item.evidence else None
        banner = f"{item.id}: {e['where']}" if e else f"{item.id}: look at the highlighted part"
        if e and (e.get("anchor") or "document") == "document":
            banner = f"{item.id} applies to the whole page: {e.get('after') or e.get('before')}"
        page = build_preview(result.paper, highlights_for(item, place),
                             st.session_state[images_key], banner=banner)
        components.html(page, height=PREVIEW_HEIGHT, scrolling=True)
        st.markdown('<p class="quiet legend"><span class="sw sw-cur"></span> this place '
                    '<span class="sw sw-all"></span> the rule\'s other places '
                    '<span class="sw sw-rel"></span> the paragraph that mentions a moved '
                    'figure or table</p>', unsafe_allow_html=True)


def _rule_card(item, run_id: int, total: int, decisions: Dict[str, bool], items) -> None:
    dec_key, idx_key, place_key = _keys(run_id)
    decided = decisions.get(item.id)
    previous = "" if decided is None else \
        f'<span class="state {"st-applied" if decided else "st-not_met"}">You said: ' \
        f'{"Yes" if decided else "No"}</span>'
    value = f'<p class="rule-value">{esc(item.value)}</p>' if item.value else ""
    st.markdown(
        f'<div class="rule-card"><div class="rule-top"><span class="rid big">{esc(item.id)}</span>'
        f'<span class="rule-group">{esc(item.group)}</span>{previous}</div>'
        f'<p class="rule-big">{esc(item.rule)}</p>{value}'
        f'<p class="engine-says">Engine: <span class="state st-{item.state}">'
        f'{esc(item.state_text)}</span> <span class="quiet">{esc(item.reason)}</span></p></div>',
        unsafe_allow_html=True)

    if item.evidence:
        count = len(item.evidence)
        place = st.session_state[place_key] % count
        e = item.evidence[place]
        st.markdown(f'<p class="place-head">Place {place + 1} of {count}: '
                    f'<b>{esc(e["where"])}</b></p>{change_html(e)}'
                    + (f'<p class="quiet">How the engine decided: {esc(e["note"])}</p>'
                       if e.get("note") else ""),
                    unsafe_allow_html=True)
        if count > 1:
            a, b = st.columns(2)
            a.button("◀ Previous place", key=f"pp_{run_id}", on_click=_place,
                     args=(run_id, -1, count), width="stretch")
            b.button("Next place ▶", key=f"np_{run_id}", on_click=_place,
                     args=(run_id, 1, count), width="stretch")
    elif item.state == "review":
        st.markdown('<p class="place-head">The engine cannot judge this rule. Look at the '
                    'highlighted part of the manuscript and decide.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="place-head">Nothing needed changing: the paper already follows '
                    'this rule. The related part is highlighted.</p>', unsafe_allow_html=True)

    st.markdown(f'<p class="question">Is this rule applied correctly in the IEEE manuscript?</p>',
                unsafe_allow_html=True)
    y, n = st.columns(2)
    y.button("✔ Yes, applied", type="primary", key=f"yes_{run_id}", on_click=_decide,
             args=(run_id, item.id, True, total), width="stretch")
    n.button("✘ No", key=f"no_{run_id}", on_click=_decide,
             args=(run_id, item.id, False, total), width="stretch")
    p, s = st.columns(2)
    p.button("◀ Previous rule", key=f"prev_{run_id}", on_click=_move, args=(run_id, -1, total),
             width="stretch", disabled=st.session_state[idx_key] == 0)
    s.button("Skip ▶", key=f"skip_{run_id}", on_click=_move, args=(run_id, 1, total),
             width="stretch")

    ids = [i.id for i in items]
    labels = [f"{'✔' if decisions.get(i.id) else '✘' if i.id in decisions else '·'} {i.id} "
              f"{i.rule[:60]}" for i in items]
    st.selectbox("Go to rule", labels, index=st.session_state[idx_key],
                 key=f"rub_jump_{run_id}", on_change=_jump, args=(run_id, ids))
    with st.expander("Finish quickly"):
        st.markdown('<p class="quiet">Answer Yes for every rule you have not reviewed that the '
                    'engine applied or found already followed, then see the score. Rules a person '
                    'must check stay "Not reviewed" and do not count as confirmed.</p>',
                    unsafe_allow_html=True)
        st.button("Accept suggestions for the rest and finish", key=f"acc_{run_id}",
                  on_click=_accept_rest, args=(run_id, items))
    st.markdown(f'<p class="quiet">Source: <a href="{esc(item.source_url)}" target="_blank" '
                f'rel="noopener">{esc(item.source)}</a></p>', unsafe_allow_html=True)


def _results(result, items, decisions: Dict[str, bool], run_id: int) -> None:
    ticks = {i.id: bool(decisions.get(i.id, False)) for i in result.rubric}
    yes, applicable, pct = rubric_score(result.rubric, ticks)
    no = sum(1 for i in items if decisions.get(i.id) is False)
    unreviewed = sum(1 for i in items if i.id not in decisions)
    engine = result.compliance.after_score if result.compliance else None
    na = sum(1 for i in result.rubric if not i.applies)

    st.markdown(
        f'<section class="rubric-final"><span class="score-cap">IEEE rubric-based evaluation score</span>'
        f'<span class="rubric-num">{pct if pct is not None else "n/a"}%</span>'
        f'<span class="score-cap">{yes} of {applicable} rules confirmed as applied correctly'
        f'{f" - {no} marked No" if no else ""}'
        f'{f" - {unreviewed} not reviewed (counted as not confirmed)" if unreviewed else ""}.'
        f' {na} rules do not apply to this paper. Engine compliance score: {engine}%.</span>'
        f'</section>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download rubric (.xlsx)",
        rubric_workbook(result.rubric, ticks, result.paper.title, engine, reviewed=decisions),
        file_name=f"{result.stem}_IEEE_rubric.xlsx", type="primary", width="stretch",
        on_click="ignore",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if result.annotated_path and os.path.exists(result.annotated_path):
        with open(result.annotated_path, "rb") as f:
            c2.download_button(
                "Word file with rule comments", f.read(),
                file_name=os.path.basename(result.annotated_path), width="stretch",
                on_click="ignore",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    c3.button("Review again from the first rule", on_click=_restart, args=(run_id,),
              width="stretch")

    rows = []
    for i in items:
        d = decisions.get(i.id)
        mark = '<span class="state st-applied">Yes</span>' if d else \
            '<span class="state st-not_met">No</span>' if d is False else \
            '<span class="state st-na">Not reviewed</span>'
        rows.append(f'<tr><td class="rid-cell"><span class="rid">{esc(i.id)}</span></td>'
                    f'<td>{esc(i.rule)}</td><td>{esc(i.state_text)}</td><td>{mark}</td></tr>')
    st.markdown('<table class="grid rules"><thead><tr><th>ID</th><th>Rule</th><th>Engine</th>'
                f'<th>Your answer</th></tr></thead><tbody>{"".join(rows)}</tbody></table>',
                unsafe_allow_html=True)
