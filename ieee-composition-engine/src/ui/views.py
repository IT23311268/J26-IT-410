"""
views.py - small functions that turn the engine's results into HTML for the
web page (app.py).

They only build text: no Streamlit calls in here, so they are easy to test
(tests/test_app.py) and app.py stays short and readable.

Colours and fonts live in theme.css next to this file.
"""

from __future__ import annotations

import html
import io
import os
import zipfile
from typing import Dict, List, Optional

from ..placement.contextual import caption_label, display_label
from ..pipeline import STAGES
from ..rules import ieee_rules as R

THEME_CSS = os.path.join(os.path.dirname(__file__), "theme.css")


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def pct(value: Optional[float]) -> str:
    """0.941 -> '94%'.  None -> 'n/a'."""
    return "n/a" if value is None else f"{round(value * 100)}%"


def plural(n: int, word: str) -> str:
    """plural(1, 'section') -> '1 section'   plural(3, 'section') -> '3 sections'"""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def theme_css() -> str:
    with open(THEME_CSS, encoding="utf-8") as f:
        return f"<style>{f.read()}</style>"


# ---------------------------------------------------------------------------
# The stage list (shown while the engine runs, and again afterwards)
# ---------------------------------------------------------------------------

def stage_fact(stage: str, r) -> str:
    """One short line saying what a stage found."""
    paper = r.paper
    if stage == "ingest":
        if r.is_package:
            return f"Verified content package with {plural(len(paper.sections), 'section')}"
        d = r.source_doc
        return (f"{d.source_format.upper()}, {plural(d.page_count, 'page')}, "
                f"{plural(len(d.blocks), 'text line')}, body text {d.body_font_size} pt")
    if stage == "structure":
        captions = sum(1 for s in paper.sections for e in s.elements if e.kind == "caption")
        return (f"{plural(len(paper.sections), 'section')}, "
                f"{plural(len(paper.references), 'reference')}, "
                f"{plural(captions, 'figure or table caption')}")
    if stage == "map":
        top = [s for s in paper.sections if s.level == 1]
        missing = ", ".join(paper.missing_sections) if paper.missing_sections else "none"
        return f"{plural(len(top), 'top-level section')} matched; missing IEEE sections: {missing}"
    if stage == "compose":
        who = r.llm.name if getattr(r.llm, "available", False) else "Rules only (no Gemini)"
        return f"{who}; {len(paper.references)} references in IEEE style"
    if stage == "visuals":
        found = paper.visual_stats or {}
        summary = paper.placement_summary or {}
        bits = []
        if found.get("captions"):
            bits.append(f"{found['found']} of {found['captions']} visual areas found")
        if summary.get("visuals"):
            bits.append(f"{summary.get('moved', 0)} moved next to their first mention")
        return "; ".join(bits) or "No figures or tables in this paper"
    if stage == "render":
        from ..render import ieee_style as S
        return f"{S.PAPER.name}, two columns, {R.FONT}, IEEE margins"
    if stage == "score":
        c = r.compliance
        before = "n/a" if c.before_score is None else f"{c.before_score}%"
        return f"{before} before, {c.after_score}% after"
    return ""


def stage_list(done: Dict[str, str], running: Optional[str] = None,
               failed: Optional[str] = None) -> str:
    """The seven stages as a list. `done` maps stage -> fact line."""
    rows = []
    for number, (key, title) in enumerate(STAGES, start=1):
        if key in done:
            state, note = "done", done[key]
        elif key == failed:
            state, note = "failed", "Stopped here - see the message below"
        elif key == running:
            state, note = "running", "Working…"
        else:
            state, note = "waiting", ""
        rows.append(
            f'<li class="stage stage-{state}">'
            f'<span class="stage-mark" aria-hidden="true">{number}</span>'
            f'<span class="stage-body"><span class="stage-title">{esc(title)}</span>'
            f'<span class="stage-note">{esc(note)}</span></span></li>'
        )
    return f'<ol class="stages" aria-label="Engine stages">{"".join(rows)}</ol>'


def next_stage(key: str) -> Optional[str]:
    keys = [k for k, _ in STAGES]
    i = keys.index(key)
    return keys[i + 1] if i + 1 < len(keys) else None


# ---------------------------------------------------------------------------
# The headline: the compliance score as one track, before -> after
# ---------------------------------------------------------------------------

def score_track(before: Optional[float], after: float) -> str:
    """Scores are percentages (0-100) here, as ComplianceReport stores them."""
    a = max(0.0, min(100.0, after))
    if before is None:
        segment = ""
        before_marker = ""
        before_text = "Not measurable for a content package"
    else:
        b = max(0.0, min(100.0, before))
        lo, hi = min(a, b), max(a, b)
        segment = f'<span class="track-gain" style="left:{lo}%;width:{hi - lo}%"></span>'
        before_marker = (f'<span class="track-dot dot-before" style="left:{b}%" '
                         f'title="Before: {b}%"></span>')
        before_text = f"{before}%"
    return f"""
<section class="score" aria-label="IEEE compliance score">
  <div class="score-figures">
    <div class="score-side side-before">
      <span class="score-num">{esc(before_text)}</span>
      <span class="score-cap">The paper as uploaded</span>
    </div>
    <div class="score-side side-after">
      <span class="score-num">{after}%</span>
      <span class="score-cap">The IEEE manuscript</span>
    </div>
  </div>
  <div class="track" role="img" aria-label="Compliance rose from {esc(before_text)} to {after}%">
    <span class="track-rail"></span>{segment}{before_marker}
    <span class="track-dot dot-after" style="left:{a}%" title="After: {a}%"></span>
  </div>
  <div class="track-scale"><span>0%</span><span>50%</span><span>100%</span></div>
</section>"""


def headline_facts(r) -> str:
    """Three sentences under the score: what the panel will ask about."""
    paper = r.paper
    summary = paper.placement_summary or {}
    b, a = summary.get("before", {}), summary.get("after", {})
    facts = []
    if summary.get("visuals") and a.get("after_first_mention") is not None:
        facts.append(
            f"<b>{a.get('after_first_mention', 0)}%</b> of figures and tables now come after "
            f"the paragraph that first mentions them (was {b.get('after_first_mention', 0)}%).")
    if paper.references:
        facts.append(f"<b>{len(paper.references)}</b> references rewritten in IEEE style, "
                     f"cited in the text as [n].")
    exported = {k: v for k, v in (r.exported or {}).items() if v}
    if exported:
        parts = ", ".join(f"{v} {k if v != 1 else k[:-1]}" for k, v in exported.items())
        facts.append(f"<b>{parts}</b> copied from the source PDF as images.")
    if not facts:
        return ""
    return '<ul class="facts">' + "".join(f"<li>{f}</li>" for f in facts) + "</ul>"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def mapping_table(paper) -> str:
    rows = []
    for s in paper.sections:
        original = f"{s.number} {s.heading}".strip() or "(text before the first heading)"
        if s.level == 1:
            ieee = (f"{s.ieee_number}. " if s.ieee_number else "") + s.ieee_heading.upper()
            conf = s.mapping_confidence or 0
            sure = (f'<span class="conf"><span class="conf-bar" style="width:{round(conf * 100)}%">'
                    f'</span></span><span class="conf-num">{round(conf * 100)}%</span>')
            why = esc(s.mapping_reason)
        else:
            ieee = f"{s.ieee_number}. {s.ieee_heading}"
            sure, why = "", ""
        cls = "lvl1" if s.level == 1 else "lvl2"
        rows.append(f'<tr class="{cls}"><td>{esc(original)}</td><td class="ieee">{esc(ieee)}</td>'
                    f'<td class="sure">{sure}</td><td class="why">{why}</td></tr>')
    missing = ""
    if paper.missing_sections:
        missing = (f'<p class="note">Missing IEEE sections: '
                   f'{esc(", ".join(paper.missing_sections))}</p>')
    return (f'<table class="grid map"><thead><tr><th>Author wrote</th><th>IEEE heading</th>'
            f'<th>Confidence</th><th>Why</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'{missing}')


def compliance_rows(report) -> str:
    out, group = [], None
    for c in report.checks:
        if c.group != group:
            group = c.group
            out.append(f'<tr class="group"><th colspan="4">{esc(group)}</th></tr>')
        dots = ['<span class="mini-rail"></span>']
        if c.before is not None:
            dots.append(f'<span class="mini-dot dot-before" style="left:{c.before * 100}%"></span>')
        if c.after is not None:
            dots.append(f'<span class="mini-dot dot-after" style="left:{c.after * 100}%"></span>')
        weak = c.after is not None and c.after < 1
        reason = c.after_reason if c.after is not None else c.before_reason
        ids = " ".join(f'<span class="rid">{esc(i)}</span>' for i in getattr(c, "rules", []))
        unscored = "" if getattr(c, "scored", True) else ' <span class="unscored">not in the score</span>'
        out.append(
            f'<tr class="{"weak" if weak else ""}"><td class="check">{esc(c.name)}{unscored}'
            f'<span class="reason">{ids} {esc(reason)}</span></td>'
            f'<td class="mini"><div class="mini-box">{"".join(dots)}</div></td>'
            f'<td class="num before">{pct(c.before)}</td><td class="num after">{pct(c.after)}</td></tr>')
    before = "n/a" if report.before_score is None else f"{report.before_score}%"
    out.append(f'<tr class="total"><td>Overall</td><td class="mini"></td><td class="num before">{before}</td>'
               f'<td class="num after">{report.after_score}%</td></tr>')
    return (f'<table class="grid checks"><thead><tr><th>Check</th><th>0–100%</th>'
            f'<th class="num">Before</th><th class="num">After</th></tr></thead>'
            f'<tbody>{"".join(out)}</tbody></table>')


def visual_cards(paper) -> List[dict]:
    """Each placed figure/table with its image path and placement details."""
    images = {}
    for s in paper.sections:
        for e in s.elements:
            if e.kind == "caption":
                label = caption_label(e)
                if label:
                    images.setdefault(display_label(*label), (e.image, e.text))
    cards = []
    for entry in paper.placement_report or []:
        image, caption = images.get(entry["label"], (None, ""))
        cards.append({**entry, "image": image if image and os.path.exists(image) else None,
                      "caption": caption})
    return cards


def distance_text(entry: dict) -> str:
    b, a = entry.get("distance_before"), entry.get("distance_after")
    if b is None and a is None:
        return "Not mentioned in the text"
    show = lambda v: "–" if v is None else str(v).replace("-", "−")
    return f"Paragraphs from first mention: {show(b)} → {show(a)}"


def equation_images(paper) -> List[str]:
    return [e.image for s in paper.sections for e in s.elements
            if e.kind == "math_region" and e.image and os.path.exists(e.image)]


def reference_list(paper) -> str:
    records = paper.reference_records or []
    items = []
    for i, ref in enumerate(paper.references):
        record = records[i] if i < len(records) and len(records) == len(paper.references) else None
        if record and record.get("segments"):
            body = "".join(f"<i>{esc(t)}</i>" if italic else esc(t)
                           for t, italic in record["segments"])
        else:
            body = esc(ref)
        items.append(f"<li><span class='refno'>[{i + 1}]</span><span>{body}</span></li>")
    by_gemini = sum(1 for r in records if r.get("source") == "gemini")
    note = ""
    if records:
        note = (f'<p class="quiet">{by_gemini} read by Gemini and {len(records) - by_gemini} by '
                f'the rules, then written in IEEE style by the engine. Titles of journals and '
                f'proceedings are in italics, as IEEE requires.</p>')
    return f'{note}<ol class="refs">{"".join(items)}</ol>'


def bullet_list(lines: List[str]) -> str:
    return '<ul class="log">' + "".join(f"<li>{esc(x)}</li>" for x in lines) + "</ul>"


def rulebook_table() -> str:
    """Every rule in the rulebook, grouped, with its status and source."""
    label = {R.ENFORCED: "Applied", R.CHECKED: "Checked", R.AUTHOR: "Author"}
    out, group = [], None
    for r in R.RULES:
        if r.group != group:
            group = r.group
            out.append(f'<tr class="group"><th colspan="4">{esc(group)}</th></tr>')
        name, url = R.SOURCES[r.source]
        value = f'<span class="reason">{esc(r.value)}</span>' if r.value else ""
        source = f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(name)}</a>'
        if r.section:
            source += f'<span class="reason">{esc(r.section)}</span>'
        out.append(f'<tr><td class="rid-cell"><span class="rid">{esc(r.id)}</span></td>'
                   f'<td>{esc(r.rule)}{value}</td>'
                   f'<td><span class="status st-{r.status}">{label[r.status]}</span></td>'
                   f'<td class="src">{source}</td></tr>')
    return ('<table class="grid rules"><thead><tr><th>ID</th><th>Rule</th><th>Status</th>'
            f'<th>Source</th></tr></thead><tbody>{"".join(out)}</tbody></table>')


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def zip_folder(folder: str) -> Optional[bytes]:
    if not folder or not os.path.isdir(folder):
        return None
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".png"))
    if not names:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(folder, n), n)
    return buffer.getvalue()
