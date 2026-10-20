"""
Baseline against layout-aware, side by side.

Pure presentation: it takes two IngestionResults and returns an HTML
string. No file or network access, which keeps it trivially testable.

Why this page exists
--------------------
Every other demo surface shows that the pipeline *runs*. This one shows
that it is *better than the alternative*, which is the proposal's claim
and the only thing a panel can actually assess. The two runs come out in
the same schema — `extraction_method` on the paper is what tells them
apart — so they can be put next to each other without any translation.

What to look at, in order:

  the section labels. Every baseline chunk is UNKNOWN, because flat text
  has no idea where it came from. "BM25 outperforms dense retrieval" in
  Related Work is somebody else's finding; in Results it is this paper's
  own, and Member 2 cannot tell those apart without the label.

  where the chunks start. The baseline cuts every N characters, so a cut
  lands wherever it lands — mid-word, mid-sentence. The layout-aware
  chunks start at a heading or at a paragraph.

This is a demo surface, not part of the Members 2/3/4 contract.
"""

from __future__ import annotations

import html

from api.ui import TOKENS
from ingestion.baseline_extractor import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP
from ingestion.chunking import TARGET_CHUNK_CHARS
from schema.ingestion_schema_v1 import Chunk, IngestionResult

_STYLES = TOKENS + """
/* This page predates the shared palette and names its colours after
   their role on it. Keep the names, take the values — three demo
   surfaces that disagree about what grey means look like three
   prototypes. */
:root {
  --bg: var(--paper);
  --surface: var(--sheet);
  --border: var(--rule);
  --text: var(--ink);
  --muted: var(--ink-soft);
  /* --blue and --green were never defined in TOKENS, so --accent and
     --good resolved to nothing and every .tag.known fell back to the
     grey it inherits from .meta. That is why the section labels on
     this page were all one colour. Point them at tokens that exist. */
  --accent: var(--series-ours);
  --bad: var(--red);
  --good: var(--brand-ink);
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px 16px 64px;
  background: var(--bg); color: var(--text);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 24px; font-size: 14px; }
.cols { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; align-items: start; }
@media (max-width: 820px) { .cols { grid-template-columns: 1fr; } }
.col > h2 { font-size: 15px; margin: 0 0 4px; }
.col > .note { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
.chunk {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
}
.meta {
  display: flex; gap: 8px; align-items: baseline;
  font-size: 12px; color: var(--muted); margin-bottom: 6px;
}
.tag {
  font-weight: 650; letter-spacing: .02em; text-transform: uppercase;
  font-size: 11px; padding: 2px 7px; border-radius: 999px;
  border: 1px solid currentColor;
}
/* UNKNOWN is the failure this page exists to show, so it is the one
   filled chip: white on #9c2b2b, 7.52:1 and 5.56:1 with the room
   lights on. Every baseline chunk carries it, so the left column
   reads as a column of red at a glance from the back of the room. */
.tag.unknown {
  color: #ffffff; background: var(--bad); border-color: var(--bad);
}

/* One hue per section, so the right column reads as a paper with
   parts rather than a list of grey boxes.

   Two numbers were solved rather than chosen.

   The lightness of each ink, per hue, against 5:1 on that chip's own
   wash with the projector's black floor added. A flat lightness
   passes in the blues and fails in the greens, because green is far
   lighter than blue at the same number — five of ten chips failed
   that way on the first pass.

   And the hues themselves. Spacing them evenly round the wheel looks
   even and is not: OKLab is crowded in the greens, and evenly spaced
   hues put introduction and method ΔE 4.3 apart, which is a guess at
   arm's length. These ten were searched for the set whose CLOSEST
   pair is furthest apart — ΔE 8.2 — and then ordered so the paper
   runs cool at the front, warm through the middle and violet at the
   back.

   Colour is still only ever reinforcement here: the chip prints the
   word too, so nothing is lost by anyone who cannot separate two of
   them. */
.tag.known { color: var(--good); background: var(--brand-wash); }
.tag.s-title        { color: #294aa6; background: #e8edfa; }
.tag.s-abstract     { color: #20587e; background: #e8f3fa; }
.tag.s-introduction { color: #196256; background: #e8faf7; }
.tag.s-method       { color: #196419; background: #e8fae8; }
.tag.s-results      { color: #615918; background: #faf8e8; }
.tag.s-discussion   { color: #804320; background: #faefe8; }
.tag.s-conclusion   { color: #77289f; background: #f4e8fa; }
.tag.s-related_work { color: #552fbe; background: #ede8fa; }
.tag.s-references   { color: #8d2374; background: #fae8f6; }
.tag.s-appendix     { color: #942543; background: #fae8ed; }
.tag.known, .tag[class*="s-"] { border-color: transparent; }
.text {
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; margin: 0;
}
.lead { font-weight: 650; }
.stats {
  display: grid; gap: 10px; grid-template-columns: repeat(4, 1fr);
  margin: 0 0 24px;
}
@media (max-width: 820px) { .stats { grid-template-columns: repeat(2, 1fr); } }
/* The page tells the reader in its own standfirst to compare the
   percentages and not the counts, and then printed the percentage in
   12px grey inside a bracket. The percentage is the figure now and
   the count is the small print under it.

   The two colours are the two the whole demo uses for these two
   paths — --series-baseline for the baseline, --series-ours for this
   component — so the colour here means what it means everywhere
   else, and a reader who has seen one chart already knows which
   column is which without reading the label. */
.stat {
  background: var(--surface); border: 1px solid var(--border);
  border-top: 3px solid var(--border);
  border-radius: 10px; padding: 12px 14px 13px;
}
.stat.base { border-top-color: var(--series-baseline); }
.stat.ours { border-top-color: var(--series-ours); }
.stat .pct {
  font-size: 34px; font-weight: 680; line-height: 1;
  letter-spacing: -0.02em; margin-bottom: 7px;
}
.stat.base .pct { color: var(--series-baseline); }
.stat.ours .pct { color: var(--series-ours); }
.stat .n { font-size: 13px; font-weight: 600; }
.stat .k { font-size: 12px; color: var(--muted); }
@media (max-width: 480px) { .stat .pct { font-size: 27px; } }
.empty { color: var(--muted); font-style: italic; }
"""


def _starts_mid_word(text: str) -> bool:
    """Does this chunk open in the middle of a word?

    The signature of fixed-size cutting: a chunk beginning with a lower
    case letter is the tail of a word the previous chunk kept the head
    of. Counting them is the cheapest honest measure of the damage.
    """
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].islower()


def _chunk_html(chunk: Chunk, limit: int = 260) -> str:
    section = chunk.section.value
    # "unknown" keeps the filled red chip; everything else gets both
    # the generic class and its own, so a section this page has no
    # colour for still lands on the readable default rather than on
    # whatever .meta happens to be.
    known = "unknown" if section == "unknown" else f"known s-{section}"
    text = chunk.text if len(chunk.text) <= limit else chunk.text[:limit] + " …"

    lead, _, rest = text.partition("\n")
    body = (
        f'<span class="lead">{html.escape(lead)}</span>'
        + (f"\n{html.escape(rest)}" if rest else "")
    )

    return (
        '<div class="chunk">'
        f'<div class="meta"><span class="tag {known}">{html.escape(section)}</span>'
        f"<span>chunk {chunk.order_index} · {len(chunk.text)} chars · "
        f"p{chunk.page_start}</span></div>"
        f'<p class="text">{body}</p>'
        "</div>"
    )


def _column(title: str, note: str, chunks: list[Chunk]) -> str:
    if not chunks:
        inner = '<p class="empty">No chunks.</p>'
    else:
        inner = "".join(_chunk_html(c) for c in chunks)
    return (
        f'<div class="col"><h2>{html.escape(title)}</h2>'
        f'<p class="note">{html.escape(note)}</p>{inner}</div>'
    )


def render_comparison(
    baseline: IngestionResult, layout: IngestionResult, limit: int = 12
) -> str:
    """Return the comparison page as HTML.

    `limit` caps how many chunks each side shows: the point lands in the
    first few, and a 100-page paper would otherwise render a wall.
    """
    b_chunks = baseline.chunks[:limit]
    l_chunks = layout.chunks[:limit]

    b_unknown = sum(1 for c in baseline.chunks if c.section.value == "unknown")
    l_unknown = sum(1 for c in layout.chunks if c.section.value == "unknown")
    b_broken = sum(1 for c in baseline.chunks if _starts_mid_word(c.text))
    l_broken = sum(1 for c in layout.chunks if _starts_mid_word(c.text))

    title = layout.paper.title or layout.paper.source_filename

    def stat(n: int, total: int, label: str, which: str) -> str:
        """`which` is "base" or "ours" — which of the two paths this
        figure measures, which is also what colours it."""
        pct = f"{(100 * n / total):.0f}%" if total else "—"
        return (
            f'<div class="stat {which}">'
            f'<div class="pct">{pct}</div>'
            f'<div class="n">{n} of {total}</div>'
            f'<div class="k">{html.escape(label)}</div></div>'
        )

    stats = (
        stat(b_unknown, len(baseline.chunks),
             "baseline chunks with no section", "base")
        + stat(l_unknown, len(layout.chunks),
               "layout-aware chunks with no section", "ours")
        + stat(b_broken, len(baseline.chunks),
               "baseline chunks starting mid-word", "base")
        + stat(l_broken, len(layout.chunks),
               "layout-aware chunks starting mid-word", "ours")
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Baseline vs layout-aware — {html.escape(title)}</title>"
        f"<style>{_STYLES}</style></head><body><div class='wrap'>"
        f"<h1>{html.escape(title)}</h1>"
        "<p class='sub'>The same PDF through both paths. Left is what you get "
        "without layout understanding; right is this component. Showing the "
        f"first {limit} chunks of each.<br>"
        "The two paths use different chunk sizes, so compare the "
        "<strong>percentages</strong>, not the counts: a fixed-size cut lands "
        "mid-word at about the same rate whatever size you choose, and flat "
        "text has no sections at any size.</p>"
        f'<div class="stats">{stats}</div>'
        '<div class="cols">'
        + _column(
            "Baseline",
            f"Flat text, cut every {DEFAULT_CHUNK_CHARS} characters with "
            f"{DEFAULT_CHUNK_OVERLAP} of overlap.",
            b_chunks,
        )
        + _column(
            "Layout-aware",
            f"Reading order recovered, cut at section and paragraph "
            f"boundaries, targeting {TARGET_CHUNK_CHARS} characters.",
            l_chunks,
        )
        + "</div></div></body></html>"
    )
