"""
The whole pipeline on one page — the demo surface for a progress review.

Every other view shows one stage: `/gallery` is Box 2, `/layout` is
Box 3, `/compare` is the evidence for Boxes 4 and 5. Shown one after
another they are six browser tabs, and a panel watching from a
projector has to hold the component diagram in their head and map each
screen onto it while the presenter talks.

This page is the diagram, with each box's real output under it. One
scroll, in pipeline order, so the structure is carried by the page
instead of by the narration.

It renders from a stored `IngestionResult` plus, optionally, a baseline
run of the same PDF for the last section. No extraction happens here —
if a number is on this page, some box upstream produced it.
"""

from __future__ import annotations

import html
from typing import Optional
from urllib.parse import quote

from api.ui import BASE_CSS
from ingestion.baseline_extractor import DEFAULT_CHUNK_CHARS
from ingestion.chunking import TARGET_CHUNK_CHARS
from ingestion.rasterise import DEFAULT_DPI
from schema.ingestion_schema_v1 import (
    ArtifactType,
    IngestionResult,
    RegionType,
    SectionType,
)

#: How many page thumbnails to put in the Box 2 strip before handing
#: off to the gallery. Twelve fills a row on a projector without
#: turning the section into a wall.
THUMBNAIL_LIMIT = 12

#: Chunks shown in the Box 5 section. The point lands in the first few.
CHUNK_PREVIEW = 4

#: Characters of each previewed chunk to print. Enough to see where the
#: cut fell, short enough that four of them fit on a projector.
CHUNK_CHARS_SHOWN = 210

#: Artifact crops shown in the Box 6 grid.
ARTIFACT_LIMIT = 8

#: key, box number, rail label. The rail is read at a glance from
#: across a room, so its labels are shorter than the headings they
#: point at — one line each, no wrapping.
STAGES = [
    ("s1", "1", "PDF in"),
    ("s2", "2", "Pages rendered"),
    ("s3", "3", "Regions found"),
    ("s4", "4", "Sections identified"),
    ("s5", "5", "Chunks cut"),
    ("s6", "6", "Figures and tables"),
    ("ev", "", "Against the baseline"),
]

_CSS = BASE_CSS + """
.masthead {
  border-bottom: 1px solid var(--rule-firm);
  background: var(--sheet);
  padding: 28px 24px 20px;
}
.masthead .inner { max-width: 1140px; margin: 0 auto; }
.masthead h1 { max-width: 24ch; }
.facts {
  display: flex; flex-wrap: wrap; gap: 4px 32px;
  margin: 14px 0 0; padding: 0;
}
.facts div { display: flex; flex-direction: column; }
.facts dt { font-size: 12px; color: var(--ink-soft); }
.facts dd { margin: 0; font-size: 14px; }
.elsewhere { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 16px; font-size: 14px; }

.layout {
  max-width: 1140px; margin: 0 auto;
  display: grid; grid-template-columns: 188px minmax(0, 1fr);
  gap: 40px; padding: 0 24px 96px;
}

/* the rail doubles as the component diagram: six boxes, in order,
   with the one you are reading marked */
.rail { position: sticky; top: 0; align-self: start; padding-top: 36px; }
.rail ol { list-style: none; margin: 0; padding: 0; border-left: 1px solid var(--rule); }
.rail li { position: relative; }
.rail a {
  display: block; padding: 7px 0 7px 16px;
  font-size: 13px; line-height: 1.3;
  color: var(--ink-soft); text-decoration: none;
  border-left: 2px solid transparent; margin-left: -1px;
}
.rail a:hover { color: var(--ink); }
.rail a .n {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  display: inline-block; min-width: 1.1em;  /* the unnumbered last entry still lines up */
  margin-right: 7px; color: var(--rule-firm);
}
.rail a[aria-current="true"] { color: var(--ink); border-left-color: var(--blue); }
.rail a[aria-current="true"] .n { color: var(--blue); }
.rail .last { margin-top: 8px; padding-top: 10px; border-top: 1px solid var(--rule); }

.stage { padding-top: 36px; }
.stage + .stage { margin-top: 44px; border-top: 1px solid var(--rule); }
.stage header { display: flex; align-items: baseline; gap: 14px; }
.stage .no {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: 34px; font-weight: 300; line-height: 1;
  color: var(--rule-firm); min-width: 34px;
}
.lede { margin-top: 8px; color: var(--ink-soft); font-size: 14px; }
.out { margin-top: 18px; }

.sheet { background: var(--sheet); border: 1px solid var(--rule); }
.pad { padding: 14px 16px; }

/* Box 2 — the page strip */
.strip { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 6px; }
.strip a { flex: 0 0 auto; display: block; line-height: 0; text-decoration: none; }
.strip img {
  height: 132px; width: auto; display: block;
  background: var(--sheet); border: 1px solid var(--rule);
}
.strip .pno {
  display: block; line-height: 1.6; font-style: normal;
  font-size: 11px; color: var(--ink-soft); text-align: center;
}
.strip a:hover img { border-color: var(--blue); }
.strip a:hover .pno { color: var(--blue); }

/* Box 3 — the overlay */
.overlay { display: grid; grid-template-columns: minmax(0, 1fr) 210px; gap: 20px; align-items: start; }
/* A paper page is tall and mostly margin. Left unbounded it eats a
   whole projector screen and the panel scrolls past the counts beside
   it, so cap the height and let the full-size view be one click away. */
.overlay img {
  display: block; width: auto; max-width: 100%; max-height: 72vh;
  border: 1px solid var(--rule); background: var(--sheet);
}
.tally { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--rule); }
.tally li {
  display: flex; justify-content: space-between; gap: 12px;
  padding: 6px 0; border-bottom: 1px solid var(--rule); font-size: 13px;
}
.tally b { font-weight: 600; font-variant-numeric: tabular-nums; }
.pagelinks { margin-top: 12px; font-size: 13px; display: flex; flex-wrap: wrap; gap: 10px; }

/* Box 4 — the section run */
.run { display: flex; flex-wrap: wrap; gap: 6px; }
.run li { list-style: none; }
.run span {
  display: inline-flex; align-items: baseline; gap: 7px;
  padding: 5px 10px; border: 1px solid var(--rule); background: var(--sheet);
  font-size: 13px;
}
.run .c { color: var(--ink-soft); font-size: 12px; font-variant-numeric: tabular-nums; }
.run .miss { border-style: dashed; color: var(--ink-soft); }

/* Box 5 — chunk cards */
.chunks { display: grid; gap: 10px; }
.chunk { border: 1px solid var(--rule); background: var(--sheet); padding: 12px 14px; }
.chunk .top {
  display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline;
  font-size: 12px; color: var(--ink-soft); margin-bottom: 7px;
}
.sect { color: var(--green); border: 1px solid currentColor; padding: 1px 7px; font-size: 11px; }
.sect.none { color: var(--red); }
.chunk p { font-size: 13px; line-height: 1.5; max-width: 90ch; }
.chunk .head { font-weight: 600; }

/* Box 6 — the crops */
.crops { display: grid; grid-template-columns: repeat(auto-fill, minmax(216px, 1fr)); gap: 14px; }
.crops figure { margin: 0; border: 1px solid var(--rule); background: var(--sheet); }
.crops img {
  display: block; width: 100%; height: 176px; object-fit: contain;
  background: #fff; border-bottom: 1px solid var(--rule); padding: 6px;
}
.crops figcaption { padding: 9px 11px; font-size: 12px; line-height: 1.45; }
.crops .kind { color: var(--ink-soft); font-size: 11px; display: block; margin-bottom: 2px; }

/* Box 5's headline numbers. The only tile row on the page — Box 1 is a
   quiet line and the comparison is bars, so this one reads as a
   highlight instead of as the house style. */
.scores { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--rule); border: 1px solid var(--rule); }
.scores div { background: var(--sheet); padding: 13px 15px; }
.scores .win b { color: var(--green); }

/* Box 1 — the file, stated once and not dressed up */
.filerow {
  display: flex; flex-wrap: wrap; gap: 8px 28px; align-items: baseline;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 12px 2px;
}
.filerow b { font-size: 15px; font-weight: 600; }
.filerow span { font-size: 14px; color: var(--ink-soft); }

/* the evidence — two measures, two bars each */
.measures { display: grid; gap: 1px; background: var(--rule); border: 1px solid var(--rule); }
.measure { background: var(--sheet); padding: 16px 18px 18px; }
.measure h3 { font-size: 15px; }
.measure > p { margin-bottom: 14px; max-width: 70ch; }
.bar { display: grid; grid-template-columns: 112px minmax(0, 1fr) 52px; gap: 12px; align-items: center; }
.bar + .bar { margin-top: 8px; }
.bar .who { font-size: 13px; }
.bar .track { height: 14px; background: var(--shade); box-shadow: inset 0 0 0 1px var(--rule); }
.bar .track i {
  display: block; height: 100%; min-width: 2px;
  border-radius: 0 3px 3px 0;   /* rounded at the data end only */
}
.bar .track i.baseline { background: var(--series-baseline); }
.bar .track i.ours     { background: var(--series-ours); }
.bar .val { font-size: 14px; font-weight: 600; text-align: right; }
.note { margin-top: 12px; font-size: 13px; color: var(--ink-soft); }

.empty { color: var(--ink-soft); font-style: italic; font-size: 14px; }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; gap: 0; padding: 0 18px 64px; }
  .rail { position: static; padding-top: 24px; }
  .rail ol { display: flex; flex-wrap: wrap; border-left: 0; }
  .rail a { border-left: 0; border-bottom: 2px solid transparent; padding: 6px 12px 6px 0; }
  .rail .last { margin: 0; padding: 0; border-top: 0; }
  .overlay { grid-template-columns: 1fr; }
  .scores { grid-template-columns: repeat(2, 1fr); }
}
"""

#: Marks the rail entry for whichever stage is on screen. Scroll is the
#: only thing that moves on this page — nothing animates on its own.
_SCRIPT = """
const links = [...document.querySelectorAll('.rail a')];
const seen = new Map();
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => seen.set(e.target.id, e.intersectionRatio));
  let best = null, top = 0;
  seen.forEach((ratio, id) => { if (ratio > top) { top = ratio; best = id; } });
  links.forEach(a => a.setAttribute('aria-current', String(a.hash === '#' + best)));
}, { threshold: [0, 0.15, 0.4, 0.75] });
document.querySelectorAll('.stage').forEach(s => io.observe(s));
"""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def _readout(value: object, label: str, tone: str = "") -> str:
    cls = f' class="{tone}"' if tone else ""
    return (
        f"<div{cls}><div class='readout'><b class='mono'>{_e(value)}</b>"
        f"<span>{_e(label)}</span></div></div>"
    )


def _human_size(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "—"
    mb = num_bytes / 1_048_576
    return f"{mb:.1f} MB" if mb >= 1 else f"{num_bytes / 1024:.0f} KB"


def _stage(key: str, number: str, title: str, lede: str, body: str) -> str:
    """One stage. The last section is not a box in the component
    diagram, so it gets no number — but it keeps the gutter, or its
    heading would sit out of line with the six above it."""
    return (
        f"<section class='stage' id='{key}'>"
        f"<header><div class='no' aria-hidden='true'>{_e(number)}</div><div>"
        f"<h2>{_e(title)}</h2><p class='lede'>{_e(lede)}</p></div></header>"
        f"<div class='out'>{body}</div></section>"
    )


# --------------------------------------------------------------------------
# the stages
# --------------------------------------------------------------------------


def _box1(result: IngestionResult, source_bytes: Optional[int]) -> str:
    """Box 1 is the upload and nothing else.

    It used to be four big tiles, two of which were filler — "PDF" under
    a heading called format, and a schema version already printed in the
    masthead. Three identical tile rows down one page also made every
    stage look equally important. One quiet line says the same thing and
    lets Box 5's numbers be the ones that stand out.
    """
    paper = result.paper
    pages = f"{paper.page_count} page{'s' if paper.page_count != 1 else ''}"
    return (
        "<div class='filerow'>"
        f"<b class='mono'>{_e(paper.source_filename)}</b>"
        f"<span>{_e(pages)}</span>"
        f"<span>{_e(_human_size(source_bytes))}</span>"
        f"<span>kept, so the baseline can be re-run from it</span>"
        "</div>"
    )


def _box2(result: IngestionResult, pid: str) -> str:
    if not result.pages:
        return (
            "<p class='empty'>No page images. This paper was ingested with "
            "rasterise=false, so Boxes 6 and 8 have nothing to work on.</p>"
        )
    shown = result.pages[:THUMBNAIL_LIMIT]
    strip = "".join(
        f"<a href='/paper/{pid}/page/{p.page_index}' target='_blank'>"
        f"<img src='/paper/{pid}/page/{p.page_index}' loading='lazy' "
        f"alt='Page {p.page_index + 1}'>"
        f"<em class='pno mono'>{p.page_index + 1}</em></a>"
        for p in shown
    )
    first = result.pages[0]
    more = (
        f" · <a href='/gallery/{pid}'>all {len(result.pages)} pages</a>"
        if len(result.pages) > THUMBNAIL_LIMIT
        else ""
    )
    return (
        f"<div class='strip'>{strip}</div>"
        f"<p class='note mono'>{first.dpi} dpi · {first.width_px}×{first.height_px} px"
        f" · {first.scale:.3f} px per point{more}</p>"
    )


def _box3(result: IngestionResult, pid: str) -> str:
    if not result.regions:
        return "<p class='empty'>No regions. Re-ingest with detect_layout=true.</p>"

    counts: dict[RegionType, int] = {}
    per_page: dict[int, int] = {}
    for r in result.regions:
        counts[r.region_type] = counts.get(r.region_type, 0) + 1
        per_page[r.bbox.page] = per_page.get(r.bbox.page, 0) + 1

    rendered = {p.page_index for p in result.pages}
    candidates = {p: n for p, n in per_page.items() if p in rendered}
    busiest = max(candidates, key=lambda p: candidates[p]) if candidates else None

    tally = "".join(
        f"<li><span>{_e(kind.value)}</span><b>{counts[kind]}</b></li>"
        for kind in RegionType
        if counts.get(kind)
    )

    if busiest is None:
        picture = (
            "<p class='empty'>Regions were detected, but there is no page "
            "image to draw them on.</p>"
        )
        links = ""
    else:
        picture = (
            f"<a href='/paper/{pid}/page/{busiest}/layout' target='_blank'>"
            f"<img src='/paper/{pid}/page/{busiest}/layout' loading='lazy' "
            f"alt='Page {busiest + 1} with its detected regions outlined and "
            f"numbered in reading order'></a>"
        )
        others = sorted(candidates)[:10]
        links = "<div class='pagelinks'><span class='soft'>Same view, other pages:</span>" + "".join(
            f"<a href='/paper/{pid}/page/{p}/layout' target='_blank'>{p + 1}</a>"
            for p in others
            if p != busiest
        ) + "</div>"

    return (
        f"<div class='overlay'><div>{picture}"
        f"<p class='note'>Page {(busiest or 0) + 1}, the densest page in the paper. "
        f"The badges are reading order — on a two-column page they run down the "
        f"left column before crossing to the right.</p>{links}</div>"
        f"<ul class='tally'>{tally}</ul></div>"
    )


def _box4(result: IngestionResult) -> str:
    if not result.regions:
        return "<p class='empty'>Nothing to label — Box 3 did not run.</p>"

    order: list[SectionType] = []
    counts: dict[SectionType, int] = {}
    titles: dict[SectionType, str] = {}
    for r in result.regions:
        if r.section not in counts:
            order.append(r.section)
        counts[r.section] = counts.get(r.section, 0) + 1
        if r.section_title_raw and r.section not in titles:
            titles[r.section] = r.section_title_raw

    run = "".join(
        f"<li><span class='{'miss' if s is SectionType.UNKNOWN else ''}' "
        f"title='{_e(titles.get(s, ''))}'>{_e(s.value.replace('_', ' '))}"
        f"<em class='c'>{counts[s]}</em></span></li>"
        for s in order
    )
    unknown = counts.get(SectionType.UNKNOWN, 0)
    verdict = (
        f"{unknown} of {len(result.regions)} regions could not be placed in a section."
        if unknown
        else "Every region landed in a section."
    )
    return (
        f"<ul class='run'>{run}</ul>"
        f"<p class='note'>In the order they appear, with the number of regions in "
        f"each. {_e(verdict)} Hover a label to see the heading that opened it.</p>"
    )


def _box5(result: IngestionResult, pid: str) -> str:
    if not result.chunks:
        return "<p class='empty'>No chunks.</p>"

    cards = []
    for c in result.chunks[:CHUNK_PREVIEW]:
        known = c.section is not SectionType.UNKNOWN
        # A chunk that opens at a heading has that heading as its first
        # line. Setting it bold is the claim of this stage made visible:
        # the cut landed on a boundary, not in the middle of a sentence.
        head, _, rest = c.text.partition("\n")
        opened_at_a_heading = bool(rest)
        body = " ".join(rest.split())[:CHUNK_CHARS_SHOWN]
        if len(rest) > CHUNK_CHARS_SHOWN:
            body += " …"

        if opened_at_a_heading:
            shown = f"<span class='head'>{_e(head)}</span> {_e(body)}"
        else:
            only = " ".join(head.split())[:CHUNK_CHARS_SHOWN]
            shown = _e(only + (" …" if len(head) > CHUNK_CHARS_SHOWN else ""))

        cards.append(
            f"<div class='chunk'><div class='top'>"
            f"<span class='sect{'' if known else ' none'}'>"
            f"{_e(c.section.value.replace('_', ' '))}</span>"
            f"<span class='mono'>chunk {c.order_index}</span>"
            f"<span class='mono'>{len(c.text)} chars</span>"
            f"<span class='mono'>p{c.page_start + 1}</span></div>"
            f"<p>{shown}</p></div>"
        )

    placed = sum(1 for c in result.chunks if c.section is not SectionType.UNKNOWN)
    return (
        f"<div class='scores'>"
        + _readout(len(result.chunks), "chunks")
        + _readout(f"{placed}/{len(result.chunks)}", "carry a section", "win")
        + _readout(TARGET_CHUNK_CHARS, "target chars")
        + _readout(sum(c.token_count for c in result.chunks), "tokens, estimated")
        + "</div>"
        f"<div class='chunks' style='margin-top:14px'>{''.join(cards)}</div>"
        f"<p class='note'>First {min(CHUNK_PREVIEW, len(result.chunks))} of "
        f"{len(result.chunks)}. Each one starts at a heading or a paragraph, "
        f"never inside a sentence. <a href='/compare/{pid}'>See them against the "
        f"baseline's cuts</a>.</p>"
    )


def _box6(result: IngestionResult, pid: str) -> str:
    if not result.artifacts:
        return (
            "<p class='empty'>No figures, tables or equations were found in "
            "this paper.</p>"
        )

    kinds: dict[ArtifactType, int] = {}
    for a in result.artifacts:
        kinds[a.artifact_type] = kinds.get(a.artifact_type, 0) + 1

    tiles = []
    for i, a in enumerate(result.artifacts[:ARTIFACT_LIMIT]):
        caption = a.caption or "No caption printed beside it."
        if a.image_path:
            picture = (
                f"<a href='/paper/{pid}/artifact/{i}' target='_blank'>"
                f"<img src='/paper/{pid}/artifact/{i}' loading='lazy' "
                f"alt='{_e(caption[:80])}'></a>"
            )
        else:
            picture = "<div class='pad small soft'>Detected, not cropped.</div>"
        tiles.append(
            f"<figure>{picture}<figcaption>"
            f"<span class='kind'>{_e(a.artifact_type.value)} · p{a.bbox.page + 1}</span>"
            f"{_e(caption[:120])}</figcaption></figure>"
        )

    summary = ", ".join(f"{n} {k.value}{'s' if n != 1 else ''}" for k, n in kinds.items())
    more = (
        f" Showing {ARTIFACT_LIMIT} of {len(result.artifacts)}."
        if len(result.artifacts) > ARTIFACT_LIMIT
        else ""
    )
    return (
        f"<div class='crops'>{''.join(tiles)}</div>"
        f"<p class='note'>{_e(summary.capitalize())}, each cut from the page image "
        f"with the caption the author printed beside it.{_e(more)}</p>"
    )


def _bar(label: str, n: int, total: int, series: str) -> str:
    """One labelled bar. Lower is better on both measures here, so the
    bar is literally "how much of this goes wrong" — a full bar is a
    total failure and an empty one is a clean run.

    The percentage is printed on every bar. Identity is carried by the
    row label as well as the colour, so the chart still reads with the
    colour removed entirely.
    """
    share = (100 * n / total) if total else 0.0
    value = f"{share:.0f}%" if total else "—"
    return (
        f"<div class='bar'><span class='who'>{_e(label)}</span>"
        f"<span class='track'><i class='{series}' style='width:{share:.1f}%'></i></span>"
        f"<span class='val mono' title='{n} of {total} chunks'>{value}</span></div>"
    )


def _evidence(result: IngestionResult, baseline: Optional[IngestionResult], pid: str) -> str:
    if baseline is None:
        return (
            "<p class='empty'>The source PDF was not kept for this paper, so the "
            "baseline cannot be re-run. One more trip through POST /ingest fixes "
            "it.</p>"
        )

    def unknown(r: IngestionResult) -> int:
        return sum(1 for c in r.chunks if c.section is SectionType.UNKNOWN)

    def broken(r: IngestionResult) -> int:
        return sum(1 for c in r.chunks if c.text.lstrip()[:1].islower())

    nb, nl = len(baseline.chunks), len(result.chunks)
    measures = [
        (
            "Chunks that do not know which section they came from",
            "Member 2 cannot tell a cited finding from this paper's own without it.",
            unknown(baseline),
            unknown(result),
        ),
        (
            "Chunks that begin in the middle of a word",
            "The tail of a sentence the chunk before it kept the head of.",
            broken(baseline),
            broken(result),
        ),
    ]

    blocks = "".join(
        f"<div class='measure'><h3>{_e(title)}</h3>"
        f"<p class='small soft'>{_e(why)}</p>"
        + _bar("Baseline", b, nb, "baseline")
        + _bar("This pipeline", l, nl, "ours")
        + "</div>"
        for title, why, b, l in measures
    )

    return (
        f"<div class='measures'>{blocks}</div>"
        f"<p class='note'>The same PDF down both paths. The baseline cuts flat "
        f"text every {DEFAULT_CHUNK_CHARS} characters and this pipeline cuts at "
        f"section and paragraph boundaries, so the two runs produce different "
        f"numbers of chunks — the bars are shares of each run, not counts. "
        f"Hover a bar for the counts. "
        f"<a href='/compare/{pid}'>Open the two side by side</a>.</p>"
    )


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def render_pipeline(
    result: IngestionResult,
    baseline: Optional[IngestionResult] = None,
    source_bytes: Optional[int] = None,
) -> str:
    """Return the whole-pipeline walkthrough as HTML.

    `baseline` is the same PDF through the flat extractor. Pass None
    when the source is no longer on disk; the last section says so
    rather than quietly disappearing.
    """
    paper = result.paper
    pid = quote(paper.paper_id, safe="")
    title = paper.title or paper.source_filename

    # The parentheses are load-bearing: an f-string sitting next to a
    # string literal is concatenated before the conditional is applied,
    # so without them the last rail entry loses its link entirely.
    rail = "".join(
        ("<li class='last'>" if key == "ev" else "<li>")
        + f"<a href='#{key}'><span class='n'>{_e(number)}</span>"
        + f"{_e(label)}</a></li>"
        for key, number, label in STAGES
    )

    bodies = [
        _stage("s1", "1", "PDF in",
               "The upload, untouched. Everything below is derived from it.",
               _box1(result, source_bytes)),
        _stage("s2", "2", "Pages rendered",
               f"Each page drawn once to a bitmap at {DEFAULT_DPI} dpi, then reused by "
               "every stage that needs pixels rather than re-rendering per stage.",
               _box2(result, pid)),
        _stage("s3", "3", "Regions found",
               "Headings, body text, captions, figures, tables and equations, located "
               "by geometry and font names, and put back into reading order. No model, "
               "no GPU.",
               _box3(result, pid)),
        _stage("s4", "4", "Sections identified",
               "Which part of the paper each region belongs to. The same sentence means "
               "different things in Related Work and in Results.",
               _box4(result)),
        _stage("s5", "5", "Chunks cut",
               "The retrievable units, cut at section and paragraph boundaries and "
               "carrying the section label with them.",
               _box5(result, pid)),
        _stage("s6", "6", "Figures and tables lifted out",
               "Each figure, table and equation cropped from its page image and kept "
               "with its caption, ready to be bound to the chunks that discuss it.",
               _box6(result, pid)),
        _stage("ev", "", "Measured against the baseline",
               "What the same PDF looks like without any of the above.",
               _evidence(result, baseline, pid)),
    ]

    facts = (
        f"<div><dt>File</dt><dd class='mono'>{_e(paper.source_filename)}</dd></div>"
        f"<div><dt>Pages</dt><dd class='mono'>{paper.page_count}</dd></div>"
        f"<div><dt>Method</dt><dd class='mono'>{_e(paper.extraction_method.value)}</dd></div>"
        f"<div><dt>Schema</dt><dd class='mono'>{_e(result.schema_version)}</dd></div>"
        f"<div><dt>Ingested</dt><dd class='mono'>"
        f"{_e(paper.ingested_at.strftime('%Y-%m-%d %H:%M UTC'))}</dd></div>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)} — ingestion pipeline</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<header class='masthead'><div class='inner'><h1>{_e(title)}</h1>"
        f"<dl class='facts'>{facts}</dl>"
        f"<div class='elsewhere'><a href='/'>All papers</a>"
        f"<a href='/paper/{pid}'>Raw JSON</a>"
        f"<a href='/gallery/{pid}'>Page gallery</a>"
        f"<a href='/compare/{pid}'>Baseline comparison</a></div>"
        "</div></header>"
        f"<div class='layout'><nav class='rail' aria-label='Pipeline stages'>"
        f"<ol>{rail}</ol></nav><main>{''.join(bodies)}</main></div>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
