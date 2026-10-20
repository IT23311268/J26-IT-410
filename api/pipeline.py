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

from api.ui import BASE_CSS, BRAND, app_rail
from ingestion.linking import caption_label, reference_span
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

#: How many artifact-to-chunk links the walkthrough prints, and how
#: much of the citing passage it shows around the reference.
LINK_PREVIEW = 4
LINK_CHARS = 150

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
    ("s7", "7", "Artifacts linked"),
    ("ev", "", "Against the baseline"),
]

_CSS = BASE_CSS + """
/* The masthead is a drawing's title block: the subject on the left,
   the sheet's particulars ruled off on the right, and the result
   across the foot of it. It used to be four stacked bands of equal
   weight, which is a list of facts rather than a composition. */
/* Pipeline-page-only visual redesign. The shared api/ui.py is
   untouched, so the other three surfaces keep what they had. */
.brand { background: #16213e; border-bottom-color: #16213e; }
.brand b { color: #ffffff; }
.brand span { color: #d2dbee; }

.masthead {
  border-bottom: 1px solid #e2e8f0;
  background: #f7f9fc;
  padding: 36px 28px 0;
}
.masthead .inner { max-width: 1240px; margin: 0 auto; }

/* Title left, the card right. Stacking them left a column of empty
   page down the middle of the screen. */
.titleblock {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 370px);
  gap: 24px 36px;
  align-items: start;
}
.titleblock > div { min-width: 0; }
.masthead h1 { max-width: 24ch; color: #16213e; }
.standfirst {
  margin-top: 14px; color: #52627c;
  font-size: 16px; max-width: 46ch;
}

/* A real information card replaces the old unboxed, left-ruled
   definition list, which floated with nothing holding it. */
.document-card {
  min-width: 0;
  width: 100%;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: 0 4px 18px rgba(22, 33, 62, 0.065);
}
.document-card__heading {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.document-card__icon {
  flex: 0 0 auto;
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  color: #087f5b;
  background: #e4f4ee;
  border-radius: 10px;
}
.document-card__icon svg {
  stroke: currentColor; fill: none; stroke-width: 1.5;
  stroke-linecap: round; stroke-linejoin: round;
}
.document-card__heading h2 {
  color: #16213e;
  font-size: 16px;
  line-height: 1.35;
  font-weight: 650;
}

/* label and value, the values aligned down a column the way a
   drawing's particulars are.

   #52627c for the labels, not a lighter slate: it measures 5.86:1 on
   this card and 4.59:1 once a projector lamp lifts the black, which
   is the surface this page is actually read from. */
.facts {
  display: grid; grid-template-columns: auto minmax(0, 1fr);
  gap: 9px 16px; margin: 0; padding: 0;
  font-size: 13px; align-items: baseline;
}
.facts dt { color: #52627c; }
.facts dd { margin: 0; color: #16213e; overflow-wrap: anywhere; }

.elsewhere {
  display: flex; flex-wrap: wrap; gap: 22px;
  margin-top: 30px; font-size: 13.5px;
}
.elsewhere a { color: var(--ink-soft); text-decoration: none; }
.elsewhere a:hover { color: var(--ink); text-decoration: underline; }

@media (max-width: 820px) {
  .titleblock { grid-template-columns: minmax(0, 1fr); gap: 20px; }
  .masthead { padding: 26px 18px 0; }
  .masthead h1 { font-size: 30px; }
}

/* The result, read before anything is scrolled. It is the foot of the
   title block, so it gets the one firm rule on the page above it and
   the largest figures anywhere. */
.headline {
  display: flex; flex-wrap: wrap; gap: 0;
  margin-top: 40px; border-top: 1px solid var(--ink);
}
.headline > div {
  flex: 1 1 150px; padding: 24px 24px 30px 0;
  display: flex; flex-direction: column; gap: 6px;
}
.headline > div + div { padding-left: 24px; border-left: 1px solid var(--rule); }
.headline b { font-size: 40px; font-weight: 600; line-height: 1; letter-spacing: -0.03em; }
.headline span { font-size: 12.5px; color: var(--ink-soft); line-height: 1.4; }
.headline .good b { color: var(--series-ours); }
/* the baseline's number sits quiet: it is the thing being beaten, and
   the gap speaks without being shouted at */
.headline .plain b { color: var(--ink-soft); font-weight: 400; }
@media (max-width: 760px) {
  .headline > div { flex-basis: 45%; padding-bottom: 22px; }
  .headline b { font-size: 29px; }
}

.layout { max-width: 1180px; margin: 0 auto; padding: 36px 24px 120px; }

/* The six boxes, in order, with the one you are reading marked. It
   lives inside the app rail now — see api/ui.py — so it has no
   position of its own; what it keeps is the look of a sub-list and
   the scroll-spy that marks your place. */
.rail { list-style: none; margin: 0; padding: 0; }
.rail li { position: relative; }
.rail a {
  display: block; padding: 5px 0 5px 10px;
  font-size: 12.5px; line-height: 1.35;
  color: var(--ink-soft); text-decoration: none;
  border-left: 2px solid transparent; margin-left: -11px;
}
.rail a:hover { color: var(--ink); }
.rail a .n {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  display: inline-block; min-width: 1.1em;  /* the unnumbered last entry still lines up */
  margin-right: 7px; color: var(--rule-firm);
}
.rail a[aria-current="true"] { color: var(--brand-ink); font-weight: 600;
                               border-left-color: var(--brand); }
.rail a[aria-current="true"] .n { color: var(--brand); }
.rail .last { margin-top: 6px; padding-top: 7px; border-top: 1px solid var(--rule); }

/* The full-size view. A thumbnail used to open the raw bitmap in a
   tab of its own, where the browser's back button is dead because
   nothing preceded it — a demo that walks into a blank tab with a
   picture in it and no way home. This never leaves the page. */
.zoom {
  position: fixed; inset: 0; z-index: 50;
  background: rgba(10, 12, 15, 0.9);
  display: flex; align-items: flex-start; justify-content: center;
  padding: 56px 20px 48px; overflow: auto; overscroll-behavior: contain;
}
.zoom[hidden] { display: none; }
/* Fitted to the WIDTH, not the height. A paper page is tall, so
   fitting it inside the viewport shrank the type to the size it was
   on the thumbnail — which is the one thing the full-size view exists
   to undo. It is readable now, and you scroll down the page. */
.zoom img {
  width: min(1080px, 94vw); max-width: none; max-height: none; height: auto;
  background: #fff; box-shadow: 0 2px 40px rgba(0,0,0,.55);
}
.zoom .shut { position: fixed; }
.zoom .shut {
  position: absolute; top: 14px; right: 18px;
  background: none; border: 1px solid rgba(255,255,255,.45); border-radius: 4px;
  color: #fff; font: inherit; font-size: 13px; padding: 5px 12px; cursor: pointer;
}
.zoom .shut:hover { background: rgba(255,255,255,.14); }

/* Each stage is its own sheet, and the paper colour shows between
   them. Before this the six ran together down one grey field with
   small white boxes floating in it: the content differed but the
   frame never did, so the page read as a long document rather than
   as six things a machine does in order. A sheet apiece gives the
   scroll a beat, and it is the same object the engine works on. */
.stage {
  background: var(--sheet); border: 1px solid var(--rule);
  border-radius: var(--r); padding: 40px 44px 46px;
}
.stage + .stage { margin-top: 18px; }
/* On a white sheet a white box is invisible, so the inner containers
   that have no bordered parent take the shade instead. */
.stage .sheet, .stage .chunk, .stage .crops figure, .stage .run span { background: var(--shade); }
/* the verdict, which is not a stage and should not look like one */
#ev { border-top: 2px solid var(--ink); margin-top: 42px; }
.stage header { display: flex; align-items: baseline; gap: 18px; }
.stage .no {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: 27px; font-weight: 400; line-height: 1.1;
  color: var(--rule-firm); min-width: 30px;
  font-variant-numeric: tabular-nums;
}
/* The stage number hangs in the margin and everything else lines up
   under the title, so the six numbers run down one edge and the
   content runs down another. */
.lede { margin-top: 12px; margin-left: 48px; color: var(--ink-soft); font-size: 15.5px; max-width: 64ch; }
.out { margin-top: 28px; margin-left: 48px; }

.sheet { background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--r); }
.pad { padding: 18px 22px; }

/* Box 2 — the page strip */
.strip { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 6px; }
.strip a { flex: 0 0 auto; display: block; line-height: 0; text-decoration: none; }
.strip img {
  height: 132px; width: auto; display: block; border-radius: 4px;
  background: var(--sheet); border: 1px solid var(--rule);
}
.strip .pno {
  display: block; line-height: 1.6; font-style: normal;
  font-size: 11px; color: var(--ink-soft); text-align: center;
}
.strip a:hover img { border-color: var(--rule-firm); }
.strip a:hover .pno { color: var(--ink); }

/* Box 3 — the overlay. The claim is prose, so its rule is ink: on
   this page a coloured mark would promise a measurement. */
.claim {
  margin: 0 0 22px; padding: 2px 0 2px 20px;
  border-left: 2px solid var(--ink);
  font-size: 16.5px; line-height: 1.6; max-width: 72ch;
}
.claim strong { font-weight: 650; }
.overlay { display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 24px; align-items: start; }
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
.run { display: flex; flex-wrap: wrap; gap: 8px; }
.run li { list-style: none; }
.run span {
  display: inline-flex; align-items: baseline; gap: 8px;
  padding: 7px 13px; border: 1px solid var(--rule); background: var(--sheet);
  font-size: 13.5px;
}
.run .c { color: var(--ink-soft); font-size: 12px; font-variant-numeric: tabular-nums; }
.run .miss { border-style: dashed; color: var(--ink-soft); }

/* Box 5 — chunk cards */
.chunks { display: grid; gap: 12px; }
.chunk { border: 1px solid var(--rule); background: var(--sheet); border-radius: var(--r);
         padding: 16px 20px 17px; }
.chunk .top {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
  font-size: 12px; color: var(--ink-soft); margin-bottom: 10px;
}
/* The section a chunk landed in. It used to be green, and red when no
   section was found — a pair that measured ΔE 6.0 apart under
   deuteranopia, and that was never carrying the meaning anyway,
   because the word is printed inside the chip. The unplaced one is a
   dashed border instead, which survives being photographed, printed
   or looked at by the one man in twelve. */
.sect { color: var(--ink-soft); border: 1px solid var(--rule-firm);
        border-radius: 3px; padding: 2px 8px; font-size: 11px; }
.sect.none { border-style: dashed; font-style: italic; }
.chunk p { font-size: 14px; line-height: 1.6; max-width: 88ch; }
.chunk .head { font-weight: 600; }

/* Box 6 — the crops */
.crops { display: grid; grid-template-columns: repeat(auto-fill, minmax(228px, 1fr)); gap: 18px; }
.crops figure { margin: 0; border: 1px solid var(--rule); background: var(--sheet);
                border-radius: var(--r); overflow: hidden; }
.crops img {
  display: block; width: 100%; height: 182px; object-fit: contain;
  background: #fff; border-bottom: 1px solid var(--rule); padding: 10px;
}
.crops figcaption { padding: 12px 14px 13px; font-size: 12.5px; line-height: 1.5; }
.crops .kind { color: var(--ink-soft); font-size: 11px; display: block; margin-bottom: 3px; }

/* Box 5's headline numbers. The only tile row on the page — Box 1 is a
   quiet line and the comparison is bars, so this one reads as a
   highlight instead of as the house style. The won figure is the same
   blue the comparison gives this pipeline, so one colour means one
   thing the whole way down. */
.scores { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--rule);
          border: 1px solid var(--rule); border-radius: var(--r); overflow: hidden; }
.scores div { background: var(--sheet); padding: 18px 20px 19px; }
.scores .win b { color: var(--series-ours); }

/* Box 7 — the artifact on the left, the sentence that cites it on the
   right, and between them the words the link was actually made from. */
.links { list-style: none; margin: 14px 0 0; padding: 0; display: grid; gap: 1px;
         background: var(--rule); border: 1px solid var(--rule);
         border-radius: var(--r); overflow: hidden; }
.link { background: var(--sheet); display: grid; gap: 4px 28px; padding: 16px 20px 18px;
        grid-template-columns: minmax(0, 7fr) minmax(0, 9fr); align-items: start; }
.link .kind { display: block; font-size: 11px; color: var(--ink-soft); margin-bottom: 4px; }
.link .from b { font-size: 14px; font-weight: 600; line-height: 1.45; }
.link .to p { font-size: 13.5px; line-height: 1.6; max-width: none; }
/* achromatic: on this page a coloured mark promises a measurement, and
   this is a quotation */
.link mark.ref { background: var(--ink); color: var(--sheet); padding: 1px 5px; }
@media (max-width: 760px) { .link { grid-template-columns: 1fr; gap: 10px; } }

/* Box 1 — the file, stated once and not dressed up */
.filerow {
  display: flex; flex-wrap: wrap; gap: 10px 32px; align-items: baseline;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 16px 2px;
}
.filerow b { font-size: 16px; font-weight: 600; }
.filerow span { font-size: 14px; color: var(--ink-soft); }

/* The evidence — two measures, two bars each. This is the only place
   on the page where colour appears, and it is the thing the demo is
   for, so the bars are taller and the figures larger than anything
   else in a card. */
.measures { display: grid; gap: 1px; background: var(--rule); border: 1px solid var(--rule);
            border-radius: var(--r) var(--r) 0 0; overflow: hidden; }
.measure { background: var(--sheet); padding: 22px 24px 24px; }
.measure h3 { font-size: 17px; }
.measure > p { margin-bottom: 20px; max-width: 70ch; font-size: 14px; color: var(--ink-soft); }
.bar { display: grid; grid-template-columns: 116px minmax(0, 1fr) 58px; gap: 14px; align-items: center; }
.bar + .bar { margin-top: 11px; }
.bar .who { font-size: 13.5px; }
.bar .track { height: 18px; background: var(--shade); border-radius: 3px;
              box-shadow: inset 0 0 0 1px var(--rule); }
.bar .track i {
  display: block; height: 100%; min-width: 2px;
}
.bar .track i.baseline { background: var(--series-baseline); }
.bar .track i.ours     { background: var(--series-ours); }
.bar .val { font-size: 15px; font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; }
/* one chunk from each run, facing each other */
.facing { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--rule);
          border: 1px solid var(--rule); border-top: 0;
          border-radius: 0 0 var(--r) var(--r); overflow: hidden; }
.side { background: var(--sheet); padding: 20px 24px 22px; }
.side h4 { margin: 0 0 12px; font-size: 13.5px; font-weight: 650; }
.side .text {
  font: 13px/1.7 ui-monospace, "Cascadia Mono", Consolas, monospace;
  margin: 0; max-width: none;
}
.side mark { padding: 2px 4px; }
.side mark.cut  { background: var(--series-baseline); color: #fff; }
.side mark.keep { background: var(--series-ours); color: #fff; }
.verdict { margin-top: 14px; font-size: 13px; line-height: 1.5; }
.verdict.bad  { color: var(--series-baseline); }
.verdict.good { color: var(--series-ours); }
@media (max-width: 760px) { .facing { grid-template-columns: 1fr; } }

.note { margin-top: 16px; font-size: 13.5px; color: var(--ink-soft); }

.empty { color: var(--ink-soft); font-style: italic; font-size: 14px; }

@media (max-width: 900px) {
  .layout { padding: 0 18px 72px; }
  .rail { display: flex; flex-wrap: wrap; }
  .rail a { border-left: 0; margin-left: 0; padding: 5px 14px 5px 0; }
  .rail .last { margin: 0; padding: 0; border-top: 0; }
  .stage { padding: 28px 20px 32px; }
  .lede, .out { margin-left: 0; }
  .overlay { grid-template-columns: 1fr; }
  .scores { grid-template-columns: repeat(2, 1fr); }
}

/* ================================================================
   RESEARCH ASSISTANT · PIPELINE PRESENTATION LAYER
   Pure CSS for this server-rendered route. Nothing in the ingestion
   engine, stored results, comparison calculations, or API is changed.
   ================================================================ */
:root {
  --dashboard-navy: #16213e;
  --dashboard-bg: #f6f8fc;
  --dashboard-blue: #1768ce;
  --dashboard-line: #e2e8f0;
  --dashboard-green: #087f5b;
  --dashboard-shadow: 0 8px 28px rgba(22, 33, 62, 0.055), 0 2px 6px rgba(22,33,62,.025);
  --paper: var(--dashboard-bg);
  --shade: #f7f9fd;
  --rule: var(--dashboard-line);
  --r: 14px;
}
body { padding-left: 248px; color: #18213a; background: var(--dashboard-bg); }
/* The shared navigation remains unchanged; style it only on /pipeline. */
.apprail {
  width: 248px; padding: 22px 13px 18px;
  background: #fff; border-right: 1px solid var(--dashboard-line);
  box-shadow: 2px 0 16px rgba(22,33,62,.018);
}
.apprail .top {
  padding: 6px 10px 24px; gap: 11px; color: var(--dashboard-navy);
  border-bottom: 1px solid #edf1f7; margin-bottom: 14px;
}
.apprail .top svg { color: var(--dashboard-green); width: 26px; height: 26px; }
.apprail .top b { font-size: 15px; letter-spacing: -.025em; }
.apprail a { border-radius: 9px; padding: 10px 12px; gap: 11px; font-size: 13.5px; }
.apprail a:hover { background: #f2f6fc; color: var(--dashboard-navy); }
.apprail a[aria-current="page"] {
  background: #e7f4ee; color: #087152; font-weight: 650;
}
.apprail .nest { margin-left: 23px; padding-left: 12px; border-left-color: #e7edf4; }
.apprail .foot { border-top-color: #e8edf5; }
.rail a { border-left-width: 2px; border-radius: 0 7px 7px 0; padding: 7px 8px 7px 11px; }
.rail a:hover { background: #f2f6fc; color: #16213e; }
.rail a[aria-current="true"] {
  background: #edf7f3; color: var(--dashboard-green);
  border-left-color: var(--dashboard-green);
}
.rail a[aria-current="true"] .n { color: var(--dashboard-green); }
/* Exact requested header color. */
.brand {
  background: #16213e; color: #fff; border-bottom: 1px solid #16213e;
  padding: 17px 32px;
}
.brand .inner { max-width: 1380px; align-items: center; }
.brand b { color: #fff; font-size: 16px; font-weight: 650; letter-spacing: -.025em; }
.brand span { color: #c9d4ea; font-size: 12px; }
.masthead {
  background: linear-gradient(180deg,#fbfcff 0%,#f6f8fc 100%);
  border-bottom: 0; padding: 34px 32px 20px;
}
.masthead .inner { max-width: 1380px; }
.titleblock {
  display: grid; grid-template-columns: minmax(0,1fr) minmax(285px,355px);
  gap: 30px 34px; align-items: center; margin-bottom: 26px;
}
.titleblock > div { min-width: 0; }
.masthead h1 {
  font-size: clamp(30px, 3.1vw, 44px); line-height: 1.14;
  color: var(--dashboard-navy); font-weight: 740;
  letter-spacing: -.035em; max-width: 30ch;
  overflow-wrap: anywhere;
}
.standfirst {
  margin-top: 13px; font-size: 15px; line-height: 1.65;
  color: #596a81; max-width: 53ch;
}
.document-card {
  position: relative; background: #fff;
  border: 1px solid #e1e9f3; border-radius: 16px;
  box-shadow: var(--dashboard-shadow);
  padding: 19px 20px 20px; max-width: 100%; min-width: 0;
}
.document-card__heading { gap: 11px; margin-bottom: 15px; }
.document-card__heading h2 { font-size: 16px; font-weight: 700; color: #16213e; }
.document-card__icon {
  width: 38px; height: 38px; border-radius: 11px;
  background: #e7f4ee; color: #087f5b;
}
.facts {
  display: grid; grid-template-columns: 62px minmax(0,1fr);
  align-items: baseline; column-gap: 13px; row-gap: 10px;
  margin: 0; padding: 0; font-size: 12.5px; line-height: 1.55;
}
.facts dt { color: #687990; font-weight: 500; }
.facts dd { color: var(--dashboard-navy); min-width: 0; max-width: none; overflow-wrap: anywhere; }
/* Measured output is legible at a glance, without a full-width grey rule. */
.headline {
  display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px; margin-top: 0; border: 0;
}
.headline > div {
  min-width: 0; background: #fff; border: 1px solid var(--dashboard-line);
  border-radius: 13px; padding: 18px 15px 19px;
  box-shadow: 0 2px 8px rgba(22,33,62,.022);
  display: flex; flex-direction: column; justify-content: center;
  gap: 0;
}
.headline > div + div { padding-left: 15px; border-left: 1px solid var(--dashboard-line); }
.headline b {
  display: block; color: #16213e; font-size: clamp(23px,2.45vw,34px);
  line-height: 1.1; letter-spacing: -.04em; font-weight: 720;
  overflow-wrap: anywhere;
}
.headline span { display: block; margin-top: 9px; font-size: 12px; color: #617187; line-height: 1.45; }
.headline .good b { color: #1768ce; }
.headline .plain b { color: #64748b; font-weight: 620; }
/* A clear heading separates the overview from the walkthrough. */
.workflow-head {
  display: flex; justify-content: space-between; align-items: end;
  gap: 12px; margin: 3px 0 20px;
}
.workflow-head h2 { font-size: 23px; letter-spacing: -.025em; color: #16213e; }
.workflow-head p { margin-top: 6px; font-size: 13px; color: #69798e; }
.workflow-head .count {
  flex: 0 0 auto; border: 1px solid #d8e7f8; background: #eff6ff;
  color: #1c63b1; border-radius: 999px; font-size: 12px;
  font-weight: 650; padding: 6px 12px;
}
.layout { max-width: 1444px; padding: 20px 32px 90px; }
.stage {
  background: #fff; border: 1px solid #e2e9f2;
  border-radius: 16px; padding: 26px 30px 30px;
  box-shadow: var(--dashboard-shadow);
  scroll-margin-top: 18px;
}
.stage + .stage { margin-top: 16px; }
.stage header { display: flex; align-items: center; gap: 16px; }
.stage .no {
  flex: 0 0 auto; display: grid; place-items: center;
  min-width: 42px; width: 42px; height: 42px;
  border-radius: 11px; background: #e5f4ee;
  color: #087f5b; font-size: 18px; font-weight: 730; line-height: 1;
}
.stage header h2 { font-size: 23px; letter-spacing: -.026em; color: #16213e; }
.stage .lede {
  margin: 9px 0 0; max-width: 77ch;
  color: #62718a; font-size: 14px; line-height: 1.65;
}
.stage .out { margin: 20px 0 0 58px; }
.stage .sheet, .stage .chunk, .stage .crops figure, .stage .run span {
  background: #f9fbff; border-color: #e4eaf3;
}
.sheet, .chunk, .crops figure, .run span { border-radius: 10px; }
.filerow {
  align-items: center; gap: 12px 26px; padding: 17px 20px;
  border: 1px solid #e2e9f3; border-radius: 11px;
  background: #f9fbff;
}
.filerow b { color: #16213e; overflow-wrap: anywhere; }
.filerow span { color: #617187; }
.strip img { border-radius: 8px; }
.strip a:hover img { border-color: #1768ce; box-shadow: 0 3px 12px rgba(23,104,206,.1); }
.overlay { grid-template-columns: minmax(0,1fr) 205px; gap: 22px; }
.tally { border-top: 0; }
.tally li { padding: 9px 0; }
.tally b { color: #16213e; }
.claim {
  border-left: 3px solid #1768ce; padding: 10px 15px;
  background: #f2f7ff; border-radius: 0 9px 9px 0;
  color: #334967; font-size: 14.5px;
}
.scores { border-radius: 11px; }
.scores div { background: #f9fbff; }
.scores .win b { color: #1768ce; }
.readout b { color: #16213e; }
.chunk { background: #fafcff; padding: 17px 19px; }
.chunk .sect { background: #fff; border-radius: 6px; }
.crops figure { background: #fafcff; overflow: hidden; }
.link { background: #fafcff; }
.link mark.ref { background: #e7f1ff; color: #155caf; border-radius: 4px; }
#ev { border-top: 1px solid #e2e9f2; margin-top: 26px; }
.measure h3 { color: #16213e; }
.measure, .side { background: #fafcff; }
.bar .track { border-radius: 999px; overflow: hidden; }
.note { color: #66768d; }
/* Responsive: long filenames cannot force horizontal scrolling. */
@media (max-width: 1480px) {
  .headline { grid-template-columns: repeat(3, minmax(0,1fr)); }
}
@media (max-width: 1120px) {
  .titleblock { grid-template-columns: minmax(0,1fr); }
  .document-card { max-width: 540px; }
  .masthead { padding: 30px 24px 18px; }
  .layout { padding: 18px 24px 70px; }
}
@media (max-width: 900px) {
  body { padding-left: 0; }
  .apprail { position: static; width: 100%; padding: 10px 15px; box-shadow: none; }
  .apprail .top { margin-bottom: 0; padding-bottom: 9px; border-bottom: 0; }
  .brand { padding: 14px 18px; }
  .brand .inner { align-items: center; }
  .masthead { padding: 25px 18px 18px; }
  .layout { padding: 18px 18px 64px; }
  .stage { padding: 24px 20px; }
  .stage .lede, .stage .out { margin-left: 0; }
  .headline { grid-template-columns: repeat(3,minmax(0,1fr)); }
  .overlay { grid-template-columns: minmax(0,1fr); }
}
@media (max-width: 600px) {
  .brand .inner { align-items: start; flex-direction: column; gap: 4px; }
  .masthead h1 { font-size: 30px; }
  .headline { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; }
  .headline > div { padding: 14px 13px; }
  .headline > div + div { padding-left: 13px; }
  .headline b { font-size: 25px; }
  .workflow-head { align-items: start; flex-direction: column; }
  .document-card { padding: 17px 16px; }
  .stage header { align-items: start; gap: 12px; }
  .stage header h2 { font-size: 20px; }
  .stage .no { width: 36px; height: 36px; min-width: 36px; }
  .scores { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .bar { grid-template-columns: 88px minmax(0,1fr) 46px; gap: 7px; }
  .filerow { padding: 13px 14px; }
}

/* Requested finishing touches: typography and two balanced header cards.
   Presentation-only; the report title and metadata remain dynamically bound. */
.brand { padding-top: 20px; padding-bottom: 20px; }
.brand b {
  font-size: 21px;
  font-weight: 740;
  line-height: 1.25;
  letter-spacing: -.025em;
}
.masthead { padding-top: 28px; }
.titleblock { align-items: stretch; gap: 20px; }
.titleblock > div:first-child {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 225px;
  padding: 25px 29px;
  background: #fff;
  border: 1px solid #e1e9f3;
  border-left: 4px solid #1768ce;
  border-radius: 16px;
  box-shadow: var(--dashboard-shadow);
}
.masthead h1 {
  font-size: clamp(25px, 2.25vw, 30px);
  line-height: 1.24;
  max-width: none;
  letter-spacing: -.027em;
}
.standfirst { max-width: 58ch; margin-top: 11px; font-size: 14px; }
.document-card { height: 100%; }
@media (max-width: 1120px) {
  .titleblock { grid-template-columns: minmax(0, 1fr); }
  .titleblock > div:first-child { min-height: auto; padding: 23px 25px; }
  .document-card { max-width: none; }
}
@media (max-width: 600px) {
  .brand b { font-size: 18px; }
  .masthead h1 { font-size: 25px; }
  .titleblock > div:first-child { padding: 20px; }
}

/* ==============================================================
   REFERENCE DASHBOARD / PRESENTATION-ONLY PIPELINE STYLING
   This is all applied only on /pipeline/{paper_id}.
   Header, sidebar, dynamic metadata, real metrics and stage previews.
   =============================================================== */
:root { --paper:#f7f9fd; --sheet:#fff; --ink:#111c37; --ink-soft:#52647e;
        --rule:#e4ebf5; --shade:#f8faff; --r:13px;
        --brand:#087f5b; --brand-ink:#087f5b; --brand-wash:#e3f3ed; }
html { scroll-behavior:smooth; }
body { padding-left:284px; background:#f7f9fd; color:#16213e; font-size:14px; line-height:1.5; }
/* Reference: a wide navy bar spanning the entire viewport ABOVE sidebar. */
.dashboard-topbar {
  position:relative; z-index:24; margin-left:-284px; width:calc(100% + 284px);
  min-height:78px; display:flex; align-items:center; justify-content:space-between;
  gap:30px; padding:13px 34px 13px 24px;
  background:#16213e; color:#fff;
  box-shadow:0 3px 16px rgba(22,33,62,.08);
}
.topbar-identity { display:flex; align-items:center; gap:18px; color:#fff; text-decoration:none; min-width:0; }
.topbar-logo { width:49px; height:49px; flex:0 0 auto; display:grid; place-items:center;
  background:linear-gradient(135deg,#67a0ff,#1b62ee); border:1px solid rgba(255,255,255,.15);
  border-radius:12px; color:white; box-shadow:0 5px 13px rgba(23,70,155,.25); }
.topbar-copy { display:flex; flex-direction:column; gap:1px; }
.topbar-copy strong { font-size:21px; font-weight:700; letter-spacing:-.02em; line-height:1.2; color:#fff; }
.topbar-copy small { font-size:12px; color:#dbe5fa; line-height:1.45; }
.topbar-links { display:flex; align-items:center; gap:27px; }
.topbar-links a { display:flex; align-items:center; gap:9px; text-decoration:none;
  color:#f5f7fd; font-size:14px; font-weight:550; white-space:nowrap; padding:5px 0; }
.topbar-links a:hover { color:#a8d3ff; }
.topbar-links svg { width:18px; height:18px; stroke-linecap:round; stroke-linejoin:round; }
.topbar-avatar { display:grid; place-items:center; width:42px; height:42px; border-radius:50%;
  color:#fff; font-weight:700; background:linear-gradient(140deg,#ca72e4,#8643bc); }
/* The only rail affected is the one on this page. */
.apprail { top:78px; width:284px; padding:17px 18px; background:#fff;
  border-right:1px solid #e5ecf5; box-shadow:none; }
.apprail .top { margin:0 0 16px; padding:8px 14px 15px; border:0; color:#14213e; }
.apprail .top b { font-size:16px; font-weight:700; }
.apprail .top svg { width:22px; height:22px; color:#16213e; }
.apprail a:not(.top) { font-size:14px; padding:10px 13px; border-radius:9px; color:#52647e; }
.apprail a[aria-current="page"] { background:#e3f3ed; color:#087f5b; }
.apprail .nest { margin-left:25px; padding-left:12px; border-left:1px solid #dbe5f1; }
.rail { margin-top:0; }
.rail a { font-size:13px; line-height:1.3; padding:9px 10px; border-radius:0 8px 8px 0; }
.rail a[aria-current="true"] { background:#e8f6f0; color:#087f5b; border-left-color:#087f5b; }
.rail a .n { color:#536c8b; min-width:20px; }
.apprail .foot { border-color:#e5ecf5; }
/* title / metadata block: no unnecessary title card or empty white space */
.masthead { background:transparent; padding:16px 30px 14px; border:0; }
.masthead .inner { max-width:1480px; }
.titleblock { display:grid; grid-template-columns:minmax(0,1fr) minmax(300px,345px);
  align-items:center; gap:22px; margin:0 0 16px; }
.titleblock > div.report-intro:first-child { min-height:0; padding:6px 4px 4px 2px; border:0; border-radius:0;
  background:transparent; box-shadow:none; display:block; }
.masthead h1 { font-size:clamp(25px,2.55vw,35px); font-weight:760; line-height:1.18;
  letter-spacing:-.035em; color:#16213e; max-width:34ch; overflow-wrap:anywhere; }
.standfirst { color:#516685; font-size:14.5px; line-height:1.65; max-width:51ch; margin-top:11px; }
.document-card { background:#fff; border:1px solid #e3ebf5; border-radius:13px;
  box-shadow:0 3px 15px rgba(21,35,65,.045); padding:12px 18px; align-self:center; height:auto; }
.document-card__heading { margin-bottom:7px; gap:10px; }
.document-card__icon { width:35px; height:35px; border-radius:9px; background:#e3f3ed; }
.document-card__icon svg { fill:none; stroke:#087f5b; stroke-width:1.7; }
.document-card__heading h2 { font-size:14px; color:#16213e; }
.facts { grid-template-columns:62px minmax(0,1fr); column-gap:10px; row-gap:4px;
  padding:0; margin:0; font-size:12px; line-height:1.4; }
.facts dt { color:#61718b; }
.facts dd { color:#16213e; font-size:12px; overflow-wrap:anywhere; }
/* Six equal KPI cells matching reference design */
.headline { display:grid; grid-template-columns:repeat(6,minmax(0,1fr));
  border:1px solid #e4ebf5; border-radius:13px; overflow:hidden; gap:0;
  background:#fff; box-shadow:0 3px 12px rgba(22,33,62,.04); }
.headline > div { background:#fff; box-shadow:none; border:0; border-radius:0;
  padding:17px 18px 19px; min-height:106px; justify-content:center; }
.headline > div + div { border:0; border-left:1px solid #e4ebf5; padding-left:18px; }
.headline b { font-size:clamp(20px,2.45vw,32px); font-weight:750; color:#16213e; }
.headline .good b { color:#1768ce; }
.headline .plain b { color:#62718a; font-weight:620; }
.headline span { font-size:11.5px; line-height:1.4; color:#63748c; margin-top:8px; }
/* Stage cards */
.layout { max-width:1540px; margin:auto; padding:0 30px 80px; }
.stage { border:1px solid #e4ebf5; border-radius:13px; padding:14px 22px;
  background:#fff; box-shadow:0 3px 12px rgba(22,33,62,.035);
  scroll-margin-top:18px; }
.stage + .stage { margin-top:14px; }
.stage header { align-items:center; gap:15px; }
.stage .no { width:42px; height:42px; min-width:42px; font-size:22px; border-radius:10px;
  background:#e2f2ec; color:#087f5b; }
.stage header h2 { font-size:19px; color:#17213c; font-weight:730; }
.stage .lede { font-size:13px; line-height:1.55; color:#61728d;
  margin:4px 0 0 57px; max-width:100ch; }
.stage .out { margin:11px 0 0 0; }
.filerow { background:#fbfcff; border:1px solid #e5ecf5; border-radius:10px;
  display:flex; align-items:center; flex-wrap:wrap; gap:15px 28px; padding:10px 15px; }
.filerow b { font-size:13.5px; flex:0 1 235px; min-width:0; color:#18213a; }
.filerow span { font-size:12.5px; color:#61718b; }
.pdf-file-icon { flex:0 0 auto; display:grid; place-items:center; width:36px; height:36px;
  border-radius:8px; background:#ffe7e9; color:#e94d60!important; }
/* Stage 2: measured run statistics left, original thumbnails right. */
.page-showcase { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(250px,.85fr);
  gap:21px; align-items:stretch; }
.page-stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
  border:1px solid #e4ebf5; border-radius:10px; overflow:hidden; }
.page-stat-grid > div { border-left:1px solid #e4ebf5; padding:17px 13px; min-width:0;
  display:flex; align-items:center; }
.page-stat-grid > div:first-child { border-left:0; }
.page-stat-grid .readout b { font-size:clamp(15px,1.7vw,24px); color:#1768ce;
  overflow-wrap:anywhere; }
.page-stat-grid .readout span { font-size:11.5px; }
.page-preview { border-radius:10px; min-width:0; overflow:hidden; }
.page-preview .strip { display:flex; gap:12px; height:128px; padding:1px 4px; }
.page-preview .strip a { flex:0 0 auto; }
.page-preview .strip img { height:100px; max-width:100%; border-radius:8px; border:1px solid #e4ebf5; }
.page-preview .note { font-size:11px; margin:0; }
/* Stage 3: claim/counts on left, visual inspection on right. */
.region-showcase { display:grid; grid-template-columns:minmax(0,1.3fr) minmax(300px,1fr);
  gap:20px; align-items:start; }
.region-copy { min-width:0; }
.region-copy .claim { font-size:12.5px; line-height:1.5; margin:0 0 12px;
  border-left:3px solid #1768ce; padding:8px 10px; }
.region-stat-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
  border:1px solid #e4ebf5; border-radius:10px; }
.region-stat-grid > div { padding:12px 14px; border-left:1px solid #e4ebf5; }
.region-stat-grid > div:first-child { border-left:0; }
.region-stat-grid .readout b { color:#1768ce; }
.region-copy .note { font-size:11.5px; margin-top:10px; }
.region-copy .pagelinks { margin-top:9px; }
.region-showcase .overlay { display:grid; grid-template-columns:minmax(125px,.8fr) minmax(125px,1fr);
  gap:15px; margin:0; align-items:start; }
.region-showcase .region-image { max-height:180px; overflow:auto; border-radius:8px;
  border:1px solid #e5ebf5; }
.region-showcase .region-image img { width:100%; height:auto; max-height:none; display:block; }
.region-showcase .tally { max-height:190px; overflow:auto; border-top:0; }
.region-showcase .tally li { font-size:11.5px; padding:5px 0; }
.stage .scores { border-radius:10px; }
.stage .chunk,.stage .crops figure { border-radius:10px; }
.note { font-size:12px; }
@media (max-width:1370px) {
  .page-showcase { grid-template-columns:1fr; }
  .page-preview .strip { height:140px; }
  .region-showcase { grid-template-columns:1fr; }
  .headline b { font-size:clamp(18px,2.35vw,30px); }
}
@media (max-width:1080px) {
  .titleblock { grid-template-columns:1fr; }
  .document-card { max-width:none; }
  .headline { grid-template-columns:repeat(3,minmax(0,1fr)); }
  .headline > div:nth-child(4) { border-left:0; border-top:1px solid #e4ebf5; }
  .headline > div:nth-child(n+5) { border-top:1px solid #e4ebf5; }
}
@media (max-width:900px) {
  body { padding-left:0; }
  .dashboard-topbar { width:100%; margin-left:0; min-height:74px; padding:12px 18px; }
  .apprail { position:static; width:100%; height:auto; padding:9px 12px; border-right:0;
    border-bottom:1px solid #e4ebf5; display:flex; flex-direction:row; gap:5px;
    align-items:center; overflow-x:auto; }
  .apprail .top { flex:0 0 auto; margin:0; padding:5px 9px; }
  .apprail .top b { display:none; }
  .apprail > a:not(.top) { flex:0 0 auto; white-space:nowrap; padding:8px 10px; }
  .apprail .nest { display:none; }
  .apprail .foot { flex:0 0 auto; margin:0; padding:0; border-top:0; }
  .masthead { padding:22px 18px 13px; }
  .layout { padding:0 18px 60px; }
  .headline { grid-template-columns:repeat(3,minmax(0,1fr)); }
  .stage { padding:19px; }
}
@media (max-width:640px) {
  .dashboard-topbar { flex-direction:column; align-items:stretch; gap:10px; }
  .topbar-logo { width:39px; height:39px; }
  .topbar-copy strong { font-size:18px; }
  .topbar-copy small { font-size:11px; }
  .topbar-links { justify-content:space-between; gap:8px; }
  .topbar-links a { font-size:11px; gap:5px; }
  .topbar-links svg { width:14px; height:14px; }
  .topbar-avatar { height:31px; width:31px; font-size:11px; }
  .masthead h1 { font-size:28px; }
  .headline { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .headline > div { padding:13px; min-height:87px; }
  .headline > div + div { padding-left:13px; }
  .headline > div:nth-child(2n+1) { border-left:0; }
  .headline > div:nth-child(n+3) { border-top:1px solid #e4ebf5; }
  .stage .lede { margin-left:0; }
  .stage header h2 { font-size:18px; }
  .stage .out { margin-top:12px; }
  .page-stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .page-stat-grid > div:nth-child(3) { border-left:0; }
  .page-stat-grid > div:nth-child(n+3) { border-top:1px solid #e4ebf5; }
  .region-showcase .overlay { grid-template-columns:1fr; }
  .region-showcase .region-image { max-height:260px; }
  .filerow { gap:9px 16px; }
  .filerow b { flex:1 1 190px; }
}


/* ==============================================================
   FINISHING PASS — clean, balanced stage timeline
   /pipeline only. No API, computation or stage-order changes.
   =============================================================== */
:root {
  --ui-navy:#16213e;
  --ui-blue:#1b65cf;
  --ui-ink:#17243e;
  --ui-mist:#f5f8fd;
  --ui-line:#e4ebf5;
  --ui-green:#067a59;
  --ui-shadow:0 6px 24px rgba(26,48,91,.048),0 1px 3px rgba(26,48,91,.025);
}
body { background:linear-gradient(120deg,#f9fbff 0%,#f4f7fc 66%,#f8faff 100%); }
.dashboard-topbar { background:#16213e; border-bottom:1px solid rgba(255,255,255,.1); }
.topbar-logo { border-radius:12px; }
.apprail { background:#fff; }
.apprail a:not(.top) { transition:background .18s ease,color .18s ease; }
.apprail a:not(.top):hover { background:#f1f6fd; }
.apprail a[aria-current="page"] { background:#e5f4ef; }
.rail a { transition:background .18s ease,color .18s ease; }
.rail .last { border-top:1px solid #e7edf5; padding-top:10px; margin-top:5px; }
.rail .last .n { font-size:0; width:20px; vertical-align:middle; }
.rail .last .n::before { content:'⇄'; display:inline-block; font-size:18px; font-weight:700;
  color:#476bd1; line-height:1; vertical-align:middle; }
/* A smaller paper heading, without an oversized empty title card. */
.masthead { padding:24px 30px 16px; }
.titleblock { grid-template-columns:minmax(0,1fr) minmax(300px,365px);
  gap:24px; align-items:center; margin-bottom:21px; }
.titleblock > .report-intro:first-child { padding:10px 0 10px 3px;
  align-self:center; }
.report-intro::before { content:'RESEARCH DOCUMENT'; display:block; width:max-content;
  color:#547096; font-size:11px; font-weight:740; letter-spacing:.115em;
  margin-bottom:11px; }
.masthead h1 { color:#16213e; font-size:clamp(24px,2.1vw,30px); line-height:1.28;
  font-weight:770; letter-spacing:-.034em; max-width:35ch; overflow-wrap:anywhere; }
.standfirst { font-size:13.5px; max-width:58ch; margin-top:9px;
  line-height:1.65; color:#60718d; }
.document-card { padding:17px 20px; border-radius:16px;
  border:1px solid #e2e9f4; box-shadow:var(--ui-shadow); }
.document-card__heading { margin-bottom:12px; }
.document-card__heading h2 { font-size:15px; }
.document-card__icon { height:37px; width:37px; border-radius:10px; }
.facts { grid-template-columns:62px minmax(0,1fr); row-gap:6px;
  column-gap:14px; font-size:12.5px; }
.facts dd { font-size:12.5px; line-height:1.4; }
/* Auto-fit means four real outputs fill the entire row; no fake blank KPIs. */
.headline { grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
  border:1px solid #e2e9f4; border-radius:16px;
  box-shadow:var(--ui-shadow); overflow:hidden; }
.headline > div { min-width:0; padding:19px 20px 19px; min-height:100px;
  box-shadow:none; border:0; background:#fff; }
.headline > div + div { border-left:1px solid #e5ecf5; padding-left:20px; }
.headline b { font-size:clamp(24px,2.25vw,34px); line-height:1.12; letter-spacing:-.035em; }
.headline span { font-size:12.5px; margin-top:7px; line-height:1.4; }
.workflow-head { margin:14px 0 15px; align-items:center; }
.workflow-head h2 { font-size:22px; color:#182847; }
.workflow-head p { font-size:13px; color:#6a7a92; }
.workflow-head .count { background:#edf5ff; border-color:#d7e7fd; padding:7px 13px; }
.layout { padding:0 30px 94px; }
/* Sequential seven-card walkthrough; full-width cards remain in order. */
.stage { position:relative; isolation:isolate; background:#fff;
  border:1px solid #e2eaf4; border-radius:16px; padding:21px 26px 23px;
  box-shadow:var(--ui-shadow); overflow:hidden; transition:border-color .2s ease,box-shadow .2s ease; }
.stage:hover { border-color:#cbd9ed; box-shadow:0 8px 28px rgba(24,44,80,.075); }
.stage + .stage { margin-top:17px; }
.stage::before { content:''; position:absolute; top:21px; bottom:21px; left:0;
  width:3px; background:#b6dfd2; border-radius:0 4px 4px 0; }
.stage header { display:flex; align-items:center; gap:15px; min-width:0; }
.stage .no { width:43px; height:43px; min-width:43px; background:#e4f4ee;
  color:#087f5b; border-radius:12px; font-size:20px; font-weight:770;
  box-shadow:inset 0 0 0 1px #d8eee5; }
.stage header > div:last-child { flex:1; min-width:0; }
.stage header h2 { font-size:19px; color:#16213e; font-weight:750; letter-spacing:-.015em; }
.stage .lede { margin:6px 0 0 58px; font-size:13.5px; line-height:1.55;
  color:#61728d; max-width:110ch; }
.stage .out { margin:17px 0 0 58px; }
/* The production template nests the subtitle inside the header text
   column. Do not indent it twice away from its heading. */
.stage header .lede { margin:6px 0 0; }
.stage > .lede { margin:6px 0 0 58px; }
.stage .out > * { max-width:100%; }
.stage .filerow { padding:14px 16px; background:#f9fbff; border-radius:10px; }
.stage .filerow b { flex:0 1 auto; font-size:13px; }
.stage .pdf-file-icon { width:34px; height:34px; }
.stage .page-stat-grid,.stage .region-stat-grid { background:#fcfdff; }
.stage .page-stat-grid > div { padding:16px 14px; }
.stage .page-stat-grid .readout b { font-size:clamp(17px,1.6vw,24px); }
.stage .page-preview { background:#f8faff; border:1px solid #e7edf6;
  padding:12px 14px; }
.stage .page-preview .strip { height:126px; }
.stage .region-image { border:1px solid #e4eaf4; }
.stage .scores,.stage .run span,.stage .chunk { border-color:#e4ebf5; }
.stage .claim { background:#f1f6ff; border-left-color:#286bd2; }
/* Baseline is an evidence/quality comparison, NOT an eighth stage. */
#ev { margin-top:27px; border:1px solid #dce5f6;
  background:linear-gradient(106deg,#f7f9ff 0%,#ffffff 57%); }
#ev::before { background:#7995dd; }
#ev .no { background:#eaf0ff; color:#345bb8; box-shadow:inset 0 0 0 1px #d9e4ff; }
#ev .no svg { width:22px; height:22px; display:block; stroke:currentColor;
  stroke-width:1.7; fill:none; stroke-linecap:round; stroke-linejoin:round; }
#ev .measure,#ev .side { background:#fff; }
#ev .measures { border-radius:11px 11px 0 0; }
#ev .facing { border-radius:0 0 11px 11px; }
/* Helpful 'step' context without changing the source data. */
#s1 header::after,#s2 header::after,#s3 header::after,
#s4 header::after,#s5 header::after,#s6 header::after,#s7 header::after {
  content:'PIPELINE STAGE'; color:#8a99b0; font-weight:690;
  font-size:10px; letter-spacing:.09em; white-space:nowrap; margin-left:auto;
}
#ev header::after { content:'BASELINE COMPARISON'; color:#5878bb;
  font-size:10px; font-weight:740; letter-spacing:.07em;
  white-space:nowrap; margin-left:auto; }
/* Tablet and mobile: maintain order and preserve real output readability. */
@media (max-width:1370px) {
  .headline { grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }
}
@media (max-width:1080px) {
  .titleblock { grid-template-columns:minmax(0,1fr) minmax(265px,320px); }
  .headline { grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); }
  .headline > div:nth-child(n) { border-top:0; }
  .stage .page-showcase,.stage .region-showcase { grid-template-columns:1fr; }
}
@media (max-width:900px) {
  .titleblock { grid-template-columns:1fr; }
  .document-card { max-width:none; }
  .headline { grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); }
  .headline > div:nth-child(n) { border-top:0; }
  .masthead { padding:23px 18px 18px; }
  .layout { padding:0 18px 64px; }
}
@media (max-width:640px) {
  .titleblock { gap:14px; }
  .report-intro::before { margin-bottom:6px; }
  .masthead h1 { font-size:25px; }
  .headline { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .headline > div { min-height:86px; padding:15px 14px; }
  .headline > div + div { padding-left:14px; }
  .headline > div:nth-child(2n+1) { border-left:0; }
  .headline > div:nth-child(n+3) { border-top:1px solid #e5ecf5; }
  .stage { padding:17px 15px 20px; }
  .stage header { gap:12px; }
  .stage .no { width:38px; height:38px; min-width:38px; font-size:18px; }
  .stage header h2 { font-size:17.5px; }
  .stage .lede { margin:9px 0 0; font-size:13px; }
  .stage .out { margin:15px 0 0; }
  .stage header::after { display:none; }
  .filerow { gap:10px; }
}
/* Keep the full-width header attached to the fixed pipeline sidebar
   while scrolling. This is presentation-only; no pipeline data changes. */
@media (min-width:901px) {
  .dashboard-topbar {
    position:sticky;
    top:0;
    z-index:50;
  }
  .apprail {
    position:fixed;
    top:78px; /* Matches the navy header's desktop height */
    bottom:0;
    z-index:40;
  }
  /* Anchor jumps must not hide the stage heading beneath the sticky bar. */
  .stage { scroll-margin-top:96px; }
}
@media (max-width:900px) {
  /* On smaller screens the rail follows the header in normal document flow. */
  .dashboard-topbar { position:relative; top:auto; }
}
@media (prefers-reduced-motion: reduce) {
  .stage,.rail a,.apprail a { transition:none; }
}
"""

#: Two things. The rail entry for whichever stage is on screen gets
#: marked, and any link to a page image or a crop opens it over the
#: page instead of navigating to the bitmap. Scroll is the only thing
#: that moves on this page — nothing animates on its own.
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

const zoom = document.querySelector('.zoom');
const shown = zoom.querySelector('img');
let cameFrom = null;

function open(href, label) {
  cameFrom = document.activeElement;
  shown.src = href;
  shown.alt = label || '';
  zoom.hidden = false;
  zoom.querySelector('.shut').focus();
}
function shut() {
  zoom.hidden = true;
  shown.removeAttribute('src');
  if (cameFrom) cameFrom.focus();
}

/* every link that ends at an image, wherever on the page it is */
document.addEventListener('click', (e) => {
  const a = e.target.closest('a[href*="/page/"], a[href*="/artifact/"]');
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
  e.preventDefault();
  const img = a.querySelector('img');
  open(a.getAttribute('href'), img ? img.alt : a.textContent.trim());
});

zoom.addEventListener('click', (e) => { if (e.target !== shown) shut(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !zoom.hidden) shut(); });
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
    # Baseline comparison is unnumbered; use a visual comparison mark,
    # not a misleading eighth step or an empty numbered badge.
    badge = (
        "<svg viewBox='0 0 24 24' aria-hidden='true'>"
        "<rect x='3' y='5' width='7' height='14' rx='1.5'/>"
        "<rect x='14' y='9' width='7' height='10' rx='1.5'/>"
        "<path d='M16 5h5m-2.5-2.5L21 5l-2.5 2.5'/>"
        "</svg>"
        if key == "ev" else _e(number)
    )
    return (
        f"<section class='stage' id='{key}'>"
        f"<header><div class='no' aria-hidden='true'>{badge}</div><div>"
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
        "<span class='pdf-file-icon' aria-hidden='true'>"
        "<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'>"
        "<path d='M6 2.5h8l4 4v15H6zM14 2.5v4h4M9 12h6M9 16h6'/></svg></span>"
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
        f"<a href='/paper/{pid}/page/{p.page_index}'>"
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
        "<div class='page-showcase'>"
        "<div class='page-stat-grid'>"
        + _readout(len(result.pages), "pages rendered", "")
        + _readout(f"{first.dpi} DPI", "rendering resolution", "")
        + _readout(f"{first.width_px}×{first.height_px}", "image size (px)", "")
        + _readout(f"{first.scale:.3f}", "pixels per point", "")
        + "</div>"
        + f"<div class='page-preview'><div class='strip'>{strip}</div>"
        + f"<p class='note'>Page previews{more}</p></div></div>"
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
            f"<a href='/paper/{pid}/page/{busiest}/layout'>"
            f"<img src='/paper/{pid}/page/{busiest}/layout' loading='lazy' "
            f"alt='Page {busiest + 1} with its detected regions outlined and "
            f"numbered in reading order'></a>"
        )
        others = sorted(candidates)[:10]
        links = "<div class='pagelinks'><span class='soft'>Same view, other pages:</span>" + "".join(
            f"<a href='/paper/{pid}/page/{p}/layout'>{p + 1}</a>"
            for p in others
            if p != busiest
        ) + "</div>"

    # The claim goes above the picture, not under it in grey. This is
    # the one thing on the page a panel can check with their own eyes
    # while the presenter is still talking, and a caption below an
    # image is read after the image — if at all.
    return (
        "<div class='region-showcase'><div class='region-copy'>"
        "<p class='claim'>The numbered badges show <strong>reading order</strong>. "
        "On two-column pages, text runs down the left column first and "
        "then the right column.</p>"
        "<div class='region-stat-grid'>"
        + _readout(len(result.regions), "regions found")
        + _readout(len(counts), "region categories")
        + "</div>"
        + f"<p class='note'>Page {(busiest or 0) + 1} has the most detected regions. "
        "Select its preview to enlarge.</p>"
        + f"{links}</div>"
        + f"<div class='overlay'><div class='region-image'>{picture}</div>"
        + f"<ul class='tally'>{tally}</ul></div></div>"
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
                f"<a href='/paper/{pid}/artifact/{i}'>"
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


def _marked(text: str, label) -> str:
    """One passage, cut down to the words around the reference, with
    the reference itself marked.

    A link a panel has to take on trust is worth less than one they can
    read, and the whole chunk would bury the three words that matter.
    """
    flat = " ".join(text.split())
    span = reference_span(flat, label)
    if span is None:
        short = flat[:LINK_CHARS]
        return _e(short + (" …" if len(flat) > LINK_CHARS else ""))

    start, end = span
    left, right = max(0, start - 62), min(len(flat), end + 74)
    before = ("… " if left else "") + flat[left:start]
    after = flat[end:right] + (" …" if right < len(flat) else "")
    return (
        f"{_e(before)}<mark class='ref'>{_e(flat[start:end])}</mark>{_e(after)}"
    )


def _box7(result: IngestionResult) -> str:
    if not result.artifacts:
        return (
            "<p class='empty'>Nothing to link: this paper produced no figures, "
            "tables or equations.</p>"
        )

    linked = [a for a in result.artifacts if a.linked_chunk_ids]
    citing = sum(1 for c in result.chunks if c.artifact_ids)
    mentions = sum(len(a.linked_chunk_ids) for a in result.artifacts)
    by_id = {c.chunk_id: c for c in result.chunks}

    rows = []
    for artifact in linked[:LINK_PREVIEW]:
        label = caption_label(artifact.caption)
        chunk = by_id.get(artifact.linked_chunk_ids[0])
        where = f"{artifact.artifact_type.value} · p{artifact.bbox.page + 1}"
        if chunk is None:
            continue
        rows.append(
            f"<li class='link'>"
            f"<div class='from'><span class='kind'>{_e(where)}</span>"
            f"<b>{_e(artifact.caption[:90])}</b></div>"
            f"<div class='to'><span class='kind'>chunk {chunk.order_index} · "
            f"{_e(chunk.section.value.replace('_', ' '))}</span>"
            f"<p>{_marked(chunk.text, label)}</p></div></li>"
        )

    unlinked = len(result.artifacts) - len(linked)
    note = (
        f"{unlinked} of {len(result.artifacts)} are left unlinked: the text never "
        "names them, and binding one to whichever chunk sits nearest would hand "
        "Member 2 something that looks like evidence and is not."
        if unlinked
        else "Every artifact in this paper is named somewhere in the text."
    )

    return (
        "<div class='scores'>"
        + _readout(f"{len(linked)}/{len(result.artifacts)}", "artifacts linked", "win")
        + _readout(citing, "chunks cite one")
        + _readout(mentions, "links in total")
        + _readout("0", "models used")
        + "</div>"
        + (f"<ul class='links'>{''.join(rows)}</ul>" if rows else "")
        + f"<p class='note'>{_e(note)}</p>"
    )


def _headline(
    result: IngestionResult, baseline: Optional[IngestionResult]
) -> str:
    """The result, before the walkthrough that explains it.

    The page used to open on the filename, the page count and the
    schema version — true, and none of it the achievement. A panel
    watching from a projector reads the top of a page and then listens
    to the presenter; whatever is up here is what they take away.

    So the top carries the answer and the stages below carry the
    argument. The two comparison figures are repeated at the end in
    full, with their bars and their caveat; here they are the headline
    they earn.
    """
    placed = sum(1 for c in result.chunks if c.section is not SectionType.UNKNOWN)
    sections = {
        r.section for r in result.regions if r.section is not SectionType.UNKNOWN
    }

    cells = [
        (f"{len(result.regions)}", "regions found", ""),
        (f"{len(sections)}", "sections identified", ""),
        (f"{placed}/{len(result.chunks)}", "chunks carry their section", "good"),
        (f"{len(result.artifacts)}", "figures and tables lifted out", ""),
    ]

    if baseline is not None and baseline.chunks and result.chunks:
        def broken(r: IngestionResult) -> float:
            n = sum(1 for c in r.chunks if c.text.lstrip()[:1].islower())
            return 100 * n / len(r.chunks)

        cells.append((f"{broken(result):.0f}%", "start mid-word", "good"))
        cells.append((f"{broken(baseline):.0f}%", "without this pipeline", "plain"))

    return "<div class='headline'>" + "".join(
        f"<div class='{tone}'><b class='mono'>{_e(value)}</b>"
        f"<span>{_e(label)}</span></div>"
        for value, label, tone in cells
    ) + "</div>"


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


def _side_by_side(result: IngestionResult, baseline: IngestionResult) -> str:
    """One chunk from each run, with its opening marked.

    The bars above say 50% against 0%. This says what the 50% is: a
    chunk that opens `lumn paragraphs, even though…` is the tail of a
    word the chunk before it kept the head of, and no amount of
    retrieval tuning recovers the sentence it came from.

    A panel reads two short paragraphs faster than two bars, and the
    broken one is the argument — so it is on the left, where reading
    starts.
    """
    broken = next(
        (c for c in baseline.chunks if c.text.lstrip()[:1].islower()), None
    )
    clean = next(
        (
            c
            for c in result.chunks
            if "\n" in c.text and c.section is not SectionType.UNKNOWN
        ),
        result.chunks[0] if result.chunks else None,
    )
    if broken is None or clean is None:
        return ""

    # The leading fragment is the tail of a split word: everything up
    # to the first space.
    tail, _, rest = broken.text.lstrip().partition(" ")
    head, _, body = clean.text.partition("\n")

    return (
        "<div class='facing'>"
        "<div class='side'><h4>Baseline</h4>"
        f"<p class='text'><mark class='cut'>{_e(tail)}</mark> "
        f"{_e(' '.join(rest.split())[:190])} …</p>"
        "<p class='verdict bad'>Opens inside a word. The sentence it belongs to "
        "is in the chunk before this one.</p></div>"
        "<div class='side'><h4>This pipeline</h4>"
        f"<p class='text'><mark class='keep'>{_e(head[:70])}</mark> "
        f"{_e(' '.join(body.split())[:170])} …</p>"
        "<p class='verdict good'>Opens at the heading that introduces it, and "
        f"carries <strong>{_e(clean.section.value.replace('_', ' '))}</strong> "
        "with it.</p></div>"
        "</div>"
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
        f"{_side_by_side(result, baseline)}"
        f"<p class='note'>The same PDF down both paths. The baseline cuts flat "
        f"text every {DEFAULT_CHUNK_CHARS} characters and this pipeline cuts at "
        f"section and paragraph boundaries, so the two runs produce different "
        f"numbers of chunks — the bars are shares of each run, not counts. "
        f"Hover a bar for the counts. "
        f"<a href='/compare/{pid}'>Open the two side by side</a>.</p>"
    )


# Pipeline-page-only brand header: does not alter ui.py or other pages.
# All links point to routes that actually exist in this project.
_PIPELINE_HEADER = """
<header class='dashboard-topbar'>
  <a href='/' class='topbar-identity' aria-label='Research Assistant home'>
    <span class='topbar-logo' aria-hidden='true'>
      <svg width='27' height='27' viewBox='0 0 28 28' fill='none'>
        <rect x='6' y='4' width='16' height='20' rx='2' stroke='currentColor' stroke-width='2'/>
        <path d='M10 10h8M10 15h7M10 20h5' stroke='currentColor' stroke-width='2' stroke-linecap='round'/>
        <path d='M19 8h5v5' stroke='currentColor' stroke-width='2' stroke-linecap='round'/>
      </svg>
    </span>
    <span class='topbar-copy'><strong>Research Assistant</strong>
      <small>From PDF to Structured Technical Report</small></span>
  </a>
  <nav class='topbar-links' aria-label='Quick navigation'>
    <a href='/'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' aria-hidden='true'><path d='m3 10 9-7 9 7v10h-6v-6H9v6H3z'/></svg>Home</a>
    <a href='/'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' aria-hidden='true'><path d='M5 4h14v16H5zM8 8h8M8 12h8M8 16h5'/></svg>All Papers</a>
    <a href='/docs'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' aria-hidden='true'><path d='m8 8-4 4 4 4m8-8 4 4-4 4m-3-11-2 14'/></svg>API Docs</a>
    <span class='topbar-avatar' aria-label='Member 1'>M1</span>
  </nav>
</header>
"""


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
    stage_rail = f"<ol class='rail'>{rail}</ol>"

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
        _stage("s7", "7", "Artifacts bound to the text",
               "Each figure, table and equation joined to the chunks that cite it, "
               "using the label the author printed rather than any guess about "
               "distance.",
               _box7(result)),
        _stage("ev", "", "Measured against the baseline",
               "What the same PDF looks like without any of the above.",
               _evidence(result, baseline, pid)),
    ]

    facts = (
        f"<dt>File</dt><dd class='mono'>{_e(paper.source_filename)}</dd>"
        f"<dt>Pages</dt><dd class='mono'>{paper.page_count}</dd>"
        f"<dt>Method</dt><dd class='mono'>{_e(paper.extraction_method.value)}</dd>"
        f"<dt>Schema</dt><dd class='mono'>{_e(result.schema_version)}</dd>"
        f"<dt>Read</dt><dd class='mono'>"
        f"{_e(paper.ingested_at.strftime('%Y-%m-%d %H:%M UTC'))}</dd>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)} — ingestion pipeline</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{_PIPELINE_HEADER}"
        f"{app_rail('pipeline', paper.paper_id, nested=stage_rail)}"
        f"<header class='masthead'><div class='inner'>"
        f"<div class='titleblock'><div class='report-intro'><h1>{_e(title)}</h1>"
        "<p class='standfirst'>Every stage, in the order it runs, with "
        "what each one produced from this paper.</p></div>"
        "<section class='document-card' aria-labelledby='doccard'>"
        "<div class='document-card__heading'>"
        "<span class='document-card__icon' aria-hidden='true'>"
        "<svg width='20' height='20' viewBox='0 0 20 20'>"
        "<path d='M11.5 2.5H6a1.5 1.5 0 0 0-1.5 1.5v12A1.5 1.5 0 0 0 6 17.5h8a1.5 "
        "1.5 0 0 0 1.5-1.5V6.5z'/><path d='M11.5 2.5v4h4'/>"
        "<path d='M7.5 10.5h5M7.5 13.5h3'/></svg></span>"
        "<h2 id='doccard'>Document Information</h2></div>"
        f"<dl class='facts'>{facts}</dl></section></div>"
        # The row of links that used to sit here — all papers, raw JSON,
        # gallery, comparison — is the app rail now, and printing both
        # was the same four destinations twice on one screen.
        f"{_headline(result, baseline)}"
        "</div></header>"
        f"<div class='layout'><main>{''.join(bodies)}</main></div>"
        # the full-size view, which replaces opening a raw image in a
        # tab of its own with nothing in it to come back by
        "<div class='zoom' hidden role='dialog' aria-modal='true' aria-label='Full size'>"
        "<button class='shut' aria-label='Close'>Close &times;</button>"
        "<img alt=''></div>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
