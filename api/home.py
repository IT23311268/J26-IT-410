"""
The front door.

Before this existed the first thing anyone saw was `/docs`, the
generated Swagger page: a list of endpoints, their request bodies and
their response schemas. That is the right page for Members 2, 3 and 4,
who are writing a client against this service. It is the wrong page for
a progress review, where nobody is going to read a JSON schema and the
question being asked is "what does this thing do".

So the front door is a place to put a PDF in, and a list of the papers
already through. Both lead to the same place: the pipeline walkthrough
for that paper. `/docs` is still there and still the contract — it is
linked from the bottom, where the people who need it will look.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from urllib.parse import quote

from api.pipeline import STAGES
from api.ui import BASE_CSS, BRAND, app_rail


@dataclass(frozen=True)
class PaperCard:
    """One row in the list of papers already read.

    A row used to be the `paper_id` and nothing else, which is a hash
    with a filename stuck on the front — the right identifier for
    Members 2/3/4 and useless to anyone choosing which paper to open.
    The title is on the page and Box 4 already found it, so the row
    leads with that and keeps the id underneath for whoever needs it.

    Everything but the id is optional: a record that fails to load
    still gets a row, so a damaged file is visible rather than silently
    missing.
    """

    paper_id: str
    title: str = ""
    page_count: int = 0
    chunk_count: int = 0
    artifact_count: int = 0

_CSS = BASE_CSS + """
body { display: flex; flex-direction: column; min-height: 100vh; }
.wrap { width: 100%; max-width: 1180px; margin: 0 auto; padding: 56px 24px 80px; flex: 1; }

/* The page used to be one narrow column down the middle of a wide
   screen: a paragraph, a box, a list, and a great deal of nothing on
   either side. The hero puts the claim and the thing you do about it
   side by side, and the width is spent on the six steps below. */
.hero {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 520px);
  gap: 32px 60px; align-items: start;
}
/* nothing ingested yet means nothing honest to show beside the claim */
.hero.alone { grid-template-columns: minmax(0, 1fr); }
.title { max-width: 15ch; font-size: 34px; }
.standfirst { margin-top: 18px; color: var(--ink-soft); max-width: 46ch; font-size: 16.5px; }

/* The page used to describe the engine and then show an empty box. It
   shows the engine instead: the most recent paper with every region
   the layout stage found drawn on it and numbered in the order they
   are read. That picture is the claim, and a reader checks it against
   the paragraph beside it without being asked to believe anything. */
.preview figure {
  margin: 0; background: var(--sheet); border: 1px solid var(--rule);
  border-radius: var(--r); padding: 10px;
  box-shadow: 0 1px 2px rgba(16, 24, 32, .05), 0 10px 28px rgba(16, 24, 32, .07);
}
.preview img {
  display: block; width: 100%; height: 340px; border-radius: 4px;
  object-fit: cover; object-position: top; background: #fff;
}
.preview figcaption {
  padding: 11px 4px 2px; font-size: 12.5px; line-height: 1.5; color: var(--ink-soft);
}
.preview .mono { color: var(--brand-ink); }
@media (prefers-color-scheme: dark) {
  .preview figure { box-shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 28px rgba(0,0,0,.5); }
}

.drop {
  margin-top: 26px; position: relative;
  border: 1px solid var(--rule-firm); background: var(--sheet);
  border-radius: var(--r); padding: 20px 22px;
  display: flex; align-items: center; gap: 18px;
}
.drop.over { border-color: var(--brand); background: var(--brand-wash); }
.drop input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop .cta {
  flex: 0 0 auto; background: var(--brand); color: #fff;
  border-radius: 6px; padding: 10px 18px; font-size: 14.5px; font-weight: 600;
}
.drop .what { min-width: 0; }
.drop strong { display: block; font-size: 15px; font-weight: 600; }
.drop span { display: block; margin-top: 3px; font-size: 13.5px; color: var(--ink-soft); }
.drop.busy { border-color: var(--brand); }
.drop.busy input { pointer-events: none; }
.drop.busy .cta { background: var(--rule-firm); }

/* a determinate bar would be a lie: the server does not stream progress */
.pulse { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; margin: 0; background: var(--rule); overflow: hidden; }
.pulse i { display: block; height: 100%; width: 36%; background: var(--brand); animation: slide 1.25s ease-in-out infinite; }
@keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(380%); } }
.fail { margin-top: 18px; color: var(--red); font-size: 14px; }

/* The six stages, across the width rather than down a column. They
   are numbered because the content is a sequence — a paper really
   does go through them in this order, and stage 3 cannot run before
   stage 2 has drawn the pixels it measures. */
.stages { margin-top: 64px; }
.stages h2 { padding-bottom: 14px; border-bottom: 2px solid var(--brand); }
.steps {
  list-style: none; margin: 0; padding: 0;
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); border-top: 0;
  border-radius: 0 0 var(--r) var(--r); overflow: hidden;
}
.steps { box-shadow: 0 1px 2px rgba(16, 24, 32, .04); }
.step { background: var(--sheet); padding: 20px 22px 24px;
        display: grid; grid-template-columns: auto minmax(0, 1fr);
        grid-template-rows: auto auto; gap: 2px 13px; align-items: start; }
.step .ico { grid-row: 1 / 3; width: 34px; height: 34px; border-radius: 7px;
             background: var(--brand-wash); color: var(--brand-ink);
             display: grid; place-items: center; margin-top: 2px; }
.step .ico svg { stroke: currentColor; fill: none; stroke-width: 1.5;
                 stroke-linecap: round; stroke-linejoin: round; }
.is-todo .ico { background: var(--shade); color: var(--ink-soft); }
/* The last card takes whatever is left of its row rather than leaving
   empty cells in the corner. Eight stages do not divide by three, and
   nor did seven; the hole read as a layout that had gone wrong rather
   than as a step. Both remainders are covered, so adding or removing a
   stage cannot reopen it. */
.steps .step:last-child:nth-child(3n + 1) { grid-column: 1 / -1; }
.steps .step:last-child:nth-child(3n + 2) { grid-column: span 2; }
.step .n { font-size: 11.5px; letter-spacing: 0.02em; color: var(--ink-soft); }
/* the stage that is not built, marked once and not dressed down: it is
   on the page because the component is eight boxes, not seven */
.step .todo {
  float: right; font-size: 11px; color: var(--ink-soft);
  border: 1px solid var(--rule-firm); border-radius: 3px; padding: 1px 8px;
}
.is-todo { background: var(--paper); }
.is-todo h3, .is-todo p { opacity: 0.72; }
.step h3 { grid-column: 2; font-size: 16px; }
.step p { grid-column: 2; margin-top: 7px; font-size: 13.5px; line-height: 1.55;
          color: var(--ink-soft); }

/* The one dark block on the page. The comparison is not a seventh
   stage, it is the reason to believe the other six, so it does not
   look like them. */
.verdict {
  margin-top: 56px; background: var(--brand-deep); color: #fff;
  border-radius: var(--r); padding: 34px 36px; display: flex; flex-wrap: wrap; gap: 24px 40px;
  align-items: center; justify-content: space-between;
}
.verdict h2 { font-size: 23px; }
.verdict p { margin-top: 10px; max-width: 64ch; font-size: 14.5px; opacity: 0.78; }
.verdict .go {
  flex: 0 0 auto; color: inherit; text-decoration: none;
  border: 1px solid currentColor; border-radius: 6px; padding: 10px 20px; font-size: 14.5px;
}
.verdict .go:hover { background: #fff; color: var(--brand-deep); }

.papers { margin-top: 64px; }
.papers h2 { padding-bottom: 14px; border-bottom: 2px solid var(--brand); }
.papers a:hover b { color: var(--brand-ink); }
.papers ul { list-style: none; margin: 0; padding: 0; }
.papers li { border-bottom: 1px solid var(--rule); }
.papers a {
  display: flex; justify-content: space-between; align-items: baseline; gap: 24px;
  padding: 20px 6px; text-decoration: none; color: var(--ink);
}
.papers a:hover { background: var(--sheet); }
.papers .what { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.papers b { font-size: 16.5px; font-weight: 600; letter-spacing: -0.012em; }
.papers .id { font-size: 12px; color: var(--ink-soft); }
.papers .go { font-size: 13px; color: var(--ink-soft); white-space: nowrap; }
.none { padding: 30px 2px; color: var(--ink-soft); font-size: 15px; }

footer {
  border-top: 1px solid var(--rule); background: var(--sheet);
  padding: 22px 24px; font-size: 13px; color: var(--ink-soft);
}
footer .inner { max-width: 840px; margin: 0 auto; display: flex; flex-wrap: wrap; gap: 24px; }
footer a { color: var(--ink-soft); }

@media (max-width: 1100px) {
  .steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .steps .step:last-child { grid-column: auto; }
  .steps .step:last-child:nth-child(2n + 1) { grid-column: 1 / -1; }
}
@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; gap: 20px; }
  .drop { grid-column: 1; grid-row: auto; }
  .title { max-width: 20ch; }
}
@media (max-width: 620px) {
  .steps { grid-template-columns: 1fr; }
  .steps .step:last-child { grid-column: auto; }
}
@media (max-width: 680px) {
  .wrap { padding-top: 40px; }
  .title { font-size: 31px; }
  .drop { padding: 40px 20px; }
  .verdict { padding: 26px 22px; }
}
"""

_SCRIPT = """
const zone = document.querySelector('.drop');
const input = zone.querySelector('input');
const fail = document.querySelector('.fail');

const stop = e => { e.preventDefault(); e.stopPropagation(); };
['dragenter','dragover'].forEach(n => zone.addEventListener(n, e => { stop(e); zone.classList.add('over'); }));
['dragleave','drop'].forEach(n => zone.addEventListener(n, e => { stop(e); zone.classList.remove('over'); }));
zone.addEventListener('drop', e => send(e.dataTransfer.files[0]));
input.addEventListener('change', () => send(input.files[0]));

async function send(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    return show('That is not a PDF. This service reads PDFs only.');
  }
  fail.textContent = '';
  zone.classList.add('busy');
  zone.querySelector('strong').textContent = 'Reading ' + file.name;
  zone.querySelector('span').textContent = 'Rendering pages, finding regions, cutting chunks.';
  zone.querySelector('.pulse').hidden = false;

  const body = new FormData();
  body.append('file', file);
  try {
    const res = await fetch('/ingest', { method: 'POST', body });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || ('The server returned ' + res.status + '.'));
    }
    const out = await res.json();
    location.href = '/pipeline/' + encodeURIComponent(out.paper.paper_id);
  } catch (err) {
    show(err.message);
  }
}

function show(message) {
  zone.classList.remove('busy');
  zone.querySelector('.pulse').hidden = true;
  zone.querySelector('strong').textContent = 'Drop a paper here';
  zone.querySelector('span').textContent = 'or click to choose a PDF';
  fail.textContent = message;
  input.value = '';
}
"""


def _row(card: PaperCard) -> str:
    counts = []
    if card.page_count:
        counts.append(f"{card.page_count} page{'s' if card.page_count != 1 else ''}")
    if card.chunk_count:
        counts.append(f"{card.chunk_count} chunks")
    if card.artifact_count:
        kind = "figure or table" if card.artifact_count == 1 else "figures and tables"
        counts.append(f"{card.artifact_count} {kind}")

    return (
        f"<li><a href='/pipeline/{quote(card.paper_id, safe='')}'><span class='what'>"
        f"<b>{html.escape(card.title or card.paper_id)}</b>"
        f"<span class='id mono'>{html.escape(card.paper_id)}</span></span>"
        f"<span class='go'>{html.escape(', '.join(counts))}</span></a></li>"
    )


#: One plain sentence per stage, said for somebody who has not seen a
#: region or a chunk yet.
#:
#: The NAMES are not here. They come from `STAGES` in api/pipeline.py,
#: the same list the walkthrough's headings and the rail are built
#: from, so the six boxes are called one thing in this service and not
#: two. An earlier version wrote its own titles — this page said "The
#: PDF arrives" where the walkthrough said "PDF in" — and a panel
#: reading one and then the other has to work out for itself that they
#: are the same six stages. tests/test_pipeline.py pins them equal.
_BLURBS: dict[str, str] = {
    "s1": "Kept exactly as uploaded, so every later stage can be re-run from it "
          "and the baseline can be measured again.",
    "s2": "Each page is drawn once to a bitmap and reused, rather than "
          "re-rendered by every stage that needs pixels.",
    "s3": "Headings, body, captions, figures, tables and equations, located by "
          "geometry and font size and put back into reading order. No model, no GPU.",
    "s4": "Which part of the paper each region belongs to. The same sentence "
          "means different things in Related Work and in Results.",
    "s5": "Cut at section and paragraph boundaries and never inside a sentence, "
          "each one carrying the section it came from.",
    "s6": "Every figure, table and equation cropped from its page, with the "
          "caption its author printed beside it.",
    "s7": "Each one joined back to the passages that cite it, using the label "
          "the author printed — “as shown in Figure 1” — and nothing else.",
}


#: The one stage that is not built. It is printed here rather than left
#: off, because the component is eight boxes and a panel shown seven
#: has to be told about the eighth anyway — better in writing, next to
#: the reason, than as an answer to a question from the floor.
#:
#: It says "Planned", not "in progress". Nothing has been written, and
#: a progress claim is the kind that gets tested with "show me".
_PLANNED = (
    "8",
    "Visual embeddings",
    "Each page image put through a late-interaction visual model, so a query "
    "can find a page by what it looks like and not only by its text. This is "
    "the one stage that needs a GPU, and nothing above depends on it.",
)


#: One line drawing per stage. The cards were eight boxes of text in a
#: hairline grid, which reads as a spreadsheet; a mark apiece gives the
#: eye somewhere to land and says what the stage is before the words do.
_STEP_ICONS = {
    "1": "<path d='M5 2.5h6.5L15 6v11.5H5z'/><path d='M11.5 2.5V6H15'/>",
    "2": "<rect x='2.5' y='3' width='6' height='14' rx='1'/>"
         "<rect x='11.5' y='3' width='6' height='14' rx='1'/>",
    "3": "<rect x='2.5' y='3' width='15' height='14' rx='1'/>"
         "<path d='M5.5 6.5h5M5.5 10h5M5.5 13.5h8.5'/>",
    "4": "<path d='M3 4.5h14M3 10h7.5M3 15.5h14'/><circle cx='14.5' cy='10' r='1.7'/>",
    "5": "<rect x='2.5' y='2.5' width='15' height='5.5' rx='1'/>"
         "<rect x='2.5' y='12' width='15' height='5.5' rx='1'/>"
         "<path d='M1.5 10h17' stroke-dasharray='2.5 2.5'/>",
    "6": "<rect x='2.5' y='4' width='15' height='12' rx='1'/>"
         "<path d='M2.5 12.5l4-4 3.5 3.5L13 9l4.5 4.5'/><circle cx='7' cy='7.5' r='1.1'/>",
    "7": "<path d='M8.5 11.5a3.4 3.4 0 0 1 0-4.8l1.8-1.8a3.4 3.4 0 0 1 4.8 4.8l-.9.9'/>"
         "<path d='M11.5 8.5a3.4 3.4 0 0 1 0 4.8l-1.8 1.8a3.4 3.4 0 0 1-4.8-4.8l.9-.9'/>",
    "8": "<circle cx='4.5' cy='4.5' r='1.5'/><circle cx='15.5' cy='4.5' r='1.5'/>"
         "<circle cx='4.5' cy='15.5' r='1.5'/><circle cx='15.5' cy='15.5' r='1.5'/>"
         "<circle cx='10' cy='10' r='1.8'/>"
         "<path d='M5.8 5.8l2.7 2.7M14.2 5.8l-2.7 2.7M5.8 14.2l2.7-2.7M14.2 14.2l-2.7-2.7'/>",
}


def _icon(number: str) -> str:
    return (
        f"<span class='ico'><svg width='19' height='19' viewBox='0 0 20 20' "
        f"aria-hidden='true'>{_STEP_ICONS[number]}</svg></span>"
    )

def _step(number: str, title: str, blurb: str, planned: bool = False) -> str:
    """One stage, named as a step so the order is read as an order.

    The word is spelled out rather than left as a bare numeral: a "3"
    alone in the corner of a card is a count of something as easily as
    it is a position in a sequence, and this is a sequence — stage 3
    cannot run before stage 2 has drawn the pixels it measures.
    """
    mark = "<span class='todo'>Planned · PP2</span>" if planned else ""
    return (
        f"<li class='step{' is-todo' if planned else ''}'>{_icon(number)}"
        f"<span class='n mono'>Step {number}</span>{mark}"
        f"<h3>{html.escape(title)}</h3><p>{blurb}</p></li>"
    )


def _rail_steps() -> str:
    """The pipeline as it stands in the rail: what the engine does,
    where the landing page would otherwise show empty space."""
    rows = "".join(
        f"<li><span class='n'>{number}</span>{html.escape(label)}</li>"
        for key, number, label in STAGES
        if key in _BLURBS
    )
    rows += (
        f"<li class='todo'><span class='n'>{_PLANNED[0]}</span>"
        f"{html.escape(_PLANNED[1])}</li>"
    )
    return f"<h2 class='group'>The pipeline</h2><ol class='railsteps'>{rows}</ol>"


def render_home(papers: list[PaperCard]) -> str:
    """Return the landing page: what the engine does, somewhere to put
    a paper, and everything already through it."""
    if papers:
        listing = f"<ul>{''.join(_row(p) for p in papers)}</ul>"
    else:
        listing = (
            "<p class='none'>Nothing ingested yet. Drop a PDF above and it will "
            "appear here.</p>"
        )

    # The hero shows the engine's own output rather than describing it:
    # the newest paper with every region drawn on it and numbered in
    # reading order. With nothing ingested there is nothing honest to
    # show, so the hero is one column until there is.
    preview = ""
    if papers:
        newest = papers[0]
        pid = quote(newest.paper_id, safe="")
        shown = html.escape(newest.title or newest.paper_id)
        preview = (
            f"<div class='preview'><figure>"
            f"<img src='/paper/{pid}/page/0/layout' alt='Page 1 of {shown}, with every "
            f"region outlined and numbered in reading order' loading='lazy'>"
            f"<figcaption>Page 1 of <b>{shown}</b> — every region the layout stage "
            f"found, <span class='mono'>numbered in the order they are read</span>. "
            f"On a two-column page the numbers run down the left column before "
            f"crossing to the right.</figcaption></figure></div>"
        )

    steps = "".join(
        _step(number, label, _BLURBS[key])
        for key, number, label in STAGES
        if key in _BLURBS  # the comparison is not a stage, and is shown below
    ) + _step(*_PLANNED, planned=True)

    # The comparison needs a paper to compare, so it points at the most
    # recent one. It carries no figures: the numbers belong to a run,
    # and printing a remembered pair here would be inventing them.
    if papers:
        measure = (
            "<section class='verdict'><div>"
            "<h2>Measured against a baseline</h2>"
            "<p>The same PDF is also read the ordinary way — all the text as one flat "
            "string, cut every 800 characters — and the two runs are put side by side. "
            "That is the only honest way to say this stage is worth having.</p></div>"
            f"<a class='go' href='/compare/{quote(papers[0].paper_id, safe='')}'>"
            "See both runs</a></section>"
        )
    else:
        measure = ""

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Ingestion engine — J26-IT-410</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{app_rail('home', tail=_rail_steps())}"
        f"{BRAND}"
        "<div class='wrap'>"
        f"<header class='hero{'' if preview else ' alone'}'><div>"
        "<h1 class='title'>A reference paper, taken apart</h1>"
        "<p class='standfirst'>This service reads a PDF the way a person reads it: "
        "columns in the right order, figures and tables kept whole, each passage "
        "labelled with the section it came from. It produces the structured JSON "
        "the rest of the pipeline is built on.</p>"
        "<div class='drop'>"
        "<input type='file' accept='application/pdf,.pdf' aria-label='Choose a PDF to ingest'>"
        "<span class='cta'>Choose a PDF</span>"
        "<span class='what'><strong>Drop a paper here</strong>"
        "<span>Any reference PDF. Nothing leaves this machine.</span></span>"
        "<div class='pulse' hidden><i></i></div>"
        "</div><p class='fail' role='alert'></p>"
        f"</div>{preview}</header>"
        f"<section class='stages'><h2>What happens to it</h2>"
        f"<ol class='steps'>{steps}</ol></section>"
        f"<section class='papers'><h2>Already read</h2>{listing}</section>"
        f"{measure}"
        "</div>"
        "<footer><div class='inner'>"
        # the project and member are named in the band at the top; a
        # footer that repeats them is a second signature on one page
        "<span>For Members 2, 3 and 4</span>"
        "<a href='/docs'>API reference</a>"
        "<a href='/papers'>Paper IDs as JSON</a>"
        "<a href='/health'>Health</a>"
        "</div></footer>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
