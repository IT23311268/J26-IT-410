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

from api.ui import BASE_CSS


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
.wrap { width: 100%; max-width: 860px; margin: 0 auto; padding: 60px 24px 40px; flex: 1; }

.title { max-width: 28ch; font-size: 34px; }
.standfirst { margin-top: 12px; color: var(--ink-soft); max-width: 62ch; }

/* the one loud thing on the page */
.drop {
  margin-top: 32px; position: relative;
  border: 1px dashed var(--rule-firm); background: var(--sheet);
  padding: 34px 28px; text-align: center;
}
.drop.over { border-color: var(--blue); background: var(--blue-wash); }
.drop input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop strong { display: block; font-size: 19px; font-weight: 600; }
.drop span { display: block; margin-top: 6px; font-size: 13px; color: var(--ink-soft); }
.drop.busy { border-style: solid; border-color: var(--blue); }
.drop.busy input { pointer-events: none; }

/* a determinate bar would be a lie: the server does not stream progress */
.pulse { height: 2px; margin-top: 18px; background: var(--rule); overflow: hidden; }
.pulse i { display: block; height: 100%; width: 36%; background: var(--blue); animation: slide 1.25s ease-in-out infinite; }
@keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(380%); } }
.fail { margin-top: 16px; color: var(--red); font-size: 14px; }

.papers { margin-top: 44px; }
.papers h2 { padding-bottom: 8px; border-bottom: 1px solid var(--rule-firm); }
.papers ul { list-style: none; margin: 0; padding: 0; }
.papers li { border-bottom: 1px solid var(--rule); }
.papers a {
  display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
  padding: 13px 4px; text-decoration: none; color: var(--ink);
}
.papers a:hover { background: var(--shade); }
.papers a:hover b { color: var(--blue); }
.papers .what { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.papers b { font-size: 15px; font-weight: 600; }
.papers .id { font-size: 12px; color: var(--ink-soft); }
.papers .go { font-size: 13px; color: var(--ink-soft); white-space: nowrap; }
.none { padding: 22px 2px; color: var(--ink-soft); font-size: 14px; }

footer {
  border-top: 1px solid var(--rule); background: var(--sheet);
  padding: 18px 24px; font-size: 13px; color: var(--ink-soft);
}
footer .inner { max-width: 860px; margin: 0 auto; display: flex; flex-wrap: wrap; gap: 20px; }
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


def render_home(papers: list[PaperCard]) -> str:
    """Return the landing page: upload, plus everything already ingested."""
    if papers:
        listing = f"<ul>{''.join(_row(p) for p in papers)}</ul>"
    else:
        listing = (
            "<p class='none'>Nothing ingested yet. Drop a PDF above and it will "
            "appear here.</p>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Ingestion engine — J26-IT-410</title>"
        f"<style>{_CSS}</style></head><body>"
        "<div class='wrap'>"
        "<h1 class='title'>A reference paper, taken apart</h1>"
        "<p class='standfirst'>This service reads a PDF the way a person reads it: "
        "columns in the right order, figures and tables kept whole, each passage "
        "labelled with the section it came from. It produces the structured JSON "
        "the rest of the pipeline is built on.</p>"
        "<div class='drop'>"
        "<input type='file' accept='application/pdf,.pdf' aria-label='Choose a PDF to ingest'>"
        "<strong>Drop a paper here</strong>"
        "<span>or click to choose a PDF</span>"
        "<div class='pulse' hidden><i></i></div>"
        "</div><p class='fail' role='alert'></p>"
        f"<section class='papers'><h2>Already read</h2>{listing}</section>"
        "</div>"
        "<footer><div class='inner'>"
        "<span>J26-IT-410 · Member 1 · layout-aware multimodal ingestion</span>"
        "<a href='/docs'>API reference</a>"
        "<a href='/papers'>Paper IDs as JSON</a>"
        "<a href='/health'>Health</a>"
        "</div></footer>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
