"""
Contact-sheet view of a paper's rendered pages.

Pure presentation: it takes an IngestionResult and returns an HTML string.
No file or network access, which keeps it trivially testable.

This is a demo surface, not part of the Members 2/3/4 contract — they
consume the JSON. It exists so the whole of Box 2's output can be shown at
once (a progress review is a poor place to be editing page numbers into a
URL one at a time).
"""

from __future__ import annotations

import html
from urllib.parse import quote

from schema.ingestion_schema_v1 import IngestionResult

_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f6f7f9;
  --surface: #ffffff;
  --border: #d8dce3;
  --text: #11161d;
  --muted: #5b6472;
  --accent: #1f3f8f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216;
    --surface: #171c23;
    --border: #2a323d;
    --text: #e8ecf1;
    --muted: #93a0b1;
    --accent: #8fb0ff;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px 64px;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
.id { font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--muted); word-break: break-all; margin: 0 0 20px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 28px; padding: 0; list-style: none; }
.stats li {
  background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 12px; font-size: 13px;
}
.stats b { color: var(--accent); font-weight: 600; }
.grid {
  display: grid; gap: 18px;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
}
.card {
  display: block; text-decoration: none; color: inherit;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden;
}
.card:hover { border-color: var(--accent); }
.card img { display: block; width: 100%; height: auto; background: #fff; }
.card figcaption {
  padding: 8px 10px; font-size: 12px; color: var(--muted);
  border-top: 1px solid var(--border); display: flex; justify-content: space-between; gap: 8px;
}
.card figcaption b { color: var(--text); font-weight: 600; }
.empty {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 24px; color: var(--muted);
}
.empty code {
  font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--bg); padding: 2px 5px; border-radius: 4px;
}
@media (max-width: 480px) {
  .grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
}
"""


def render_gallery(result: IngestionResult) -> str:
    """Build the contact sheet for one ingested paper."""
    paper = result.paper
    title = html.escape(paper.title or paper.source_filename)
    # The id goes into both markup and a URL, and real filenames carry
    # spaces and ampersands — so escape for one and percent-encode for the
    # other. Never reuse one for the other.
    safe_id_text = html.escape(paper.paper_id)
    safe_id_url = quote(paper.paper_id, safe="")

    stats = [
        f"<li><b>{len(result.pages)}</b> page images</li>",
        f"<li><b>{paper.page_count}</b> pages in PDF</li>",
        f"<li><b>{len(result.chunks)}</b> chunks</li>",
        f"<li>method <b>{html.escape(paper.extraction_method.value)}</b></li>",
        f"<li>schema <b>{html.escape(result.schema_version)}</b></li>",
    ]
    if result.pages:
        stats.insert(2, f"<li><b>{result.pages[0].dpi}</b> dpi</li>")

    if result.pages:
        cards = "\n".join(
            f'<a class="card" href="/paper/{safe_id_url}/page/{p.page_index}" target="_blank">'
            f'<figure style="margin:0">'
            f'<img loading="lazy" src="/paper/{safe_id_url}/page/{p.page_index}" '
            f'alt="Page {p.page_index + 1}" width="{p.width_px}" height="{p.height_px}">'
            f"<figcaption><b>Page {p.page_index + 1}</b>"
            f"<span>{p.width_px}&times;{p.height_px}</span></figcaption>"
            f"</figure></a>"
            for p in result.pages
        )
        body = f'<div class="grid">{cards}</div>'
    else:
        body = (
            '<div class="empty">No page images for this paper. It was ingested with '
            "<code>rasterise=false</code> — re-ingest it with rasterising on to populate "
            "this view.</div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — page images</title>
<style>{_STYLES}</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <p class="id">{safe_id_text}</p>
  <ul class="stats">{"".join(stats)}</ul>
  {body}
</div>
</body>
</html>"""
