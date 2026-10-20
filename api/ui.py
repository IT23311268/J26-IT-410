"""
Shared look for the HTML demo surfaces.

Three pages are shown at a progress review — the home page, the
pipeline walkthrough and the baseline comparison — and they have to
read as one system, because a panel that sees three different designs
assumes three different prototypes.

No web fonts and no CDN. These pages are opened from `localhost` on a
laptop plugged into a projector, and a lecture room's network is not
something to bet a demo on. Everything here is a system font or a
system stack.

Two rules carry the whole design, and both of them are information
rather than decoration.

**Colour means a measurement.** Every piece of chrome on these pages —
the masthead, the rules, the stage numbers, the rail, every card
border — is ink, paper or grey. The only coloured things anywhere are
the two data series in the comparison and the status marks that say
whether a region landed in a section. A panel scrolling the
walkthrough therefore meets colour exactly once, at the measurement
the demo exists to show. An earlier version put a solid blue band
across the top of both pages, which spent the page's one loud note on
a label.

**Monospace means the engine measured it.** Coordinates, pixel sizes,
dpi, character counts, paper ids and extracted text are set in the
mono face; prose and labels are not. So the face itself says which
numbers came out of the pipeline and which are English.
"""

from __future__ import annotations

#: One theme. See the note in BASE_CSS for why there is no dark one.
#:
#: The greys are neutral rather than warm. A warm off-white reads as
#: stationery, and these pages are closer to an instrument's readout
#: than to a letter.
TOKENS = """
:root {
  color-scheme: light;
  --paper:    #f1f2f0;
  --sheet:    #ffffff;
  --ink:      #16181a;
  --ink-soft: #60666d;
  --rule:     #e4e5e3;
  --rule-firm:#b7bab7;
  --brand:      #0d7a5f;
  --brand-ink:  #0a5f4a;
  --brand-wash: #e6f2ed;
  --brand-deep: #0a4f3e;
  --on-brand:   #ffffff;
  --red:      #9c2b2b;
  --shade:    #f5f6f4;
  --series-baseline: #b8641a;
  --series-ours:     #1b4f9c;
  --r: 8px;
}
/* There is no dark theme, on purpose.

   It followed the laptop's own setting, which means the demo machine
   decides. A dark page is washed out by a projector — the lamp adds
   light, so a dark background comes out grey and the contrast the
   design was measured for is gone. The supervisor saw this and said
   so.

   Light is the only theme now. It is the one that survives the room. */
"""

#: Three colours, and each one means a different kind of thing.
#:
#: --brand is the application: the mark, the page you are on, a link, a
#: button. It never appears on a measurement.
#:
#: --series-baseline and --series-ours are the comparison, and nothing
#: else: amber against blue, two identities rather than a verdict.
#: ΔE 24.5 apart under protanopia, and passing under deuteranopia and
#: tritanopia too — which is why the brand colour had to be chosen
#: around them rather than the other way about.
#:
#: Measured against that pair, #0d7a5f comes back at ΔE 18.5 from the
#: blue and 21.0 from the amber, 12.9 at its worst under simulated
#: deuteranopia, protanopia and tritanopia, and 5.29:1 on white. Teal
#: and petrol were tried first and sat too close to the blue (16.2 and
#: 12.2), and a wine accent collapsed into --red at 4.3.
#:
#: There used to be a green/red STATUS pair as well, marking a region
#: placed or unplaced. It is gone, for two reasons: that pair measured
#: ΔE 6.0 apart under deuteranopia, a fail; and it was never carrying
#: the meaning anyway, because the word is printed beside it. Those
#: marks are grey now, which freed enough of the wheel for an accent
#: that clears the comparison.
#:
#: --red survives for one job: text that reports a failure.


#: Segoe UI first because the demo machine is a Windows laptop; the
#: rest of the stack keeps it reasonable anywhere else. Cascadia Mono
#: ships with modern Windows and sets figures better than Consolas.
#:
#: The scale steps 13 / 16 / 20 / 27 / 40. An earlier version ran
#: 13 / 15 / 21 / 30, where body and heading sit a factor of 1.4 apart
#: — close enough that the page reads as one undifferentiated weight
#: from the back of a lecture room. 16 to 27 is 1.7, and the headings
#: carry from a distance.
BASE_CSS = TOKENS + """
*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  padding-left: 232px;     /* the app rail, which is fixed */
  background: var(--paper);
  color: var(--ink);
  font: 16px/1.65 "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

.mono, code {
  font-family: ui-monospace, "Cascadia Mono", "Cascadia Code", Consolas,
               "SF Mono", "Roboto Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.92em;   /* mono runs large next to Segoe UI at the same size */
}

a { color: var(--brand-ink); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:focus-visible, button:focus-visible, summary:focus-visible {
  outline: 2px solid var(--brand);
  outline-offset: 3px;
}

h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: -0.02em; }
h1 { font-size: 40px; line-height: 1.08; }
h2 { font-size: 27px; line-height: 1.2; letter-spacing: -0.022em; }
h3 { font-size: 16px; letter-spacing: -0.01em; }
p  { margin: 0; max-width: 68ch; }

.soft { color: var(--ink-soft); }
.small { font-size: 13px; }

/* a value and what it counts, set as one unit */
.readout { display: flex; flex-direction: column; gap: 2px; }
.readout b { font-size: 26px; font-weight: 600; line-height: 1.05; letter-spacing: -0.025em; }
.readout span { font-size: 12.5px; color: var(--ink-soft); line-height: 1.4; }

/* The line across the top of every page.

   The page used to take its identity from whatever the paper was
   called, which fails the moment a paper arrives titled
   `arXiv:2005.11401v4 [cs.CL]` — a stamp arXiv prints down the margin
   rather than a title. The engine is named here, where it reads the
   same on every page and whatever the paper turns out to be.

   It is a rule and a wordmark, not a filled band. The band was the
   loudest thing on a page whose loudest thing should be the
   measurement at the bottom of it. */
.brand {
  background: var(--sheet);
  border-bottom: 1px solid var(--rule);
  padding: 13px 24px;
}
.brand .inner {
  max-width: 1140px; margin: 0 auto;
  display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
}
.brand b { font-size: 14.5px; font-weight: 650; letter-spacing: -0.012em; }
.brand span { font-size: 12.5px; color: var(--ink-soft); }

/* The app rail — every destination this service has, in one fixed
   column, so the pages read as one application rather than as a set
   of HTML files that happen to link to each other.

   Every entry carries a word under its icon. An icon-only rail works
   in a tool somebody uses daily; it does not work in front of a panel
   meeting the service for the first time, who would have to hover
   six shapes to find out what they are.

   It does not replace the pipeline's own rail of stages. That one
   lives inside the walkthrough and lists the six boxes; this one
   lists pages. Two rails, two jobs — an editor does the same thing
   with an activity bar beside a file tree. */
.apprail {
  position: fixed; left: 0; top: 0; bottom: 0; width: 232px; z-index: 20;
  background: var(--sheet); border-right: 1px solid var(--rule);
  display: flex; flex-direction: column;
  padding: 16px 12px 18px; overflow-y: auto;
}
/* the engine's mark and name, which is also the way home */
.apprail .top {
  display: flex; align-items: center; gap: 9px;
  padding: 2px 8px 16px; color: var(--ink); text-decoration: none;
}
.apprail .top svg { color: var(--brand); }
.apprail .top b { font-size: 13.5px; font-weight: 650; line-height: 1.25; letter-spacing: -0.01em; }
.apprail a {
  padding: 8px 10px; border-radius: 6px;
  display: flex; align-items: center; gap: 10px;
  color: var(--ink-soft); text-decoration: none;
  font-size: 13.5px; line-height: 1.3;
}
.apprail a svg { flex: 0 0 auto; stroke: currentColor; fill: none; stroke-width: 1.5;
                 stroke-linecap: square; stroke-linejoin: miter; }
.apprail a:hover { color: var(--ink); background: var(--shade); }
.apprail a[aria-current="page"] {
  color: var(--brand-ink); background: var(--brand-wash); font-weight: 600;
}
.apprail a[aria-current="page"] svg { stroke-width: 1.9; }
.apprail .foot { margin-top: auto; padding-top: 12px; border-top: 1px solid var(--rule); }

/* The stages of whichever page you are on, nested under it — the one
   place a second rail would otherwise have to go. Indented to the
   width of an icon plus its gap, so the stage names hang off the page
   name the way a sub-list should. */
.apprail .nest { margin: 2px 0 6px 20px; padding-left: 10px; border-left: 1px solid var(--rule); }

/* The pipeline, standing in the rail on a page that has no paper yet.
   Not links: there is nothing to scroll to until something has been
   read, and a dead link is worse than a plain line. */
.apprail .group { margin: 22px 0 0; padding: 0 10px 8px; font-size: 11.5px;
                  font-weight: 600; color: var(--brand-ink); letter-spacing: 0.01em; }
.apprail .railsteps { list-style: none; margin: 0 0 0 10px; padding: 0 0 0 11px;
                      border-left: 1px solid var(--rule); }
.apprail .railsteps li { display: flex; gap: 9px; padding: 5px 0; font-size: 12.5px;
                         line-height: 1.3; color: var(--ink-soft); }
.apprail .railsteps .n { flex: 0 0 auto; color: var(--brand);
                         font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
/* the stage that is not built: said once, in the one place a reader
   would otherwise assume it was */
.apprail .railsteps .todo { opacity: 0.55; }
.apprail .railsteps .todo .n::after { content: " ·"; }

@media (max-width: 900px) {
  body { padding-left: 0; }
  .apprail {
    position: static; width: 100%;
    border-right: 0; border-bottom: 1px solid var(--rule);
    padding: 10px 14px;
  }
  .apprail .top { padding-bottom: 8px; }
  .apprail .foot { margin-top: 0; border-top: 0; padding-top: 4px; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""

#: The band's markup, so the two pages cannot drift apart.
BRAND = (
    "<div class='brand'><div class='inner'>"
    "<b>Layout-aware ingestion engine</b>"
    "<span>J26-IT-410 · Member 1</span>"
    "</div></div>"
)

#: Line drawings rather than pictograms: at 20px a filled symbol turns
#: into a blob, and these have to survive a projector.
_ICONS = {
    # a sheet with its corner turned
    "papers": "<path d='M5 2.5h6.5L15 6v11.5H5z'/><path d='M11.5 2.5V6H15'/>",
    # stages, stacked in the order they run
    "pipeline": "<rect x='3' y='3' width='14' height='3.5'/>"
                "<rect x='3' y='8.25' width='14' height='3.5'/>"
                "<rect x='3' y='13.5' width='14' height='3.5'/>",
    # the pages of one paper, laid out
    "pages": "<rect x='3' y='3' width='6' height='6'/><rect x='11' y='3' width='6' height='6'/>"
             "<rect x='3' y='11' width='6' height='6'/><rect x='11' y='11' width='6' height='6'/>",
    # two runs, two heights
    "compare": "<rect x='3.5' y='8' width='5' height='9'/><rect x='11.5' y='3' width='5' height='14'/>",
    # the contract Members 2, 3 and 4 read
    "json": "<path d='M8 3c-2 0-2 1.5-2 3.5S4.5 10 4.5 10s1.5.5 1.5 3.5S6 17 8 17'/>"
            "<path d='M12 3c2 0 2 1.5 2 3.5s1.5 3.5 1.5 3.5-1.5.5-1.5 3.5.5 3.5-1.5 3.5'/>",
    "api": "<path d='M7 6 3 10l4 4M13 6l4 4-4 4'/>",
}

#: The engine's mark and name, at the head of the rail. It links home,
#: because the mark in the corner of an application is the way back
#: everywhere else, and a demo should not be the exception.
_TOP = (
    "<a class='top' href='/'>"
    "<svg width='24' height='24' viewBox='0 0 26 26' aria-hidden='true' "
    "fill='none' stroke='currentColor' stroke-width='1.6'>"
    "<rect x='3.5' y='2.5' width='19' height='21'/>"
    "<path d='M7 7h5M7 11h5M7 15h5M14 7h5M14 11h5'/>"
    "</svg><b>Ingestion engine</b></a>"
)


def _entry(key: str, href: str, label: str, current: bool) -> str:
    mark = " aria-current='page'" if current else ""
    return (
        f"<a href='{href}'{mark}>"
        f"<svg width='18' height='18' viewBox='0 0 20 20' aria-hidden='true'>{_ICONS[key]}</svg>"
        f"<span>{label}</span></a>"
    )


def app_rail(
    current: str = "", paper_id: str = "", nested: str = "", tail: str = ""
) -> str:
    """The fixed column of destinations, shared by every HTML page.

    `paper_id` is what decides the length of the rail. Four of the six
    destinations are views of one paper, and before a paper is chosen
    they have nothing to point at — so on the home page they are left
    out rather than shown greyed. A disabled control that is never
    enabled is a dead end with a tooltip; an absent one is simply the
    rail that fits where you are.

    `nested` is markup hung underneath whichever entry is current: the
    walkthrough passes its stages, so one rail carries both which page
    you are on and where you are inside it. The alternative was two
    rails down the left of the same screen, which is a navigation
    scheme explaining itself twice.

    `tail` is a block below the destinations, for whatever the page
    wants standing in the rail that is not a place to go. The landing
    page puts the pipeline there: with no paper chosen the rail is two
    entries and a great deal of nothing, and what belongs in that space
    is the thing the service does.
    """
    from urllib.parse import quote

    def nest_after(key: str) -> str:
        return f"<div class='nest'>{nested}</div>" if nested and current == key else ""

    out = [_TOP, _entry("papers", "/", "Papers", current == "home"), nest_after("home")]

    if paper_id:
        pid = quote(paper_id, safe="")
        out += [
            _entry("pipeline", f"/pipeline/{pid}", "Pipeline", current == "pipeline"),
            nest_after("pipeline"),
            _entry("pages", f"/gallery/{pid}", "Pages", current == "gallery"),
            nest_after("gallery"),
            _entry("compare", f"/compare/{pid}", "Compare", current == "compare"),
            _entry("json", f"/paper/{pid}", "JSON", False),
        ]

    if tail:
        out.append(tail)

    out.append("<div class='foot'>" + _entry("api", "/docs", "API", False) + "</div>")
    return f"<nav class='apprail' aria-label='Sections'>{''.join(out)}</nav>"
