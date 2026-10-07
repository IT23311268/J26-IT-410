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

The visual language comes from what the pipeline actually does: it
measures a page. Points, pixels, bounding boxes, a scale factor. So
the page is set like a drafting sheet — hairline rules instead of
shadows, square corners, and every number in a monospace face so
coordinates line up in columns the way they would on a drawing.
"""

from __future__ import annotations

#: Dark is a media query, not a toggle: a demo should match whatever
#: the room's laptop is already set to without anyone touching a
#: switch.
TOKENS = """
:root {
  color-scheme: light dark;
  --paper:    #eceff4;
  --sheet:    #ffffff;
  --ink:      #141c26;
  --ink-soft: #5a6674;
  --rule:     #ccd4de;
  --rule-firm:#9aa7b6;
  --blue:     #1b4f9c;
  --blue-wash:#e3ebf8;
  --green:    #0f6b4f;
  --red:      #9c2b2b;
  --shade:    #f4f6f9;
  --series-baseline: #b8641a;
  --series-ours:     #1b4f9c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:    #0e1319;
    --sheet:    #151c25;
    --ink:      #e7edf3;
    --ink-soft: #93a1b2;
    --rule:     #27313d;
    --rule-firm:#3d4a59;
    --blue:     #85adf2;
    --blue-wash:#17273f;
    --green:    #5fc79b;
    --red:      #e58b8b;
    --shade:    #111820;
    --series-baseline: #c0832c;
    --series-ours:     #5e8ed6;
  }
}
"""

#: --green and --red are a STATUS pair: they mark a region as placed or
#: unplaced, and they never appear without the word beside them
#: ("abstract", "unknown"), so nothing is carried by colour alone. They
#: are deliberately not used for the baseline-vs-us comparison, for two
#: reasons.
#:
#: The first is measurable. Run them through the palette validator and
#: #9c2b2b against #0f6b4f comes back at ΔE 6.0 under deuteranopia —
#: a fail. Around one man in twelve cannot separate that pair, and a
#: progress panel is three or four people.
#:
#: The second is that red-for-them, green-for-us is an argument rather
#: than a measurement, and the numbers are strong enough without one.
#: --series-baseline and --series-ours are two identities, not a verdict:
#: blue against amber, ΔE 24.5 under protanopia and 23.3 in dark mode,
#: all six checks passing in both.


#: Segoe UI first because the demo machine is a Windows laptop; the
#: rest of the stack keeps it reasonable anywhere else. Cascadia Mono
#: ships with modern Windows and sets figures better than Consolas.
BASE_CSS = TOKENS + """
*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 15px/1.6 "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.mono, code {
  font-family: ui-monospace, "Cascadia Mono", "Cascadia Code", Consolas,
               "SF Mono", "Roboto Mono", monospace;
  font-variant-numeric: tabular-nums;
}

a { color: var(--blue); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:focus-visible, button:focus-visible, summary:focus-visible {
  outline: 2px solid var(--blue);
  outline-offset: 2px;
}

h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: -0.015em; }
h1 { font-size: 30px; line-height: 1.15; }
h2 { font-size: 21px; line-height: 1.25; }
h3 { font-size: 15px; }
p  { margin: 0; max-width: 68ch; }

.soft { color: var(--ink-soft); }
.small { font-size: 13px; }

/* a value and what it counts, set as one unit */
.readout { display: flex; flex-direction: column; gap: 1px; }
.readout b { font-size: 24px; font-weight: 600; line-height: 1.1; }
.readout span { font-size: 12px; color: var(--ink-soft); }

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""
