# Panel questions and answers — Member 1 (ingestion)

Read this before a progress presentation. Answers are written the way
you would say them out loud: short, concrete, no hedging. If you only
have time for one section, read **Box 3** — it is the part with real
engineering decisions in it, so it attracts the most questions.

---

## The one-minute version

> "My component is the ingestion engine. It takes a reference PDF and
> produces the structured JSON that Members 2, 3 and 4 build on. Right
> now it does three things: it accepts and stores the PDF, it renders
> every page to an image, and it detects the layout regions on each page
> and recovers the correct reading order. Everything is tested — 73
> tests — and it runs as a FastAPI service you can try in a browser."

---

## Box 1 — PDF input and the schema contract

**Q: What does your component output?**
A single JSON object per paper, `IngestionResult`. It holds the paper's
metadata, the text chunks, the page images, and the layout regions.
Every field is defined in `schema/ingestion_schema_v1.py`.

**Q: Why did you lock the schema in week one?**
Because three other people build against it. If the shape of my output
is still moving, Member 2 cannot write the grounding verifier and Member
3 cannot build the graph. Locking it early means they can start before
my implementation is finished.

**Q: What happens when you need to change it?**
The version number goes up and the change is additive — new fields, never
renamed or removed ones. It has gone 1.0.0 → 1.1.0 → 1.2.0 so far, and
code written against 1.0.0 still works, because the new fields default to
empty. The changelog is at the top of the schema file.

**Q: Why PyMuPDF and not PyPDF2 or pdfplumber?**
PyMuPDF gives me three things in one library: text with per-span font
sizes, block bounding boxes, and page rendering. The layout work needs
all three. pdfplumber has no renderer; PyPDF2 gives neither geometry nor
rendering.

---

## Box 2 — page rasterisation

**Q: Why turn pages into images at all? You already have the text.**
Three later stages work on pixels, not text: layout detection reads the
page as an image, figure extraction crops from it, and the ColPali visual
retriever embeds page images directly. If each rendered its own copy we
would pay the cost three times and risk three slightly different
coordinate systems.

**Q: Why 200 dpi?**
It is the point where 8pt caption text and thin table rules are still
legible, while a twenty-page paper stays under about 100 MB of PNGs. It
is a parameter, not a constant — the endpoint takes a `dpi` argument.

**Q: PDFs are in points and images are in pixels. How do you convert?**
`PageImage.scale`, which is `dpi / 72`. It is defined once, on the
schema, and every stage uses it. Getting that conversion wrong in one
place would misplace every figure crop in the paper, so it does not get
re-derived anywhere.

---

## Box 3 — layout region detection

**Q: What problem does this solve?**
A two-column paper parsed naively comes out interleaved — first line of
the left column, first line of the right, back to the left. The
sentences are shredded and every chunk built from that text is noise.
This stage recovers the order a human actually reads in.

**Q: How does it work?**
Four steps, all geometry:
1. Get every block on the page with its bounding box and font size.
2. Find the **gutters** — vertical strips almost no text sits in. The
   columns are what lies between them.
3. Blocks that cross a gutter (a title, a full-width abstract) split the
   page into bands. Inside a band, read column by column, top to bottom.
4. Classify each block from its font size and text shape.

**Q: Did you use a layout detection model — LayoutParser, Detectron2?**
No. Those need a GPU and we do not have one yet. Geometry is enough for a
standard scholarly layout, and it has two advantages for this stage: it
is deterministic, and it runs in milliseconds. The vision model comes in
at Box 8 for the pages geometry cannot read — heavily designed layouts
and scans with no text layer.

**Q: How do you find the gutters?**
I build a profile of how much text *height* sits at each horizontal
position across the page. Columns show as tall plateaus, the gutter as a
deep notch between them.

**Q: Why two passes?**
Because of a chicken-and-egg problem. A gutter is a strip no *column*
block touches — but you cannot tell which blocks belong to a column until
you know where the columns are. My first version demanded a completely
empty strip and failed on the very first test paper: a centred title
crosses the gutter, so the strip was never empty and the page read as one
column. So pass one uses a loose threshold to get a rough idea of the
columns, which is enough to spot the blocks that span them; pass two
removes those and re-measures, and the gutter is then almost empty.

**Q: What if the page is single-column? Does it break?**
No, and it needs no special case. The loose first pass might pick up a
dip in the text, but then nearly every block spans those phantom columns,
so pass two is left with almost no coverage, the whole width reads as one
run touching both page edges, and that is discarded as margin. No
gutters, one column, plain top-to-bottom order. There is a test for it.

**Q: How do you classify a block as a heading?**
Font size relative to the page's body size, plus length. The body size is
the **median** size across the page weighted by character count — median,
so one huge title cannot drag the baseline up and make every real heading
look like body text. A block at least 12% larger than that and under 120
characters is a heading. Longer than that and it is display text, not a
heading.

**Q: How do you find captions?**
A regular expression for "Figure 3", "Fig. 2", "Table 1" at the start of
the block. The caption test runs **before** the heading test, because a
caption set in a large font would otherwise be classified as a heading.

**Q: Most figures in papers are vector charts, not images. Do you catch
those?**
Yes, and that was a real gap I had to close. PyMuPDF's text dictionary
only reports *embedded raster images* as blocks, so matplotlib and TikZ
figures were invisible. I read the vector drawing list separately, merge
paths that sit close together — a chart is drawn as dozens of separate
paths — and anything large enough becomes a figure region.

**Q: What about tables?**
`TABLE` exists in the enum but this stage never emits it. Telling a table
apart from a column of short text lines needs ruling-line analysis, which
belongs with the artifact work in Box 6. A ruled table currently merges
into one region labelled FIGURE. I would rather emit no label than put a
wrong one into the contract three other components read.

**Q: How do you know it works?**
Two ways. There are 73 tests, including one that asserts the four marked
paragraphs of a synthetic two-column paper come out in the order
ALPHA, BETA, GAMMA, DELTA — and a second test that confirms the naive
top-to-bottom order *would* have been wrong, so the first test cannot
pass for the wrong reason. And there is a visual check: the
`/layout` endpoint draws the detected regions on the page, numbered in
reading order, so you can see at a glance whether the numbers run down
the left column before crossing to the right.

**Q: What happens on a scanned PDF?**
There is no text layer, so geometry has nothing to read. It returns one
figure region covering the page rather than inventing text regions. That
is the honest answer, and it is the signal that the page needs the visual
path in Box 8. I found this by accident while testing — a scanned
52-page document produced 52 page images but only one text chunk. It is
the clearest demonstration I have of why the pipeline needs to be
multimodal.

---

## Questions about engineering practice

**Q: How do you know you have not broken anything?**
73 tests, run before every commit. Eight of them exist because of a bug
I hit: the dev server's auto-reloader restarted the process while a
record was being written and left a zero-byte file, so every later read
of that paper failed. I fixed it by writing to a temporary file and
renaming it into place, which is atomic — an interrupted save now leaves
either the old complete file or the new one, never a half-written one.

**Q: How does your work reach the rest of the team?**
Git. I work on `feature/m1-ingestion` and push there; the schema goes to
`main` so the others build against the same contract.

**Q: What is next?**
Box 4, section identification — grouping the detected headings into an
IMRaD section tree, so chunks can be tagged with the section they came
from. Then Box 5, chunking that respects those section boundaries
instead of fixed-size windows.

---

## If you do not know the answer

Say so, then say what you would do to find out. "I have not measured that
yet — I would test it on the arXiv corpus and compare against the
baseline extractor" is a better answer than a guess. The panel is
assessing whether you understand your own system, and admitting a gap you
can name is evidence that you do.
