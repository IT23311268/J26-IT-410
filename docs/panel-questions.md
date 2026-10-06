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
> and recovers the correct reading order. Everything is tested — 254
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

**Q: Not every heading is printed larger. How do you catch those?**
By weight as well as size. "2.3  Generator: BART" in the RAG paper is
bold at exactly body size, and a size-only rule missed it. A block is a
heading when it is short *and* either larger than the body text or set
predominantly bold. Both conditions are needed: a long bold passage is
emphasis, and a paragraph that opens with a bold run-in is only a few
percent bold, so it stays body text.

This one matters more than it sounds. Box 4 builds the section tree out
of the heading regions, so a missed heading does not just mislabel one
block — it loses a whole section.

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

**Q: The proposal says you extract equations. Do you?**
Yes. A display equation is detected by the font its characters are set
in: typesetters switch to a separate maths face, and that is the
giveaway. I check for CMMI, CMSY and CMEX — the Computer Modern maths
faces — plus the AMS, STIX, XITS and Latin Modern equivalents.

**Q: How do you avoid labelling every paragraph with a symbol in it as
an equation?**
I measure the *share* of a block's characters set in a maths face, not
whether any are. A body paragraph in the RAG paper had 170 ordinary
characters and 7 in CMMI — a ratio of 0.04. A display equation is
essentially all maths. The threshold is a half, and nothing real sits
near it.

**Q: Why not just match any Computer Modern font?**
Because CMR and CMBX are also the *body* face of any paper typeset
wholly in Computer Modern, which is most older LaTeX work. Matching them
would label every paragraph of those papers as an equation. Only the
faces used exclusively for mathematics are on the list.

**Q: A PDF does not store equations as units. How do you get one back?**
It does not — the RAG paper's first equation came back as ten separate
blocks: the summation sign, each subscript, each variable, each bracket.
I detect the maths blocks first, then merge the ones that sit close
together into a single region. The gap is deliberately small, so two
display equations on separate lines stay separate and fragments in
different columns cannot reach across the gutter.

**Q: How did you choose those font names?**
I measured them. There is a script in the repo,
`scripts/inspect_fonts.py`, that prints every block on a page with the
fonts it uses. I ran it on a real paper and built the rule from what was
actually there, rather than from an assumption about what might be.

**Q: A chart is full of text — axis labels, a legend. What happens to
all of that?**
It is folded into the figure. One page of the GPT-4 report has a chart
with twenty language names and twenty percentages inside it, and each
came back as its own region — about seventy on one page. Left alone
they would have become seventy chunks: "Telugu", "25.0%". That is not
prose, and it would have gone straight into the corpus Members 2 and 3
retrieve from. A text block that sits mostly inside a figure now
belongs to that figure, and its text is kept on the figure rather than
thrown away, so Box 6 and Box 8 can still use it. Captions are exempt —
one printed inside a figure's border is still a caption.

**Q: What about tables?**
Detected, from their rules. A scholarly table is set in the booktabs
style — a horizontal rule above the header, one below it, one under the
last row, and no vertical lines. Those rules are far too thin to be
figures, so the figure detector throws them away; read on their own they
are the clearest signal on the page. Two or more rules that line up
horizontally, with text between them, is a table.

This one is not cosmetic. Tables go to Member 4 for the IEEE output, and
before this every cell came back as its own region — a results table
dissolved into a scatter of numbers in the retrieval corpus.

**Q: A chart's axes are horizontal lines too. How do you keep the two
detectors apart?**
By company. An axis sits inside the drawing its own chart is made of
and a table's rule does not, so the figures are found first and any rule
inside one is that figure's axis. Without it the chain ran from the
table's top rule, straight past the caption, down to the chart's
baseline, and one region came back covering two different artifacts —
which matters because tables and figures go to Member 4 as separate
things. It was visible the moment I put the layout overlay on a page of
the GPT-4 report that happens to carry both.

**Q: A chart's tick labels are printed outside its frame. Do you lose
them?**
No, but I did at first. The containment rule only reaches text *inside*
a figure, and matplotlib draws the ticks, the axis title and the chart
title outside the plot frame, so "gpt-3.5-base", "0-shot" and "Model"
each came back as a region of their own. The chart title was bold and
short, so it was labelled a heading — which would have opened a section
in Box 4 that does not exist in the paper.

So the figure is grown to its real extent before anything is absorbed:
text that all but touches its edge is part of the graphic. I measured
it on that page — the furniture sits 4.5 to 11.6 pt from the frame,
and the nearest thing that is *not* part of the chart, its caption, is
30.3 pt away. Nothing real sits in between, so the halo is 12 pt. The
growth repeats, because the x-axis labels bring the frame down far
enough that the axis title beneath them comes within reach on the next
pass, and it only ever takes in short text, never a caption.

**Q: The caption says "Table 4". Why work it out from the drawing at
all?**
I should have asked that sooner — it is the best signal on the page
and it is the author's own. Geometry has to *infer* whether a box of
ruled lines is a table or a chart; the caption simply says. So the
division of labour is: geometry decides *where* an artifact is,
because a caption cannot give you a bounding box, and the caption
decides *what it is called*, overruling the drawing when they
disagree. A caption claims the artifact nearest to it — measured
across these pages, a caption sits 1.9 to 8.5 pt from its own
artifact and 33 pt or more from any other, so the pairing is not close
to ambiguous.

It also reaches a case no structural test can. A table pasted into a
paper as a screenshot has no rules to read and is a figure by every
geometric measure there is. Its caption still says Table, and that is
now what it is filed as — which matters, because tables go to Member 4
for the IEEE output.

**Q: Then is the drawing analysis redundant?**
No, for two reasons. It is what finds the artifact in the first place
and sets its bounding box. And not every artifact has a caption, so it
is still the fallback — the caption corrects a label, it does not
replace the detector.

**Q: A table can be drawn as a box rather than booktabs rules. Does
that confuse the two?**
It did, and the way it broke is the part worth telling. Reading a
chart's axes as part of its drawing is what makes the tick labels
reachable — but a boxed table's border is straight lines too, so it
merged into a figure as well, and a figure's rules are excluded from
the table detector. The table lost its own rules and disappeared
completely. One fix caused the other failure.

What separates them is artwork. A figure contains something that is
not a straight line — a filled bar, a curve, a marker. A box, a frame
and a set of rules are furniture, and a cluster made only of those is
not a picture of anything. Measured on that page: the table
contributes 6 straight lines and no other shape, the chart 4 lines and
8 bars. Text coverage, the guard that was already there, reads 0.17
against 0.07 — a table is mostly white space between its columns, so
it could not separate them.

**Q: Why is the size of the type part of the test, and not just the
length?**
Because PyMuPDF returns a chart's whole row of x-axis labels as one
block, and on a real chart that runs past a hundred characters —
"Anthropic-LM 0-shot Anthropic-LM RLHF gpt-3.5-base 0-shot ...". A
length test threw it out and the row came back as a paragraph. Chart
furniture is always set smaller than the body text, 6 pt against 9 pt
on that page, so size catches the long row and length catches a short
axis title set at body size. Prose fails both, because it is long
*and* at body size.

**Q: Then why does a figure's bounding box not just come from PyMuPDF?**
Because it reports one rectangle per path and a chart is dozens of
them. There was also a quieter bug here: an axis line has zero area, so
PyMuPDF marks its rectangle "empty" and I was skipping those. The
figure then stopped at the edge of the *bars* rather than the axes,
which is wrong on its own terms and is why the labels outside the frame
could not be reached in the first place.

**Q: What if a table has no ruling lines?**
It is missed, and comes back as body text. A table drawn with vertical
lines or with no lines at all has no signal this stage can read. That is
a known limit, and it is the honest place for the vision model in Box 8
to take over.

**Q: How do you know it works?**
Two ways. There are 254 tests, including one that asserts the four marked
paragraphs of a synthetic two-column paper come out in the order
ALPHA, BETA, GAMMA, DELTA — and a second test that confirms the naive
top-to-bottom order *would* have been wrong, so the first test cannot
pass for the wrong reason. And there is a visual check: the
`/layout` endpoint draws the detected regions on the page, numbered in
reading order, so you can see at a glance whether the numbers run down
the left column before crossing to the right.

**Q: You measured those thresholds on one or two pages. How do you know
they hold on other papers?**
I do not yet, and that is the right question. The *rules* carry over —
"a rule inside a figure is that figure's axis" has no number in it at
all — and two of the thresholds have a wide margin either side of them:
text coverage measured 0.71 against 0.00, the maths ratio 0.04 against
almost 1.0. The one I would not defend yet is the 12 pt figure halo,
because it came from a single page and the gap it measures scales with
the body font size.

So I wrote `scripts/check_corpus.py`. It runs the stage over a folder
of papers and reports the pages carrying the shapes every bug so far
produced — short text scattered around a figure, a table and a figure
overlapping, a page with no text layer. It does not know the right
answer; it narrows a corpus down to the pages worth opening in the
overlay. A clean run across the arXiv set is the evidence, and until I
have it the honest answer is that these numbers are measured, not yet
validated.

**Q: How do you know the checker itself works?**
By reintroducing the bugs. `tests/test_check_corpus.py` turns the halo
off and asserts the checker reports scattered labels, then lets a
chart's axes count as table rules again and asserts it reports the
overlap. A checker that cannot see the bugs it was written for is worse
than no checker, because then a clean run reads as evidence.

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
254 tests, run before every commit. Eight of them exist because of a bug
I hit: the dev server's auto-reloader restarted the process while a
record was being written and left a zero-byte file, so every later read
of that paper failed. I fixed it by writing to a temporary file and
renaming it into place, which is atomic — an interrupted save now leaves
either the old complete file or the new one, never a half-written one.

**Q: How does your work reach the rest of the team?**
Git. I work on `feature/m1-ingestion` and push there; the schema goes to
`main` so the others build against the same contract.

---

## Box 4 — section identification

**Q: What does this stage add?**
The section each region belongs to. Box 3 says a block is BODY; Box 4
says it is BODY *in the Method*. That label goes on every region and
from there onto every chunk.

**Q: Why does it matter? It sounds like metadata.**
Because the same sentence means different things in different
sections. "BM25 outperforms dense retrieval on short queries" in
Related Work is somebody else's finding being cited; in Results it is
this paper's own. Member 2 verifies claims against this text and
cannot tell those apart without the label — which is most of the point
of grounding verification. Box 5 needs it too: a chunk that straddles
a section boundary is half Method and half Results, and useless.

**Q: How do you classify a heading?**
Keywords, with the number stripped first — a heading's number says
where it sits, never what it is. "2." opens Method in one paper and
Related Work in another. The keyword must then be at the *start* of
what is left.

**Q: Why at the start, rather than anywhere in the heading?**
Because anywhere was my first version and it was wrong. "Limitations
of Prior Methods" came back METHOD — it is a subsection discussing
other people's work and merely contains the word. A real heading opens
with its own word.

**Q: Papers do not say "Method". How do you catch "Our Approach"?**
A list of the wordings papers really use: approach, model,
architecture, implementation, experimental setup for METHOD;
evaluation, experiments, findings for RESULTS. Plus the leading
"our"/"the" comes off, so "Our Approach" and "The Proposed Model" both
land.

**Q: What about a heading that names two sections — "Results and
Discussion"?**
Those cannot be decided by what is at the start, so they fall through
to the order of the keyword list. DISCUSSION sits last on purpose: a
heading pairing it with Results or Conclusion is opening a section
whose substance is the other one. Four of those combinations are
pinned by tests, so reshuffling the list fails there rather than
silently relabelling a corpus.

**Q: What happens to a heading you cannot classify?**
It returns UNKNOWN, and UNKNOWN deliberately does *not* start a new
section. "3.2 Generator: BART" is a subsection of the Method, and
letting it move us would drop the rest of the Method into a section
that does not exist. Only a recognised heading moves us — which is
what makes UNKNOWN the useful answer rather than a gap.

**Q: A page number came back as a heading. How did that happen?**
It is short, and at 10.9pt against a 9pt body it cleared the size test,
so both halves of the heading rule said yes. I found it in the JSON on
page 99 of the GPT-4 report: a centred "100" labelled HEADING.

Box 4 absorbed it without damage — "100" classifies as UNKNOWN, so the
section tree did not move, which is the UNKNOWN rule earning its keep a
second time. But Box 5 turns regions into chunks, and "100" is not
something anybody should retrieve.

A block whose text is only a number, sitting in the top or bottom
margin, is now dropped as page furniture. Both halves are needed: a
number alone is never a heading wherever it sits, but a stray figure in
the text block could be an equation number or a table remnant, and
dropping content is worse than mislabelling furniture. Measured on that
page, the number sits 37pt from the foot while the page's real heading
starts 58pt from the head, so a one-inch margin separates them
comfortably.

**Q: And the text before the first heading?**
TITLE — the title, the authors, the affiliations. That is where they
belong, and it keeps them out of the abstract.

**Q: Did the schema change?**
Yes, 1.3.0 → 1.4.0: `LayoutRegion.section` and `.section_title_raw`.
Additive, as always — both default to UNKNOWN and "" on a run that
stops at Box 3, so nothing written against 1.3.0 changes behaviour.
The raw heading is kept beside the classified label so a
misclassification loses nothing.

---

## Box 5 — section-aware chunking

**Q: What is a chunk, and why not just cut every thousand characters?**
A chunk is the unit Members 2 and 3 retrieve. Cutting every N
characters is what the baseline does, and on a real paper the cut lands
mid-word: "…interleaves the column | s and shreds every sentence…".
Three of the four baseline chunks on my test paper open in the middle of
a word. It also puts the end of the Method and the start of the Results
in one chunk, so nothing downstream can say which section a retrieved
passage came from.

**Q: Where do you cut instead?**
Two boundaries and nowhere else. A section boundary, always — a chunk is
never half Method and half Results. And a region boundary when a section
runs longer than one chunk should be; a region is a paragraph, so the
cut falls between paragraphs.

**Q: How long is a chunk?**
A target of 1200 characters, about 300 tokens. Characters rather than
tokens because tokenising here would tie this stage to one model's
tokeniser and Members 2 and 3 may not use the same one. It is a target,
not a limit: a single paragraph longer than it comes out whole, because
splitting it would put the cut inside a sentence, which is the thing
this stage exists to prevent.

**Q: What happens to figures and tables?**
They are not chunks. A figure is an Artifact — Box 6 crops it, Box 7
binds it back to the chunks that discuss it. Retrieving one as prose
would return an empty passage. Captions *are* chunks, but their own:
retrieving "Figure 7 shows accuracy on TruthfulQA" should not drag a
page of unrelated Method text with it.

**Q: Captions are short. Doesn't your minimum-size rule eat them?**
It did. A one-line paragraph or a stray heading is folded into the chunk
before it — it matches nothing alone and dilutes the corpus — and the
first version of that rule swallowed every caption, which is exactly the
coupling the caption chunk exists to avoid. Captions are now exempt.

**Q: Can you show the improvement, not just describe it?**
`/compare/{paper_id}`. The same PDF through both paths, side by side,
with the two numbers that carry the claim: chunks with no section, and
chunks starting mid-word. On my two-column test paper the baseline
scores 100% and 75%; the layout-aware run scores 0% and 0%. The baseline
is re-run live rather than stored, so it always reflects the baseline as
it stands today — a stored copy would quietly flatter us.

**Q: What is next?**
Box 6, artifact extraction — cropping each figure out of the page image
Box 2 rendered, using the bounding box Box 3 found.

---

## If you do not know the answer

Say so, then say what you would do to find out. "I have not measured that
yet — I would test it on the arXiv corpus and compare against the
baseline extractor" is a better answer than a guess. The panel is
assessing whether you understand your own system, and admitting a gap you
can name is evidence that you do.
