# PP1 walkthrough · 80–90 seconds

Before presenting, start both services using the README. Open
http://127.0.0.1:5173 at a desktop viewport. Keep Real data selected and wait
for the first artifact verification to finish. Confirm the displayed graph
version under provenance. Keep a second tab ready at `/#graph` if helpful.

**0–15 seconds — Research overview**

“This isolated dashboard reads the published Stage 06 artifacts. The current
bounded corpus contains 2,000 papers, 6,406 source-local author records and
17 provisional topics. There are no linked citation facts in this graph yet.”

Point at the counts and the citation-linkage notice. Clarify that author
records do not establish a count of globally unique researchers.

**15–30 seconds — Citation intents**

Open Citation intents and inspect one context. “The three SciCite categories
are Background, Method and Result. This export contains 10,985 baseline
predictions from TF-IDF logistic regression. It is not a SciBERT export.
Canonical paper links are unresolved; Result is not investigation ground truth.”

**30–55 seconds — Interactive knowledge graph**

Open Knowledge graph. Zoom, select a paper and show the metadata/provenance
panel. Click Explore neighborhood, then toggle an author or topic filter.
“The graph connects actual paper, source-local author and topic records.
Topic membership remains separate from investigation. We load bounded
neighborhoods, and derived reverse edges are excluded from factual totals.”

**55–70 seconds — Conceptual research gaps**

Open Research gaps. “This page previews a future review experience. Attention,
investigation and uncertainty will need scientifically reviewed definitions.
Research-gap scoring will be available after Stage 08 implementation.
There are no calculated scores, rankings or discovered gaps here.”

**70–90 seconds — Validation & progress**

Open Validation & progress. Point to the missing evidence and temporal/expert
plans. “Engineering artifact checks and scientific validation are separate.
The graph remains exploratory; verified citation linkage, investigation
annotations, historical evidence and expert evaluation are still required.
Stages 07–12 are planned, and this dashboard does not execute them.”

If genuine artifacts are missing on the presentation computer, show the
unavailable state first. Select Demo explicitly and say: “These are synthetic
interface examples, not research results.” Keep the demo banner visible.
Do not repeat the real corpus counts as if they were the displayed demo fixture.
