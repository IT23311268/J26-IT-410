# IEEE Manuscript Composition Engine

Component 4 of *An AI Research Assistant Pipeline Using RAG* (SLIIT IT4010).

Takes an unstructured academic paper (PDF or Word) and produces a
publication-ready IEEE-formatted manuscript, together with a compliance score
that says how closely the output matches the IEEE template.

## Pipeline

| Stage | Folder | What it does |
|---|---|---|
| 1. Ingest | `src/ingest/` | Reads the file into text lines **with their font and position clues** |
| 2. Classify | `src/structure/` | Labels each line: title, heading, body, caption, reference |
| 3. Map | `src/structure/` | Matches the author's sections onto the IEEE skeleton |
| 4. Render | `src/render/` | Builds the two-column IEEE `.docx` (all measurements in `ieee_style.py`) |
| 5. Compose | `src/compose/` | Gemini: IEEE-style rewriting, abstract and index terms, IEEE references |
| 6. Visuals | `src/visuals/`, `src/placement/` | Crops figures, tables, algorithms and equations from the PDF; places each visual after its first mention (**the novelty**) |
| Score | `src/evaluate/` | IEEE format compliance score: 17 checks, before vs. after |
| 7. Web page | `app.py`, `src/ui/` | Dark Streamlit front end: pick or drop a paper, watch the stages, see the score, download the Word file |

All stages run through one function, `run_pipeline()` in `src/pipeline.py`, so
the command line and the web page always do exactly the same thing.

## The IEEE rulebook

Every IEEE rule the engine follows is in **one file: `src/rules/ieee_rules.py`**.
Each rule has an ID (e.g. `LAY-02`), its value, the official IEEE source it
comes from, a status, the file that applies it, and the compliance check that
tests it. All values (page size, margins, fonts, sizes, labels, word limits,
journal abbreviations, units, contractions) are defined there and nowhere
else; the renderer, the text rules and the compliance checker read them.

| Status | Meaning |
|---|---|
| Applied automatically | The engine changes the manuscript to follow the rule |
| Checked and reported | Measured and reported; a program cannot fix it safely |
| Author's judgement | Written down for completeness; needs a human decision |

A readable table of every rule: [`docs/IEEE_RULES.md`](docs/IEEE_RULES.md),
regenerated with `python tools/export_rules.py`. The web page also shows it
in the **IEEE rulebook** tab, and every compliance check shows the rule IDs
it tests.

## The IEEE rubric (rubric-based evaluation)

Every change the engine makes leaves a **receipt**: the rule ID, the exact
place in the paper, and the text before and after
(`src/rubric/evidence.py`). From these, `src/rubric/rubric.py` builds a
rubric with one line per rulebook rule:

| State | Meaning | Box starts |
|---|---|---|
| Applied by the engine | the engine changed the paper to follow the rule | ticked |
| Already followed | nothing needed changing | ticked |
| Partly / not followed | measured, and a program cannot fix it safely | unticked |
| Check yourself | the author's judgement - a person decides | unticked |
| Does not apply | e.g. table rules for a paper with no tables | not counted |

On the web page, the **IEEE rubric** tab is a guided review, one rule at a time:

1. **Left:** the rule, what the engine did, and how it changed the text
   (before → after) - "Place 3 of 19", with buttons to step through every place.
2. **Right:** the finished IEEE manuscript (`src/ui/preview.py`), with that
   rule's places **highlighted** and the current one scrolled into view. A
   moved figure is outlined and the paragraph that mentions it is marked too.
3. **"Yes, applied" / "No"**, and on to the next rule.

At the end: the **IEEE rubric-based evaluation score = rules confirmed Yes /
rules that apply to this paper**, next to the engine's automatic compliance
score. Downloads:

- `<paper>_IEEE_rubric.xlsx` - the filled rubric (Summary + Rubric sheets)
- `<paper>_IEEE_annotated.docx` - the manuscript with every rule as a
  **margin comment** at the exact paragraph, heading, caption or reference

The command line also saves `<paper>_rubric.json` and the annotated Word file.

## Setup (once)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run the web page (React)

```bash
python server.py
```

Your browser opens at <http://localhost:8000>. Any paper you put in
`data/input/` appears in the list; you can also drag in your own file. Press
Ctrl+C in the terminal to stop. The page works without an internet connection.

How it fits together:

- `frontend/` - the React page (Vite). The built page is in `frontend/dist/`
  and is already included, so **Node.js is not needed to run the demo**.
- `server.py` - a small web API. It only calls the engine (`run_pipeline`)
  and sends the results to the page as JSON. No engine logic lives here.
- `src/ui/serialize.py` - turns the engine's result into JSON for the page.

To change the React code you need Node.js (LTS) once:

```bash
cd frontend
npm install        # once
npm run build      # rebuilds frontend/dist - then run python server.py again
```

The older dark Streamlit page still works: `streamlit run app.py`.

## Run on the command line

```bash
# build a messy test paper to experiment with
python tools/make_sample_paper.py

# run the engine - the IEEE Word file appears in data/output/
python main.py data/input/sample_paper.pdf

# A4 paper instead of US Letter (rule LAY-01)
python main.py data/input/sample_paper.pdf --paper a4

# without Gemini (rules only)
python main.py data/input/sample_paper.pdf --offline

# let Gemini polish the wording (every change is checked)
python main.py data/input/sample_paper.pdf --polish

# input from the other agents: a verified content package
python main.py data/input/sample_verified_content.json

# the tests (Gemini is never called - a fake stands in)
pytest -q
```

## Two ways in

1. **A paper** (`.pdf` / `.docx`) - the engine reads it itself. This stands in
   for Member 1's ingestion while the components are not integrated.
2. **A verified content package** (`.json`) - TAF task 1, "receive verified
   content from other AI agents". The format is documented at the top of
   `src/ingest/package_reader.py`. Items marked `"verified": false` by the
   grounding gatekeeper are left out.

The IEEE manuscript is written to `data/output/<name>_IEEE.docx`.

## Gemini API key

Copy `.env.example` to a new file named `.env` in this folder and put your key in it:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite      # optional
```

Without a key the engine still runs, using rules only. Gemini answers are
cached in `data/cache/`, so re-running the same paper costs no extra requests.

Get a free key at <https://aistudio.google.com>. The `.env` file is listed in
`.gitignore` and must never be committed.

## Status

- [x] Stage 1 - ingest (PDF + Word, single and two-column)
- [x] Stage 2 - block classification and paper reconstruction
- [x] Stage 3 - IEEE section mapping
- [x] Stage 4 - two-column IEEE Word document (tested on a real ICLR-style arXiv paper)
- [x] Stage 5 - IEEE references and citations, wording rules, index terms, abstract, polish (Gemini, every output checked); verified content package input
- [x] Stage 6 - visual extraction, contextual placement, compliance score
- [x] Stage 7 - web page: bright React front end (`python server.py`); the older Streamlit page remains (`streamlit run app.py`)
- [x] IEEE rulebook - 99 rules in `src/rules/ieee_rules.py`, read by the whole engine
- [x] IEEE author block - each author listed separately with their own affiliation (TTL-06)
- [x] IEEE rubric - every rule, where it was applied, reviewer ticks and a rubric score

## Results so far

| Input | Compliance before | after | Visuals after first mention |
|---|---|---|---|
| `sample_paper.pdf` (3 pages) | 25.0% | 95.5% | 0% -> 100% |
| arXiv 2610.20597v1 (31 pages, ICLR style) | 37.6% | 98.4% | 67% -> 100% (mean distance 5.4 -> 0 paragraphs) |

Scores use 22 checks tied to the rulebook (the acronym check is reported but
not scored). With the original 17 checks the scores were 8.8% -> 94.1% and
33.0% -> 99.7%.
