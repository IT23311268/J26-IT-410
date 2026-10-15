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
| 6. Place and score | `src/evaluate/` | Figure/table placement and IEEE format compliance score |

## Setup (once)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
# build a messy test paper to experiment with
python tools/make_sample_paper.py

# run the engine - the IEEE Word file appears in data/output/
python main.py data/input/sample_paper.pdf

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
- [ ] Stage 6 - figure placement and compliance score
