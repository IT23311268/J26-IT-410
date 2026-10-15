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

# the tests
pytest -q
```

The IEEE manuscript is written to `data/output/<name>_IEEE.docx`.

## Gemini API key

Create a file named `.env` in this folder:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com>. The `.env` file is listed in
`.gitignore` and must never be committed.

## Status

- [x] Stage 1 - ingest (PDF + Word, single and two-column)
- [x] Stage 2 - block classification and paper reconstruction
- [x] Stage 3 - IEEE section mapping
- [x] Stage 4 - two-column IEEE Word document (tested on a real ICLR-style arXiv paper)
- [ ] Stage 5 - composition with Gemini
- [ ] Stage 6 - figure placement and compliance score
