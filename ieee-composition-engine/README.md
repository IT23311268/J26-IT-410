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
| 4. Compose | `src/compose/` | Rewrites into IEEE register, generates abstract and index terms, places figures |
| 5. Render | `src/render/` | Builds the two-column IEEE `.docx` |
| 6. Score | `src/evaluate/` | Runs the IEEE format compliance checks |

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

# stage 1: see what the reader pulls out
python main.py data/input/sample_paper.pdf

# the tests
pytest -q
```

Stage 1 writes its full result to `data/output/<name>_stage1_blocks.json`.

## Gemini API key

Create a file named `.env` in this folder:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com>. The `.env` file is listed in
`.gitignore` and must never be committed.

## Status

- [x] Stage 1 - ingest (PDF + Word)
- [ ] Stage 2 - block classification
- [ ] Stage 3 - IEEE section mapping
- [ ] Stage 4 - composition
- [ ] Stage 5 - rendering
- [ ] Stage 6 - compliance score
