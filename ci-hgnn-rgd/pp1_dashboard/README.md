# CI-HGNN-RGD · PP1 Dashboard

An isolated, read-only presentation prototype for the outputs published after
Stage 06. The dashboard consumes artifact contracts; the research pipeline does
not import or depend on it. Removing this directory leaves Stages 01–12 independent.

## Capabilities

- Research overview with measured corpus counts, publication years, citation-intent
  distribution, graph statistics, coverage, artifact status and provenance.
- Citation-intent explorer with Background, Method and Result predictions, text
  search, pagination, saved confidence, source annotations and paper identifiers.
- Topic catalog with keywords, associated/representative papers, provenance,
  independent investigation evidence tiers and uncertainty.
- Cytoscape graph with zoom, pan, search, filters, one/two-hop exploration, bounded
  loading, keyboard-accessible node selection, metadata and source traces.
- Stage 08 conceptual research-gap interface with no candidate scores or rankings.
- Engineering/scientific progress, verified saved benchmark metrics and future
  temporal/expert validation plans.

Stack: React, TypeScript, Vite, Tailwind CSS, Lucide, Recharts, Cytoscape.js;
FastAPI and Pydantic. Pandas is available in the isolated backend environment;
SQLite and JSON suffice for the published graph, so PyArrow is unnecessary.
There are no external runtime fonts, image services or model downloads.

## Installation

Use Python 3.11 and Node 22.12+ (verified with Python 3.11.9 / Node 22.21.1).
Run from the repository root in PowerShell:

```powershell
cd ci-hgnn-rgd\pp1_dashboard
python -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.lock
cd frontend
npm.cmd ci
cd ..
```

Dependencies and their locks live only here. Do not install these dependencies
in the research component's `.venv`. `backend/requirements.txt` declares direct
dependency ranges; `backend/requirements.lock` records the tested resolved set.
No credentials or environment variables are required. Installation requires
package-registry access; normal use is local and offline.

## Start the dashboard

On Windows, double-click **Start Dashboard.cmd** in this directory. It starts the
backend and frontend in the background, checks that the website can reach the
API, and opens http://127.0.0.1:5173/. Running it again reuses healthy servers.
Use **Stop Dashboard.cmd** to stop only processes started by this launcher.
Logs and process records are stored locally in the ignored `.run/` directory.

This address belongs to your own computer. Internet data or access to YouTube
does not start the dashboard. If you see `ERR_CONNECTION_REFUSED`, run
**Start Dashboard.cmd** again. The servers must be started again after Windows
restarts; this launcher does not install an automatic startup service.

For a connection check without opening a browser, run from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ci-hgnn-rgd\pp1_dashboard\scripts\dashboard.ps1 -Action Status
```

The launcher refuses conflicting ports and does not terminate unrelated programs.
If startup fails, its console stays open and identifies the log to inspect.

Alternatively, open two terminals from the repository root and keep them open.

Terminal 1 — backend:

```powershell
cd ci-hgnn-rgd\pp1_dashboard\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:

```powershell
cd ci-hgnn-rgd\pp1_dashboard\frontend
npm.cmd run dev
```

Open **http://127.0.0.1:5173**. API health:
http://127.0.0.1:8000/api/health; OpenAPI documentation:
http://127.0.0.1:8000/docs. Stop each terminal with Ctrl+C.
The optional `scripts/start.ps1 -Service backend|frontend` wrappers perform the
same actions when invoked from the dashboard directory.

On macOS/Linux, use `backend/.venv/bin/python`, `npm`, and `/` path separators.
The local consumed manifests currently contain Windows absolute paths, so real
mode requires those published inputs at their recorded paths. Demo mode is portable.

`npm.cmd run build` type-checks and produces `frontend/dist/`. The demonstration
uses Vite's development proxy; `vite preview` alone has no backend proxy and is
not the documented full-stack launch path. This is a local prototype, not a
public deployment with authentication or production hosting.

## Real artifacts and provenance

Real mode is always initially selected. The read-only adapter discovers:

```text
../data/processed/stage06/latest.json
../data/processed/stage06/<graph_version>/manifests/graph_manifest.json
../data/processed/stage06/<graph_version>/graph.sqlite
../data/processed/stage06/<graph_version>/reports/graph_integrity.json
../data/processed/stage06/<graph_version>/reports/graph_statistics.json
```

The manifest must be a completed `stage06-v1` artifact. The adapter verifies the
database, consumed report checksums and all recorded input checksums. It rejects
pending graphs, count/report disagreement and synthetic provenance in any graph
node, relation or evidence row. SQLite opens with `mode=ro&immutable=1` and
`query_only=ON`; no research pipeline imports, writes, DDL, model loading or
deserialization run in application code. Published files are assumed immutable;
mtime/size changes invalidate the cached reader and require a backend restart.
Checksums establish consistency with the supplied manifest, not external
authenticity. Use trusted published manifests.

The selected Stage 06 manifest identifies the consumed Stage 03 arXiv corpus,
Stage 05 topic/evidence/embedding exports and Stage 04 intent export. Stage 04
context text is joined by context ID only after checking its recorded SHA-256
in the completed model run. Evaluation sidecars are accepted only after their
model-run checksums match. Missing/incompatible sidecars omit text/metrics;
saved graph predictions remain distinguishable from source annotations.
The acquisition registry is displayed with its own historical status; it can
predate separately published runs.

Inspected local graph:
`graph-2026-10-22T09-45-15.768394+00-00-c19bfcea`.

| Observed artifact | Count / interpretation |
| --- | --- |
| Papers | 2,000 admitted arXiv papers; all recorded publication years are 2007 |
| Author nodes | 6,406 source-local records; not verified unique people |
| Topic nodes | 17 provisional BERTopic clusters |
| Authorships | 6,406 factual edges |
| Topic associations | 1,653 `associated_with_topic` edges; 347 outlier papers |
| Citation facts / investigation positives | 0 / 0 |
| Context predictions | 10,985, unlinked; TF-IDF logistic regression, not SciBERT |
| Investigation evidence | 304 RULE_CANDIDATE, 1,349 UNKNOWN; no positive graph edges |

Observed input versions are retained in the graph manifest. Metrics shown are
the consumed run's saved baseline evaluations, not newly computed experiments.
The adapter does not scan every unrelated model run or consume a live/pending
SciBERT checkpoint.

## Real mode and demo mode

Missing real artifacts produce an unavailable state. The application never
silently substitutes demo records. Select **Demo** explicitly in the top bar to
load the small hand-authored `demo_data/dataset.json` fixture.

Demo mode keeps this banner on every page:

> DEMO MODE — SYNTHETIC DATA — NOT RESEARCH RESULTS

Synthetic record views also show **Synthetic Demo Data — Not Research Findings**.
Demo records carry `synthetic: true` and a notice in API responses. No demo
confidence, scientific evaluation metrics, expert feedback or gap ranking is
generated. Switching modes remounts page state and cancels stale requests.
Synthetic data is never copied to upstream artifact directories.

## Scientific limitations and future boundaries

- A topic membership is distinct from a textual mention and direct investigation.
  The UI preserves the actual `associated_with_topic` relation.
- Result citation intent is not an investigation label. Raw model confidence is
  uncalibrated and does not establish correctness or epistemic uncertainty.
- Citation contexts have unresolved canonical endpoints; no citation links are
  invented. Source IDs remain separate from canonical IDs.
- The current graph is `EXPLORATORY_NO_TEMPORAL_CLAIM`. Historical source
  availability and temporal generalization remain unverified. Current topic
  production dates cannot be silently used as historical features.
- Saved baseline benchmark metrics describe filtered official SciCite splits.
  Quarantines and known overlap risks limit comparisons and interpretation.
- Stage 01–06 artifact availability does not establish full stage acceptance or
  a percentage of research completion. Stages 01/02/03/05 remain “Validation
  Pending” in this viewer's independent acceptance audit; verified consumed
  evidence does not automatically certify every upstream stage check.
- The viewer initially shows a small neighborhood. Node limits are 1–150,
  depth 1–2 and at most 400 edges. Individual adjacency reads are capped;
  truncated views explicitly identify this. Page sizes are 1–100, offsets at
  most 100,000, search strings at most 160 characters. Citation predictions are
  capped at 100,000 cached records. Validation streams graph rows once at first
  real request, so startup cost scales with the selected graph.
- Stage 07 models, Stage 08 scoring, Stage 09 ablations, Stage 10 robustness,
  Stage 11 temporal/expert evaluation and Stage 12 explanations/reporting remain
  future research work. This viewer adds no scientific implementations there.
- No shared API/schema, root dependency file, existing CLI or upstream source
  was modified. No commit, push, merge or branch change was performed.

## Verification

Run from `pp1_dashboard/`:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp reports\api-tmp --junitxml reports\api-tests.xml
cd frontend
npm.cmd run build
npm.cmd run test:e2e
```

Browser tests use installed Microsoft Edge in headless mode, start their own
backend/frontend, and require ports 8000/5173 to be free. On hosts without Edge,
install a reviewed Playwright browser and update `channel` in the isolated test
configuration. The browser tests cover desktop 1440×1000 and mobile 390×844,
all six pages, graph selection/controls, topic/context panels, mode switching,
missing-artifact states and horizontal overflow. Screenshots and reports remain
under `reports/` and are ignored by Git.

Genuine artifact tests skip on a fresh checkout without published data; no
synthetic fixture is reported as genuine data coverage. Adversarial fixtures
under tests deliberately test checksum and synthetic-provenance rejection.

The unchanged upstream suite can be run from `ci-hgnn-rgd/` using its existing
environment, with all new test output kept in this prototype:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp pp1_dashboard\reports\upstream-tmp --junitxml pp1_dashboard\reports\upstream-tests.xml
```

`scripts/snapshot_isolation.py snapshot` records a local before-work baseline;
it refuses to overwrite one. `verify` compares those hashes. This session's
baseline covers existing source/test/config files, the selected published graph,
consumed inputs and completed citation-intent model runs. It does not hash the
entire raw multi-gigabyte archive or monitor independently running pending jobs.
See [verification evidence](docs/VERIFICATION.md), the
[API contract](docs/API_CONTRACT.md), and the
[60–90 second walkthrough](PP1_DEMO_GUIDE.md).
