# Isolated PP1 dashboard plan

All paths below are relative to `ci-hgnn-rgd/pp1_dashboard/`. No pre-existing
source, configuration, dependency, test, model, or dataset file may be edited.
Active branch: `feature/-IT23239470-Gap`; no branch changes or Git publication.

1. Snapshot pre-existing source hashes and the consumed published artifacts.
2. Build a standalone FastAPI reader for checksum-verified Stage 06 SQLite and
   JSON, plus verified Stage 04 context text and evaluation sidecars. Use only
   SELECT queries against read-only SQLite connections; bounded queries and
   explicit unavailable states. No scientific pipeline imports or model loading.
3. Build a responsive six-page React interface with local assets, Recharts and
   Cytoscape. Separate real and synthetic modes at every API request and view.
4. Implement isolated synthetic fixtures and tests, install separate environments,
   verify desktop/mobile navigation and graph interactions in a real browser.
5. Re-run the unchanged research tests, compare isolation hashes, and document
   exact launch commands and the PP1 walkthrough.

Intended new files:

- `README.md`, `PP1_DEMO_GUIDE.md`, `.gitignore`, `.env.example`, `pyproject.toml`
- `docs/IMPLEMENTATION_PLAN.md`, `docs/API_CONTRACT.md`, `docs/VERIFICATION.md`
- `backend/requirements.txt`, `backend/requirements.lock`
- `backend/app/__init__.py`, `backend/app/main.py`, `backend/app/schemas.py`
- `backend/app/adapters/__init__.py`, `backend/app/adapters/artifacts.py`
- `demo_data/dataset.json`
- `frontend/package.json`, `frontend/package-lock.json`, `frontend/index.html`
- `frontend/tsconfig.json`, `frontend/vite.config.ts`
- `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/styles.css`
- `frontend/src/types.ts`, `frontend/src/api.ts`
- `frontend/src/components/UI.tsx`, `frontend/src/components/GraphCanvas.tsx`
- `frontend/src/pages/Overview.tsx`, `frontend/src/pages/Intents.tsx`
- `frontend/src/pages/Topics.tsx`, `frontend/src/pages/Graph.tsx`
- `frontend/src/pages/Gaps.tsx`, `frontend/src/pages/Validation.tsx`
- `tests/conftest.py`, `tests/test_api.py`, `tests/test_isolation.py`
- `tests/test_artifact_failures.py` for checksum and synthetic-provenance rejection
- `tests/dashboard.spec.ts`, `tests/playwright.config.ts`
- `scripts/snapshot_isolation.py`, `scripts/start.ps1`

Generated environments, dependency caches, build output, screenshots, baseline
hashes and test reports stay within this directory and are locally ignored.
Small additional files are permitted only inside this prototype if required to
make these features independently installable and testable.

Observed input: complete graph
`graph-2026-10-22T09-45-15.768394+00-00-c19bfcea`, 2,000 papers, 6,406
source-local authors, 17 topics, 6,406 authorships, 1,653 topic associations.
The 10,985 Stage 04 predictions use `tfidf_logistic_regression` and are unlinked.
There are no verified citation facts or investigation positives in this graph.
Stage 08 remains conceptual. Stages 07–12 are not implemented by this dashboard.
