# PP1 verification · 22 October 2026

Local-only implementation on branch `feature/-IT23239470-Gap`.
The working tree already contained Stage 04–06 changes and a partial dashboard
scaffold before this work. Existing upstream changes were preserved. All edits,
dependencies, test additions, build output and reports for this task remain in
`ci-hgnn-rgd/pp1_dashboard/`. No Git commit, push, merge or branch switch occurred.

## Executed checks

| Check | Result | Local evidence |
| --- | --- | --- |
| Existing research suite, unchanged | 542 passed, 16 opt-in integration tests deselected | `reports/upstream-tests.xml` |
| Dashboard API, read-only, contract, bounds, mode and isolation tests | 60 passed | `reports/api-tests.xml` |
| Browser tests, desktop and mobile Edge | 6 passed | `reports/browser-tests.json` |
| TypeScript and Vite production build | Passed | `frontend/dist/` |
| Backend Ruff check and format | Passed | Reproduce with commands below |
| Isolated backend dependency consistency | Passed | `pip check` |
| Source / selected artifact isolation hashes | 376 checked, no changes | `reports/isolation-after.json` |

The upstream suite took 456.11 seconds and used its existing ML environment.
Its four warnings concern intentionally invalid/isolated PyG test fixtures.
The dashboard suite emits one upstream Starlette TestClient/httpx deprecation
warning; all request checks pass. Browser runs also emit Node color notices.
These warnings do not represent empirical research evaluation outcomes.

## Browser checks and visual review

Verified at 1440×1000 desktop and 390×844 mobile:

- All six navigation destinations render with no uncaught page errors.
- Frontend requests successfully reach the separate backend through Vite.
- Real mode remains the default and contains no synthetic record labels.
- Cytoscape creates a canvas; zoom/fit controls operate; direct center-node
  clicks and keyboard-accessible node buttons open metadata/provenance details.
- Neighborhood exploration and visible-node limits update the graph.
- Citation filtering and context details, plus topic details, operate.
- Demo selection is explicit; the banner persists across navigation; returning
  to real mode clears demo records and their labels.
- A simulated missing-artifact API response shows unavailable without fallback.
- Document width remains inside the viewport for every page.

Screenshots were visually inspected. Files are retained locally as
`reports/overview-desktop.png`, `overview-mobile.png`, `graph-desktop.png` and
`graph-mobile.png`. Synthetic fixtures are test/interface evidence only.

## Real artifacts inspected

Selected graph:
`graph-2026-10-22T09-45-15.768394+00-00-c19bfcea`.

The real API reproduced 2,000 papers, 6,406 source-local author records,
17 topics, 6,406 authorships and 1,653 topic associations. There are no citation
facts or supported investigation edges. Derived reverse edges are not counted
as additional scholarly facts. The 10,985 citation predictions come from the
recorded TF-IDF logistic-regression baseline and retain unresolved endpoints.
The viewer joins verified source contexts and displays saved evaluation
sidecars; it runs no new inference, training or held-out evaluation.

The existing 376-file before-work baseline includes tracked/untracked source,
tests/configuration, the selected complete Stage 06 graph, consumed input files
and completed citation-intent run files. `snapshot_isolation.py verify` compares
those bytes. Unconsumed raw archives and independently active pending jobs are
outside that baseline. Application code contains no filesystem write operations
and never imports `ci_hgnn_rgd`. SQLite rejects an attempted write in a dedicated
read-only guard test. Adversarial synthetic graphs and tampered checksums fail
closed in the real-mode API.

## Reproduce

From `pp1_dashboard/`:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp reports\api-tmp --junitxml reports\api-tests.xml
.\backend\.venv\Scripts\python.exe -m ruff check backend scripts tests
.\backend\.venv\Scripts\python.exe -m ruff format --check backend scripts tests
.\backend\.venv\Scripts\python.exe -m pip check
python scripts\snapshot_isolation.py verify
cd frontend
npm.cmd run build
npm.cmd run test:e2e
```

Browser tests own and stop their servers; ports 8000 and 5173 must be free.
The Windows execution sandbox blocked initial package-network access and Vite's
parent-directory resolution. Package installation, builds and browser testing
were rerun with the execution tool's permission flow. No root-environment
installation or sandbox workaround was added to the application.

No GNN, gap scoring, real gap ranking, expert assessment or historical backtest
was executed. Scientific validation and full upstream stage acceptance remain
separate from this engineering verification.
