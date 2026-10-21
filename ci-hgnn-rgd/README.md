# CI-HGNN-RGD

Citation-Intent-Aware Heterogeneous Graph Neural Network for Research-Gap Discovery.
**Stages 01–04 are implemented.** Stages 02–03 provide acquisition, streaming
staging, schema/quality checks, conservative identity/provenance controls, and
temporal audits. Stage 04 adds SciCite citation-intent data views, leakage audits,
baselines, SciBERT fine-tuning infrastructure, safe artifacts, inference, and
reports. Raw scholarly data and generated model artifacts remain ignored by Git.

See [Stage 02 commands, findings and limitations](docs/stage02.md) for the data workflow.
See [Stage 03 cleaning, identity, integration and temporal controls](docs/stage03.md)
for versioned canonical staging, offline commands and the limits of actual linkage.
See [Stage 04 citation-intent classification](docs/stage04.md) for the actual data
view, model boundary, baseline run, controls, reports, and unexecuted SciBERT work.

## Research objective and hypothesis

The proposed system combines citation-intent classification, topic identification,
heterogeneous scholarly graphs, graph representation learning, and explainable
citation-asymmetry scoring to identify **potential** underexplored topics.
High scholarly attention paired with comparatively low direct investigation is a
hypothesis to validate independently, not an established indicator of a true gap.
Citation intent and direct investigation are separate tasks and separate contracts.

## Installation

Use CPython **3.11**. Runtime dependencies are Pydantic 2 and PyYAML 6. Python
3.11.9 on Windows x64 was used for verification. Install from the repository root:

```powershell
cd ci-hgnn-rgd
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

For development with the exact verified dependency versions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Alternatively, `pip install -e ".[dev]"` resolves the supported development
dependency ranges. The lock pins runtime, test, lint, and build dependencies for
the verified Windows/Python environment; it does not include hashes or promise
compatibility on other platforms. Linux/macOS use `.venv/bin/python` and
`.venv/bin/ci-hgnn-rgd` in place of Windows paths.

The core package remains lightweight. Stage 04 adds an opt-in `citation-intent`
extra for NumPy, scikit-learn, Matplotlib, safetensors, Transformers, and a bounded
PyTorch version. Install the host-appropriate CPU/CUDA PyTorch wheel, then use
`requirements-stage04.lock` or `pip install -e ".[citation-intent,parquet]"`.
PyG, BERTopic, sentence-transformers, Neo4j, and FastAPI remain later-stage choices.
The local Stage 04 environment was verified with CPU PyTorch; CUDA execution remains
supported by the code but requires a compatible CUDA PyTorch installation.

## Commands

From `ci-hgnn-rgd/`, without activating the virtual environment:

```powershell
.\.venv\Scripts\ci-hgnn-rgd.exe --help
.\.venv\Scripts\ci-hgnn-rgd.exe check-env
.\.venv\Scripts\ci-hgnn-rgd.exe check-env --device cpu
.\.venv\Scripts\ci-hgnn-rgd.exe validate-config
.\.venv\Scripts\ci-hgnn-rgd.exe validate-config --config configs/base.yaml
.\.venv\Scripts\ci-hgnn-rgd.exe show-pipeline
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m ci_hgnn_rgd --version
```

Stage 04 commands are explicit: training does not evaluate the held-out test
split, and test evaluation requires a separate command after selection is frozen.

```powershell
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent inspect-data
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent audit-leakage
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent train-baselines
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent train --model scibert --mode quick --offline
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent evaluate --run-id RUN_ID --split dev
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent evaluate --run-id RUN_ID --split test
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent report --run-id RUN_ID
```

See [Stage 04 documentation](docs/stage04.md) for inference, calibration,
provenance export, dependency setup, and exact command behavior.

Commands return JSON on stdout (except help/version), structured JSON logs on
stderr, and nonzero status on validation failure. Foundation commands remain
read-only; explicit `data` commands perform their named actions. Explicit
`citation-intent train` commands are the only model-training entrypoints and never
evaluate the held-out test split implicitly. `validate-config` finds defaults from the component
or repository root; from another directory, pass an explicit `--config` path.
Installed wheels require access to the external project configuration files.

`configs/base.yaml` includes seven same-directory section files. Unknown fields,
duplicate keys/sections, unsafe YAML tags, aliases, invalid labels, invalid time
boundaries, and paths outside the project root fail validation. Paths resolve
relative to the project root (the parent of `configs/`), not the working directory.
`--project-root` overrides that root explicitly. Loading creates no directories.
Later-stage `enabled` fields must remain false in Stage 01.

## Architecture and interfaces

The `src/ci_hgnn_rgd` package is the reusable integration boundary; the CLI is a
thin adapter. No team service, graph database, web framework, or shared identifier
scheme existed in the initial repository, so none is assumed.

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Frozen Pydantic contracts; JSON schema/serialization support |
| `config.py`, `logging_utils.py`, `runtime.py` | YAML, scoped JSON logging, explicit seeds/device detection |
| `data/` | Source adapters, cleaning, conservative identity, traceable integration and temporal audits |
| `nlp/` | Stage 04 citation-intent data contracts, audits, baselines, SciBERT, inference, calibration and reporting; later topic/investigation boundaries |
| `graph/`, `models/` | Typed backend-neutral graph schema and future training interfaces |
| `scoring/`, `evaluation/` | Hypothesis scoring boundary and temporal validation; no empirical metrics |
| `explainability/`, `reporting/` | Future evidence explanations and reporting interfaces |
| `pipeline.py` | Proposed stage registry, validation, input/config fingerprints and in-memory audit manifest |

Contracts include `Paper`, `Author`, `Topic`, `CitationEdge`, `CitationContext`,
`CitationIntentPrediction`, `PaperTopicEvidence`, `GraphManifest`,
`ResearchGapCandidate`, and `EvidenceReference`. Known SciCite classes are exactly
**Background, Method, Result**. Missing intent remains null. Extended taxonomies
require a separate identifier and explicit class list. Predictions are not human
ground truth. `PaperTopicEvidence` distinguishes `mentions_topic`,
`investigates_topic`, and `unknown`, retaining confidence and source evidence;
`investigates_topic` based only on citation frequency is rejected.

Graph nodes are paper, author, and topic, with optional institution/research-area
nodes. Relations include cites, authored_by, mentions_topic, and investigates_topic.
Citation intent is an optional edge attribute with its own provenance. A gap
candidate retains both signals, an explicitly versioned score, supporting evidence,
uncertainty, model version, coverage counts, and an unvalidated-hypothesis status.
Stage 01 defines these shapes; it does not compute their scientific contents.

The Python API `prepare_run(config, corpus)` validates a corpus and returns a
JSON-compatible audit manifest without writing files. It rejects synthetic evidence
unless `allow_synthetic=True` is explicitly passed. Downstream adapters must retain
raw permitted source bytes, verify hashes, and supply evidence-backed identifiers.

## Proposed twelve-stage workflow

This implementation sequence is a planning boundary, not a validated research protocol.

| Stage | Scope | Status |
| --- | --- | --- |
| 01 | Project foundation and architecture | Implemented |
| 02 | Dataset acquisition and source schema audits | Implemented; real-data blockers documented |
| 03 | Entity resolution, provenance and temporal splits | Implemented; genuine SciCite integration is partial |
| 04 | Citation-intent classification | Implemented; real SciCite baseline run and SciBERT training path documented |
| 05 | Topic identification | Stub |
| 06 | Independent direct-investigation evidence | Stub |
| 07 | Heterogeneous graph construction | Stub |
| 08 | Heterogeneous graph representation learning | Stub |
| 09 | Baselines and ablation protocols | Stub |
| 10 | Citation-asymmetry hypothesis scoring | Stub |
| 11 | Independent temporal and expert evaluation | Stub |
| 12 | Evidence explanations and reporting | Stub |

Stages 05–12 interfaces raise `NotImplementedError`. Stage 04 preserves official
SciCite assignments, fits learned baseline vocabulary on train only, selects on dev,
and reserves held-out test evaluation for an explicit command. Temporal splitting
must precede fitting any vocabulary, topic model, normalization, embeddings, intent
model, or GNN.

## Planned data sources and limitations

The catalogue records planned sources; executed acquisition/inspection state is
persisted separately in `artifacts/reports/dataset_registry.json`. See the Stage 02
report for actual local source availability. Schema expectations remain audit hints.

| Source | Intended role and constraints | Primary reference |
| --- | --- | --- |
| SciCite | Context/intent source. Verify raw text field names and label mapping on the pinned archive. | [Maintainer repository](https://github.com/allenai/scicite) |
| Cornell arXiv metadata | Titles, abstracts, authors, dates and categories; no citation contexts assumed. | [arXiv bulk access](https://info.arxiv.org/help/bulk_data.html) |
| MAG-Year / H2GB | Candidate heterogeneous graph source; exact variant, schema, IDs, temporal fields and dataset terms remain unverified. | [H2GB repository](https://github.com/junhongmit/H2GB) |
| OGBN-ArXiv | Homogeneous paper citation graph prototype, not intent or gap ground truth. | [Official dataset description](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) |
| OpenAlex / Semantic Scholar | Optional metadata/ID resolution only, requiring explicit mappings and access-policy review. | [OpenAlex](https://openalex.org/), [Semantic Scholar API](https://api.semanticscholar.org/api-docs/) |

Do not assume shared IDs, one-to-one paper versions, citation contexts, or merge
compatibility. The registry distinguishes source namespaces and explicit versions;
it never picks an implicit latest release. A code repository licence is not blanket
permission for every underlying dataset. Retain source-specific licence notes.

Strict temporal eligibility requires known publication dates plus source-version
availability and retrieval at or before the observation cutoff. Retrieving an old
paper today does not prove that today's snapshot is suitable for a historical run.
Verified historical-snapshot support would require an explicit policy extension.
Model training cutoffs are checked independently of inference timestamps.

## Security and reproducibility

- Read API credentials from environment variables in future adapters; `.env.example`
  contains names only and is not automatically loaded. Actual `.env` files are ignored.
- Raw/interim/processed data, generated artifacts, model weights, pickles, caches,
  and local environments are ignored; `.gitkeep` files preserve the requested layout.
- YAML uses a restricted SafeLoader. No code path deserializes model objects. Future
  loaders must avoid untrusted pickle/joblib and unrestricted `torch.load`, pin
  model revisions, verify checksums, and avoid `trust_remote_code=True` for untrusted models.
- Imports do not seed RNGs, import heavy ML backends, or download assets.
  `set_random_seeds` acts only when called. Set `PYTHONHASHSEED` before process start;
  seed NumPy Generator instances separately. GPU and cross-version determinism are
  not guaranteed, even when deterministic algorithms are requested.
- Manifests record versions, normalized config, counts, cutoff, seed, stable
  fingerprints, UUID and creation time. Retain exact source/environment snapshots
  separately; a hash alone cannot reconstruct the inputs or uncommitted code.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build --no-isolation --outdir artifacts/dist
```

Tests use explicitly synthetic fixtures on example.org and never stand in for
scientific validation. Coverage includes YAML safety/path handling, contracts,
taxonomies, independent investigation evidence, graph consistency, registry
integrity/checksums, temporal guards, CLI installation/import behavior, seeds,
logging and fail-fast future-stage interfaces.

See [the original foundation architecture](docs/architecture.md) for detailed
assumptions, integration duties and the next-stage checklist.
