# Stage 02: scholarly data acquisition and auditing

Stage 02 extends the existing src package, configuration, catalogue, CLI and tests.
It performs no model training, fuzzy entity resolution, data cleaning, resplitting,
graph learning, gap scoring or expert evaluation. Stage 03 requires separate approval.

## Setup and commands

Python 3.11 and the existing Stage 01 environment are sufficient. No new runtime
dependencies were installed. NumPy/PyTorch/PyG/OGB/H2GB and Kaggle remain optional;
their actual graph/download integrations are not silently replaced by other data.

From `ci-hgnn-rgd/` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\ci-hgnn-rgd.exe data list
.\.venv\Scripts\ci-hgnn-rgd.exe data acquire --dataset scicite
.\.venv\Scripts\ci-hgnn-rgd.exe data inspect --dataset scicite --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data validate --dataset scicite --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data inspect --dataset arxiv --path data/raw/arxiv-metadata/arxiv-metadata-oai-snapshot.json --max-records 1000 --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data inspect --dataset arxiv --path data/raw/arxiv-metadata/arxiv-metadata-oai-snapshot.json --sample-size 100 --seed 42 --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data acquire --dataset arxiv
.\.venv\Scripts\ci-hgnn-rgd.exe data inspect --dataset mag_year --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data inspect --dataset ogbn_arxiv --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data validate --dataset ogbn_arxiv --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data audit-identifiers --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data report --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data readiness --offline
```

The arXiv paths above must point to genuine manually downloaded local files; they
are not bundled. Dataset aliases map to the existing catalogue keys. Every data
command accepts `--config`, `--project-root`, `--data-dir`, `--output-dir`, and
`--offline`. Inspection/validation also accept record limits, reservoir sample size,
seed, verified upstream `--version-id`, graph `--max-edges`, and explicit
`--trusted-cache`. `--synthetic` labels software-test inputs and prevents real
readiness claims. Check command-specific `--help` for scope.

Inspection never downloads automatically. `data acquire` is an explicit acquisition
action; the Stage 01 `allow_downloads: false` setting continues to forbid implicit
downloads. The default 64 MiB budget is configurable as `data.max_download_bytes`.
Larger acquisitions require deliberate budget configuration and resource review.
Kaggle acquisition first verifies credential/CLI availability and source size; it
uses the official CLI and never uses `--force` or `--unzip`. Existing user files
are not overwritten or deleted. Resume uses owned `.part` metadata plus HTTP
Range/If-Range; servers that ignore Range restart only the owned partial.

Exit status 0 means the command executed its declared scope. Inspection can return
0 with partial/invalid data findings; `data validate` returns 1 for failed checks,
including cross-split exact-context overlap. Missing sources, blocked dependencies,
checksum failures and acquisition failures return nonzero. Reporting remains usable
when independent optional datasets are blocked.

## Evidence and generated files

Every inspection writes a unique directory under `artifacts/reports/inspections/`
containing `inspection.json`, canonical `staging.jsonl`, and `identifiers.sqlite`.
Staging envelopes retain valid/invalid status, issue codes and exact raw record
locators/checksums. Invalid records are not silently discarded. Raw text and raw
identifiers are preserved; derived comparison keys are separate. Missing identifiers,
dates, availability times and unknown upstream versions remain null.

`DatasetLoader.iter_staging` streams canonical staging records; `DatasetLoader.inspect`
and source-specific adapters provide audits. The old `load(...)->Corpus` interface
still refuses to fabricate the resolved IDs/full provenance required by the stricter
Stage 01 graph-ready `Corpus`. Resolved Corpus creation belongs after Stage 03.
Graph adapters expose official constructors, bounded array checks and `iter_edges`
with explicit dataset-specific node-index namespaces. They never convert whole
graphs to NetworkX or infer citation intent from graph labels.

The persistent registry is `artifacts/reports/dataset_registry.json`. It records
actual actions, status history, provenance, checksums, counts, schema, quality,
identifier coverage, errors and report locations. The Stage 01 catalogue remains
the planned-source interface. Registry/report JSON writes are atomic. The registry
uses a single writer; parallel writers need external serialization. Input files
must remain unchanged throughout an inspection.

`data report`, `data readiness` and `data audit-identifiers` generate these pairs:

- `dataset_schema_report.json` / `.md`
- `data_quality_report.json` / `.md`
- `identifier_compatibility.json` / `.md`
- `stage02_readiness.json` / `.md`

JSON includes inspection configuration, fingerprints, scope, timestamps and source
paths. JSON is authoritative; Markdown interprets the findings. Generated reports
and raw data remain ignored by Git. Preserve a reviewed copy separately if sharing.
Report regeneration updates summary files; per-inspection directories remain intact.
Source access observations are retained in `data/raw/source_audits/`; fetched Python
files are inspected as text and never executed.

## Actual integration findings

The [official SciCite archive](https://github.com/allenai/scicite) was acquired:
23,189,911 bytes, local SHA-256
`711ece2c4e61d116c8ae5bb07e9fbb2ee9ff7bba004b4cab7fbd0ac3af499193`.
This is a locally computed checksum, not an independently published trusted hash.
The upstream release version is unknown. The repository's Apache-2.0 licence and
citation instructions were inspected; retain release-specific attribution.

The three principal split files contain 11,020 inspected records. All principal
labels were background/method/result. The actual text key is `string`; README
prose says `context`. There are 35 records containing nonstandard JSON numeric
constants; they are flagged while preserving their usable context text and raw
locators. Inspection found one exact duplicate object, four repeated-context
occurrences, and two context groups spanning splits. The original train/dev/test
assignments remain unchanged. These are data-quality findings, not model results.

Auxiliary scaffold files exist in the archive but are different tasks. They are
not interpreted as principal citation-intent splits or additional SciCite classes.
Principal source `id` values repeat; `unique_id` is profiled separately as a candidate
annotation key. Repeated IDs alone do not prove duplicated scholarly entities.
Identifier presence/40-character hexadecimal patterns do not prove global resolution.

The [Cornell Kaggle source](https://www.kaggle.com/datasets/Cornell-University/arxiv)
was accessed for file metadata. It advertised `arxiv-metadata-oai-snapshot.json`
at 5,630,408,394 bytes. No Kaggle CLI/credentials or local metadata file were
configured. The full snapshot was not downloaded; its actual record schema and
quality remain unverified. Local streaming and bounded sampling are unit-tested.

The supplied [SpaceLearner/H2GB URL](https://github.com/SpaceLearner/H2GB) returned
404. The accessible [H2GB maintainer repository](https://github.com/junhongmit/H2GB)
was inspected as an explicitly documented alternative reference, not a substituted
dataset. Its `MAGDataset(root, name="mag-year", rand_split=False)` code uses a
processed `data.pt` cache and PyTorch/PyG/torch_sparse/OGB dependencies. Source code
also derives year labels, changes edge representation and creates random splits;
do not assume its output is temporally safe. No real MAG-Year graph was loaded.

[Official OGB documentation](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
and loading code were inspected. OGBN-ArXiv is a homogeneous directed citation
graph, with citing-to-cited direction preserved. The official API imports torch
and may load processed pickle-based caches. Neither OGB/torch nor actual graph
files were installed here. Real node counts, feature tensors, splits and raw paper
mapping remain unverified by execution. Miniature mock tests are not graph validation.

No cross-dataset overlap percentage is reported: only one genuine dataset produced
an identifier index. Missing graph mappings and arXiv input prevent a measured join.

## Identifier, temporal and resource limitations

DOI normalization removes recognized DOI prefixes/URLs and compares casefolded
values. Modern/legacy arXiv parsing separates version suffixes from base IDs. Raw
values stay intact. Bare IDs from different namespaces never match. SQLite indexed
joins aggregate common keys before pairing; title equality yields CANDIDATE, never
EXACT_VERIFIED. Multiple records per common key yield ambiguity. Exact status means
unique common reliable identifier agreement only, not an online verification or
entity merge. Coverage denominators are inspected record occurrences, not all
papers in an unavailable full dataset. No fuzzy entity resolution runs.

Schema/quality profiling is bounded in memory by record size, selected sample,
schema/category limits and a SQLite cache. Duplicate/ID indexes grow on disk.
JSONL lines are bounded by `data.max_line_bytes`; overlong/malformed lines produce
invalid locators. Full reservoir sampling scans the complete source unless a prefix
limit is supplied. Sample counts are never extrapolated as total dataset counts.
Nested field inspection is explicitly bounded. Large immutable input hashes require
a streaming pass over the source even when record inspection is limited.

Ingestion timestamps are not historical availability evidence. Stage 01 cutoff
validation stays strict; imported staging records cannot automatically qualify for
the example 2025 observation cutoff. No split allocation or learned preprocessing
is performed. Later stages must resolve publication-date precision, snapshot
availability, duplicated versions and split leakage before fitting anything.

Official graph constructors may materialize full graphs before bounded inspection.
This environment deliberately blocks missing/untrusted caches and dependencies.
Install a reviewed compatible graph stack and verify resource budgets/source
artifacts before explicitly using `--trusted-cache`. Trusted cache is a user assertion,
not a cryptographic verification. Never pass untrusted pickle/model files to it.

Optional OpenAlex/Semantic Scholar clients are opt-in, offline by default, cache
bounded JSON responses, respect configurable call/spacing budgets and bounded
429/server-error retries. API keys come from environment variables and are excluded
from stored URLs/logs. OpenAlex contact comes from `OPENALEX_CONTACT_EMAIL`.
No live enrichment was executed. No complete citation-context availability is assumed.

## Tests and handoff

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
$env:CI_HGNN_SCICITE_PATH = (Resolve-Path data/raw/scicite/scicite.tar.gz).Path
.\.venv\Scripts\python.exe -m pytest -q -m integration
```

The default suite excludes integration tests. They skip unless genuine files are
configured through `CI_HGNN_SCICITE_PATH`, `CI_HGNN_ARXIV_PATH`, `CI_HGNN_OGB_ROOT`
and `CI_HGNN_MAG_ROOT`. Real graph tests additionally require
`CI_HGNN_TRUST_GRAPH_CACHE=1`. Unit graph fixtures are explicitly synthetic.

Executed verification on Windows / CPython 3.11.9: **179 unit tests passed**, with
four integration cases deselected by default. The explicit integration run produced
**one passed (genuine SciCite), three skipped** (no configured genuine arXiv,
OGBN-ArXiv or MAG-Year files). Ruff checks and formatting passed. Version 0.2.0
built successfully as a wheel and source distribution; `pip check` reported no
broken requirements. These are software checks, not scientific model results.

Before Stage 03: review invalid/nonstandard records without overwriting raw data,
verify source identifier grain and authoritative mappings, approve version/date
precision policies and leakage handling, and obtain missing genuine inputs. The
attachment's downstream stage numbering differs from the existing proposed
registry; readiness reports retain the requested target labels without renumbering
or executing later pipeline stages.
