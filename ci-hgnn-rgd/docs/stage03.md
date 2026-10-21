# Stage 03: cleaning, conservative identity resolution and integration

Stage 03 creates traceable scholarly staging tables. It does not train citation-intent,
topic or graph models, create investigation labels, score research gaps, or establish
forecasting performance. Stages 01 and 02 retain their contracts and commands. The new
configuration and CLI commands are additive.

The working integration may contain independent citation contexts, canonical paper
clusters and source-local graphs. Missing mappings remain unresolved. A disconnected
source is retained rather than forced into a single scholarly identity space.

## Architecture and existing interfaces

The package extends the existing `src/ci_hgnn_rgd/data/` modules:

| Responsibility | Implementation |
| --- | --- |
| Source-aware cleaning and recoverable staging | `cleaning.py`, `canonical_records.py` |
| Local source adapters over Stage 02 streams | `integration_adapters.py` |
| Namespaced identifiers and bounded SQLite candidate indexes | `normalization.py`, `candidate_generation.py` |
| Evidence-based match features, decisions and compatible clusters | `match_scoring.py`, `entity_resolution.py`, `deduplication.py` |
| Point-in-time eligibility, official-split auditing and graph checks | `temporal_integrity.py`, `split_management.py`, `graph_integrity.py` |
| Versioned outputs, trace index and atomic publication | `integration_manifest.py` |
| Offline orchestration and reports | `integration.py`, `stage03_reporting.py` |
| Separate validated settings and additive CLI | `stage03_config.py`, `stage03_cli.py` |

The source adapters reuse Stage 02 `RecordEvent` streaming and existing SciCite/arXiv
staging adapters. `--input-format stage02` explicitly accepts earlier canonical JSONL
envelopes. These envelopes omit some original source fields; their original raw source
locator and checksum remain the recovery path. Raw source files are never overwritten.

The earlier `EntityResolver.resolve(tuple[Paper, ...])` remains guarded because that
legacy graph-oriented schema cannot supply the provenance needed for source identity
decisions. Use `EntityResolver.resolve_staging(..., index_path=...)` or `IdentityEngine`
for provenance-bearing normalized records. No guessed identity bridges that contract.

## Local inputs and scientific meaning

| Source | Accepted Stage 03 input | Identity and integration rules |
| --- | --- | --- |
| SciCite | Official local JSONL splits or archive; Stage 02 envelopes | Original context, three labels and split are retained. Source paper IDs stay in `scicite-paper-id`; a 40-character value alone is not a verified Semantic Scholar ID. |
| Cornell arXiv | Local metadata JSONL; Stage 02 envelopes | Titles, abstracts, ordered authors, categories, raw IDs and all observed versions remain recoverable. `v1.created` and `update_date` remain distinct. |
| OpenAlex | Explicit local work metadata JSONL | Only documented source identifier fields assert DOI/OpenAlex/MAG namespaces. Inverted abstracts remain raw; they are not silently presented as original text. |
| Semantic Scholar | Explicit local paper metadata JSONL | Paper IDs and observed external mappings retain distinct namespaces. Ordinary citations do not supply context text or intent labels. |
| OGBN-ArXiv | Explicit `graph_tables` JSONL exports | Integer node indices require dataset/release/node-type scope. Original split values and directed edges are retained; no implicit benchmark download or binary deserialization occurs. |
| MAG-Year/H2GB | Explicit `graph_tables` JSONL exports | Observed node/relation types remain source-local; verified mappings are required for cross-source scholarly identities. |

SciCite is the only genuine local dataset established during Stage 02. The arXiv
metadata file, OGBN-ArXiv graph and MAG-Year graph were unavailable for actual record
integration. Optional source metadata is not fetched during Stage 03. The originally
supplied H2GB repository was unavailable during the recorded source inspection;
the documented alternative repository is source-code evidence, not an integrated graph.
Each run records its own actual source availability and processed denominators.

Official source references: [SciCite](https://github.com/allenai/scicite),
[Cornell arXiv metadata](https://www.kaggle.com/datasets/Cornell-University/arxiv),
[OGBN-ArXiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv),
[H2GB implementation](https://github.com/junhongmit/H2GB),
[OpenAlex works](https://docs.openalex.org/api-entities/works/work-object), and
[Semantic Scholar Graph API](https://api.semanticscholar.org/api-docs/graph).

For graph exports, use one observed object per JSONL line. Nodes supply `kind: node`,
`node_type`, and `node_id`; edges supply `kind: edge`, `source_node_id`,
`target_node_id`, `source_node_type`, `target_node_type`, and observed `edge_type`.
Paper, author and source-field declarations are also accepted. A `source_split` or
`split` value is retained without claiming its official status was independently
verified. Declare the actual release through source configuration or `--source-version`;
an unknown release is kept unknown, and incomplete index scope is reported.

## Cleaning and recoverability

Cleaning uses NFC Unicode composition and whitespace normalization in separate
normalized text fields. It retains original scientific text, mathematical symbols,
punctuation, citation markers, title case and source author ordering. It does not remove
stop words, transliterate names, infer author identities or guess encoding repairs.

Every cleaned row includes `raw_values`, normalized fields, transformations, issues,
missing fields, a quarantine decision and provenance. Non-finite source numbers receive
an explicit `__source_nonfinite__` marker so output is standards-compliant JSON; the
unchanged source bytes and per-record SHA-256 remain authoritative. Malformed rows
retain source locators and hashes even when their object cannot be parsed.

Invalid required contexts and unsupported labels are quarantined. Missing publication
dates remain unknown; a year does not create an invented day. Date precision is kept
as `day`, `month`, `year` or `unknown`. Suspected encoding damage is flagged and retained.
The cleaning report counts actual input, valid, corrected, quarantined and duplicate
occurrences, transformations and missing fields. Quarantines remain recoverable.

## Identifier and merge policy

DOIs retain punctuation while recognized prefixes are removed for matching. Modern
and legacy arXiv identifiers retain version suffixes separately from the work key.
OpenAlex Work IDs, Semantic Scholar Paper IDs, MAG Paper IDs and dataset-local IDs
remain distinct. Internal indices additionally require dataset, release and node type.
Source-declared identifier evidence is not an assertion that an external authority was
contacted during this offline stage.

Candidate generation uses indexed identifier and normalized-title/year blocks, bounded
candidate neighborhoods, configurable batches and SQLite. Title, author and year
similarity are review signals. Their heuristic scores are not calibrated probabilities.
Bounded retrieval may miss candidates; it does not establish exhaustive matching recall.

Decisions are `EXACT_VERIFIED`, `PROBABLE_REVIEW`, `AMBIGUOUS`, `CONFLICTED`, `UNMATCHED`
or `UNRESOLVABLE`. Only compatible source-backed authoritative identifier evidence may
automatically merge records. Conflicting verified IDs, materially contradictory metadata,
ambiguous candidates and incompatible transitive clusters remain separate for review.
Within-source exact duplicates receive decisions while raw occurrences are retained.
Distinct citation contexts remain distinct even when citing/cited paper pairs match.

Canonical paper IDs are deterministic for identical cluster source keys. Record keys
include source locators and source content; moving input files or changing cluster
membership can change rebuilt IDs. This is not a persistent global identity registry.
Incremental runs must retain their versioned identifier mappings instead of treating
an earlier cluster ID as permanently stable. Author names remain source-local and
unverified; categories remain source fields, not validated topic-investigation labels.

## Temporal integrity and split interpretation

`STRICT_POINT_IN_TIME` is the default policy. The cutoff inherits `run.cutoff` from
the base configuration when no Stage 03 cutoff is supplied. The supplied base value
is `2025-12-31T23:59:59Z`; that example cutoff does not establish that a 2025 dataset
snapshot exists. Publication year, present-day retrieval and ingestion do not establish
historical visibility of a citation, later abstract or enrichment field.

| Policy | Interpretation |
| --- | --- |
| `STRICT_POINT_IN_TIME` | Requires sufficient historical availability evidence by cutoff; unknown or future evidence is excluded from eligible historical use. |
| `PUBLICATION_DATE_PROXY` | Uses publication as an explicitly uncertain proxy; does not claim strict point-in-time availability. |
| `EXPLORATORY_NO_TEMPORAL_CLAIM` | Supports descriptive staging; does not establish leakage-free forecasting. |

All records remain in staging with temporal decisions. Excluded rows are additionally
indexed in `excluded_records`; their retention does not permit them to be used as
historical features. Canonical identity/metadata tables are staging views, not approved
historical feature matrices. Consumers must enforce recorded eligibility before fitting,
constructing snapshots or computing aggregates. Stage 03 creates no learned aggregates.

Official SciCite assignments and graph split values are preserved. The split audit
reports exact record/text overlap, normalized and bounded near-text overlap, observed
source-paper ID reuse and resolved identity overlap separately. Shared papers are an
experiment-dependent risk rather than an automatic leakage verdict. Near-duplicate
retrieval is approximate and exposes comparison limits. Official OGBN-ArXiv splits do
not become a strict historical forecasting experiment. Optional temporal/group-split
helpers do not silently replace source assignments in the integration pipeline.

Graph checks audit endpoint existence, namespace/type scope, duplicate occurrences,
self-citations, authorship relations and temporal eligibility. Self-citations are
reported rather than automatically rejected. Ordinary graph edges receive no invented
citation intent or investigation relationship. Unresolved context endpoints stay in
the independent intent dataset and remain outside the resolved canonical graph.

## Configuration and default budgets

Use the unchanged base configuration with `--config configs/base.yaml` and the separate
Stage 03 configuration with `--stage03-config configs/integration.yaml`. Includes reuse
the existing duplicate-key/alias-safe YAML loader and remain in the configuration
directory. Configured paths resolve relative to the component root; CLI output paths
are explicit overrides.

| Setting | Default |
| --- | --- |
| Input discovery | Existing Stage 02 local registry when `inputs: []` |
| Explicit inputs | One entry per dataset: `dataset`, `path`, `version`, `format`, `synthetic` |
| Processing batch / memory budget | 1,000 records / 128 MiB |
| Maximum development records | `null` (read full configured sources); set `--max-records` for bounded development |
| Sampling seed | 42; prefix record limits do not claim random sampling |
| Candidate methods / maximum candidates | `identifier`, `title_year` / 50 |
| Maximum identity cluster size | 1,000 |
| Probable title / material conflict thresholds | 0.94 / 0.35; heuristic settings only |
| Publication-year tolerance | 1 year |
| Automatic merge / conflict handling | Exact verified only / retain separate |
| Author policy / duplicate policy | Source-local only / retain all with decisions |
| Temporal policy / source splits | Strict point in time / preserve official |
| Near-text threshold / per-context candidates / comparison budget | 0.94 / 20 / 100,000 |
| Storage | `auto`: Parquet when optional PyArrow is available, otherwise explicit JSONL fallback |
| Network enrichment / overwrite | Offline local metadata only / no previous-run overwrite |

The memory setting bounds processing buffers and index caches; it is not a guarantee
that the entire interpreter or optional libraries stay below that resident-memory value.
Large, pathological rows and candidate/cluster expansions have additional guards.
Bounded synthetic scaling tests demonstrate indexed candidate behavior, not validated
throughput or capacity for millions of genuine records.

An explicit input entry can be configured as:

```yaml
integration:
  inputs:
    - dataset: scicite
      path: data/raw/scicite/scicite.tar.gz
      version: null
      format: raw
      synthetic: false
```

Keep the complete `integration.yaml` sections/includes when editing configuration.
Do not invent a release value to make unknown source evidence appear verified.

## Commands

Run from `ci-hgnn-rgd/` with the existing environment. An editable installation of
the component exposes `ci-hgnn-rgd`; the Python module form avoids shell activation:

```powershell
.venv\Scripts\python.exe -m ci_hgnn_rgd check-env
.venv\Scripts\python.exe -m ci_hgnn_rgd validate-config
.venv\Scripts\python.exe -m ci_hgnn_rgd data stage03-run --dry-run --offline
```

Processing commands execute a fresh, declared pipeline prefix, writing a new run.
They do not mutate an earlier validated run or pretend that unavailable inputs succeeded:

```text
ci-hgnn-rgd data clean --dataset scicite --offline
ci-hgnn-rgd data clean --dataset arxiv --max-records 1000 --offline
ci-hgnn-rgd data normalize-identifiers --offline
ci-hgnn-rgd data generate-match-candidates --offline
ci-hgnn-rgd data resolve-entities --offline
ci-hgnn-rgd data deduplicate --offline
ci-hgnn-rgd data integrate --offline
ci-hgnn-rgd data stage03-run --offline
```

`clean` performs cleaning only, `normalize-identifiers` adds identifier staging,
`generate-match-candidates` creates bounded candidates, `resolve-entities` produces
identity decisions and canonical papers, `deduplicate` retains deduplication decisions,
and `integrate`/`stage03-run` also creates relations, temporal and split/graph audits.
Earlier prefixes still retain their raw/provenance/temporal observations and mark
unexecuted integration outputs accordingly.

Audit commands read a complete selected run, using `latest.json` when `--run-dir` is
omitted. `integration-report` copies that run's reports to the configured report directory:

```text
ci-hgnn-rgd data audit-leakage --run-dir PATH_TO_COMPLETE_RUN
ci-hgnn-rgd data validate-integrated --run-dir PATH_TO_COMPLETE_RUN
ci-hgnn-rgd data integration-report --run-dir PATH_TO_COMPLETE_RUN
ci-hgnn-rgd data trace-record --run-dir PATH_TO_COMPLETE_RUN --id RECORD_OR_PAPER_ID
```

All commands accept `--config`, `--project-root`, `--stage03-config`, `--offline`,
`--dataset`, `--path`, `--input-format`, `--source-version`, `--synthetic`,
`--max-records`, `--batch-size`, `--memory-budget-mb`, `--seed`, `--temporal-cutoff`,
`--temporal-policy`, `--output-dir`, `--report-dir`, `--storage-format`, `--run-dir`
and `--dry-run`; processing overrides apply to processing commands. `trace-record`
also requires `--id` and accepts `--limit` from 1 to 1,000. `--path` requires a dataset.
Use `--help` for command-specific parser details. Dry runs perform no publication.

For one explicit local source:

```text
ci-hgnn-rgd data stage03-run --dataset scicite --path data/raw/scicite/scicite.tar.gz --max-records 1000 --offline
ci-hgnn-rgd data stage03-run --dataset ogbn-arxiv --path LOCAL_NODES_AND_EDGES_DIRECTORY --input-format graph_tables --source-version VERIFIED_RELEASE --offline
```

The second example needs genuine local graph table exports and a documented actual
release. It does not fetch a graph or claim that such exports are bundled.

Missing required input files, invalid configuration and failed operations return nonzero.
`validate-integrated` returns nonzero for checksum mismatches, quarantines, invalid
canonical graph edges or eligible future-information violations. A completed integration
can therefore coexist with failed quality validation; completion is not a clean-data claim.

Thin script equivalents forward all arguments to these same CLI implementations:

```text
python scripts/run_stage03.py --max-records 1000 --offline
python scripts/inspect_entity_matches.py --run-dir PATH_TO_COMPLETE_RUN --id RECORD_OR_PAPER_ID
python scripts/generate_integration_report.py --run-dir PATH_TO_COMPLETE_RUN
```

## Versioned storage and audit trail

Each run is published under `data/processed/stage03/run-.../` by default. Processing
occurs in a new `.pending-...` directory; successful publication renames that directory
and atomically updates `latest.json`. Failed runs keep a recoverable failure manifest
and do not replace the latest complete pointer. Automatic resume is not implemented;
restart processing into a new run, preserving the failed evidence when needed.

Every run creates JSONL tables for cleaned records, quarantines, papers, source-local
papers/nodes, authors, authorships, citations, citation contexts, source fields,
identifier mappings, candidates, decisions, unresolved records, temporal eligibility,
excluded records and provenance. Empty tables represent no observed rows, not missing
source acquisition. With PyArrow, nonempty tables also receive bounded Parquet
partitions under `TABLE.parquet/part-*.parquet`; nested metadata remains losslessly
available through `record_json`, with common scalar columns separately queryable.

`trace.sqlite` indexes source/canonical record references for bounded audit lookup.
`integration_manifest.json` records configuration and SHA-256, input checksums, processing
times, component/resolver metadata, actual counts, blocked sources and output checksums.
Its mapping and decision rows explain merges, retained conflicts, source field values,
transformations and temporal decisions without logging credentials.

The six report pairs are retained inside the run and copied atomically per file to
`artifacts/reports/`:

- `stage03_cleaning_report.{json,md}`
- `stage03_entity_resolution.{json,md}`
- `stage03_integration_coverage.{json,md}`
- `stage03_temporal_leakage.{json,md}`
- `stage03_graph_integrity.{json,md}`
- `stage03_readiness.{json,md}`

Each report identifies real versus synthetic observations, actual denominators,
unverified fields, unavailable sources and limitations. Undefined coverage is null,
not an invented zero-percent cross-dataset match rate.

## Verification and next-stage handoff

The recorded Stage 01/02 baseline was **179 passing offline tests**, with four genuine
integration tests deselected by the default marker policy. New synthetic tests check
cleaning, namespaces, conservative identity, bounded candidates, temporal/split/graph
controls, storage and CLI behavior. Synthetic fixtures are software evidence only.
Final executed Stage 03 results belong in the generated run reports and task handoff;
no unexecuted test, match rate or scientific result is implied by this document.

Executed verification on CPython 3.11.9 / Windows: **331 offline tests passed**,
with eight integration cases deselected by default. The explicit genuine-data run
produced **two passed (Stage 02 and Stage 03 SciCite), six skipped** because arXiv,
OGBN-ArXiv and MAG-Year inputs/exports were unavailable. Ruff checks and formatting,
version 0.3.0 wheel/source builds, `pip check`, isolated-wheel installation and its
Stage 03 dry-run command passed. PyArrow 26.0.0 was installed as the optional reviewed
binary storage extra; the core dependencies remain Pydantic and PyYAML.

The complete genuine SciCite principal-split run processed **11,020 records**,
retained **10,985 citation contexts**, quarantined **35 nonstandard-JSON records**,
and recorded **896 normalized-text corrections** and **one raw-record duplicate**.
No raw record or context occurrence was deleted. The 35 quarantines retain their
raw values, source locators and hashes. Quality validation returns nonzero because
those quarantines remain; all published output checksums matched.

Original split totals remained train **8,243**, dev **916**, test **1,861**, including
quarantines in the split audit. Observed cross-split findings were **two exact-text
groups**, **19 normalized-text groups**, and **33 heuristic near-duplicate pairs**.
The near search performed **92,468 comparisons** with **3,424 candidate-capped
contexts**; it is approximate and does not establish exhaustive duplicate recall.
The audit also found **1,302 source-local paper-ID overlap groups**, reported as
experiment-dependent risks, not globally verified identities or automatic leakage.

No paper metadata source was available: **zero canonical papers or verified paper
mappings** were produced, and cross-dataset paper integration coverage is undefined.
The **21,970 endpoints** of retained nonquarantined contexts remain unresolved.
No genuine canonical graph was formed. All **11,020 records** were excluded from
strict historical use because availability was unknown; the default cutoff does
not turn these present-day records into a historical snapshot. These are data
processing findings, not citation-intent model or research-gap evaluation results.

Run verification from the component root:

```text
pytest -q
ruff check .
ruff format --check .
pytest -q -m integration
```

Real integration tests require explicitly configured genuine local files; missing
datasets are skipped with reasons. Optional dependencies do not convert mock graph
fixtures into real-data verification.

Stage 04 can consume `citation_contexts` with original text, observed labels, source
splits and provenance, while keeping quarantines separate. Before any training, review
exact/near-duplicate split overlap, release/licence provenance and a train-only fitting
protocol. Canonical metadata and verified relationships provide later stage interfaces,
subject to temporal eligibility and unresolved mapping constraints. Citation intent is
not direct investigation; research categories are not validated investigation labels.

Stage 04 now consumes the published `citation_contexts` view through a separate,
provenance-preserving modelling adapter. See [Stage 04](stage04.md) for its split
audits, baseline results, SciBERT resource blocker, and export contract. Stages 05
and later scientific modelling remain unstarted pending approval.
