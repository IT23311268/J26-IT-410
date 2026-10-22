# Stage 06 scholarly graph construction

Baseline before implementation: **472 passed, 15 deselected**, 161.00 seconds
on Windows/Python 3.11 with the previously requested SciBERT job running.

Implementation plan:

1. Validate actual Stage 03/04/05 manifests, stable IDs, topic versions and text hashes.
2. Stage nodes, relations and original evidence in indexed SQLite, with bounded reads.
3. Reassess all temporal evidence; exclude future models and unknown historical relations.
4. Build deterministic mappings, masked features and separate evidence-tier relations.
5. Validate and atomically publish Parquet sidecars plus safe HeteroData tensors.
6. Exercise offline regressions and bounded real builds; report observed limitations.

The frozen `configs/graph.yaml` and Stage 01 schema remain unchanged. Stage 06
uses `configs/graph_stage06.yaml` with `graph_features.yaml` and
`graph_temporal.yaml`. Stage 07 training and Stage 08 gap scoring are not run.

## Architecture and compatibility

`graph/stage06_config.py` validates all options and project paths. `inputs.py`
checks published manifests and consumed file checksums. Stage 05's recorded
Stage 03 corpus hashes must match, and topic model, snapshot, prediction model,
embedding dimensions and paper text hashes must agree. `assembly.py` joins by
stable identity, not row position. `store.py` uses disk-backed SQLite indexes,
bounded Parquet batches, and a single materialized ranking pass for node indices.
`temporal.py` reassesses all source contributors at each cutoff. `audit.py`,
`pyg.py`, `workflow.py` and `cli.py` implement validation, tensors, publication,
reports and the CLI respectively. The additive `Stage06GraphBuilder` API uses
this pipeline; the old frozen `GraphBuilder.build(Corpus, ...)` guard remains.

Automatic topic selection chooses the newest completed BERTopic fit whose
training corpus names the selected Stage 03 run. Newer fits for other corpora
are skipped. An explicitly configured incompatible topic run still raises an
error, and the selected run's source checksums are verified before construction.

The pipeline registry now follows the requested numbering: Stage 06 constructs
graphs, Stage 07 is future graph learning, and Stage 08 is future gap scoring.
The old proposed Stage 06 investigation boundary was already implemented in
Stage 05; no additional model fitting is performed here. Only the registry and
its authorization expectations changed in earlier-stage regression tests.

Node IDs retain canonical or source-local namespaces. Source-local author
records with the same name remain different identities. Topics use the tuple
`(model_version, temporal_snapshot_id, topic_id)`, hashed into a stable graph key.
An outlier assignment creates no invented topic. Source categories can become
optional `research_field` nodes, separately from learned topics. Affiliations
are preserved in author sidecars; institution nodes are not constructed without
a validated source-specific affiliation adapter.

Relations preserve citing-to-cited direction, author order, individual source
citation observations, and self-citations. Exact repeated source-edge IDs are
deduplicated into an explicit duplicate-observation audit. Different source
observations with the same endpoints remain separate edges. Reverse relations
are marked as derived and refer to their original edge; they are never counted
as new citations or independent scholarly observations.

Paper-topic association is separate from human-reviewed `investigates_topic`
and `mentions_topic`. Model evidence uses `predicted_` relations. Rule candidates
stay in evidence tables by default; an explicit option permits separately named
`candidate_` relations. UNKNOWN never becomes a verified or predicted positive.
Human labels require a reviewed, matching source-text span. Numeric assignment
and evidence scores retain their method-specific semantics.

Citation intent can use edge attributes, verified context nodes, or both. Every
per-context prediction remains available. Only a verified paper pair and an
existing directed citation fact permit linkage. The mean probability vector,
context count, missing-intent mask, per-context calibration and model provenance
are retained. Aggregation is explicitly **not validated calibration**.

## Features, snapshots and leakage controls

Paper `x` contains the recorded scientific embedding; `has_embedding` identifies
missing vectors, represented as zeros. Publication year, its availability mask,
text availability, temporal eligibility and original graph index are separate
tensors. No random vectors, projections, scalers or embeddings are trained.
Author `x` is the eligible authorship count. Topic `x` is the eligible association
count, recomputed from the constructed graph; original catalog size remains
source metadata. The supplied Stage 05 catalog has no topic embeddings, so none
are fabricated.

The edge feature columns are `prob_background`, `prob_method`, `prob_result`,
`context_count`, `score`, `has_score`. Sidecars preserve the score's meaning,
label/evidence tier, calibration, model version, temporal evidence and source
references. Features from different embedding backends or dimensions are never
silently mixed. The default scientific embeddings retain Stage 05's recorded
dynamic INT8 encoder and 256-token truncation; output vectors are float32.

The default policy inherits the actual Stage 03 exploratory policy and cutoff.
It allows current descriptive analysis and explicitly carries **no historical
forecasting claim**. Future updates and model-production dates remain visible in
the temporal report. `build-snapshot` defaults to STRICT_POINT_IN_TIME unless
an explicit policy is supplied. Strict mode excludes unknown historical
availability. Publication-date proxy mode reports its approximation and still
excludes known future updates, relations and learned representations. Original
ingestion dates never establish historical availability.

Explicit `temporal.split_cutoffs` create chronological masks. The `temporal_view`
API removes invisible nodes and edges, hides later-produced embeddings, computes
intent aggregates using only eligible contexts, and recomputes author/topic
counts. `original_index` preserves the mapping to the published graph sidecars.
Official source split metadata is retained; no OGBN or SciCite benchmark split is
silently reinterpreted as a heterogeneous graph split.

`masked_relations` and loader-time masks hide target relations and their derived
reverse edges. Masking citations also hides context paths that could reveal
those pairs. Masking authorship or topic association clears the corresponding
count feature. A Stage 07 experiment must explicitly define its prediction task,
label visibility and related relation masks; the graph is not an automatic
leakage-free train/test experiment.

## Storage and API

Each `data/processed/stage06/GRAPH_VERSION/` contains:

- `nodes/*.parquet`, `edges/*.parquet`, and `mappings/*_index.parquet`.
- `evidence/*.parquet`, including original predictions, unlinked contexts,
  assignments, investigation evidence, feature provenance and exclusions.
- `graph.sqlite`, an indexed, lossless construction and trace database.
- `tensors/heterodata.safetensors` and `tensors/heterodata.json`.
- `manifests/graph_manifest.json`, temporal, feature and provenance manifests.
- `reports/` with measured schema, statistics, coverage, integrity and readiness.

Parquet includes queryable IDs and a lossless `record_json` column. Small Arrow
row groups bound transport buffers. The full HeteroData object is reconstructed
from safetensors and JSON, avoiding pickle and `torch.load(...weights_only=False)`.
Checksums establish local artifact integrity, not the authenticity of an
arbitrarily replaced manifest. Use outputs from trusted source workflows.
Publication uses an atomic directory rename with bounded Windows lock retries;
failed pending directories are recoverable and never accepted by the loader.
Existing validated versions cannot be overwritten. Interrupted builds restart
into a new version; automatic phase resume is not implemented.

```python
from ci_hgnn_rgd.graph.pyg import load_hetero_graph, temporal_view

graph = load_hetero_graph("GRAPH_VERSION", root="data/processed/stage06")
# For an explicitly configured chronological experiment:
# train_graph = temporal_view(graph, "train")
# For a link-prediction experiment, supply masked_relations=("cites",).
```

`export_heterodata(data, directory)` exports an independently validated tensor
bundle. `Stage06GraphBuilder().build(config)` uses the same versioned pipeline
as the CLI. Reports are also copied into
`artifacts/reports/stage06/GRAPH_VERSION/`; `latest.json` points to the primary
exploratory graph after the real temporal audits.

The PyG API follows the official
[HeteroData documentation](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.data.HeteroData.html).
Install the local optional stack after the appropriate Torch wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage06.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

## CLI

Run from `ci-hgnn-rgd/`. All commands use the Stage 06 configuration, while
foundation `validate-config` continues to use the frozen foundation config.

```powershell
ci-hgnn-rgd graph inspect-inputs
ci-hgnn-rgd graph validate-contracts
ci-hgnn-rgd graph estimate-resources
ci-hgnn-rgd graph build --config configs/graph_stage06.yaml --max-papers 10000
ci-hgnn-rgd graph build-snapshot --cutoff 2025-12-31
ci-hgnn-rgd graph build-snapshot --cutoff 2025-12-31 --temporal-policy PUBLICATION_DATE_PROXY
ci-hgnn-rgd graph validate --graph-version VERSION
ci-hgnn-rgd graph statistics --graph-version VERSION
ci-hgnn-rgd graph export-pyg --graph-version VERSION
ci-hgnn-rgd graph trace-node --graph-version VERSION --id NODE_ID
ci-hgnn-rgd graph trace-edge --graph-version VERSION --id EDGE_ID
ci-hgnn-rgd graph audit-temporal --graph-version VERSION
ci-hgnn-rgd graph report --graph-version VERSION
```

Applicable commands support `--config`, `--project-root`, `--output-dir`,
`--max-papers`, `--field`, `--start-year`, `--end-year`, `--snapshot-id`,
`--temporal-policy`, `--cutoff`, `--mask-relation`, `--offline` and `--dry-run`.
Invalid arguments and incompatible inputs return nonzero exit status.

The 10,000-paper default is an upper limit, not a claim that this many papers are
available. The current validated source corpus has 2,000 distinct papers.
SQLite cache, Arrow batches, per-record bytes and tensor payload have separate
budgets. JSON metadata is capped at 64 MiB. Tensors remain an in-memory sparse
edge representation with an explicit byte budget; there is no distributed
partitioned tensor runtime or million-paper performance claim. Connected
components use sparse union-find up to 200,000 total nodes and are explicitly
reported as uncomputed above that budget. No NetworkX or dense adjacency is used.

## Actual inputs and observed real graph

The primary published graph is
`graph-2026-10-22T08-49-03.594435+00-00-feaa91f5`.
Its safe HeteroData bundle is at
`data/processed/stage06/graph-2026-10-22T08-49-03.594435+00-00-feaa91f5/tensors/heterodata.safetensors`.
The corresponding report directory is
`artifacts/reports/stage06/graph-2026-10-22T08-49-03.594435+00-00-feaa91f5/`.

Consumed inputs were:

- Stage 03 real arXiv run `run-2026-10-21T16-41-01.298401+00-00-918c4a52`.
  It contains 2,000 canonical papers and the same 2,000 source-local copies;
  verified canonical IDs prevent duplicate paper nodes. It supplies 6,406
  source-local authorships and source provenance. No citation table observations
  exist in this corpus. Stage 06 reads these processed records, not the original
  multi-gigabyte raw snapshot again.
- Stage 05 BERTopic run `run-2026-10-21T17-33-36.960285+00-00-7ee9c89c` and its
  evidence export `run-2026-10-21T17-35-32.916891+00-00-c623bb3e`: 17 topics,
  2,000 assignments including 347 outliers, 1,653 investigation evidence records,
  and 2,000 aligned 768-dimensional SPECTER2 vectors.
- Stage 04 full-data baseline prediction export from
  `run-2026-10-21T15-42-06.722377+00-00-e4c36987`: 10,985 real SciCite context
  predictions. Its recorded training view contains 8,222 real training contexts.
  This export is independent of the continuing SciBERT training process.

| Observed node or relation | Count |
| --- | ---: |
| paper nodes | 2,000 |
| source-local author nodes, not verified unique people | 6,406 |
| discovered topic nodes | 17 |
| `paper -> authored_by -> author` | 6,406 |
| `paper -> associated_with_topic -> topic` | 1,653 |
| derived `author_of` reverse edges | 6,406 |
| derived `associated_with_paper` reverse edges | 1,653 |
| citation facts | 0 |
| verified or model-predicted investigation edges | 0 |
| citation-context predictions preserved as unlinked | 10,985 |

Paper-topic coverage is **1,653/2,000 = 82.65%**; the remaining 347 papers are
Stage 05 outliers. Citation-intent linkage is **0/10,985 = 0%**. Investigation
evidence contains **304 RULE_CANDIDATE** and **1,349 UNKNOWN** records, with no
human-verified or model-predicted positive labels. None becomes a positive graph
relation. No field or institution nodes were enabled in this real build.

All 2,000 scientific embeddings aligned by stable ID and verified text hash.
The missing-embedding rate is **0%**; text and publication year are also present
for all admitted papers. The tensor payload is **6,936,455 bytes**. The graph
has **364 weak connected components** and zero isolated nodes under the combined
forward scholarly relationships. Reverse edges do not increase factual counts.
Integrity and provenance validation passed with **zero critical violations**.

The build took **149.40 seconds** on this host while the previously authorized
SciBERT CPU job continued. Windows reported peak physical working set
**295,686,144 bytes** and peak process-private committed memory
**809,144,320 bytes** for the graph worker. These are observed host-specific
measurements, not a 296 MB memory guarantee. Resource JSON preserves the full
measurement. No embeddings were regenerated and no GPU was needed.

## Actual temporal audits and readiness

At the **2025-12-31** cutoff:

| Policy | Papers | Source-local authors | Topics | Result |
| --- | ---: | ---: | ---: | --- |
| exploratory, primary graph | 2,000 | 6,406 | 17 | descriptive current evidence only |
| strict point in time | 0 | 0 | 0 | historical paper availability is unverified |
| publication-date proxy | 1,985 | 6,378 | 0 | 15 later-updated papers excluded |

Strict audit version is `graph-2026-10-22T08-47-02.827881+00-00-9f7f4335`;
proxy audit version is `graph-2026-10-22T08-47-28.256657+00-00-1936dae8`.
Both exclude the later-fitted topic representation and scientific embedding
features. The strict empty artifact is loadable and explicitly does not claim
a validated historical dataset. Proxy authorship is an approximation, not
verified historical relationship visibility.

All 2,000 original papers and 6,406 source-local authorships have unknown
historical availability. The exploratory graph retains 15 future paper updates,
28 corresponding authorship observations and 17 later-produced topic nodes;
these are explicitly counted in temporal reports. The production time of a
topic model is known, while its underlying historical corpus availability is
not established. Ingestion timestamps are never substituted for that evidence.

The graph is technically loadable for Stage 07 through the documented API.
The intended citation-intent-aware learning experiment still lacks verified
citations in this arXiv corpus, verified links for SciCite predictions, and human
investigation annotations. Real citation aggregation, human investigation and
institution graphs cannot be empirically verified with these supplied sources;
their supported paths are covered by explicit synthetic fixtures. Larger
10,000–50,000-paper builds and distributed partitions were not executed because
the validated corpus contains 2,000 papers. No GNN training, gap scoring or
research-gap validation was performed.

## Files and execution evidence

Created files are the nine Stage 06 graph modules, three graph config files,
`requirements-stage06.lock`, this document, `tests/stage06_helpers.py`,
`test_stage06_graph.py`, `test_stage06_cli.py`, and
`test_stage06_real_integration.py`. Existing files changed are graph/builder.py,
cli.py, pipeline.py, pyproject.toml, both repository READMEs, and the two existing
pipeline/CLI registry authorization tests. Earlier-stage source tables, model
weights and source manifests were not modified.

All 12 requested CLI actions were exercised successfully against the real
artifacts, including real node/edge traces. Strict and proxy snapshot builds are
recorded separately. Command arguments, actual stdout, stderr and exit codes
are under `artifacts/reports/stage06/verification-20261022/verification.json`.
This bundle also retains the initial build before the indexing optimization.

The separate synthetic indexing measurement used **5,000 nodes in 0.052 s** and
**20,000 nodes in 0.155 s**. It measures indexing only and does not establish
full-graph or million-record scalability. Its JSON is `index_scaling.json` in
the verification directory. An offline 20,000-node test additionally bounds
SQLite virtual-machine work to catch the former correlated rescan pattern.

Final verification:

- Full offline suite: **540 passed, 16 deselected**, 357.53 seconds. This includes
  Stages 01–05 regressions and the Stage 06 synthetic tests.
- Following the final verified-author source adapter and richer trace output,
  the complete Stage 06 test selection was rerun: **69 passed**, 131.97 seconds.
  These are **68 synthetic/offline checks and one genuine published-graph
  integration check**, not 69 real-data experiments.
- The genuine graph round trip also passed independently: **1 passed**,
  35.02 seconds. It rejects synthetic source provenance and loads/validates the
  actual published HeteroData bundle.
- Ruff lint and formatting passed (**155 Python files**); dependency consistency
  and `git diff --check` passed. Expected PyG warnings cover deliberately invalid
  indices, legitimate isolated node types, and its upstream import deprecation.

JUnit files `pytest-offline.xml`, `pytest-stage06-final.xml` and `pytest-real.xml`
and the parsed `test_summary.json` are in the verification directory. Other
earlier-stage opt-in real integration experiments were not rerun in the bounded
Stage 06 selection. Real citation-linkage, human-investigation and institution
experiments remain unexecuted because their verified inputs are unavailable.

Verified global authors can be shared across independently dated canonical
authorships. Source-local author identities still require their original scope;
the fallback never matches names. Trace-node output includes the embedding
model, revision, text hash and source index. Trace-edge output returns supporting
context predictions when linked, including for a derived reverse edge.

## Command rerun on 2026-10-22

The requested inspect, contract, resource estimate, build, validation,
statistics, PyG export and temporal audit commands all exited successfully.
The new graph version is
`graph-2026-10-22T09-45-15.768394+00-00-c19bfcea` and is now selected by
`latest.json`. The 10,000-paper cap admitted all 2,000 available distinct
papers, with the same node and relation counts documented above. Build time
was 182.25 seconds; peak Windows working set was 247,324,672 bytes and peak
private committed memory was 809,091,072 bytes.

During review, automatic topic selection was found to choose a newer fit for
an unrelated Stage 03 corpus. Two failing regression cases reproduced it.
Selection now filters by the configured corpus before choosing the latest fit;
an explicit incompatible run remains an error. Post-fix inspection selected
the same actual inputs used for this build.

Final `pytest -q` result: **542 passed, 16 deselected**, 431.06 seconds.
The separately enabled genuine-data graph round trip passed: **1 passed**,
32.09 seconds. Ruff, formatting (155 Python files), and dependency checks
passed. The default pytest marker excludes opt-in integration tests; the
genuine graph test was explicitly enabled with this new graph path.
Expected synthetic edge-bound/isolated-node warnings and an upstream PyG
deprecation remain informational.

Exact arguments, exit codes, stdout, stderr, regression reproduction and JUnit
results are saved in
`artifacts/reports/stage06/command-check-2026-10-22T09-43-16-884Z/`.
The exploratory temporal audit still flags future evidence and unknown
historical availability; successful execution does not establish a
leakage-free historical graph.
