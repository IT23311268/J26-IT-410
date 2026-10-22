# Stage 05: scientific topics and independent investigation evidence

Stage 05 adds paper preparation, scientific encoders, topic models and diagnostics,
annotation workflows, conservative evidence candidates, and an annotation-gated
scientific pair classifier. Citation intent, semantic relatedness, mentions, and
direct investigation remain separate. No graph, GNN, or research-gap score is built.

The original twelve-stage registry is preserved for compatibility: its later
stages retain their legacy names/numbers. This request moves independent
investigation evidence into Stage 05 and describes the next graph handoff as
Stage 06. No later registry stage is marked implemented by this work.

## Inspection and actual execution

Before implementation the existing offline suite produced **382 passed,
11 deselected**, in 168.73 seconds. Stage 01 schemas/configs, Stage 02 registry and
adapters, Stage 03 canonical/identity/temporal/provenance outputs, Stage 04 evidence
contracts, CLI boundaries, and test coverage were inspected. Existing uncommitted
Stage 04 work was preserved.

The initial published Stage 03 run is
`run-2026-10-18T19-08-09.492600+00-00-663447b5`. Its genuine source is SciCite:
11,020 raw context records and 10,985 admitted contexts. It contains **zero paper
records with title/abstract metadata**. That run had no configured arXiv metadata
corpus, real investigation annotations, or cached scientific encoder weights.
Citation contexts were not repurposed as paper abstracts.

During the command repair, a real arXiv snapshot was located at
`C:\Users\Hafzan\Desktop\J26-IT-410\CC-GNN-RGD\data\raw\arxiv\arxiv-metadata-oai-snapshot.json`.
It contains 5,452,730,833 bytes; its locally computed SHA-256 is
`ae71fb6476eabae4ceeda3f78fcfd62fd1879df3fe679dc61afdea59f3a97062`.
The first 2,000 records were read through the existing Stage 03 pipeline and
published separately as
`data/processed/stage03-topics/run-2026-10-21T16-41-01.298401+00-00-918c4a52/`.
All 2,000 have titles and abstracts. Stage 05 removes the duplicate identities
present in both canonical and source-local tables, retaining 2,000 papers.
The default Stage 05 config now reads that separate root. The original SciCite
run and Stage 04 input path are preserved. This is a bounded file-prefix sample,
not a representative sample of the entire arXiv snapshot.

Real NMF and count-LDA fitting completed on these papers. The first fitted run is
`run-2026-10-21T16-42-11.242823+00-00-0062bc2a`, with eight topics per model.
NMF NPMI was 0.271159 and diversity 0.9625; LDA NPMI was 0.105771 and diversity
0.725. Both assigned every input paper. These are descriptive corpus metrics,
not validated topic quality or direct-investigation ground truth.

The subsequent low-memory repair completed real SPECTER2 embedding of all
2,000 papers, published as `run-2026-10-21T17-30-43.647398+00-00-2a610ff1`.
It uses the pinned base and proximity adapter, CPU dynamic INT8 linear weights,
256-token inputs, and normalized 768-dimensional float32 output. All vectors
were finite and nonzero; no duplicate vectors were found. The input cap truncated
395 abstracts (19.75%), recorded individually. This is not a full-precision,
512-token result. An offline reload with the active proximity adapter reproduced
three saved real-paper vectors exactly.

Actual quantized parameter storage was estimated at 184,024,320 bytes, versus
443,251,968 original float32 parameter bytes (97 quantized linear layers).
The observed process peak physical working set was 1,011,548,160 bytes; parameter
storage estimates exclude runtime, activation, and committed virtual memory.
The evidence is under `artifacts/reports/stage05/low-memory-20261021T172346Z/`.
The initial publication exposed a Windows open-mmap directory-rename failure;
handles are now closed before publication, failed pending runs are rejected by
the resolver, and all 2,000 validated vectors were recovered through the cache.

Real BERTopic fitting also completed, initially as
`run-2026-10-21T17-31-51.858102+00-00-72f8d161`, discovering 17 topics and retaining
347 outliers (17.35%). NPMI was 0.129251 and diversity 0.847059. Topic stability
and expert review remain unevaluated. No investigation accuracy is claimed.
The installed BERTopic/UMAP/HDBSCAN stack also passed a separate synthetic-vector
integration check; that software result is distinct from the real-paper run.
Real human annotation status is `NO_HUMAN_ANNOTATIONS`; supervised investigation
status is `NOT_TRAINED`.

The new source run explicitly uses `EXPLORATORY_NO_TEMPORAL_CLAIM`. Historical
availability is unknown for all 2,000 papers; 15 source records have updates after
the recorded 2025-12-31 cutoff. No leakage-free historical snapshot is claimed.
Stage 05 temporal reports retain the source audit and separately count selected
papers' temporal evidence. Without a current cutoff, future violations are
`null` (not evaluable), rather than a claimed zero.

The environment is Python 3.11.9, Windows, PyTorch 2.6.0+cpu, no CUDA. Available RAM
was only about 0.08–0.60 GiB during inspection, below the 1.5 GiB encoder floor and
3 GiB classifier floor. `adapters`, BERTopic, UMAP, and HDBSCAN were absent at
initial inspection; they were installed during the subsequent command repair.
No multi-gigabyte model or dataset download was started. The genuine resource
audit is under
`data/processed/stage05/run-2026-10-21T16-02-49.732943+00-00-6256b460/` and its
report bundle under the corresponding `artifacts/reports/stage05/` directory.
Later verification runs may capture different free RAM; those measurements are
preflights, not inference/training benchmarks.

Offline tests execute real NMF/LDA fitting and frozen numeric-component inference,
tiny local Transformer encoding/classification, and mocked SPECTER2/BERTopic loading
interfaces on explicitly synthetic fixtures. Their metrics are software checks,
not real scholarly findings. Optional genuine corpus/encoder/BERTopic tests are
separately marked `integration` and require explicit data/model environment flags.

Initial Stage 05 verification produced **443 passed, 14 deselected** in 76.19 seconds. All
Stages 01–04 regressions passed; stage-status assertions were updated only for
the newly authorized Stage 05. Ruff lint passed, all 141 Python files passed
format checks, `pip check` found no broken requirements, and `git diff --check`
passed. Explicit genuine Stage 05 integration execution produced **3 skipped**,
with missing corpus/cached-model prerequisites reported; it did not count those
skips as real model validation.

With network connections disabled, seven genuine CLI preflights checked inspection,
resource estimation, temporal audit, both encoder requests, BERTopic training and
baseline training. They recorded the paper-text blocker and stopped before model
loading. This execution record is
`artifacts/reports/stage05/verification-20261021T161237Z/verification.json`.
The same verification validated **10,985 genuine Stage 04 prediction records**
against `CitationIntentEvidence`, retained 10,985 distinct context IDs, and found
zero resolved paper pairs. All 10,985 outputs were explicitly uncalibrated and
remained unlinked. None was used as investigation supervision or to create topic
links. The detailed result is `stage04_prediction_audit.json` in that directory.

The final low-memory verification completed all eight requested commands,
including a successful retry of a comparison-model publication that encountered
a transient Windows folder lock. Final regressions were **466 passed, 15
deselected** in 69.90 seconds; lint, formatting (141 Python files), dependency
consistency, and diff checks passed. The complete record, including recovered
failures, is
`artifacts/reports/stage05/low-memory-20261021T172346Z/verification.json`.
The final primary BERTopic run is
`run-2026-10-21T17-33-36.960285+00-00-7ee9c89c`. Its Stage 06 evidence export is
`run-2026-10-21T17-35-32.916891+00-00-c623bb3e`, retaining all 2,000 assignments,
347 outliers, and 1,653 unreviewed UNKNOWN evidence rows with provenance.
No graph was created and no UNKNOWN candidate was promoted to a human label.

Reproduce software checks with:

The subsequent command repair finished with **454 passed, 15 deselected** in
96.26 seconds; Ruff and `pip check` passed. Six of the eight requested CLI
commands completed on real data. SPECTER2 embedding and BERTopic fitting were
blocked before model loading by the 1.5 GiB available-RAM floor (about 0.8-0.95
GiB measured). The real-paper integration check passed, with two genuine encoder
tests skipped for unavailable cached models/compute. The separately selected
installed-stack BERTopic integration test passed on synthetic vectors.

The complete command/test logs are in
`artifacts/reports/stage05/command-check-20261021T164406Z/verification.json`.
That directory also records metadata discovery and a checksum-validated export
audit: eight NMF topics, 2,000 assignments, and 2,000 UNKNOWN paper-topic evidence
records, including 296 rule candidates and 1,704 records with no evidence span.
All rows retain source provenance. No human annotations, graph, or supervised
investigation model was created.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -m integration tests/test_stage05_real_integration.py
```

To enable the genuine integration checks later, set
`CI_HGNN_STAGE05_STAGE03_RUN` to a published run containing permitted paper text;
set `CI_HGNN_STAGE05_REAL_MODELS=1` only with cached pinned encoders, the optional
topic stack and adequate memory. Default unit tests never fetch pretrained models.

## Configuration and models

`configs/topic_model_stage05.yaml` and `configs/investigation_detection.yaml`
are additive typed settings. The original `configs/topic_model.yaml` remains
the frozen Stage 01 section with `enabled: false`. `validate-config` stays compatible.
Library defaults are defined in `nlp/topics/config.py`; unknown keys fail validation.
The repository YAML selects the low-memory CPU profile described below.

| Setting | Default |
| --- | --- |
| Input | Published Stage 03 papers/source-local papers; checked table hashes |
| Selection | Deterministic seeded hash sample, at most 2,000 eligible unique papers |
| Text | NFC/whitespace only; title plus separator plus abstract; title-only allowed and flagged |
| Extremely long text | Exclude above 100,000 source characters, with a reason; never silently clip source text |
| Language | Preserve metadata; no language filter unless explicitly configured |
| Encoder | SPECTER2 base + proximity adapter, 768 dimensions, CLS pooling |
| Alternative | Explicit SciBERT masked mean pooling, including nonpadding special tokens |
| Encoding | Local YAML: batch 1, max length 256, CPU dynamic INT8 linear weights, fp32 activations, normalized float32 output; library defaults retain batch 8/max 512/full precision |
| Cache | Enabled; content/model/tokenization/normalization addressed, checksummed numeric `.npy` files |
| BERTopic | External embeddings; UMAP 5 dimensions, 15 neighbors, min_dist 0, cosine; HDBSCAN min cluster 20/min samples 5; c-TF-IDF; top 10 keywords |
| NMF | TF-IDF + NMF, 8 configured components, bounded 20,000-feature vocabulary |
| LDA | CountVectorizer + batch LDA, 8 components; raw counts explicitly replace inappropriate TF-IDF likelihood input |
| Keywords | Separate English stop-word pipeline, min_df 2, unigrams/bigrams; transformer text is unaffected |
| Temporal | Inherit Stage 03 policy/cutoff; frozen historical topic representations |
| Reproducibility | Seed 42; explicit UMAP/HDBSCAN config; ordered IDs, source hashes and environment |
| Stability | Opt-in repeated fits at seeds 42 and 73; align keywords before comparison |
| Investigation | Three labels; min 30 validated train pairs; three epochs; weighted loss; dev macro-F1 selection |
| Model access | Offline by default; `--allow-model-download` is explicit and resource-gated |

SPECTER2 uses the documented [base model](https://huggingface.co/allenai/specter2_base)
and [proximity adapter](https://huggingface.co/allenai/specter2), not the base
alone or a classification/search-query adapter. Tokenizer and CLS pooling follow
those model cards; the base/adapter cards declare Apache-2.0. Revisions are pinned:

- Base: `3447645e1def9117997203454fa4495937bfbd83`.
- Proximity adapter: `2081559630a80fc5851d8f798a05ba81e9468089`.
- SciBERT: `952d056ace307abd2cda7cee822a1d608deb20ed`.

Adapter loading uses the official `AutoAdapterModel` interface and a pinned local
adapter snapshot. The official SPECTER2 legacy tensor checkpoints require
PyTorch >=2.6 and adapters 1.3.0's restricted tensor loading. SciBERT and newly
trained classifier checkpoints use safetensors. Model source licences do not
establish redistribution rights for arbitrary input paper text.

The optional stack is bounded in `pyproject.toml` under `topics`; adapters 1.3.0
supports the existing Transformers 4.57.6. Install it on a suitable host without
upgrading unrelated dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade-strategy only-if-needed -e ".[topics]"
.\.venv\Scripts\python.exe -m pip check
```

The command repair installed adapters 1.3.0, BERTopic 0.17.4, UMAP 0.5.12,
HDBSCAN 0.8.44, and sentence-transformers 5.7.0 while retaining PyTorch
2.6.0+cpu and Transformers 4.57.6. The selected optional dependencies are recorded
in `requirements-stage05.lock` and can be installed with
`python -m pip install -r requirements-stage05.lock` after choosing the
host-appropriate PyTorch wheel. Dependency installation does not supply the
missing scholarly corpus or pretrained encoder weights.

## Input, temporal and caching controls

The local default YAML and `topics embed --model specter2 --low-memory` use
dynamic INT8 linear layers, one paper per batch, at most 256 input tokens, two
CPU threads, and a 0.5 GiB available-RAM preflight floor. This profile changes
the numeric representation and may truncate more abstracts than the 512-token
default. It records quantization in the encoder identity and cache key, retains
truncation counts, and never substitutes different model weights. Quantization
converts weights in place to avoid a second full float32 model. Empty LoRA
wrappers around ordinary BERT projections can be flattened, while actual LoRA
weights are rejected rather than discarded. Bottleneck proximity adapters are
retained. The floor is a startup heuristic, not a guarantee of total process RAM.

Pretrained base and adapter files are cached inside
`data/interim/stage05_embeddings/models/`. First-time acquisition requires
`--allow-model-download`; subsequent runs can remain offline. Per-paper numeric
embedding caches survive failed runs, so retrying the same profile resumes from
the validated cached vectors. Reuse a completed embedding run for training:

```powershell
.\.venv\Scripts\ci-hgnn-rgd.exe topics embed --model specter2 --low-memory --allow-model-download
.\.venv\Scripts\ci-hgnn-rgd.exe topics train --model bertopic --low-memory --embeddings-run-id EMBEDDING_RUN --offline
```

Canonical paper IDs are retained. Source-local records receive a clearly scoped
**record key**, while `canonical_paper_id` stays null if absent. Such keys are not
invented scholarly identities. A disk-backed uniqueness index avoids double
counting canonical papers present in both Stage 03 paper tables. Original titles,
abstracts and source provenance remain in the prepared table. Missing abstracts,
invalid dates, unrecoverable encoding, invalid source IDs/provenance, temporal
exclusions, duplicate identities and sample exclusions are reported. Synthetic
records are rejected by research CLI workflows.

Input sampling is bounded; transformer batches and mmap matrices avoid loading
all vectors on the GPU. File assignment streams bounded batches and rejects input
above the explicit document limit rather than silently discarding rows. Embedding
cache keys include paper ID/key, raw text/preprocessing hash, base revision,
adapter revision, pooling, tokenizer/max-length settings, normalization, precision,
and dimension. Cache bytes are checksummed; stale or corrupt records are not reused.
Interrupted runs leave pending directories; completed vector cache entries can be
reused in a new run. Exact optimizer-state resume is not implemented.

Stage 03 contributor evidence is reassessed against the selected cutoff. Unknown
historical availability fails strict mode; publication proxy and exploratory modes
carry their weaker interpretation. Current ingestion time is never publication
time. A strict encoder run also requires an explicitly verified
`embedding.verified_model_available_at` no later than the cutoff; an old paper
alone cannot validate a modern pretrained encoder historically. Topic IDs include
snapshot and run version. New documents use frozen components; future documents
never update historical keyword frequencies or centroids. Earlier inference
cutoffs cannot use later-fitted topic representations.

## Topics, scores and quality

BERTopic fits external scientific embeddings with UMAP, HDBSCAN, CountVectorizer
and c-TF-IDF. Training assignments retain HDBSCAN outliers (`topic_id=null`) and
uncalibrated strength semantics. Topic catalogs contain provisional keyword labels,
representative paper keys, assigned counts/coverage, year/category distributions,
method, version, snapshot, and provenance. NMF/LDA catalogs retain configured latent
components even when a component has zero hard assignments; those zero counts are
explicit. A provisional label is not a verified concept.

Safe persistence exports JSON and numeric NPZ components, not executable pickle.
The BERTopic export saves keywords and scientific centroids plus train-only
rejection thresholds. Its persisted new-document inference is explicitly a
**frozen centroid approximation**, with cosine scores; it does not restore UMAP/
HDBSCAN objects or claim identical HDBSCAN prediction semantics. NMF inference
uses nonnegative least squares against frozen components; LDA uses frozen-component
variational inference. No inference operation refits the model. These weights,
cosines and strengths are not calibrated topic-membership probabilities.
See the upstream [BERTopic serialization discussion](https://maartengr.github.io/BERTopic/getting_started/serialization/serialization.html)
for the tradeoffs of reduced versus full estimator serialization.

Quality includes document-boolean NPMI on the recorded corpus, topic diversity,
assigned topic sizes and outlier rate. Zero word-pair cooccurrence scores -1;
universal cooccurrence uses a documented 0 convention for the undefined limit.
The tokenizer/reference-corpus hash accompanies coherence. Comparisons reject
different corpora or tokenizers. Stability uses Hungarian maximum keyword-Jaccard
alignment, unmatched topics and potential split/merge counts. Keyword redundancy,
bounded embedding-neighbor examples, and blank expert-review forms support
inspection. They do not establish superiority or supply fabricated expert ratings.
Year distributions are descriptive retrospective counts, not prospective topic
emergence or research-gap detection.

## Independent investigation evidence

Read [the annotation guideline](investigation_annotation_guidelines.md). Labels are
`INVESTIGATES_TOPIC`, `MENTIONS_TOPIC`, and `UNKNOWN`. The system distinguishes:

1. Topic-aligned rule spans, always exported as unreviewed UNKNOWN candidates.
2. Scientific span-to-topic-centroid ranking, optionally enabled with
   `--rank-evidence`; similarity remains an unvalidated candidate signal.
3. Human evidence after independent agreement or explicit adjudication.
4. A supervised SciBERT paper–topic pair classifier, only after validated labels
   and paper-disjoint train/dev/test sets exist.

Exact span offsets refer to original title/abstract characters. Generic cues alone
are insufficient; negation, background and future-work cues stay review candidates.
Rules are conservative retrieval aids and cannot resolve every attribution or
methodological ambiguity. UNKNOWN does not assert absence of investigation.

Annotation tasks are stratified by topic, year, category, source, text completeness
and score level; exported tasks contain no completed labels. Completed imports
validate frozen text, topic definitions, taxonomy and offsets. Cohen's kappa is
computed only from actual independent overlapping responses. Disagreements require
an independent adjudicator and grounded label. Supervised training rejects synthetic
fixtures in research workflows and never predicts test during training. The
separate evaluation command is explicit. If evaluated, reports include the actual
three-class metrics and confusion matrix; absent training produces neither.

The Stage 01 inference facade remains backward compatible in shape and returns
UNKNOWN when no independent annotated/model evidence is supplied. Stage 04 citation
predictions can be exported alongside topic evidence, preserving their probabilities,
calibration, versions, provenance and nullable endpoints. They never become
investigation labels. Repeated citation contexts remain separate.

## CLI and a later genuine run

Run from `ci-hgnn-rgd/`. All actions accept `--config`, `--project-root`, `--offline`,
`--device`, `--batch-size`, `--max-records`, `--seed`, `--output-dir`,
`--temporal-cutoff`, `--temporal-policy`, and `--stage03-run` where meaningful.
Use subcommand help for required run/input arguments. Failures return nonzero.

`generate-investigation-candidates` and `export-graph-evidence` accept an optional
`--run-id`. Without it they select the latest completed fitted topic model, ignoring
newer inspection, embedding, audit, and export runs. An explicit `--model-name`
limits selection to a run containing that algorithm. The selected run's checksums
are verified before use; corrupt artifacts do not cause fallback to older weights.
If no topic model has been fitted, both commands report the missing prerequisite.

Acquire a permitted scholarly metadata file, retain its licence/source and trusted
checksum, and pass it through the existing Stage 02/03 workflow. For example:

```powershell
# INPUT_PATH is a real, permitted arXiv metadata JSONL/JSON file, not citation contexts.
.\.venv\Scripts\ci-hgnn-rgd.exe data acquire --dataset arxiv-metadata --path INPUT_PATH --offline
.\.venv\Scripts\ci-hgnn-rgd.exe data stage03-run --dataset arxiv-metadata --path INPUT_PATH --max-records 2000 --temporal-policy EXPLORATORY_NO_TEMPORAL_CLAIM --output-dir data/processed/stage03-topics --report-dir artifacts/reports/stage03-topics --offline
.\.venv\Scripts\ci-hgnn-rgd.exe topics inspect-data --offline
.\.venv\Scripts\ci-hgnn-rgd.exe topics estimate-resources --offline
# After installing the optional stack and providing adequate memory/cache:
.\.venv\Scripts\ci-hgnn-rgd.exe topics embed --model specter2 --offline
.\.venv\Scripts\ci-hgnn-rgd.exe topics embed --model scibert --offline
.\.venv\Scripts\ci-hgnn-rgd.exe topics train --model bertopic --embedding-model specter2 --embeddings-run-id EMBEDDING_RUN
.\.venv\Scripts\ci-hgnn-rgd.exe topics train --model bertopic --embedding-model scibert --embeddings-run-id SCIBERT_EMBEDDING_RUN
.\.venv\Scripts\ci-hgnn-rgd.exe topics train-baselines
.\.venv\Scripts\ci-hgnn-rgd.exe topics evaluate --run-id TOPIC_RUN --stability
.\.venv\Scripts\ci-hgnn-rgd.exe topics assign --run-id TOPIC_RUN --input NEW_PAPERS.jsonl
.\.venv\Scripts\ci-hgnn-rgd.exe topics export-catalog --run-id TOPIC_RUN
.\.venv\Scripts\ci-hgnn-rgd.exe topics generate-investigation-candidates --run-id TOPIC_RUN
.\.venv\Scripts\ci-hgnn-rgd.exe topics export-annotation-sample --run-id TOPIC_RUN
# Complete real independent annotation before the following commands:
.\.venv\Scripts\ci-hgnn-rgd.exe topics import-annotations --run-id TOPIC_RUN --input ANNOTATIONS.jsonl
.\.venv\Scripts\ci-hgnn-rgd.exe topics evaluate-annotations --run-id TOPIC_RUN --input ANNOTATIONS.jsonl
.\.venv\Scripts\ci-hgnn-rgd.exe topics train-investigation-classifier --run-id TOPIC_RUN --input ANNOTATIONS.jsonl
.\.venv\Scripts\ci-hgnn-rgd.exe topics predict-investigation --run-id TOPIC_RUN --input PAIRS.jsonl --classifier-run-id CLASSIFIER_RUN
.\.venv\Scripts\ci-hgnn-rgd.exe topics evaluate-investigation-classifier --run-id CLASSIFIER_RUN --split test
.\.venv\Scripts\ci-hgnn-rgd.exe topics audit-temporal
.\.venv\Scripts\ci-hgnn-rgd.exe topics export-graph-evidence --run-id TOPIC_RUN --evidence-run-id EVIDENCE_RUN --citation-intent-input STAGE04_PREDICTIONS.jsonl
.\.venv\Scripts\ci-hgnn-rgd.exe topics report --run-id TOPIC_RUN
```

For first-time model acquisition, `--allow-model-download` permits the requested
pinned model only after resource checks; it never changes encoder choice. No
extra user approval is built into the research CLI. For historical evaluation,
configure verified availability and cutoff evidence instead of switching to
exploratory mode and claiming a strict result.

## Output and handoff

Each operation publishes `data/processed/stage05/RUN_ID/` atomically with
`stage05_manifest.json`, config/environment, checked inputs, temporal snapshot and
applicable tables. Embeddings use `.npy` mmap storage and a separate ID/row index.
Other tables have portable JSONL plus, when PyArrow is installed, Parquet audit
envelopes with queryable IDs and lossless `record_json`. Failed `.pending-*` runs
never replace completed runs.
Fitted topic/classifier artifacts are mirrored under the requested
`artifacts/models/topic_discovery/` or `investigation_detection/` run root only
when actually fitted. Reports include only applicable real/synthetic execution
artifacts and clearly identify unavailable metrics.

Created files are the `nlp/topics/` modules (`contracts`, `config`, `data`,
`embeddings`, `models`, `quality`, `investigation`, `investigation_model`,
`artifacts`, `workflow`, `reporting`, `cli`), additive YAMLs, this document and the
annotation guideline, and `tests/stage05_helpers.py`/`tests/test_stage05_*.py`.
Modified integration files are `cli.py`, `pipeline.py`, the two legacy NLP facades,
`pyproject.toml`, README links, and the stage-status expectations in existing CLI/
pipeline tests. No Stage 03/04 scientific algorithms were rewritten.

Stage 06 readiness:

- [x] Versioned topic, assignment and independent-evidence contracts and exports.
- [x] Explicit outliers, UNKNOWN and nullable canonical IDs.
- [x] Stage 04 prediction compatibility without fabricated edges or labels.
- [x] Temporal snapshots, provenance, checksums and safe local model components.
- [x] Bounded genuine canonical paper corpus available on this host.
- [x] Real NMF/LDA comparison models on the same 2,000-paper corpus.
- [x] Real quantized SPECTER2 embeddings and BERTopic/NMF/LDA experiments.
- [ ] Optional full-precision/512-token and SciBERT encoder comparisons.
- [ ] Independent human annotations, expert topic review and supervised held-out validation.
- [ ] Scientific review of topic quality and historical encoder eligibility.

The software handoff is implemented. Empirical readiness remains blocked by the
items above. Stage 06 graph construction was not started. Cross-platform/library/
hardware bitwise reproducibility remains unguaranteed; retain exact data, pins,
model/adapter snapshots, config and code revision alongside the run manifests.
