# Stage 04: SciBERT citation-intent classification

Stage 04 predicts the purpose of a citation context. It uses only the three
SciCite labels `background`, `method`, and `result`, with fixed IDs `0`, `1`,
and `2`. Citation intent is neither direct-investigation evidence nor a
research-gap score.

This implementation stops before Stage 05 topic identification, Stage 06
direct-investigation evidence, graph construction, GNN training, and gap scoring.

## Data boundary and provenance

The modelling view reads the immutable published Stage 03 `citation_contexts`
table. It retains original context text, source label, official source split,
source record ID, identifiers, transformations, temporal metadata, and source
provenance. Model text receives NFC Unicode composition and whitespace collapse
only; punctuation, citation markers, stop words, notation, and terminology remain.

The locally inspected SciCite archive is `data/raw/scicite/scicite.tar.gz` with
locally computed SHA-256
`711ece2c4e61d116c8ae5bb07e9fbb2ee9ff7bba004b4cab7fbd0ac3af499193`.
The source audit captures the upstream repository README and Apache-2.0 license
text in `data/raw/source_audits/`. Retain attribution and independently review
the terms for redistribution of dataset-derived text or artifacts.

The selected published Stage 03 run has 10,985 nonquarantined contexts:

| Split | Background | Method | Result | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 4,840 | 2,285 | 1,097 | 8,222 |
| dev | 538 | 255 | 122 | 915 |
| test | 994 | 600 | 254 | 1,848 |

Stage 03 retained 35 malformed/nonstandard-JSON records in quarantine. Stage 04
does not silently add them to a modelling split. Source paper IDs were not
verified as canonical identities, so all 10,985 contexts remain unlinked:
`citing_paper_id` and `cited_paper_id` are null.

## Split, leakage, and time controls

Official SciCite assignments remain unchanged. Train fits vectorizers and model
parameters; dev selects candidates and fits optional temperature scaling; test is
available only through the explicit `evaluate --split test` command after model
selection is frozen.

The Stage 04 model-view audit found 627 scoped source-record-ID overlap groups,
one exact-text overlap group, 18 normalized-text overlap groups, and 26
approximate cross-split near-duplicate pairs. It also reports 1,300 source-local
paper-ID overlap groups and zero verified canonical paper-identity overlap groups.
The bounded near-duplicate search compared 35,909 pairs at a 0.94 threshold; it
is nonexhaustive and 1,471 contexts reached the candidate cap. Findings are
reported without changing benchmark assignments. An opt-in exact-normalized-text
sensitivity training view remains explicitly non-benchmark.

The official split is not a chronological claim. No verified publication-date or
historical-availability coverage is available for these contexts, so temporal
generalization is `NOT_EVALUABLE`. Ingestion time is never treated as publication
time. The optional chronological allocator refuses incomplete temporal evidence.

## Model and training design

The primary implementation uses Hugging Face `AutoTokenizer` and
`AutoModelForSequenceClassification` with
`allenai/scibert_scivocab_uncased`, pinned to revision
`952d056ace307abd2cda7cee822a1d608deb20ed`. It fixes three output logits and
implements dynamic padding, attention masks, AdamW or Adafactor, warmup scheduling, gradient
clipping, accumulation, early stopping on dev macro F1, deterministic ordering,
safe `safetensors` checkpoints, and CPU/CUDA selection. Imports never download
weights.

Full settings are 256 tokens, learning rate `2e-5`, up to four epochs, effective
train batch 16, weight decay 0.01, warmup ratio 0.1, and seed 42.
Class weighting supports `none`, `inverse`, and `smoothed`; weighted sampling is
optional and cannot be combined with class weighting. Quick mode uses a fixed,
label-balanced training subset and is for pipeline debugging only. Multiple seeds
publish separate runs and summarize dev macro-F1 variation without reusing test
results for selection.

The local environment has `torch 2.6.0+cpu` and no CUDA device. The repository
configuration selects a lower-memory CPU profile: Adafactor with fixed learning
rate, gradient checkpointing, one-record microbatches, 16 accumulation steps,
one-record evaluation batches, and two CPU threads. All encoder and classifier
parameters remain trainable in FP32. This changes the optimizer from the library's
AdamW default; compare results using the recorded optimizer and configuration.
`--low-memory` explicitly selects the same profile with a custom configuration.

The pinned pretrained checkpoint is cached under
`data/interim/citation_intent_models/`; cached training can use `--offline`.
The repository profile checks at least 0.25 GiB immediately available RAM and,
on Windows, 3 GiB of available committed-memory capacity. These startup checks
are heuristics; the measured quick run used about 1.7 GB peak physical RAM and
1.93 GB process-private committed memory. No operating-system page-file settings
are changed. Library defaults retain the original 3 GiB physical-memory floor.
Blocked runs preserve `resource_preflight.json` with the measured values.

During training, `training_progress.json` and structured log messages record
actual examples, optimizer steps, epoch timing, and optimizer-state bytes.
The full corpus contains 8,222 admitted training contexts. Quick mode uses 768
real contexts, balanced across the three labels, and validates on all 915 dev
contexts. It is a pipeline check rather than a full-data performance result.

Actual quick run on 2026-10-21:
`run-2026-10-21T17-52-54.140709+00-00-d34d16eb`. One epoch and 48 optimizer
updates completed in about 10.6 minutes including audits, validation and artifact
publication. Dev macro F1 was **0.757874**, dev accuracy **0.784699**. The test
split was not evaluated. The measured Adafactor state was 1,297,676 bytes.
The saved model and tokenizer are under that published run directory; execution
logs and status are in `artifacts/reports/stage04/scibert-real-quick-20261021/`.

The full-data command was launched at 2026-10-21 18:05 UTC and verified through
actual optimizer updates on all 8,222 train contexts. It is still running at this
documentation update, with an initial CPU estimate of 5–6 hours for four epochs.
Read `artifacts/reports/stage04/scibert-real-full-20261021/status.json` and
`training.log` for live progress and eventual completion or failure; no completed
full-data metric is claimed yet. The durable logging wrapper is
`scripts/run_scibert_training.py`.

## Baselines and executed results

Implemented train-only baselines are majority class, TF-IDF plus logistic
regression, and TF-IDF plus linear SVM. Vectorizers fit training contexts only;
`C` and class weighting select by dev macro F1 only. The SVM exposes decision
scores but does not invent probabilities. Baseline artifacts are JSON plus NumPy
`.npz` arrays loaded with `allow_pickle=False`, never pickle or joblib.

Actual baseline run: `run-2026-10-19T14-49-47.567642+00-00-74ca5ac8`.
It selected logistic regression with `C=2.0` and no weighting, plus linear SVM
with `C=0.5` and inverse weights (`background=0.398396`, `method=0.843867`,
`result=1.757737`). Its held-out test was invoked once only after selection.

| Model | Dev macro F1 | Held-out test macro F1 | Test accuracy |
| --- | ---: | ---: | ---: |
| Majority | 0.246846 | 0.233169 | 0.537879 |
| TF-IDF + logistic regression | 0.810450 | 0.815649 | 0.837662 |
| TF-IDF + linear SVM | 0.811142 | 0.804018 | 0.823052 |

These are local benchmark measurements for this recorded data view. They do not
establish SciBERT performance, temporal generalization, direct investigation, or
a research-gap claim.

## Commands

Run from `ci-hgnn-rgd/`:

```powershell
ci-hgnn-rgd citation-intent inspect-data
ci-hgnn-rgd citation-intent audit-leakage
ci-hgnn-rgd citation-intent train-baselines
ci-hgnn-rgd citation-intent train --model scibert --mode quick --offline
ci-hgnn-rgd citation-intent train --model scibert --mode full --offline
ci-hgnn-rgd citation-intent evaluate --run-id RUN_ID --split dev
ci-hgnn-rgd citation-intent evaluate --run-id RUN_ID --split test
ci-hgnn-rgd citation-intent calibrate --run-id RUN_ID
ci-hgnn-rgd citation-intent predict --run-id RUN_ID --text "Citation context"
ci-hgnn-rgd citation-intent predict-file --run-id RUN_ID --input INPUT.jsonl --output OUTPUT.jsonl
ci-hgnn-rgd citation-intent export-graph-evidence --run-id RUN_ID
ci-hgnn-rgd citation-intent report --run-id RUN_ID
```

`--config`, `--project-root`, `--offline`, `--device`, `--batch-size`, `--seed`,
`--output-dir`, `--resume`, `--low-memory`, and `--max-records` apply where meaningful. Paths
stay inside the component root; existing model runs and exports are not silently
overwritten, and failed training keeps a recoverable pending directory.

Install a host-appropriate PyTorch wheel, then the remaining optional stack:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
.\.venv\Scripts\python.exe -m pip install -r requirements-stage04.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

## Artifacts and later handoff

Each run publishes atomically under `artifacts/models/citation_intent/RUN_ID/`.
It records the frozen config, label mapping, data fingerprint, leakage/temporal
audits, environment, safe model/baseline artifacts, history, and explicit
evaluation updates. Resume verifies model revision, tokenizer metadata, mapping,
and dataset fingerprint.

Inference emits `CitationIntentEvidence` in input order: context/source IDs,
nullable endpoints, three probabilities when a model supplies them, confidence,
entropy, calibration status, model/tokenizer version, provenance, timestamp,
truncation state, and identity status. Empty input becomes `skipped_empty_text`;
raw softmax values remain `uncalibrated` until dev-only temperature scaling, and
entropy is not presented as epistemic uncertainty.

`export-graph-evidence` produced 10,985 provenance-preserving logistic-baseline
records under the actual run's `prediction_exports/` directory. All are unlinked
contexts, and no graph was created. A later stage may attach an evidence record
only to its verified matching paper pair, must preserve repeated contexts, and
must keep intent separate from topic, investigation, and gap signals.

The generated report bundle under `artifacts/reports/stage04/RUN_ID/` includes the
data profile, leakage/temporal audit, baseline comparison, per-class metrics,
context-reference error analysis, confusion-matrix PNGs, evaluation summary, model
comparison, and readiness note. Generated artifacts are ignored by Git.

## Remaining-work audit, 2026-10-21

See [the completion audit](stage04_completion_audit.md) for the inspected gaps,
changes, actual verification results, and experiments that remain unexecuted.
This follow-up keeps the existing Stage 01–03 contracts and later-stage stubs.

File inference now streams JSONL and Parquet through bounded CPU/model batches.
JSONL and Parquet publication is atomic, including when inference fails midway.
Parquet readers restore serialized provenance and the original string/integer
source-ID type. Empty exports retain their contract schema. Each output includes
an input context reference; text excerpts require `include_text_excerpt: true`.
Prediction commands report truncated and skipped-empty context counts.

Training now normalizes accumulated gradients by the actual effective batch
denominator, including partial batches and weighted loss. Device-based batch
adjustment preserves the configured effective batch size and records the actual
microbatch and accumulation settings. `auto_adjust_batch_size: false` retains
explicit settings; the library CPU microbatch cap is two and the local YAML cap
is one. Memory estimates are
conservative heuristics rather than guarantees. Quick mode caps training at
`quick_epochs: 1` by default; `--max-records` requires quick mode.

Training/dev truncation counts and completed-epoch histories are persisted.
Early stopping counts consecutive non-improving validation epochs. A model-only
resume additionally verifies the exact training-record selection and ordering;
optimizer/scheduler/scaler state is not restored. Reports include actual per-split
metrics JSON and calibration JSON when their source artifacts exist, and runs
include a standalone environment snapshot.

Leakage JSON includes all affected source-ID, exact/normalized-text records and
all detected near-duplicate pairs. Near-duplicate discovery remains bounded and
nonexhaustive. Chronological allocation rechecks historical availability and
future-derived metadata against each allocation cutoff; an old eligibility flag
alone cannot authorize earlier features. The current SciCite view still has no
verified temporal coverage.

The 35 quarantined records mean this is a filtered view of the official splits.
Compare measured scores only against runs using the same admitted records;
do not equate them with results on all 11,020 source rows.
