# Stage 04 remaining-work audit — 2026-10-21

Stage 04 already contained adapters, three baselines, a SciBERT trainer,
calibration, inference, CLI commands, and reports. This follow-up addresses
implementation gaps and verifies the existing subsystem. The initial audit below
predates the low-memory SciBERT execution recorded in the final section. Initial
resource blockers and test counts describe that earlier verification phase.
The SciBERT follow-up changes Stage 04 only.

## Inspection and initial verification

Reviewed the package/configuration layout, Stage 01 schemas and runtime guards,
Stage 02 registry/loaders and SciCite adapter, Stage 03 canonical records,
integration/identity/split/temporal interfaces, published manifests and context
table, Stage 04 modules and CLI, and existing test coverage.

Before implementation, `python -m pytest` produced **350 passed, 1 failed,
8 deselected**. The failure was an incomplete resume-test fixture lacking model
metadata and a SciBERT run type. The strict production guard was retained and
the fixture corrected. Ruff initially reported three line-length violations.

The execution environment is CPython 3.11.9 on Windows, PyTorch 2.6.0+cpu, with no
CUDA device. Available RAM fluctuated around 0.9 GiB during initial inspection,
below the configured 3 GiB full-training floor. No cached SciBERT model directory
was found in the default Hugging Face cache. No pretrained weights were fetched
during that initial audit.

## Real data and scientific boundary

The archive `data/raw/scicite/scicite.tar.gz` was inspected directly. SHA-256:
`711ece2c4e61d116c8ae5bb07e9fbb2ee9ff7bba004b4cab7fbd0ac3af499193`.
It contains `scicite/train.jsonl`, `scicite/dev.jsonl`, and `scicite/test.jsonl`.
Actual context text uses `string`; labels use `label`. Observed fields also
include `id`, `unique_id`, `citingPaperId`, `citedPaperId`, section and citation
offsets, and annotation metadata. The upstream repository documents this data
and its attribution: [AllenAI SciCite](https://github.com/allenai/scicite).
The local source audit retains its README and Apache-2.0 license text.

| Split | Source rows | Quarantined | Background | Method | Result | Admitted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8,243 | 21 | 4,840 | 2,285 | 1,097 | 8,222 |
| dev | 916 | 1 | 538 | 255 | 122 | 915 |
| test | 1,861 | 13 | 994 | 600 | 254 | 1,848 |

The 35 nonstandard-JSON source rows remain in Stage 03 quarantine. Official split
assignments are preserved for the admitted 10,985 records. This is a filtered
official-split view, so its scores should not be compared as if the evaluation
contained every official source row. Source paper IDs exist, but verified
canonical endpoints do not: all current context predictions remain unlinked.
No graph edges, topics, direct-investigation labels, or gap scores were created.

## Files and fixes

| Files | Changes |
| --- | --- |
| `src/ci_hgnn_rgd/nlp/intent_inference.py`, `intent_cli.py` | Bounded JSONL/Parquet input, inference and output; global ordering; empty-input handling; atomic publication; overwrite rejection; restored Parquet provenance and source-ID types; truncation summaries; explicit text-excerpt setting |
| `src/ci_hgnn_rgd/nlp/intent_contracts.py` | Additive nullable text reference and opt-in excerpt fields |
| `src/ci_hgnn_rgd/nlp/intent_transformer.py` | Correct accumulated weighted/unweighted loss and partial batches; resource-based microbatch adjustment; effective batch/epoch recording; quick-mode epoch cap; truncation counts; early stopping; per-epoch history; stricter resume selection check |
| `src/ci_hgnn_rgd/nlp/intent_data.py` | Exclusion reasons for malformed envelopes; all detected affected text/source-ID records and near pairs; per-window temporal reassessment |
| `src/ci_hgnn_rgd/nlp/intent_metrics.py`, `intent_baselines.py` | Fixed-taxonomy class-weight validation; calibration-bin validation; unknown bootstrap units cannot collide with verified group names; accurate resampling-unit reporting |
| `src/ci_hgnn_rgd/nlp/intent_artifacts.py`, `intent_reporting.py` | Standalone environment snapshot; actual per-split metrics/calibration JSON; empty error-analysis CSV for evaluated models without errors |
| `src/ci_hgnn_rgd/nlp/intent_config.py`, `configs/citation_intent_stage04.yaml` | Typed quick-epoch and automatic batch settings |
| `tests/test_stage04_*.py`, `pyproject.toml` | Expanded offline regression coverage; separately marked genuine-data and opt-in real SciBERT checks |
| `docs/stage04.md`, this audit | Behavior, execution evidence, commands, limits, and later-stage handoff |

The label mapping remains `background=0`, `method=1`, `result=2`. SciBERT remains
`allenai/scibert_scivocab_uncased` at revision
`952d056ace307abd2cda7cee822a1d608deb20ed`, using three logits, maximum length 256,
AdamW at `2e-5`, four full-mode epochs, configured batch 16/evaluation batch 32,
weight decay 0.01, warmup ratio 0.1, and seed 42. Its pinned revision includes a
[safetensors checkpoint](https://huggingface.co/allenai/scibert_scivocab_uncased/tree/952d056ace307abd2cda7cee822a1d608deb20ed).
Class weighting defaults to none; inverse and smoothed strategies and exclusive
weighted sampling remain configurable. No new full SciBERT imbalance experiment
or dev-based strategy selection is claimed.

## Evaluation and artifact interpretation

Final offline verification: **382 passed, 11 deselected**. Separately, the genuine
Stage 04 integration checks produced **2 passed, 1 deselected**; the deselected
case is opt-in real SciBERT training. Ruff lint and format checks passed, `pip
check` reported no broken requirements, and `git diff --check` passed. After the
last Parquet transport-field adjustment, its focused regression also passed.

A persisted CLI smoke run trained all three baselines on **60 real training
contexts**, selected/evaluated them on the **915 official dev contexts**, and
never evaluated the held-out test. It used no class weighting, a 1,000-feature
TF-IDF cap and `C=1.0`. Its dev macro F1 values below are **smoke-only**:

| Model | Dev macro F1, smoke only |
| --- | ---: |
| Majority | 0.246846 |
| TF-IDF logistic regression | 0.518824 |
| TF-IDF linear SVM | 0.519536 |

Safe baseline artifacts and the standalone environment/config/data audit live in
`artifacts/models/citation_intent/run-2026-10-21T15-03-15.350008+00-00-ece8424f/`.
Seven genuine contexts were exported in both JSONL and Parquet using batches of
two, with exact context order and provenance preserved. Local execution evidence
is in `artifacts/reports/stage04/verification-20261021T150111Z/`, including
`smoke_training.json`, `smoke_exports.json`, both prediction exports, config and
resource preflight. The initial measured preflight found **1.179 GiB available
RAM**, no GPU and `insufficient_available_system_memory`; that full-training
attempt was blocked before the later low-memory profile was introduced.

The previously recorded full baseline run remains
`run-2026-10-19T14-49-47.567642+00-00-74ca5ac8`. Its measured results are documented
in [Stage 04](stage04.md). The follow-up regenerates reports from those saved
evaluations without invoking its held-out test again. At that initial audit no
new SciBERT dev or test metric existed. Its actual confusion matrices remain under
`artifacts/reports/stage04/run-2026-10-19T14-49-47.567642+00-00-74ca5ac8/`, with
filenames such as `test_tfidf_logistic_regression_confusion_matrix.png` and
`test_tfidf_logistic_regression_normalized_confusion_matrix.png`.

The existing logistic baseline was also invoked through the single-text CLI on
`We use the method described in [12].` It returned `method`, probabilities
`[0.001053, 0.998223, 0.000724]` in fixed label order, entropy `0.014227`, and
`calibration_status=uncalibrated`. Both paper endpoints remained null. This is
an inference example with no asserted ground-truth label or SciBERT result.

Existing leakage findings remain: 627 scoped source-ID overlap groups, one exact
text group, 18 normalized-text groups, 26 detected near-duplicate pairs, 1,300
source-local paper overlap groups, and zero verified canonical-paper overlap
groups. Affected context identifiers now appear in the JSON audit. Near discovery
is nonexhaustive; official assignments remain unchanged. Temporal validation is
`NOT_EVALUABLE`, because verified dates and historical availability are absent.

## Commands and remaining experiments

All existing Stage 04 actions remain available: `inspect-data`, `audit-leakage`,
`train-baselines`, `train`, `evaluate`, `calibrate`, `predict`, `predict-file`,
`export-graph-evidence`, and `report`. Use `--help` on each for applicable flags.
The following PowerShell commands run from `ci-hgnn-rgd/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m pip check
$env:CI_HGNN_STAGE04_STAGE03_RUN = (Get-Content -Raw data\processed\stage03\latest.json | ConvertFrom-Json).run_dir
.\.venv\Scripts\python.exe -m pytest -q -m 'integration and not scibert_training' tests/test_stage04_real_data.py
```

Training commands (the pinned weights are now cached locally):

```powershell
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent train --model scibert --mode quick --max-records 60
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent train --model scibert --mode full --device cpu
# For the bounded class-weight comparison, add --search to full training.
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent evaluate --run-id RUN_ID --split dev
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent calibrate --run-id RUN_ID
# Freeze selection/calibration before this explicit final evaluation:
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent evaluate --run-id RUN_ID --split test
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent predict --run-id RUN_ID --text "We use the method described in [12]."
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent export-graph-evidence --run-id RUN_ID
.\.venv\Scripts\ci-hgnn-rgd.exe citation-intent report --run-id RUN_ID
```

Use `--device cuda` only after installing a compatible CUDA PyTorch build. Use
`--offline` only after caching the pinned tokenizer/model. The execution section
below records completed and running SciBERT jobs. Dataset acquisition is already
complete locally; a replacement archive is available from the upstream SciCite
download linked above and must pass the Stage 02/03 manifest workflow.

Resume is a verified model-only warm start, not exact optimizer-state recovery.
Cross-device/version/kernel bitwise reproducibility remains unguaranteed. Seeds,
data fingerprints, config, environment, and actual batch decisions are recorded.
Calibration uses dev only and cannot be refitted after test observation; raw
softmax probabilities remain uncalibrated. Entropy does not measure validated
epistemic uncertainty.

Scientific model-quality readiness requires a completed full-data SciBERT run,
dev selection, optional calibration, and a frozen held-out evaluation. The quick
training check below does not satisfy full-data model-quality evaluation.

## Real SciBERT execution with lower memory, 2026-10-21

The requested quick command completed successfully against the genuine Stage 03
SciCite contexts. Published run:
`run-2026-10-21T17-52-54.140709+00-00-d34d16eb`.
It trained all encoder and classifier parameters for one epoch on 768
deterministically balanced real train contexts (256 per label), then validated
on all 915 admitted official dev contexts. It completed 48 optimizer updates;
dev macro F1 was **0.757874**, accuracy **0.784699**. The held-out test was not
evaluated. This is a quick pipeline result rather than full-data performance.
The published safetensors checkpoint was independently reloaded successfully
and used for one illustrative prediction without an asserted ground-truth label.

The pinned pretrained revision is now cached inside the project. The repository
YAML selects CPU FP32, Adafactor with the existing fixed `2e-5` learning rate and
warmup scheduler, gradient checkpointing, one-context microbatches, accumulation
16, evaluation batch one, and two CPU threads. Effective training batch remains
16, and maximum token length remains 256. AdamW remains the typed library
default and a configurable alternative. Inference INT8 is not used for training.
The measured quick process peak physical RAM was **1,705,263,104 bytes**; a
sampled private committed-memory value was **1,937,928,192 bytes**. The measured
optimizer-state tensors occupied **1,297,676 bytes**. These observations are
specific to this host and workload.

Full mode now checks at least 0.25 GiB available physical RAM and, on Windows,
3 GiB available committed-memory capacity with this local profile. It records
those observations before loading the model. These are startup heuristics, not
guarantees of sufficient memory throughout a run. No OS page-file setting was
modified. New tests verify that both encoder and head update, Adafactor state is
smaller than AdamW state for a tiny model, commit-capacity checks work, and cache
location changes do not invalidate an otherwise compatible model-only resume.
Atomic model publication retries bounded transient Windows directory-lock errors;
pending or failed runs cannot be resolved as completed artifacts.

Final regression verification after these changes: **472 passed, 15 deselected**
in 91.45 seconds. Ruff lint, formatting (142 files), dependency consistency, and
`git diff --check` passed. Synthetic unit tests and the genuine quick training
run are distinct observations; neither implies completed full training.

Full training was launched at **2026-10-21 18:05 UTC** on all **8,222** admitted
training contexts, with up to four epochs and dev-based early stopping.
Completion is pending; at launch, the CPU estimate was roughly **5–6 hours**.
The durable supervisor in `scripts/run_scibert_training.py` writes live status,
stderr and final command JSON to
`artifacts/reports/stage04/scibert-real-full-20261021/`. Read `status.json` for
`RUNNING`, `TRAINED` or `FAILED`; `training.log` contains actual optimizer-step
progress. Each pending model directory also contains `training_progress.json`
and completed-epoch `training_history.json`. On success, the supervisor records
the published run ID in its final result. It does not evaluate test data.

The quick run's corresponding execution evidence is in
`artifacts/reports/stage04/scibert-real-quick-20261021/`, including
`status.json`, `memory_observation.json` and `checkpoint_reload_check.json`.
