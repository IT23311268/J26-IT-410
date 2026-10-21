"""Research-facing reports generated only from published Stage 04 artifacts.

The module intentionally does not manufacture empty scorecards.  A report lists
which evaluations were actually run, retains the official split provenance, and
marks unavailable evidence as unavailable.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..data.manifests import atomic_json
from .intent_artifacts import resolve_intent_run
from .intent_contracts import INTENT_LABELS
from .intent_data import PreparedIntentData


class IntentReportError(ValueError):
    """Raised when an otherwise published run cannot be reported faithfully."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".tmp-" + path.name)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _write_markdown(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".tmp-" + path.name)
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)
    return path


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntentReportError(
            f"cannot read report artifact {path.name}: {type(error).__name__}"
        ) from None
    if not isinstance(value, dict):
        raise IntentReportError(f"report artifact {path.name} must be a JSON object")
    return value


def _profile_markdown(prepared: PreparedIntentData) -> str:
    profile = prepared.dataset_profile
    label_note = (
        "The source labels are genuine SciCite labels. They are supervised labels for citation "
        "intent, not research-gap or direct-investigation ground truth."
    )
    lines = [
        "# Stage 04 dataset profile",
        "",
        f"Generated: `{_timestamp()}`",
        "",
        f"- Source: `{profile['source']}`",
        f"- Stage 03 run: `{profile['stage03_run_id']}`",
        f"- Contexts admitted to the modelling view: **{profile['records_admitted']}**",
        f"- Stage 03 quarantined records: **{profile['stage03_quarantined_records']}**",
        f"- Stage 04 exclusions: **{profile['records_excluded_by_stage04']}**",
        "",
        "## Preserved official splits",
        "",
        "| Split | Contexts | Background | Method | Result |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "dev", "test"):
        labels = profile["label_distribution"][split]
        lines.append(
            f"| {split} | {profile['split_counts'][split]} | {labels['background']} | "
            f"{labels['method']} | {labels['result']} |"
        )
    linkage = profile["paper_identity_linkage"]
    lines.extend(
        [
            "",
            "## Identity coverage",
            "",
            f"- Verified paper-pair links: {linkage['linked_contexts']}",
            f"- Unlinked citation contexts: {linkage['unlinked_contexts']}",
            f"- Policy: {linkage['interpretation']}",
            "",
            label_note,
        ]
    )
    return "\n".join(lines) + "\n"


def _leakage_markdown(prepared: PreparedIntentData) -> str:
    audit = prepared.leakage_audit["stage04_model_view"]
    near = audit["near_audit"]
    normalized_overlap = audit["normalized_text_overlap_groups"]
    resolved_overlap = audit["resolved_paper_identity_overlap_groups"]
    source_local_overlap = audit["source_local_paper_overlap_groups"]
    identity_note = (
        "A source-local identifier is not promoted to a verified global paper identity. "
        "Shared verified paper identity is a design-dependent risk, not automatically a "
        "benchmark-invalidating event."
    )
    lines = [
        "# Stage 04 split and leakage audit",
        "",
        f"Generated: `{_timestamp()}`",
        "",
        "The official SciCite train/dev/test assignments are retained. These findings are reported "
        "without silently filtering benchmark examples.",
        "",
        "| Finding | Count |",
        "| --- | ---: |",
        f"| Contexts inspected | {audit['contexts_inspected']} |",
        f"| Shared source record identifiers across splits | "
        f"{audit['shared_source_record_identifier_overlap_groups']} |",
        f"| Exact text overlap groups | {audit['exact_text_overlap_groups']} |",
        f"| Whitespace/case-normalized overlap groups | {normalized_overlap} |",
        f"| Near-duplicate cross-split pairs | {audit['near_duplicate_cross_split_pairs']} |",
        f"| Resolved paper-identity overlap groups | {resolved_overlap} |",
        f"| Source-local identifier overlap groups | {source_local_overlap} |",
        "",
        "## Near-duplicate method limits",
        "",
        f"- Method: `{near['method']}`",
        f"- Similarity threshold: `{near['threshold']}`",
        f"- Comparisons executed: `{near['comparisons']}` / `{near['max_comparisons']}`",
        f"- Candidate-capped contexts: `{near['candidate_capped_contexts']}`",
        f"- Budget-skipped contexts: `{near['budget_skipped_contexts']}`",
        f"- Long contexts skipped: `{near['long_text_skipped']}`",
        "",
        identity_note,
    ]
    return "\n".join(lines) + "\n"


def write_data_reports(prepared: PreparedIntentData, output_dir: Path) -> dict[str, str]:
    """Publish source-profile, leakage, and temporal reports for an actual data view."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset_profile_json": output_dir / "dataset_profile.json",
        "dataset_profile_markdown": output_dir / "dataset_profile.md",
        "leakage_audit_json": output_dir / "leakage_audit.json",
        "leakage_audit_markdown": output_dir / "leakage_audit.md",
        "temporal_validation_json": output_dir / "temporal_validation.json",
    }
    atomic_json(paths["dataset_profile_json"], prepared.dataset_profile)
    _write_markdown(paths["dataset_profile_markdown"], _profile_markdown(prepared))
    atomic_json(paths["leakage_audit_json"], prepared.leakage_audit)
    _write_markdown(paths["leakage_audit_markdown"], _leakage_markdown(prepared))
    atomic_json(paths["temporal_validation_json"], prepared.temporal_validation)
    return {name: str(path) for name, path in paths.items()}


def _evaluation_documents(run_dir: Path) -> list[dict]:
    root = run_dir / "evaluations"
    documents = []
    if not root.is_dir():
        return documents
    for path in sorted(root.glob("*/*.json")):
        document = _read_json(path)
        if document is None:
            continue
        document["_path"] = path
        documents.append(document)
    return documents


def _per_class_rows(evaluations: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        metrics = evaluation.get("metrics", {})
        for label in INTENT_LABELS:
            item = metrics.get("per_class", {}).get(label)
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "split": evaluation.get("split"),
                    "model": evaluation.get("model"),
                    "label": label,
                    "precision": item.get("precision"),
                    "recall": item.get("recall"),
                    "f1": item.get("f1"),
                    "support": item.get("support"),
                }
            )
    return rows


def _error_rows(evaluations: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        for item in evaluation.get("errors", []):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "split": evaluation.get("split"),
                    "model": evaluation.get("model"),
                    "citation_context_id": item.get("citation_context_id"),
                    "source_record_id": item.get("source_record_id"),
                    "true_label": item.get("true_label"),
                    "predicted_label": item.get("predicted_label"),
                    "prob_background": item.get("prob_background"),
                    "prob_method": item.get("prob_method"),
                    "prob_result": item.get("prob_result"),
                    "model_version": item.get("model_version"),
                    "text_reference": item.get("text_reference"),
                }
            )
    return rows


def _plot_confusion_matrices(evaluations: list[dict], output_dir: Path) -> dict[str, str]:
    """Render actual matrices when matplotlib is installed; never substitute placeholders."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    paths: dict[str, str] = {}
    for evaluation in evaluations:
        metrics = evaluation.get("metrics", {})
        matrix = metrics.get("confusion_matrix")
        normalized = metrics.get("normalized_confusion_matrix")
        if not isinstance(matrix, list) or not isinstance(normalized, list):
            continue
        split = str(evaluation.get("split", "unknown"))
        model = str(evaluation.get("model", "unknown"))
        safe_model = "".join(char if char.isalnum() or char in "-_" else "_" for char in model)
        for name, values, fmt in (
            ("confusion_matrix", matrix, "d"),
            ("normalized_confusion_matrix", normalized, ".2f"),
        ):
            figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(
                values, cmap="Blues", vmin=0, vmax=1 if "normalized" in name else None
            )
            figure.colorbar(image, ax=axis)
            axis.set_xticks(range(3), INTENT_LABELS)
            axis.set_yticks(range(3), INTENT_LABELS)
            axis.set_xlabel("Predicted intent")
            axis.set_ylabel("True intent")
            axis.set_title(f"{model}: {split} ({metrics.get('samples', 0)} contexts)")
            for row in range(3):
                for column in range(3):
                    axis.text(
                        column, row, format(values[row][column], fmt), ha="center", va="center"
                    )
            figure.tight_layout()
            path = output_dir / f"{split}_{safe_model}_{name}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths[f"{split}:{model}:{name}"] = str(path)
    return paths


def _comparison_markdown(run_manifest: dict, evaluations: list[dict]) -> str:
    no_evaluation_note = (
        "No explicit post-training evaluation artifacts are present. "
        "No held-out performance is claimed."
    )
    interpretation = (
        "Citation-intent results describe citation-context language only. They are not "
        "direct-investigation or research-gap scores, and no result-class prediction is exported "
        "as direct-investigation evidence."
    )
    lines = [
        "# Stage 04 model comparison",
        "",
        f"Generated: `{_timestamp()}`",
        "",
        f"- Run type: `{run_manifest.get('run_type', 'unknown')}`",
        "- Selection metric: validation macro F1 only.",
        "- Held-out test results appear only when an explicit `evaluate --split test` command ran.",
        "",
    ]
    baseline = run_manifest.get("selected_models", {})
    if baseline:
        lines.extend(["## Selected baseline configurations", ""])
        for name, item in baseline.items():
            validation = item.get("validation_metrics", {})
            lines.append(
                f"- `{name}`: dev macro F1 = `{validation.get('macro_f1', 'not recorded')}`; "
                f"class weighting = `{item.get('class_weighting', 'not applicable')}`."
            )
        lines.append("")
    if evaluations:
        lines.extend(
            [
                "## Actually executed evaluations",
                "",
                "| Split | Model | Samples | Accuracy | Macro F1 | Weighted F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in evaluations:
            metrics = item.get("metrics", {})
            lines.append(
                f"| {item.get('split')} | {item.get('model')} | {metrics.get('samples')} | "
                f"{metrics.get('accuracy')} | {metrics.get('macro_f1')} | "
                f"{metrics.get('weighted_f1')} |"
            )
    else:
        lines.extend(
            [
                "## Evaluation status",
                "",
                no_evaluation_note,
            ]
        )
    lines.extend(
        [
            "",
            interpretation,
        ]
    )
    return "\n".join(lines) + "\n"


def _readiness_markdown(prepared: PreparedIntentData, evaluations: list[dict]) -> str:
    temporal = prepared.temporal_validation
    integration_note = (
        "- Stage 05/06 integration: consume the exported `CitationIntentEvidence` "
        "contract only after their own evidence policies are implemented; do not infer "
        "topics, direct investigation, or "
        "gaps here."
    )
    final_performance_note = (
        "- Stage 04 does not claim model superiority or final performance without a recorded "
        "held-out evaluation."
    )
    lines = [
        "# Stage 04 readiness",
        "",
        f"Generated: `{_timestamp()}`",
        "",
        "- Genuine SciCite modelling records: available through the published Stage 03 run.",
        "- Official train/dev/test assignments: preserved.",
        f"- Temporal generalization: `{temporal['status']}` — {temporal['reason']}",
        "- Graph endpoints: missing endpoints remain unlinked citation-context records.",
        integration_note,
        "",
        f"- Explicit evaluations present: {len(evaluations)}.",
        final_performance_note,
    ]
    return "\n".join(lines) + "\n"


def generate_run_reports(
    run_root: Path,
    run_id: str,
    report_root: Path,
    prepared: PreparedIntentData,
) -> dict[str, Any]:
    """Generate a report bundle from one real, published model run."""
    run_dir = resolve_intent_run(run_root, run_id)
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest is None:
        raise IntentReportError("published run manifest is missing")
    output_dir = Path(report_root).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    base = write_data_reports(prepared, output_dir)
    evaluations = _evaluation_documents(run_dir)
    generated: dict[str, Any] = {**base, "report_dir": str(output_dir)}
    baseline = _read_json(run_dir / "baseline_results.json")
    if baseline is not None:
        rows = []
        for candidate in baseline.get("candidate_results", []):
            metrics = candidate.get("validation_metrics", {})
            rows.append(
                {
                    "model": candidate.get("model"),
                    "selected": candidate.get("selected"),
                    "C": candidate.get("C"),
                    "class_weighting": candidate.get("class_weighting"),
                    "macro_f1": metrics.get("macro_f1"),
                    "macro_precision": metrics.get("macro_precision"),
                    "macro_recall": metrics.get("macro_recall"),
                    "accuracy": metrics.get("accuracy"),
                    "probability_status": candidate.get("probability_status"),
                }
            )
        path = _write_csv(
            output_dir / "baseline_comparison.csv",
            rows,
            [
                "model",
                "selected",
                "C",
                "class_weighting",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "accuracy",
                "probability_status",
            ],
        )
        generated["baseline_comparison"] = str(path)
    history = _read_json(run_dir / "training_history.json")
    if history is not None:
        rows = []
        for item in history.get("epochs", []):
            validation = item.get("validation", {})
            rows.append(
                {
                    "epoch": item.get("epoch"),
                    "train_loss": item.get("train_loss"),
                    "validation_macro_f1": validation.get("macro_f1"),
                    "validation_accuracy": validation.get("accuracy"),
                    "learning_rate": item.get("learning_rate"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                }
            )
        path = _write_csv(
            output_dir / "training_history.csv",
            rows,
            [
                "epoch",
                "train_loss",
                "validation_macro_f1",
                "validation_accuracy",
                "learning_rate",
                "elapsed_seconds",
            ],
        )
        generated["training_history"] = str(path)
    per_class = _per_class_rows(evaluations)
    if per_class:
        generated["per_class_metrics"] = str(
            _write_csv(
                output_dir / "per_class_metrics.csv",
                per_class,
                ["split", "model", "label", "precision", "recall", "f1", "support"],
            )
        )
    errors = _error_rows(evaluations)
    if errors:
        generated["error_analysis"] = str(
            _write_csv(
                output_dir / "error_analysis.csv",
                errors,
                [
                    "split",
                    "model",
                    "citation_context_id",
                    "source_record_id",
                    "true_label",
                    "predicted_label",
                    "prob_background",
                    "prob_method",
                    "prob_result",
                    "model_version",
                    "text_reference",
                ],
            )
        )
    plots = _plot_confusion_matrices(evaluations, output_dir)
    if plots:
        generated["confusion_matrices"] = plots
    evaluation_summary = [
        {
            "split": item.get("split"),
            "model": item.get("model"),
            "evaluated_at": item.get("evaluated_at"),
            "metrics": item.get("metrics"),
        }
        for item in evaluations
    ]
    atomic_json(output_dir / "evaluation_summary.json", evaluation_summary)
    generated["evaluation_summary"] = str(output_dir / "evaluation_summary.json")
    comparison = _write_markdown(
        output_dir / "model_comparison.md", _comparison_markdown(manifest, evaluations)
    )
    readiness = _write_markdown(
        output_dir / "stage04_readiness.md", _readiness_markdown(prepared, evaluations)
    )
    generated["model_comparison"] = str(comparison)
    generated["stage04_readiness"] = str(readiness)
    return generated
