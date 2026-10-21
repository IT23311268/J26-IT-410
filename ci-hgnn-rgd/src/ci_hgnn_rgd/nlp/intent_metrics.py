"""Deterministic three-class metrics and uncertainty summaries for Stage 04.

The implementation is intentionally dependency-light so metrics remain available
for audit/report commands even when optional ML backends are not installed.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

from .intent_contracts import INTENT_LABELS, IntentDataError

LABEL_IDS = tuple(range(len(INTENT_LABELS)))


def _validate_targets(values: Sequence[int], *, name: str) -> None:
    if any(type(value) is not int or value not in LABEL_IDS for value in values):
        raise IntentDataError(f"{name} must contain only fixed SciCite class IDs 0, 1, 2")


def _validate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> None:
    if len(y_true) != len(y_pred):
        raise IntentDataError("true and predicted label counts differ")
    if not y_true:
        raise IntentDataError("evaluation requires at least one record")
    _validate_targets(y_true, name="true labels")
    _validate_targets(y_pred, name="predicted labels")


def _matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    matrix = [[0 for _ in LABEL_IDS] for _ in LABEL_IDS]
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[actual][predicted] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]] | None = None,
    *,
    ece_bins: int = 15,
) -> dict:
    """Return complete fixed-order metrics with zero-division defined as zero."""
    _validate_predictions(y_true, y_pred)
    if ece_bins < 2:
        raise IntentDataError("expected calibration error requires at least two bins")
    matrix = _matrix(y_true, y_pred)
    per_class = {}
    precision_values, recall_values, f1_values, supports = [], [], [], []
    for label_id, label in enumerate(INTENT_LABELS):
        true_positive = matrix[label_id][label_id]
        false_positive = sum(matrix[row][label_id] for row in LABEL_IDS if row != label_id)
        false_negative = sum(matrix[label_id][column] for column in LABEL_IDS if column != label_id)
        support = sum(matrix[label_id])
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "label_id": label_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        supports.append(support)
    total = len(y_true)
    normalized = [[_safe_divide(value, sum(row)) for value in row] for row in matrix]
    result = {
        "label_order": list(INTENT_LABELS),
        "label_ids": list(LABEL_IDS),
        "samples": total,
        "accuracy": _safe_divide(sum(matrix[index][index] for index in LABEL_IDS), total),
        "macro_precision": fmean(precision_values),
        "macro_recall": fmean(recall_values),
        "macro_f1": fmean(f1_values),
        "weighted_f1": _safe_divide(
            sum(score * support for score, support in zip(f1_values, supports, strict=True)), total
        ),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "normalized_confusion_matrix": normalized,
        "zero_division": 0,
    }
    if probabilities is not None:
        result["calibration"] = calibration_metrics(y_true, probabilities, bins=ece_bins)
    return result


def _validate_probability_rows(probabilities: Sequence[Sequence[float]], expected: int) -> None:
    if len(probabilities) != expected:
        raise IntentDataError("probability rows do not match evaluation records")
    for values in probabilities:
        if len(values) != len(LABEL_IDS):
            raise IntentDataError("each probability row must contain three classes")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise IntentDataError("probabilities must be finite nonnegative values")
        if abs(sum(float(value) for value in values) - 1.0) > 1e-6:
            raise IntentDataError("each probability row must sum to one")


def calibration_metrics(
    y_true: Sequence[int], probabilities: Sequence[Sequence[float]], *, bins: int = 15
) -> dict:
    """NLL, multiclass Brier score and top-label expected calibration error."""
    if not y_true:
        raise IntentDataError("calibration metrics require at least one record")
    _validate_targets(y_true, name="true labels")
    _validate_probability_rows(probabilities, len(y_true))
    nll = -fmean(
        math.log(max(float(probabilities[index][label]), 1e-15))
        for index, label in enumerate(y_true)
    )
    brier = fmean(
        sum(
            (float(probabilities[row][label]) - (1.0 if y_true[row] == label else 0.0)) ** 2
            for label in LABEL_IDS
        )
        for row in range(len(y_true))
    )
    bucket_count = [0] * bins
    bucket_confidence = [0.0] * bins
    bucket_accuracy = [0.0] * bins
    for actual, row in zip(y_true, probabilities, strict=True):
        prediction = max(LABEL_IDS, key=lambda label: float(row[label]))
        confidence = float(row[prediction])
        bucket = min(int(confidence * bins), bins - 1)
        bucket_count[bucket] += 1
        bucket_confidence[bucket] += confidence
        bucket_accuracy[bucket] += float(prediction == actual)
    ece = 0.0
    details = []
    total = len(y_true)
    for index, count in enumerate(bucket_count):
        if not count:
            continue
        confidence = bucket_confidence[index] / count
        accuracy = bucket_accuracy[index] / count
        ece += (count / total) * abs(accuracy - confidence)
        details.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "negative_log_likelihood": nll,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "ece_bins": bins,
        "bins": details,
        "interpretation": (
            "These describe calibration of supplied probabilities; raw softmax values are not "
            "automatically calibrated confidence estimates."
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise IntentDataError("cannot calculate a percentile of no bootstrap samples")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _bootstrap_units(length: int, groups: Sequence[str | None] | None) -> list[list[int]]:
    if groups is None:
        return [[index] for index in range(length)]
    if len(groups) != length:
        raise IntentDataError("bootstrap group count does not match predictions")
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if group is None:
            buckets[f"__record__:{index}"].append(index)
        else:
            buckets[str(group)].append(index)
    return [buckets[key] for key in sorted(buckets)]


def bootstrap_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    groups: Sequence[str | None] | None = None,
    seed: int = 42,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap a macro-F1 interval by evaluation unit or supplied cluster."""
    _validate_predictions(y_true, y_pred)
    if n_resamples < 1 or not 0 < confidence < 1:
        raise IntentDataError(
            "bootstrap resamples must be positive and confidence must be in (0, 1)"
        )
    units = _bootstrap_units(len(y_true), groups)
    rng = random.Random(seed)
    scores = []
    for _ in range(n_resamples):
        indices = [index for _ in units for index in rng.choice(units)]
        scores.append(
            evaluate_predictions([y_true[i] for i in indices], [y_pred[i] for i in indices])[
                "macro_f1"
            ]
        )
    alpha = (1 - confidence) / 2
    return {
        "metric": "macro_f1",
        "point_estimate": evaluate_predictions(y_true, y_pred)["macro_f1"],
        "lower": _percentile(scores, alpha),
        "upper": _percentile(scores, 1 - alpha),
        "confidence": confidence,
        "resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "verified_group" if groups is not None else "record",
        "assumption": (
            "Bootstrap intervals are sampling estimates. Group clusters are used only when a "
            "verified group identifier was supplied."
        ),
    }


def paired_bootstrap_difference(
    y_true: Sequence[int],
    left_pred: Sequence[int],
    right_pred: Sequence[int],
    *,
    groups: Sequence[str | None] | None = None,
    seed: int = 42,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap of left minus right macro F1 on an identical evaluation set."""
    _validate_predictions(y_true, left_pred)
    _validate_predictions(y_true, right_pred)
    if n_resamples < 1 or not 0 < confidence < 1:
        raise IntentDataError(
            "bootstrap resamples must be positive and confidence must be in (0, 1)"
        )
    units = _bootstrap_units(len(y_true), groups)
    rng = random.Random(seed)
    differences = []
    for _ in range(n_resamples):
        indices = [index for _ in units for index in rng.choice(units)]
        left = evaluate_predictions([y_true[i] for i in indices], [left_pred[i] for i in indices])[
            "macro_f1"
        ]
        right = evaluate_predictions(
            [y_true[i] for i in indices], [right_pred[i] for i in indices]
        )["macro_f1"]
        differences.append(left - right)
    alpha = (1 - confidence) / 2
    point = (
        evaluate_predictions(y_true, left_pred)["macro_f1"]
        - evaluate_predictions(y_true, right_pred)["macro_f1"]
    )
    return {
        "metric": "macro_f1_difference_left_minus_right",
        "point_estimate": point,
        "lower": _percentile(differences, alpha),
        "upper": _percentile(differences, 1 - alpha),
        "confidence": confidence,
        "resamples": n_resamples,
        "seed": seed,
        "resampling_unit": "verified_group" if groups is not None else "record",
        "interpretation": "An interval crossing zero is not a significance claim either way.",
    }
