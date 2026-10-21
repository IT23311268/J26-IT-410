"""Dev-only temperature scaling and calibration summaries for Stage 04."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .intent_contracts import IntentDataError, probabilities_for_label_order
from .intent_metrics import calibration_metrics


def softmax(logits: Sequence[float], *, temperature: float = 1.0) -> list[float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise IntentDataError("temperature must be a positive finite number")
    if len(logits) != 3 or any(not math.isfinite(float(value)) for value in logits):
        raise IntentDataError("exactly three finite logits are required")
    ceiling = max(float(value) / temperature for value in logits)
    values = [math.exp(float(value) / temperature - ceiling) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    return -sum(
        math.log(max(softmax(row, temperature=temperature)[label], 1e-15))
        for row, label in zip(logits, labels, strict=True)
    ) / len(labels)


@dataclass(frozen=True)
class TemperatureScaling:
    temperature: float
    fitted_on_split: str
    examples: int
    optimization: str = "bounded_log_temperature_ternary_search"

    def apply(self, logits: Sequence[Sequence[float]]) -> list[list[float]]:
        return [softmax(row, temperature=self.temperature) for row in logits]

    def to_dict(self) -> dict:
        return {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "fitted_on_split": self.fitted_on_split,
            "examples": self.examples,
            "optimization": self.optimization,
            "calibration_status": "temperature_scaled",
        }


def fit_temperature_scaling(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    split: str,
    ece_bins: int = 15,
) -> tuple[TemperatureScaling, dict]:
    """Fit one scalar temperature strictly on development predictions.

    The scalar search is deterministic and dependency-free.  It is deliberately
    prohibited on held-out test predictions.
    """
    if split != "dev":
        raise IntentDataError(
            "temperature scaling may only fit on the designated development split"
        )
    if not logits or len(logits) != len(labels):
        raise IntentDataError("calibration logits and labels must be nonempty and aligned")
    if any(type(label) is not int or label not in {0, 1, 2} for label in labels):
        raise IntentDataError("calibration labels must be fixed SciCite class IDs")
    for row in logits:
        softmax(row)  # validates shape and finiteness
    # Search log(T) over a wide but bounded range.  The interval captures practical
    # over/under-confidence without allowing degenerate zero or infinite scales.
    low, high = math.log(0.05), math.log(20.0)
    for _ in range(80):
        first = low + (high - low) / 3
        second = high - (high - low) / 3
        if _nll(logits, labels, math.exp(first)) <= _nll(logits, labels, math.exp(second)):
            high = second
        else:
            low = first
    scaler = TemperatureScaling(
        temperature=math.exp((low + high) / 2), fitted_on_split="dev", examples=len(labels)
    )
    raw = [softmax(row) for row in logits]
    calibrated = scaler.apply(logits)
    report = {
        **scaler.to_dict(),
        "uncalibrated": calibration_metrics(labels, raw, bins=ece_bins),
        "temperature_scaled": calibration_metrics(labels, calibrated, bins=ece_bins),
        "test_split_used": False,
        "interpretation": (
            "Temperature scaling is fitted on development data only. It calibrates a scalar "
            "probability transformation and does not measure epistemic uncertainty."
        ),
    }
    return scaler, report


def apply_temperature_to_probabilities(
    logits: Sequence[Sequence[float]], scaler: TemperatureScaling | None
) -> tuple[list[list[float]], str]:
    if scaler is None:
        return [softmax(row) for row in logits], "uncalibrated"
    return scaler.apply(logits), "temperature_scaled"


def validate_probability_export(
    rows: Sequence[Sequence[float]],
) -> list[tuple[float, float, float]]:
    """Validate output rows at the model-boundary before writing inference evidence."""
    return [probabilities_for_label_order(tuple(row)) for row in rows]
