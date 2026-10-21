"""Stage 04 data and prediction contracts for SciCite citation intent.

These contracts deliberately stay separate from the Stage 01 graph contracts.  A
prediction can be useful evidence for a later graph stage while its citation
endpoints remain unresolved, and a citation-intent label is never direct
investigation evidence.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import UTC
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, ConfigDict, Field, StrictInt, field_validator, model_validator

from ..schemas import FiniteNumber, FrozenModel, NonEmpty, Probability

IntentLabel = Literal["background", "method", "result"]
SourceSplit = Literal["train", "dev", "test"]

INTENT_LABELS: tuple[IntentLabel, ...] = ("background", "method", "result")
LABEL_TO_ID: dict[IntentLabel, int] = {label: index for index, label in enumerate(INTENT_LABELS)}
ID_TO_LABEL: dict[int, IntentLabel] = {index: label for label, index in LABEL_TO_ID.items()}


class IntentDataError(ValueError):
    """Raised when a source record cannot enter the Stage 04 modelling view."""


class IntentFrozenModel(FrozenModel):
    """Frozen contract that does not trim source text during validation."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=False, allow_inf_nan=False
    )


def normalize_context_text(value: str) -> str:
    """Apply only NFC and whitespace normalization without erasing scientific text."""
    if not isinstance(value, str):
        raise IntentDataError("citation context must be text")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def normalize_intent_label(value: object) -> IntentLabel:
    """Map SciCite's three source labels deterministically and reject everything else."""
    if not isinstance(value, str):
        raise IntentDataError("citation-intent label must be text")
    normalized = value.strip().casefold()
    if normalized not in LABEL_TO_ID:
        raise IntentDataError(f"unsupported SciCite label: {value!r}")
    return normalized  # type: ignore[return-value]


def entropy_from_probabilities(probabilities: tuple[float, float, float]) -> float:
    """Shannon entropy in natural units; it is not epistemic uncertainty."""
    return -sum(value * math.log(value) for value in probabilities if value > 0)


class IntentDatasetRecord(IntentFrozenModel):
    """One non-quarantined Stage 03 citation context prepared for Stage 04."""

    citation_context_id: NonEmpty
    source_dataset: NonEmpty
    source_record_id: str | int | None = None
    dataset_version: str | None = None
    source_split: SourceSplit
    original_text: str
    model_text: str
    # Preserve the source spelling verbatim.  ``normalize_intent_label`` validates
    # it later without replacing the original raw value in the modelling record.
    original_label: str
    normalized_label: IntentLabel
    label_id: Annotated[StrictInt, Field(ge=0, le=2)]
    citing_paper_id: str | None = None
    cited_paper_id: str | None = None
    citing_identifier: dict[str, Any] | None = None
    cited_identifier: dict[str, Any] | None = None
    identity_resolution_status: NonEmpty = "UNRESOLVED"
    provenance: tuple[dict[str, Any], ...] = ()
    temporal_eligibility: dict[str, Any] = Field(default_factory=dict)
    transformations: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def exact_label_mapping(self):
        if LABEL_TO_ID[self.normalized_label] != self.label_id:
            raise ValueError("label_id does not match the fixed SciCite label mapping")
        if normalize_intent_label(self.original_label) != self.normalized_label:
            raise ValueError("original label disagrees with normalized SciCite label")
        if not self.original_text or not self.original_text.strip():
            raise ValueError("original text must not be empty")
        if not self.model_text or not self.model_text.strip():
            raise ValueError("model text must not be empty")
        return self


class CitationIntentEvidence(FrozenModel):
    """Provenance-preserving Stage 04 output for a single citation context.

    Empty inference inputs are retained as ``skipped_empty_text`` records.  They
    carry no prediction or probability instead of a fabricated class.
    """

    citation_context_id: NonEmpty
    source_dataset: NonEmpty
    source_record_id: str | int | None = None
    citing_paper_id: str | None = None
    cited_paper_id: str | None = None
    source_split: SourceSplit | None = None
    predicted_intent: IntentLabel | None = None
    prob_background: Probability | None = None
    prob_method: Probability | None = None
    prob_result: Probability | None = None
    confidence: Probability | None = None
    predictive_entropy: FiniteNumber | None = Field(default=None, ge=0)
    calibration_status: Literal["uncalibrated", "temperature_scaled", "not_applicable"]
    low_confidence_review: bool = False
    model_id: NonEmpty
    model_revision: str | None = None
    model_version: NonEmpty
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    dataset_version: str | None = None
    provenance: tuple[dict[str, Any], ...] = ()
    identity_resolution_status: NonEmpty = "UNRESOLVED"
    prediction_timestamp: AwareDatetime
    input_order: Annotated[StrictInt, Field(ge=0)] = 0
    prediction_status: Literal["completed", "skipped_empty_text"] = "completed"
    truncation_applied: bool = False

    @field_validator("prediction_timestamp")
    @classmethod
    def prediction_time_is_utc(cls, value):
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def valid_prediction(self):
        values = (self.prob_background, self.prob_method, self.prob_result)
        if self.prediction_status == "skipped_empty_text":
            if self.predicted_intent is not None or any(value is not None for value in values):
                raise ValueError("skipped input must not have a fabricated prediction")
            if self.confidence is not None or self.predictive_entropy is not None:
                raise ValueError("skipped input must not have confidence or entropy")
            if self.calibration_status != "not_applicable":
                raise ValueError("skipped input requires not_applicable calibration status")
            return self
        if self.predicted_intent is None or any(value is None for value in values):
            raise ValueError("completed prediction requires all three probabilities and a label")
        probabilities = tuple(float(value) for value in values)
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("citation-intent probabilities must sum to one")
        if self.confidence is None or self.predictive_entropy is None:
            raise ValueError("completed prediction requires confidence and entropy")
        if abs(self.confidence - max(probabilities)) > 1e-6:
            raise ValueError("confidence must equal the maximum class probability")
        if self.predicted_intent != ID_TO_LABEL[max(range(3), key=probabilities.__getitem__)]:
            raise ValueError("predicted label must match the maximum class probability")
        return self


def probabilities_for_label_order(
    values: list[float] | tuple[float, ...],
) -> tuple[float, float, float]:
    """Validate and freeze probabilities in Background/Method/Result order."""
    if len(values) != len(INTENT_LABELS):
        raise IntentDataError("exactly three class probabilities are required")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in result):
        raise IntentDataError("probabilities must be finite values in [0, 1]")
    if abs(sum(result) - 1.0) > 1e-6:
        raise IntentDataError("probabilities must sum to one")
    return result  # type: ignore[return-value]
