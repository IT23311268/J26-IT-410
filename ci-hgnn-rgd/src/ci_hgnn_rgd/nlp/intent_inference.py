"""Order-preserving Stage 04 inference and provenance-safe prediction exports."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..pipeline import canonical_json
from .intent_baselines import BaselinePredictor
from .intent_contracts import (
    ID_TO_LABEL,
    CitationIntentEvidence,
    IntentDatasetRecord,
    entropy_from_probabilities,
    normalize_context_text,
    probabilities_for_label_order,
)
from .intent_transformer import TransformerPredictor


class InferenceInputError(ValueError):
    """Raised when an external inference input cannot be represented honestly."""


def _read_run_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        raise InferenceInputError("Stage 04 run manifest is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictor(
    run_dir: Path, *, model_name: str | None = None, device: str = "auto"
) -> tuple[Any, dict]:
    """Load a local, published Stage 04 model; never contact a model hub."""
    run_dir = Path(run_dir)
    manifest = _read_run_manifest(run_dir)
    run_type = manifest.get("run_type")
    if run_type == "scibert":
        predictor = TransformerPredictor.load(run_dir, device=device)
        metadata = predictor.metadata
        return predictor, {
            "model_id": metadata["model_id"],
            "model_revision": metadata.get("model_revision"),
            "model_version": manifest["run_id"],
            "tokenizer_id": metadata.get("tokenizer_id"),
            "tokenizer_revision": metadata.get("tokenizer_revision"),
            "calibration_status": "temperature_scaled"
            if predictor.calibration is not None
            else "uncalibrated",
        }
    if run_type == "baselines":
        selected = manifest.get("selected_models", {})
        if model_name is None:
            model_name = (
                "tfidf_logistic_regression"
                if "tfidf_logistic_regression" in selected
                else next(iter(selected), None)
            )
        if model_name not in selected:
            raise InferenceInputError("requested baseline is not a selected artifact in this run")
        predictor = BaselinePredictor.load(run_dir / "baselines" / selected[model_name]["artifact"])
        return predictor, {
            "model_id": model_name,
            "model_revision": None,
            "model_version": manifest["run_id"],
            "tokenizer_id": "sklearn_tfidf" if model_name != "majority" else None,
            "tokenizer_revision": None,
            "calibration_status": "uncalibrated",
        }
    raise InferenceInputError("Stage 04 run does not contain a supported inference model")


def _value(input_row: IntentDatasetRecord | dict, field: str, default=None):
    if isinstance(input_row, IntentDatasetRecord):
        return getattr(input_row, field, default)
    return input_row.get(field, default)


def _text(input_row: IntentDatasetRecord | dict) -> str:
    if isinstance(input_row, IntentDatasetRecord):
        return input_row.model_text
    value = input_row.get(
        "text", input_row.get("normalized_context", input_row.get("original_context"))
    )
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InferenceInputError("inference text must be a string")
    return normalize_context_text(value)


def _context_id(input_row: IntentDatasetRecord | dict, index: int) -> str:
    value = _value(input_row, "citation_context_id", None)
    if value is None and isinstance(input_row, dict):
        value = input_row.get("context_id")
    if value is None:
        return f"manual-input:{index}"
    if not isinstance(value, str) or not value.strip():
        raise InferenceInputError("citation-context identifier must be a nonempty string")
    return value


def predict_contexts(
    inputs: Sequence[IntentDatasetRecord | dict],
    predictor,
    model_info: dict,
    *,
    batch_size: int,
    low_confidence_threshold: float,
) -> tuple[CitationIntentEvidence, ...]:
    """Predict nonempty contexts in batches and retain the caller's input order."""
    if not 0 <= low_confidence_threshold <= 1:
        raise InferenceInputError("low-confidence threshold must be between zero and one")
    nonempty: list[tuple[int, str]] = []
    texts: list[str] = []
    for index, row in enumerate(inputs):
        text = _text(row)
        if text:
            nonempty.append((index, text))
            texts.append(text)
    # The reconstructed TF-IDF predictors are CPU/vectorized and do not accept a
    # batching keyword.  Transformer inference does, so keep the public evidence
    # function backend-neutral without swallowing arbitrary TypeErrors.
    model_output = (
        predictor.predict(texts)
        if isinstance(predictor, BaselinePredictor)
        else predictor.predict(texts, batch_size=batch_size)
    )
    probabilities = model_output.get("probabilities")
    if probabilities is None:
        raise InferenceInputError(
            "this selected model has uncalibrated decision scores but no probabilities; "
            "use SciBERT or TF-IDF logistic regression for provenance prediction exports"
        )
    if len(probabilities) != len(nonempty):
        raise RuntimeError("predictor returned a different number of probability rows")
    result_by_index = {}
    for offset, (index, _) in enumerate(nonempty):
        result_by_index[index] = (
            probabilities_for_label_order(tuple(probabilities[offset])),
            bool(model_output.get("truncated", [False] * len(nonempty))[offset]),
            model_output.get("calibration_status", model_info["calibration_status"]),
        )
    timestamp = datetime.now(UTC)
    outputs = []
    for index, row in enumerate(inputs):
        context_id = _context_id(row, index)
        source_dataset = _value(row, "source_dataset", "manual-input")
        provenance = _value(row, "provenance", ())
        if not isinstance(provenance, (list, tuple)):
            provenance = ()
        if index not in result_by_index:
            outputs.append(
                CitationIntentEvidence(
                    citation_context_id=context_id,
                    source_dataset=str(source_dataset),
                    source_record_id=_value(row, "source_record_id"),
                    citing_paper_id=_value(row, "citing_paper_id"),
                    cited_paper_id=_value(row, "cited_paper_id"),
                    source_split=_value(row, "source_split"),
                    calibration_status="not_applicable",
                    model_id=model_info["model_id"],
                    model_revision=model_info.get("model_revision"),
                    model_version=model_info["model_version"],
                    tokenizer_id=model_info.get("tokenizer_id"),
                    tokenizer_revision=model_info.get("tokenizer_revision"),
                    dataset_version=_value(row, "dataset_version"),
                    provenance=tuple(item for item in provenance if isinstance(item, dict)),
                    identity_resolution_status=_value(
                        row, "identity_resolution_status", "UNRESOLVED"
                    ),
                    prediction_timestamp=timestamp,
                    input_order=index,
                    prediction_status="skipped_empty_text",
                )
            )
            continue
        probability_row, truncated, calibration_status = result_by_index[index]
        predicted_id = max(range(3), key=probability_row.__getitem__)
        confidence = max(probability_row)
        outputs.append(
            CitationIntentEvidence(
                citation_context_id=context_id,
                source_dataset=str(source_dataset),
                source_record_id=_value(row, "source_record_id"),
                citing_paper_id=_value(row, "citing_paper_id"),
                cited_paper_id=_value(row, "cited_paper_id"),
                source_split=_value(row, "source_split"),
                predicted_intent=ID_TO_LABEL[predicted_id],
                prob_background=probability_row[0],
                prob_method=probability_row[1],
                prob_result=probability_row[2],
                confidence=confidence,
                predictive_entropy=entropy_from_probabilities(probability_row),
                calibration_status=calibration_status,
                low_confidence_review=confidence < low_confidence_threshold,
                model_id=model_info["model_id"],
                model_revision=model_info.get("model_revision"),
                model_version=model_info["model_version"],
                tokenizer_id=model_info.get("tokenizer_id"),
                tokenizer_revision=model_info.get("tokenizer_revision"),
                dataset_version=_value(row, "dataset_version"),
                provenance=tuple(item for item in provenance if isinstance(item, dict)),
                identity_resolution_status=_value(row, "identity_resolution_status", "UNRESOLVED"),
                prediction_timestamp=timestamp,
                input_order=index,
                truncation_applied=truncated,
            )
        )
    return tuple(outputs)


def _atomic_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(canonical_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_prediction_export(path: Path, predictions: Sequence[CitationIntentEvidence]) -> Path:
    """Write JSONL or Parquet predictions without serializing executable objects."""
    path = Path(path)
    rows = [prediction.model_dump(mode="json") for prediction in predictions]
    if path.suffix.lower() == ".jsonl":
        _atomic_jsonl(path, rows)
        return path
    if path.suffix.lower() != ".parquet":
        raise InferenceInputError("prediction output must end in .jsonl or .parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InferenceInputError(
            "Parquet prediction exports require the optional parquet extra"
        ) from error
    flattened = []
    for row in rows:
        flattened.append({**row, "provenance": canonical_json(row["provenance"])})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".tmp-" + path.name)
    pq.write_table(pa.Table.from_pylist(flattened), temporary, compression="zstd")
    os.replace(temporary, path)
    return path


def read_prediction_inputs(path: Path) -> list[dict]:
    """Read JSONL or Parquet inference rows without loading model artifacts."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise InferenceInputError(f"invalid JSONL at line {number}") from error
                if not isinstance(row, dict):
                    raise InferenceInputError(f"JSONL row {number} is not an object")
                rows.append(row)
        return rows
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise InferenceInputError(
                "Parquet input requires the optional parquet extra"
            ) from error
        return pq.read_table(path).to_pylist()
    raise InferenceInputError("prediction input must be JSONL or Parquet")
