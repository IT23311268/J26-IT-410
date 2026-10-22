"""Train-only TF-IDF baselines with safe JSON/NPZ persistence.

Scikit-learn objects are never loaded from pickle/joblib.  Selected estimators are
exported as audited vocabulary, IDF, and coefficient arrays so a later inference
process can reconstruct their prediction function without deserializing code.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..data.manifests import atomic_json
from .intent_config import IntentBaselineSettings
from .intent_contracts import (
    INTENT_LABELS,
    IntentDataError,
    IntentDatasetRecord,
    normalize_context_text,
)
from .intent_metrics import evaluate_predictions


class BaselineDependencyError(RuntimeError):
    """Raised only when an explicitly requested sklearn baseline lacks its dependency."""


def _require_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
    except ImportError as error:
        raise BaselineDependencyError(
            "TF-IDF baselines require the optional Stage 04 citation-intent dependencies"
        ) from error
    return TfidfVectorizer, LogisticRegression, LinearSVC


def calculate_class_weights(
    labels: Sequence[int], *, strategy: str, smoothing: float = 1.0
) -> dict[int, float] | None:
    """Calculate train-only class weights for the exact three-class taxonomy."""
    if any(type(label) is not int or label not in {0, 1, 2} for label in labels):
        raise IntentDataError("class weights require fixed SciCite class IDs 0, 1, 2")
    if strategy == "none":
        return None
    if strategy not in {"inverse", "smoothed"}:
        raise IntentDataError("class-weight strategy must be none, inverse, or smoothed")
    if smoothing <= 0 or not math.isfinite(smoothing):
        raise IntentDataError("class-weight smoothing must be positive")
    counts = [labels.count(label) for label in range(3)]
    if any(count == 0 for count in counts):
        raise IntentDataError("class weighting requires every SciCite class in the training split")
    total = len(labels)
    denominators = [count if strategy == "inverse" else count + smoothing for count in counts]
    raw = [total / (3 * value) for value in denominators]
    # Keep the average weight one across classes for stable optimizer scale.
    mean = sum(raw) / len(raw)
    return {label: value / mean for label, value in enumerate(raw)}


def _label_counts(records: Sequence[IntentDatasetRecord]) -> dict[str, int]:
    return {
        label: sum(record.label_id == index for record in records)
        for index, label in enumerate(INTENT_LABELS)
    }


def _softmax(values: Sequence[float]) -> list[float]:
    ceiling = max(values)
    exponentials = [math.exp(value - ceiling) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _vectorizer(settings: IntentBaselineSettings):
    TfidfVectorizer, _, _ = _require_sklearn()
    try:
        import numpy as np
    except ImportError as error:
        raise BaselineDependencyError("TF-IDF baselines require NumPy") from error
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(settings.ngram_min, settings.ngram_max),
        min_df=settings.min_document_frequency,
        max_df=settings.max_document_frequency,
        max_features=settings.max_features,
        dtype=np.float64,
    )


def _fit_estimator(
    kind: str,
    x_train,
    y_train: list[int],
    *,
    c_value: float,
    class_weight: dict[int, float] | None,
    settings: IntentBaselineSettings,
    seed: int,
):
    _, LogisticRegression, LinearSVC = _require_sklearn()
    if kind == "tfidf_logistic_regression":
        estimator = LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=settings.max_iter,
            random_state=seed,
            solver="lbfgs",
        )
    elif kind == "tfidf_linear_svm":
        estimator = LinearSVC(
            C=c_value,
            class_weight=class_weight,
            max_iter=settings.max_iter,
            random_state=seed,
            dual="auto",
        )
    else:
        raise IntentDataError(f"unknown TF-IDF baseline: {kind}")
    estimator.fit(x_train, y_train)
    if list(estimator.classes_) != [0, 1, 2]:
        raise IntentDataError("baseline training did not retain the complete fixed label order")
    return estimator


def _save_linear_artifact(
    destination: Path,
    *,
    kind: str,
    vectorizer,
    estimator,
    selected: dict,
) -> None:
    """Persist only JSON metadata and NPZ numeric arrays; never pickle estimators."""
    try:
        import numpy as np
    except ImportError as error:
        raise BaselineDependencyError("baseline persistence requires NumPy") from error
    destination.mkdir(parents=True, exist_ok=False)
    vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
    (destination / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        destination / "weights.npz",
        idf=np.asarray(vectorizer.idf_, dtype="float64"),
        coefficients=np.asarray(estimator.coef_, dtype="float64"),
        intercept=np.asarray(estimator.intercept_, dtype="float64"),
    )
    atomic_json(
        destination / "metadata.json",
        {
            "schema_version": "1.0",
            "model_kind": kind,
            "label_order": list(INTENT_LABELS),
            "classes": [0, 1, 2],
            "vectorizer": {
                "lowercase": True,
                "ngram_min": vectorizer.ngram_range[0],
                "ngram_max": vectorizer.ngram_range[1],
                "token_pattern": vectorizer.token_pattern,
                "normalization": "l2",
                "use_idf": True,
                "smooth_idf": True,
            },
            "selected_on": "dev_macro_f1_only",
            "selected": selected,
            "serialization": {
                "format": "JSON plus NumPy NPZ with allow_pickle=False at load time",
                "untrusted_artifacts": (
                    "rejected by policy; only local run artifacts should be used"
                ),
            },
        },
    )


def _save_majority_artifact(destination: Path, *, labels: Sequence[int], selected: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    counts = [labels.count(label) for label in range(3)]
    majority = max(range(3), key=lambda label: (counts[label], -label))
    atomic_json(
        destination / "metadata.json",
        {
            "schema_version": "1.0",
            "model_kind": "majority",
            "label_order": list(INTENT_LABELS),
            "classes": [0, 1, 2],
            "class_counts": counts,
            "class_priors": [count / len(labels) for count in counts],
            "majority_label_id": majority,
            "selected_on": "no_tunable_parameters",
            "selected": selected,
            "serialization": "JSON only",
        },
    )


@dataclass(frozen=True)
class BaselinePredictor:
    """A safe reconstructed baseline predictor for batch inference and evaluation."""

    artifact_dir: Path
    metadata: dict

    @classmethod
    def load(cls, artifact_dir: Path) -> BaselinePredictor:
        artifact_dir = Path(artifact_dir)
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise IntentDataError("baseline artifact metadata is missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("label_order") != list(INTENT_LABELS) or metadata.get("classes") != [
            0,
            1,
            2,
        ]:
            raise IntentDataError("baseline artifact label mapping is incompatible with SciCite")
        kind = metadata.get("model_kind")
        if kind not in {"majority", "tfidf_logistic_regression", "tfidf_linear_svm"}:
            raise IntentDataError("unknown baseline artifact type")
        return cls(artifact_dir=artifact_dir, metadata=metadata)

    def _linear_arrays(self):
        try:
            import numpy as np
        except ImportError as error:
            raise BaselineDependencyError("baseline inference requires NumPy") from error
        vocabulary = json.loads((self.artifact_dir / "vocabulary.json").read_text(encoding="utf-8"))
        with np.load(self.artifact_dir / "weights.npz", allow_pickle=False) as arrays:
            idf = arrays["idf"].astype("float64", copy=True)
            coefficients = arrays["coefficients"].astype("float64", copy=True)
            intercept = arrays["intercept"].astype("float64", copy=True)
        if coefficients.shape != (3, len(vocabulary)) or idf.shape != (len(vocabulary),):
            raise IntentDataError("baseline artifact shapes are invalid")
        return vocabulary, idf, coefficients, intercept

    def _features(self, text: str, vocabulary: dict[str, int], idf):
        import numpy as np

        settings = self.metadata["vectorizer"]
        tokens = re.findall(settings["token_pattern"], normalize_context_text(text).lower())
        terms = []
        for size in range(settings["ngram_min"], settings["ngram_max"] + 1):
            terms.extend(
                " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
            )
        counts: dict[int, int] = {}
        for term in terms:
            index = vocabulary.get(term)
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
        features = np.zeros(len(vocabulary), dtype="float64")
        for index, count in counts.items():
            features[index] = count * idf[index]
        norm = float(np.linalg.norm(features))
        return features / norm if norm else features

    def predict(self, texts: Sequence[str]) -> dict:
        kind = self.metadata["model_kind"]
        if kind == "majority":
            priors = [float(value) for value in self.metadata["class_priors"]]
            label = int(self.metadata["majority_label_id"])
            return {
                "predicted_ids": [label] * len(texts),
                "scores": [priors] * len(texts),
                "probabilities": [priors] * len(texts),
                "calibration_status": "not_applicable",
            }
        vocabulary, idf, coefficients, intercept = self._linear_arrays()
        scores = []
        for text in texts:
            features = self._features(text, vocabulary, idf)
            scores.append((coefficients @ features + intercept).tolist())
        predicted = [max(range(3), key=row.__getitem__) for row in scores]
        if kind == "tfidf_logistic_regression":
            probabilities = [_softmax(row) for row in scores]
            return {
                "predicted_ids": predicted,
                "scores": scores,
                "probabilities": probabilities,
                "calibration_status": "uncalibrated",
            }
        return {
            "predicted_ids": predicted,
            "scores": scores,
            "probabilities": None,
            "calibration_status": "not_applicable",
        }


def train_baselines(
    train_records: Sequence[IntentDatasetRecord],
    dev_records: Sequence[IntentDatasetRecord],
    destination: Path,
    settings: IntentBaselineSettings,
    *,
    seed: int,
) -> dict:
    """Fit baseline candidates on train and choose each model only by dev macro F1."""
    if not train_records or not dev_records:
        raise IntentDataError("baseline training requires nonempty official train and dev splits")
    if any(record.source_split != "train" for record in train_records):
        raise IntentDataError("baseline fitting accepts only official training records")
    if any(record.source_split != "dev" for record in dev_records):
        raise IntentDataError("baseline selection accepts only official development records")
    train_texts = [record.model_text for record in train_records]
    dev_texts = [record.model_text for record in dev_records]
    y_train = [record.label_id for record in train_records]
    y_dev = [record.label_id for record in dev_records]
    if set(y_train) != {0, 1, 2}:
        raise IntentDataError("official training split must contain all three SciCite labels")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    candidate_results = []
    selected_models = {}
    started = time.perf_counter()
    if "majority" in settings.models:
        majority = max(range(3), key=lambda label: (y_train.count(label), -label))
        predicted = [majority] * len(y_dev)
        metrics = evaluate_predictions(y_dev, predicted)
        selected = {
            "model": "majority",
            "majority_label_id": majority,
            "majority_label": INTENT_LABELS[majority],
            "validation_metrics": metrics,
            "probability_status": "class_prior_not_instance_calibrated",
        }
        _save_majority_artifact(destination / "majority", labels=y_train, selected=selected)
        selected_models["majority"] = {**selected, "artifact": "majority"}
        candidate_results.append({**selected, "selected": True})
    for kind in ("tfidf_logistic_regression", "tfidf_linear_svm"):
        if kind not in settings.models:
            continue
        vectorizer = _vectorizer(settings)
        x_train = vectorizer.fit_transform(train_texts)
        x_dev = vectorizer.transform(dev_texts)
        best = None
        for weighting in settings.class_weighting_candidates:
            class_weight = calculate_class_weights(
                y_train, strategy=weighting, smoothing=settings.class_weight_smoothing
            )
            for c_value in settings.candidate_c:
                estimator = _fit_estimator(
                    kind,
                    x_train,
                    y_train,
                    c_value=c_value,
                    class_weight=class_weight,
                    settings=settings,
                    seed=seed,
                )
                predicted = [int(value) for value in estimator.predict(x_dev)]
                probabilities = (
                    [[float(item) for item in row] for row in estimator.predict_proba(x_dev)]
                    if kind == "tfidf_logistic_regression"
                    else None
                )
                metrics = evaluate_predictions(y_dev, predicted, probabilities)
                candidate = {
                    "model": kind,
                    "C": c_value,
                    "class_weighting": weighting,
                    "class_weights": class_weight,
                    "validation_metrics": metrics,
                    "probability_status": "uncalibrated"
                    if probabilities is not None
                    else "not_available_for_linear_svm",
                }
                candidate_results.append(candidate)
                rank = (metrics["macro_f1"], metrics["macro_recall"], -c_value)
                if best is None or rank > best[0]:
                    best = (rank, candidate, estimator)
        assert best is not None
        _, chosen, estimator = best
        artifact = destination / kind
        _save_linear_artifact(
            artifact,
            kind=kind,
            vectorizer=vectorizer,
            estimator=estimator,
            selected=chosen,
        )
        selected_models[kind] = {**chosen, "artifact": kind}
    elapsed = time.perf_counter() - started
    selected_names = set(selected_models)
    for candidate in candidate_results:
        model = candidate["model"]
        candidate["selected"] = bool(
            model in selected_names
            and candidate.get("C") == selected_models[model].get("C")
            and candidate.get("class_weighting") == selected_models[model].get("class_weighting")
        )
    return {
        "schema_version": "1.0",
        "training_split": "train",
        "selection_split": "dev",
        "test_split_used": False,
        "label_distribution_train": _label_counts(train_records),
        "label_distribution_dev": _label_counts(dev_records),
        "selected_models": selected_models,
        "candidate_results": candidate_results,
        "training_seconds": elapsed,
        "serialization_policy": (
            "selected sklearn models are exported as JSON and NPZ; pickle/joblib is not used"
        ),
    }


def evaluate_baseline(
    artifact_dir: Path, records: Sequence[IntentDatasetRecord], *, ece_bins: int = 15
) -> tuple[dict, dict]:
    """Evaluate an already selected model without refitting or choosing parameters."""
    predictor = BaselinePredictor.load(artifact_dir)
    texts = [record.model_text for record in records]
    result = predictor.predict(texts)
    metrics = evaluate_predictions(
        [record.label_id for record in records],
        result["predicted_ids"],
        result["probabilities"],
        ece_bins=ece_bins,
    )
    return metrics, result
