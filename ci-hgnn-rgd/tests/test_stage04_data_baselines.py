"""Synthetic offline Stage 04 data, baseline, and inference tests.

No fixture in this file is a SciCite record or a scholarly performance result.
"""

import json
from pathlib import Path

import pytest

from ci_hgnn_rgd.data.provenance import file_sha256
from ci_hgnn_rgd.nlp.citation_intent import CitationIntentClassifier
from ci_hgnn_rgd.nlp.intent_baselines import (
    BaselinePredictor,
    calculate_class_weights,
    evaluate_baseline,
    train_baselines,
)
from ci_hgnn_rgd.nlp.intent_config import Stage04Config
from ci_hgnn_rgd.nlp.intent_data import prepare_intent_data
from ci_hgnn_rgd.nlp.intent_inference import (
    InferenceInputError,
    predict_contexts,
    read_prediction_inputs,
    write_prediction_export,
)
from ci_hgnn_rgd.schemas import CitationContext


def source_row(split: str, label: str, identifier: str, text: str) -> dict:
    return {
        "context_id": f"synthetic:{identifier}",
        "source_dataset": "scicite",
        "source_record_id": f"source:{identifier}",
        "dataset_version": "synthetic-fixture-v1",
        "source_split": split,
        "quarantine": False,
        "original_context": text,
        "normalized_context": text,
        "original_intent_label": label,
        "normalized_intent_label": label.capitalize(),
        "citing_paper_id": None,
        "cited_paper_id": None,
        "identity_resolution_status": "UNRESOLVED",
        "provenance": [
            {
                "synthetic": True,
                "source_dataset": "scicite",
                "dataset_version": "synthetic-fixture-v1",
                "source_file": "synthetic-fixture.jsonl",
            }
        ],
        "temporal_eligibility": {"eligible": False, "evidence": {}},
        "transformations": [],
    }


def stage03_fixture(tmp_path: Path, *, include_invalid: bool = False) -> tuple[Stage04Config, Path]:
    root = tmp_path / "stage03"
    run = root / "run-synthetic"
    run.mkdir(parents=True)
    rows = []
    for split, count in (("train", 2), ("dev", 1), ("test", 1)):
        for label in ("background", "method", "result"):
            for index in range(count):
                token = "devonlytoken" if split == "dev" else f"{split}token"
                rows.append(
                    source_row(
                        split,
                        label,
                        f"{split}-{label}-{index}",
                        f"{label} fixture text {token} item {index}",
                    )
                )
    if include_invalid:
        invalid = source_row("train", "background", "invalid", "invalid fixture context")
        invalid["original_intent_label"] = "unsupported"
        rows.append(invalid)
    table = run / "citation_contexts.jsonl"
    table.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "status": "COMPLETE",
        "run_id": "synthetic-stage03-run",
        "real_sources": ["scicite"],
        "synthetic_sources": ["software-fixture"],
        "output_checksums": {"citation_contexts.jsonl": file_sha256(table)},
        "table_counts": {
            "cleaned_records": len(rows),
            "citation_contexts": len(rows),
            "quarantine": 0,
        },
        "split_audit": {"synthetic_fixture": True},
        "temporal": {"synthetic_fixture": True},
    }
    (run / "integration_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "latest.json").write_text(json.dumps({"run_dir": str(run)}), encoding="utf-8")
    config = Stage04Config.model_validate(
        {
            "dataset": {"stage03_output_dir": root, "stage03_run": run},
            "outputs": {
                "model_root": tmp_path / "models",
                "report_root": tmp_path / "reports",
                "work_root": tmp_path / "work",
            },
            "baselines": {"max_features": 100, "candidate_c": [1.0], "max_iter": 100},
        }
    )
    return config, run


def test_prepare_view_preserves_official_splits_and_marks_temporal_limit(tmp_path):
    config, run = stage03_fixture(tmp_path)
    prepared = prepare_intent_data(config, tmp_path, stage03_run=run)
    assert [len(prepared.split(split)) for split in ("train", "dev", "test")] == [6, 3, 3]
    assert prepared.dataset_manifest["official_splits_preserved"] is True
    assert prepared.temporal_validation["status"] == "NOT_EVALUABLE"
    assert prepared.split("train")[0].original_label in {"background", "method", "result"}
    assert all(record.citing_paper_id is None for record in prepared.all_records)


def test_prepare_view_records_invalid_rows_as_exclusions(tmp_path):
    config, run = stage03_fixture(tmp_path, include_invalid=True)
    prepared = prepare_intent_data(config, tmp_path, stage03_run=run)
    assert prepared.dataset_profile["records_excluded_by_stage04"] == 1
    assert "unsupported SciCite label" in prepared.exclusions[0]["reason"]


def test_prepare_view_rejects_a_stage03_table_checksum_mismatch(tmp_path):
    config, run = stage03_fixture(tmp_path)
    table = run / "citation_contexts.jsonl"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        prepare_intent_data(config, tmp_path, stage03_run=run)


def test_train_only_tfidf_baselines_and_safe_reconstructed_inference(tmp_path):
    config, run = stage03_fixture(tmp_path)
    prepared = prepare_intent_data(config, tmp_path, stage03_run=run)
    weights = calculate_class_weights([0, 0, 0, 1, 2], strategy="inverse")
    assert weights is not None and weights[1] == weights[2] > weights[0]
    assert calculate_class_weights([0, 1, 2], strategy="none") is None
    result = train_baselines(
        prepared.split("train"),
        prepared.split("dev"),
        tmp_path / "baselines",
        config.baselines,
        seed=7,
    )
    assert set(result["selected_models"]) == {
        "majority",
        "tfidf_logistic_regression",
        "tfidf_linear_svm",
    }
    vocabulary = json.loads(
        (tmp_path / "baselines" / "tfidf_logistic_regression" / "vocabulary.json").read_text()
    )
    assert "devonlytoken" not in vocabulary
    artifact = tmp_path / "baselines" / "tfidf_logistic_regression"
    predictor = BaselinePredictor.load(artifact)
    prediction = predictor.predict(["method fixture text traintoken"])
    assert len(prediction["probabilities"][0]) == 3
    metrics, output = evaluate_baseline(artifact, prepared.split("dev"))
    assert metrics["samples"] == 3 and len(output["predicted_ids"]) == 3


class FakeProbabilityPredictor:
    def predict(self, texts, *, batch_size):
        probabilities = [[0.1, 0.8, 0.1] for _ in texts]
        return {
            "predicted_ids": [1 for _ in texts],
            "probabilities": probabilities,
            "truncated": [False for _ in texts],
            "calibration_status": "uncalibrated",
        }


def test_public_facade_returns_evidence_not_a_graph_edge(evidence):
    classifier = CitationIntentClassifier(
        FakeProbabilityPredictor(),
        {
            "model_id": "fixture-model",
            "model_version": "fixture-run",
            "calibration_status": "uncalibrated",
        },
    )
    context = CitationContext(
        context_id="synthetic:context",
        edge_id="synthetic:edge",
        text="A synthetic method context.",
        evidence=(evidence,),
    )
    prediction = classifier.predict((context,))[0]
    assert prediction.predicted_intent == "method"
    assert prediction.citing_paper_id is None
    assert prediction.provenance[0]["reference_id"] == "synthetic:ref1"


def test_order_preserving_prediction_and_jsonl_parquet_exports(tmp_path):
    inputs = [
        {
            "citation_context_id": "synthetic:first",
            "source_dataset": "fixture",
            "source_record_id": "one",
            "original_context": "A method fixture citation.",
            "provenance": [{"synthetic": True}],
        },
        {
            "citation_context_id": "synthetic:empty",
            "source_dataset": "fixture",
            "text": "  ",
        },
        {
            "citation_context_id": "synthetic:third",
            "source_dataset": "fixture",
            "text": "A second method fixture citation.",
        },
    ]
    model = {
        "model_id": "fixture-model",
        "model_version": "fixture-run",
        "model_revision": None,
        "tokenizer_id": "fixture-tokenizer",
        "tokenizer_revision": None,
        "calibration_status": "uncalibrated",
    }
    predictions = predict_contexts(
        inputs, FakeProbabilityPredictor(), model, batch_size=2, low_confidence_threshold=0.5
    )
    assert [item.citation_context_id for item in predictions] == [
        "synthetic:first",
        "synthetic:empty",
        "synthetic:third",
    ]
    assert predictions[1].prediction_status == "skipped_empty_text"
    assert predictions[0].citing_paper_id is None
    jsonl = tmp_path / "predictions.jsonl"
    parquet = tmp_path / "predictions.parquet"
    assert write_prediction_export(jsonl, predictions) == jsonl
    assert write_prediction_export(parquet, predictions) == parquet
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 3
    assert len(read_prediction_inputs(parquet)) == 3


def test_prediction_rejects_linear_svm_without_probabilities(tmp_path):
    class ScoresOnlyPredictor:
        def predict(self, texts, *, batch_size):
            return {"predicted_ids": [0 for _ in texts], "probabilities": None}

    with pytest.raises(InferenceInputError, match="no probabilities"):
        predict_contexts(
            [{"citation_context_id": "synthetic", "source_dataset": "fixture", "text": "text"}],
            ScoresOnlyPredictor(),
            {"model_id": "scores", "model_version": "v1", "calibration_status": "not_applicable"},
            batch_size=1,
            low_confidence_threshold=0.5,
        )
