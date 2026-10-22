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
    iter_context_predictions,
    iter_prediction_inputs,
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


@pytest.mark.parametrize(
    "bad_row",
    [
        [],
        None,
        12,
        {"provenance": 12},
        {"provenance": ["invalid"]},
        {"temporal_eligibility": {"evidence": []}},
    ],
)
def test_malformed_modelling_rows_are_excluded_with_reasons(tmp_path, bad_row):
    config, run = stage03_fixture(tmp_path)
    if isinstance(bad_row, dict):
        bad_row = {**source_row("train", "method", "bad", "fixture"), **bad_row}
    table = run / "citation_contexts.jsonl"
    with table.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(bad_row) + "\n")
    manifest_path = run / "integration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_checksums"]["citation_contexts.jsonl"] = file_sha256(table)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prepared = prepare_intent_data(config, tmp_path)
    assert prepared.dataset_profile["records_excluded_by_stage04"] == 1
    assert prepared.exclusions[0]["reason"]


def test_streaming_inference_is_bounded_and_preserves_global_order():
    consumed = []
    observed_batch_sizes = []

    def rows():
        for index in range(7):
            consumed.append(index)
            yield {"text": "" if index % 3 == 0 else "method fixture"}

    class BoundedPredictor(FakeProbabilityPredictor):
        def predict(self, texts, *, batch_size):
            assert len(texts) <= batch_size
            observed_batch_sizes.append(len(texts))
            return super().predict(texts, batch_size=batch_size)

    output = iter_context_predictions(
        rows(),
        BoundedPredictor(),
        {"model_id": "fixture", "model_version": "v1", "calibration_status": "uncalibrated"},
        batch_size=2,
        low_confidence_threshold=0.5,
    )
    first = next(output)
    assert len(consumed) == 2
    predictions = [first, *output]
    assert [item.input_order for item in predictions] == list(range(7))
    assert [item.citation_context_id for item in predictions] == [
        f"manual-input:{i}" for i in range(7)
    ]
    assert observed_batch_sizes and max(observed_batch_sizes) <= 2


def test_all_empty_inputs_do_not_call_predictor():
    class ForbiddenPredictor:
        def predict(self, *args, **kwargs):
            pytest.fail("empty inputs must not invoke the model")

    model = {"model_id": "fixture", "model_version": "v1"}
    outputs = predict_contexts(
        [{"text": "  "}], ForbiddenPredictor(), model, batch_size=1, low_confidence_threshold=0.5
    )
    assert outputs[0].prediction_status == "skipped_empty_text"
    assert (
        predict_contexts(
            [], ForbiddenPredictor(), model, batch_size=1, low_confidence_threshold=0.5
        )
        == ()
    )


def test_parquet_streaming_schema_handles_late_nonnull_fields_and_provenance(tmp_path):
    model = {"model_id": "fixture", "model_version": "v1", "calibration_status": "uncalibrated"}
    inputs = [
        {"text": "  ", "source_record_id": None},
        {
            "text": "method fixture",
            "source_record_id": 42,
            "citing_paper_id": "verified:p1",
            "provenance": [{"source": "fixture"}],
        },
    ]
    output = tmp_path / "stream.parquet"
    predictions = iter_context_predictions(
        inputs, FakeProbabilityPredictor(), model, batch_size=1, low_confidence_threshold=0.5
    )
    write_prediction_export(output, predictions, batch_size=1)
    rows = list(iter_prediction_inputs(output, batch_size=1))
    assert rows[0]["confidence"] is None
    assert rows[1]["source_record_id"] == 42
    assert rows[1]["provenance"] == [{"source": "fixture"}]
    assert rows[1]["citing_paper_id"] == "verified:p1"
    from ci_hgnn_rgd.nlp.intent_contracts import CitationIntentEvidence

    assert CitationIntentEvidence.model_validate(rows[1]).source_record_id == 42


@pytest.mark.parametrize("suffix", [".jsonl", ".parquet"])
def test_failed_stream_export_never_publishes_partial_output(tmp_path, suffix):
    model = {"model_id": "fixture", "model_version": "v1", "calibration_status": "uncalibrated"}
    prediction = predict_contexts(
        [{"text": "method fixture"}],
        FakeProbabilityPredictor(),
        model,
        batch_size=1,
        low_confidence_threshold=0.5,
    )[0]

    def failing_rows():
        yield prediction
        raise RuntimeError("synthetic interrupted stream")

    output = tmp_path / ("predictions" + suffix)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_prediction_export(output, failing_rows(), batch_size=1)
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_export_rejects_overwrite_and_empty_parquet_has_contract_schema(tmp_path):
    import pyarrow.parquet as pq

    output = tmp_path / "empty.parquet"
    write_prediction_export(output, iter(()))
    assert "citation_context_id" in pq.read_schema(output).names
    assert read_prediction_inputs(output) == []
    with pytest.raises(InferenceInputError, match="already exists"):
        write_prediction_export(output, [])


def test_text_excerpts_are_explicitly_opt_in():
    model = {"model_id": "fixture", "model_version": "v1", "calibration_status": "uncalibrated"}
    inputs = [{"citation_context_id": "fixture:context", "text": "method fixture text"}]
    default = predict_contexts(
        inputs, FakeProbabilityPredictor(), model, batch_size=1, low_confidence_threshold=0.5
    )[0]
    excerpt = predict_contexts(
        inputs,
        FakeProbabilityPredictor(),
        model,
        batch_size=1,
        low_confidence_threshold=0.5,
        include_text_excerpt=True,
        text_excerpt_characters=6,
    )[0]
    assert default.text_excerpt is None
    assert default.text_reference == "fixture:context"
    assert excerpt.text_excerpt == "method"


def test_reports_export_actual_split_metrics_without_inventing_test_results(tmp_path):
    from ci_hgnn_rgd.nlp.intent_artifacts import IntentRunPublication
    from ci_hgnn_rgd.nlp.intent_reporting import generate_run_reports

    config, run = stage03_fixture(tmp_path)
    prepared = prepare_intent_data(config, tmp_path)
    publication = IntentRunPublication(config.outputs.model_root, config={}, project_root=tmp_path)
    publication.write_json(
        "evaluations/dev/fixture.json",
        {
            "model": "fixture",
            "split": "dev",
            "metrics": {"macro_f1": 0.25},
            "evaluated_at": "2026-01-01T00:00:00Z",
        },
    )
    directory = publication.publish({"run_type": "synthetic-test"})
    generated = generate_run_reports(
        config.outputs.model_root, directory.name, config.outputs.report_root, prepared
    )
    metrics = json.loads(Path(generated["validation_metrics"]).read_text())
    assert metrics["models"]["fixture"]["metrics"]["macro_f1"] == 0.25
    assert "test_metrics" not in generated
    assert not (Path(generated["report_dir"]) / "test_metrics.json").exists()
    assert (directory / "environment.json").is_file()
