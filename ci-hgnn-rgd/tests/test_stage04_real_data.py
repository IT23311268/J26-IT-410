"""Opt-in genuine Stage 04 checks, distinct from synthetic offline unit tests."""

import os
from pathlib import Path

import pytest

from ci_hgnn_rgd.nlp.intent_baselines import BaselinePredictor, train_baselines
from ci_hgnn_rgd.nlp.intent_config import Stage04Config
from ci_hgnn_rgd.nlp.intent_contracts import LABEL_TO_ID
from ci_hgnn_rgd.nlp.intent_data import prepare_intent_data, quick_training_subset

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def genuine_view(tmp_path_factory):
    value = os.environ.get("CI_HGNN_STAGE04_STAGE03_RUN")
    if not value or not (Path(value) / "integration_manifest.json").is_file():
        pytest.skip("CI_HGNN_STAGE04_STAGE03_RUN must identify a genuine published Stage 03 run")
    root = Path(value).resolve()
    settings = Stage04Config.model_validate(
        {
            "dataset": {"stage03_output_dir": root.parent, "stage03_run": root},
            "baselines": {
                "max_features": 1000,
                "candidate_c": [1.0],
                "class_weighting_candidates": ["none"],
            },
        }
    )
    return settings, prepare_intent_data(settings, tmp_path_factory.mktemp("real-stage04"))


def test_real_stage04_official_splits_provenance_and_leakage(genuine_view):
    _, prepared = genuine_view
    assert prepared.dataset_manifest["official_splits_preserved"]
    assert prepared.dataset_manifest["label_mapping"] == LABEL_TO_ID
    assert set(prepared.records) == {"train", "dev", "test"}
    for split, records in prepared.records.items():
        assert records and all(record.source_split == split for record in records)
        assert all(record.provenance for record in records)
        assert all(
            not reference.get("synthetic", False)
            for record in records
            for reference in record.provenance
        )
    assert prepared.leakage_audit["stage04_model_view"]["contexts_inspected"] == len(
        prepared.all_records
    )


def test_bounded_real_baseline_smoke_uses_only_train_and_dev(tmp_path, genuine_view):
    config, prepared = genuine_view
    # This is a pipeline smoke check, never final research performance.
    train = quick_training_subset(prepared.split("train"), limit=60)
    dev = prepared.split("dev")[:30]
    result = train_baselines(train, dev, tmp_path / "baselines", config.baselines, seed=42)
    assert result["test_split_used"] is False
    assert len(result["selected_models"]) == 3
    artifact = tmp_path / "baselines" / "tfidf_logistic_regression"
    output = BaselinePredictor.load(artifact).predict([row.model_text for row in dev])
    assert len(output["probabilities"]) == len(dev)
    assert all(sum(row) == pytest.approx(1) for row in output["probabilities"])


@pytest.mark.scibert_training
def test_opt_in_real_scibert_fine_tuning(tmp_path, genuine_view):
    if os.environ.get("CI_HGNN_RUN_SCIBERT_TRAINING") != "1":
        pytest.skip("set CI_HGNN_RUN_SCIBERT_TRAINING=1 with cached weights and adequate memory")
    from ci_hgnn_rgd.nlp.intent_transformer import train_scibert

    _, prepared = genuine_view
    settings = Stage04Config.model_validate(
        {
            "offline": True,
            "model": {"max_length": 64},
            "training": {"mode": "quick", "epochs": 1, "train_batch_size": 1, "eval_batch_size": 1},
        }
    )
    result = train_scibert(
        quick_training_subset(prepared.split("train"), limit=12),
        prepared.split("dev")[:12],
        tmp_path / "real-scibert",
        settings,
        prepared.dataset_manifest,
    )
    assert (result.model_dir / "model.safetensors").is_file()
    assert result.validation_metrics["samples"] == 12
