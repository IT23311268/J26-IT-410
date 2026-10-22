"""Synthetic offline tests for Stage 04 contracts, audits, and metrics.

These fixtures exercise software behavior only; they are not scholarly examples,
model-performance claims, or citation-intent ground truth.
"""

from datetime import UTC, datetime

import pytest

from ci_hgnn_rgd.nlp.intent_calibration import fit_temperature_scaling
from ci_hgnn_rgd.nlp.intent_config import Stage04Config
from ci_hgnn_rgd.nlp.intent_contracts import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    CitationIntentEvidence,
    IntentDataError,
    IntentDatasetRecord,
    entropy_from_probabilities,
    normalize_context_text,
    normalize_intent_label,
    probabilities_for_label_order,
)
from ci_hgnn_rgd.nlp.intent_data import (
    audit_intent_leakage,
    chronological_split_if_supported,
    quick_training_subset,
)
from ci_hgnn_rgd.nlp.intent_metrics import (
    bootstrap_macro_f1,
    evaluate_predictions,
    paired_bootstrap_difference,
)


def record(
    split: str,
    label: str,
    identifier: str,
    text: str,
    *,
    source_record_id: str | None = None,
    citing_paper_id: str | None = None,
    cited_paper_id: str | None = None,
    temporal_eligibility: dict | None = None,
) -> IntentDatasetRecord:
    return IntentDatasetRecord(
        citation_context_id=f"synthetic:{identifier}",
        source_dataset="synthetic-stage04-fixture",
        source_record_id=source_record_id or identifier,
        dataset_version="fixture-v1",
        source_split=split,
        original_text=text,
        model_text=normalize_context_text(text),
        original_label=label,
        normalized_label=normalize_intent_label(label),
        label_id=LABEL_TO_ID[normalize_intent_label(label)],
        citing_paper_id=citing_paper_id,
        cited_paper_id=cited_paper_id,
        identity_resolution_status="UNRESOLVED" if citing_paper_id is None else "VERIFIED",
        provenance=({"synthetic": True},),
        temporal_eligibility=temporal_eligibility or {},
    )


def test_fixed_label_taxonomy_rejects_unsupported_or_missing_labels():
    assert LABEL_TO_ID == {"background": 0, "method": 1, "result": 2}
    assert ID_TO_LABEL == {0: "background", 1: "method", 2: "result"}
    assert normalize_intent_label(" Method ") == "method"
    with pytest.raises(IntentDataError, match="unsupported"):
        normalize_intent_label("future_work")
    with pytest.raises(IntentDataError, match="must be text"):
        normalize_intent_label(None)
    with pytest.raises(ValueError, match="cannot be enabled together"):
        Stage04Config.model_validate(
            {"training": {"class_weighting": "inverse", "weighted_sampling": True}}
        )


def test_text_preservation_and_empty_context_rejection():
    original = "  α = β\tusing Ångström [12].  "
    assert normalize_context_text(original) == "α = β using Ångström [12]."
    with pytest.raises(ValueError, match="original text"):
        record("train", "background", "empty", "   ")


def test_evidence_contract_preserves_missing_graph_endpoints_and_empty_input_state():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    skipped = CitationIntentEvidence(
        citation_context_id="synthetic:empty",
        source_dataset="synthetic",
        calibration_status="not_applicable",
        model_id="synthetic-model",
        model_version="synthetic-run",
        prediction_timestamp=now,
        prediction_status="skipped_empty_text",
    )
    assert skipped.citing_paper_id is None and skipped.predicted_intent is None
    complete = CitationIntentEvidence(
        citation_context_id="synthetic:prediction",
        source_dataset="synthetic",
        predicted_intent="method",
        prob_background=0.1,
        prob_method=0.8,
        prob_result=0.1,
        confidence=0.8,
        predictive_entropy=entropy_from_probabilities((0.1, 0.8, 0.1)),
        calibration_status="uncalibrated",
        model_id="synthetic-model",
        model_version="synthetic-run",
        prediction_timestamp=now,
    )
    assert complete.cited_paper_id is None
    assert probabilities_for_label_order((0.1, 0.8, 0.1)) == (0.1, 0.8, 0.1)
    with pytest.raises(IntentDataError, match="sum"):
        probabilities_for_label_order((0.1, 0.8, 0.2))


def test_metrics_calibration_and_bootstrap_use_complete_label_order():
    truth = [0, 1, 2, 2]
    prediction = [0, 0, 2, 1]
    probabilities = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.1, 0.7, 0.2]]
    metrics = evaluate_predictions(truth, prediction, probabilities)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]
    assert metrics["per_class"]["method"]["precision"] == 0.0
    assert metrics["zero_division"] == 0
    interval = bootstrap_macro_f1(truth, prediction, seed=3, n_resamples=20)
    assert interval["resampling_unit"] == "record"
    paired = paired_bootstrap_difference(truth, prediction, prediction, seed=3, n_resamples=20)
    assert paired["point_estimate"] == 0
    with pytest.raises(IntentDataError, match="development"):
        fit_temperature_scaling([[1.0, 0.0, 0.0]], [0], split="test")
    _, calibration = fit_temperature_scaling(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], [0, 1, 2], split="dev"
    )
    assert calibration["test_split_used"] is False


def test_leakage_audit_reports_overlap_without_reassigning_splits(tmp_path):
    rows = (
        record(
            "train",
            "background",
            "one",
            "The synthetic method uses a test signal in fixture form.",
            source_record_id="shared-source-record",
            citing_paper_id="verified:p1",
        ),
        record(
            "dev",
            "method",
            "two",
            "The synthetic method uses a test signal in fixture form.",
            source_record_id="shared-source-record",
            citing_paper_id="verified:p1",
        ),
        record(
            "test",
            "result",
            "three",
            "The synthetic method uses a test signal in fixture forms.",
        ),
    )
    audit = audit_intent_leakage(rows, Stage04Config(), sqlite_path=tmp_path / "audit.sqlite")
    assert audit["official_splits_modified"] is False
    assert audit["shared_source_record_identifier_overlap_groups"] == 1
    assert audit["exact_text_overlap_groups"] == 1
    assert audit["resolved_paper_identity_overlap_groups"] == 1
    assert audit["near_duplicate_cross_split_pairs"] >= 1
    assert audit["near_audit"]["exhaustive"] is False
    assert len(audit["affected_records"]["source_record_id"]) == 2
    assert (
        len(audit["affected_records"]["near_duplicate_pairs"])
        == audit["near_duplicate_cross_split_pairs"]
    )


def test_quick_subset_and_chronological_validation_are_deterministic():
    rows = tuple(
        record("train", label, f"{label}-{index}", f"{label} synthetic {index}")
        for label in ("background", "method", "result")
        for index in range(3)
    )
    selected = quick_training_subset(rows, limit=5)
    assert [item.normalized_label for item in selected] == [
        "background",
        "method",
        "result",
        "background",
        "method",
    ]
    dated = (
        record(
            "train",
            "background",
            "date-one",
            "one",
            temporal_eligibility={
                "eligible": True,
                "evidence": {"publication_date": "2020-01-01", "available_at": "2020-01-02"},
            },
        ),
        record(
            "dev",
            "method",
            "date-two",
            "two",
            temporal_eligibility={
                "eligible": True,
                "evidence": {"publication_date": "2021-01-01", "available_at": "2021-01-02"},
            },
        ),
        record(
            "test",
            "result",
            "date-three",
            "three",
            temporal_eligibility={
                "eligible": True,
                "evidence": {"publication_date": "2022-01-01", "available_at": "2022-01-02"},
            },
        ),
    )
    chronological = chronological_split_if_supported(
        dated, train_end="2020-06-01", dev_end="2021-06-01", test_end="2022-06-01"
    )
    assert {name: len(values) for name, values in chronological["records"].items()} == {
        "train": 1,
        "dev": 1,
        "test": 1,
    }
    with pytest.raises(IntentDataError, match="not supported"):
        chronological_split_if_supported(
            rows[:3], train_end="2020-06-01", dev_end="2021-06-01", test_end="2022-06-01"
        )


@pytest.mark.parametrize(
    "future_field", ["available_at", "feature_available_at", "enrichment_date"]
)
def test_chronological_validation_rechecks_each_cutoff(future_field):
    rows = tuple(
        record(
            split,
            label,
            split,
            "synthetic context",
            temporal_eligibility={
                "eligible": True,
                "evidence": {
                    "publication_date": f"{year}-01-01",
                    "available_at": f"{year}-01-02",
                    **({future_field: "2022-03-01"} if split == "train" else {}),
                },
            },
        )
        for split, label, year in (
            ("train", "background", 2020),
            ("dev", "method", 2021),
            ("test", "result", 2022),
        )
    )
    with pytest.raises(IntentDataError, match="not supported"):
        chronological_split_if_supported(
            rows, train_end="2020-06-01", dev_end="2021-06-01", test_end="2022-06-01"
        )


def test_bootstrap_unknown_groups_remain_independent_units():
    from ci_hgnn_rgd.nlp.intent_metrics import _bootstrap_units, calibration_metrics

    assert len(_bootstrap_units(2, [None, "__record__:0"])) == 2
    result = bootstrap_macro_f1([0, 1, 2], [0, 1, 2], groups=[None] * 3, n_resamples=5)
    assert result["resampling_unit"] == "record"
    with pytest.raises(IntentDataError, match="at least two bins"):
        calibration_metrics([0], [[1.0, 0.0, 0.0]], bins=0)
