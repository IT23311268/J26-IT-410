"""Synthetic temporal evidence; no historical availability is empirically claimed."""

from datetime import UTC, datetime

import pytest

from ci_hgnn_rgd.data.temporal_integrity import (
    TemporalEvidence,
    assess_temporal,
    historical_aggregate,
    parse_cutoff,
)

CUTOFF = "2020-12-31"


def test_publication_and_ingestion_do_not_establish_historical_availability():
    result = assess_temporal(
        {"publication_date": "2019-01-01", "ingested_at": "2019-01-01"}, CUTOFF
    )
    assert not result["eligible"]
    assert result["unknown_availability"]
    assert "unknown_historical_availability" in result["reasons"]


def test_strict_verified_content_availability_is_eligible():
    result = assess_temporal(
        TemporalEvidence(publication_date="2019-01-01", available_at="2020-01-01"), CUTOFF
    )
    assert result["eligible"] and result["strict_historical_claim"]


@pytest.mark.parametrize(
    "field",
    [
        "publication_date",
        "version_date",
        "update_date",
        "available_at",
        "feature_available_at",
        "citation_observed_at",
        "enrichment_date",
        "training_data_cutoff",
    ],
)
def test_future_information_never_enters_strict_snapshot(field):
    result = assess_temporal({"available_at": "2019-01-01", field: "2022-01-01"}, CUTOFF)
    assert not result["eligible"]
    assert f"future_{field}" in result["reasons"]


def test_publication_proxy_is_explicit_and_cannot_override_known_future_version():
    result = assess_temporal({"publication_date": "2019-01-01"}, CUTOFF, "PUBLICATION_DATE_PROXY")
    assert result["eligible"] and result["proxy"]
    assert not result["strict_historical_claim"]
    future = assess_temporal(
        {"publication_date": "2019-01-01", "version_date": "2022-01-01"},
        CUTOFF,
        "PUBLICATION_DATE_PROXY",
    )
    assert not future["eligible"]


def test_year_precision_cannot_establish_availability_inside_that_year():
    result = assess_temporal({"publication_year": 2020}, "2020-06-30", "PUBLICATION_DATE_PROXY")
    assert not result["eligible"]
    assert "uncertain_publication_date_precision_at_cutoff" in result["reasons"]
    assert assess_temporal({"publication_year": 2019}, CUTOFF, "PUBLICATION_DATE_PROXY")["eligible"]


def test_unknown_dates_and_missing_cutoff_fail_closed():
    assert not assess_temporal({}, CUTOFF)["eligible"]
    assert not assess_temporal({"available_at": "2019-01-01"}, None)["eligible"]
    result = assess_temporal({}, CUTOFF, "PUBLICATION_DATE_PROXY")
    assert "publication_proxy_unavailable" in result["reasons"]


def test_exploratory_retains_future_records_without_forecasting_claim():
    result = assess_temporal(
        {"publication_date": "2024-01-01"}, CUTOFF, "EXPLORATORY_NO_TEMPORAL_CLAIM"
    )
    assert result["eligible"] and result["status"] == "EXPLORATORY_ONLY"
    assert not result["strict_historical_claim"]
    assert "future_publication_date" in result["reasons"]


def test_future_snapshot_is_not_automatically_the_content_availability_date():
    result = assess_temporal(
        {"available_at": "2019-01-01", "snapshot_date": "2025-01-01", "ingested_at": "2025-01-02"},
        CUTOFF,
    )
    assert result["eligible"]
    assert "snapshot_after_cutoff_requires_independent_availability" in result["warnings"]


def test_naive_timestamps_are_rejected_not_assigned_a_local_timezone():
    with pytest.raises(ValueError, match="timezone"):
        parse_cutoff(datetime(2020, 1, 1))
    result = assess_temporal({"available_at": "2019-01-01T00:00:00"}, CUTOFF)
    assert not result["eligible"]
    assert "invalid_available_at" in result["reasons"]
    assert parse_cutoff("2020-12-31T23:59:59Z").tzinfo == UTC


def test_citation_and_feature_need_their_own_availability_evidence():
    edge = assess_temporal(
        {"publication_date": "2019-01-01", "citation_observed_at": "2020-01-01"},
        CUTOFF,
        kind="citation_edge",
    )
    assert edge["eligible"]
    feature = assess_temporal(
        {"publication_date": "2019-01-01", "feature_available_at": "2021-01-01"},
        CUTOFF,
        kind="feature",
    )
    assert not feature["eligible"]


def test_historical_aggregate_recomputed_from_eligible_evidence_only():
    result = historical_aggregate(
        [
            {"feature_available_at": "2019-01-01", "value": 2},
            {"feature_available_at": "2022-01-01", "value": 100},
            {"value": 1000},
        ],
        CUTOFF,
    )
    assert result == {"eligible_observations": 1, "excluded_observations": 2, "sum": 2.0}


def test_month_precision_uses_entire_calendar_month_including_leap_day():
    evidence = {"publication_date": "2020-02", "date_precision": "month"}
    within = assess_temporal(evidence, "2020-02-28", "PUBLICATION_DATE_PROXY")
    assert not within["eligible"]
    assert "uncertain_publication_date_precision_at_cutoff" in within["reasons"]
    complete = assess_temporal(evidence, "2020-02-29", "PUBLICATION_DATE_PROXY")
    assert complete["eligible"]
    assert not complete["invalid_temporal_fields"]
    invalid = assess_temporal({"publication_date": "2020-13"}, CUTOFF)
    assert "invalid_publication_date" in invalid["reasons"]
