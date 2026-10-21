"""Conservative point-in-time gates; ingestion is never historical availability.

Dates with limited precision are intervals. Eligibility requires their upper bound
to precede the cutoff. ``available_at`` describes the exact record/feature version,
not the work's original publication. Exclusions are returned, never deleted here.
"""

from calendar import monthrange
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class TemporalPolicy(StrEnum):
    STRICT_POINT_IN_TIME = "STRICT_POINT_IN_TIME"
    PUBLICATION_DATE_PROXY = "PUBLICATION_DATE_PROXY"
    EXPLORATORY_NO_TEMPORAL_CLAIM = "EXPLORATORY_NO_TEMPORAL_CLAIM"


class TemporalEvidence(BaseModel):
    """Loss-aware evidence; invalid input dates are surfaced by the assessment."""

    model_config = ConfigDict(extra="allow", frozen=True)

    publication_date: Any = None
    publication_year: Any = None
    date_precision: str | None = None
    version_date: Any = None
    update_date: Any = None
    snapshot_date: Any = None
    ingested_at: Any = None
    available_at: Any = None
    citation_observed_at: Any = None
    feature_available_at: Any = None
    enrichment_date: Any = None
    training_data_cutoff: Any = None


def _interval(value: Any) -> tuple[datetime, datetime, str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a date")
    if isinstance(value, int) or (isinstance(value, str) and len(value) == 4):
        year = int(value)
        return (
            datetime(year, 1, 1, tzinfo=UTC),
            datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
            "year",
        )
    if isinstance(value, str):
        value = value.strip()
        if len(value) == 7:
            first = date.fromisoformat(value + "-01")
            last = first.replace(day=monthrange(first.year, first.month)[1])
            return (
                datetime.combine(first, time.min, tzinfo=UTC),
                datetime.combine(last, time.max, tzinfo=UTC),
                "month",
            )
        if len(value) == 10:
            value = date.fromisoformat(value)
        else:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("datetime must carry an explicit timezone")
        moment = value.astimezone(UTC)
        return moment, moment, "instant"
    if isinstance(value, date):
        return (
            datetime.combine(value, time.min, tzinfo=UTC),
            datetime.combine(value, time.max, tzinfo=UTC),
            "day",
        )
    raise ValueError("unrecognized temporal value")


def parse_cutoff(cutoff: datetime | date | str) -> datetime:
    """A date-only cutoff includes that UTC day; naive timestamps are rejected."""
    interval = _interval(cutoff)
    if interval is None:
        raise ValueError("cutoff is required")
    return interval[1]


def _json_evidence(evidence: dict) -> dict:
    def convert(value):
        if isinstance(value, datetime | date):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [convert(item) for item in value]
        return value

    return convert(evidence)


def assess_temporal(
    evidence: dict | TemporalEvidence,
    cutoff: datetime | date | str | None,
    policy: str | TemporalPolicy = TemporalPolicy.STRICT_POINT_IN_TIME,
    *,
    kind: str = "paper",
) -> dict:
    """Assess one record, feature, or relation without guessing missing evidence.

    A verified availability timestamp is sufficient for the exact supplied content.
    A later snapshot/ingestion timestamp alone neither proves nor disproves that
    earlier availability. Known future version/update/enrichment/edge evidence
    always excludes content from historical policies. Exploratory eligibility
    explicitly carries no forecasting or leakage-free claim.
    """
    policy = TemporalPolicy(policy)
    raw = evidence.model_dump() if isinstance(evidence, TemporalEvidence) else dict(evidence)
    raw = dict(raw)
    # Support Stage 01/02 naming without replacing original evidence.
    raw.setdefault("publication_date", raw.get("published_on"))
    raw.setdefault("version_date", raw.get("version_created_at"))
    raw.setdefault("snapshot_date", raw.get("dataset_snapshot_date"))
    raw.setdefault("ingested_at", raw.get("acquired_at"))
    raw.setdefault("update_date", raw.get("updated_at"))
    end = parse_cutoff(cutoff) if cutoff is not None else None
    reasons: list[str] = []
    warnings: list[str] = []
    invalid: list[str] = []
    intervals = {}
    fields = (
        "publication_date",
        "publication_year",
        "version_date",
        "update_date",
        "snapshot_date",
        "ingested_at",
        "available_at",
        "citation_observed_at",
        "feature_available_at",
        "enrichment_date",
        "training_data_cutoff",
    )
    for field in fields:
        try:
            intervals[field] = _interval(raw.get(field))
        except (ValueError, TypeError, OverflowError):
            intervals[field] = None
            invalid.append(field)
            reasons.append(f"invalid_{field}")
    publication = intervals["publication_date"] or intervals["publication_year"]
    if raw.get("date_precision") == "year" and publication is not None:
        publication = _interval(publication[0].year)
    elif raw.get("date_precision") == "month" and publication is not None:
        publication = _interval(publication[0].strftime("%Y-%m"))
    if publication is None:
        warnings.append("unknown_publication_date")
    if end is None:
        reasons.append("missing_temporal_cutoff")
    else:
        # Publication dates never establish the availability of particular content.
        applicable = {
            name: value
            for name, value in intervals.items()
            if name not in {"snapshot_date", "ingested_at", "publication_date", "publication_year"}
        }
        applicable["publication_date"] = publication
        for field, interval in applicable.items():
            if interval is None:
                continue
            if interval[0] > end:
                reasons.append(f"future_{field}")
            elif interval[1] > end:
                reasons.append(f"uncertain_{field}_precision_at_cutoff")
        if intervals["snapshot_date"] and intervals["snapshot_date"][1] > end:
            warnings.append("snapshot_after_cutoff_requires_independent_availability")
        if intervals["ingested_at"] and intervals["ingested_at"][1] > end:
            warnings.append("ingested_after_cutoff_not_used_as_historical_evidence")
    specific_field = (
        "citation_observed_at"
        if kind in {"citation", "citation_edge", "edge"}
        else "feature_available_at"
        if kind in {"feature", "metadata", "aggregate"}
        else "available_at"
    )
    availability = intervals[specific_field] or intervals["available_at"]
    unknown_availability = availability is None
    if unknown_availability:
        warnings.append("unknown_historical_availability")
    proxy = False
    if policy == TemporalPolicy.STRICT_POINT_IN_TIME:
        if unknown_availability:
            reasons.append("unknown_historical_availability")
        eligible = not reasons
        status = "ELIGIBLE" if eligible else "EXCLUDED"
    elif policy == TemporalPolicy.PUBLICATION_DATE_PROXY:
        proxy = unknown_availability
        if proxy and publication is None:
            reasons.append("publication_proxy_unavailable")
        if proxy:
            warnings.append("publication_date_proxy_not_verified_historical_availability")
        eligible = not reasons
        status = "PROXY_ELIGIBLE" if eligible and proxy else "ELIGIBLE" if eligible else "EXCLUDED"
    else:
        eligible = True
        status = "EXPLORATORY_ONLY"
        warnings.append("no_leakage_free_forecasting_claim_permitted")
    return {
        "eligible": eligible,
        "status": status,
        "policy": policy.value,
        "kind": kind,
        "cutoff": end.isoformat() if end else None,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "unknown_availability": unknown_availability,
        "proxy": proxy,
        "strict_historical_claim": bool(eligible and policy == TemporalPolicy.STRICT_POINT_IN_TIME),
        "invalid_temporal_fields": invalid,
        "evidence": _json_evidence(raw),
    }


def historical_aggregate(
    observations,
    cutoff: datetime | date | str,
    *,
    value_field: str = "value",
    policy: str | TemporalPolicy = TemporalPolicy.STRICT_POINT_IN_TIME,
) -> dict:
    """Recompute an aggregate from individually eligible evidence, never a current total."""
    count = excluded = 0
    total = 0.0
    for observation in observations:
        result = assess_temporal(observation, cutoff, policy, kind="feature")
        if result["eligible"]:
            value = observation.get(value_field, 1)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("aggregate values must be numeric")
            count += 1
            total += value
        else:
            excluded += 1
    return {"eligible_observations": count, "excluded_observations": excluded, "sum": total}
