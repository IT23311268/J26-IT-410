"""Reassess every contributing record and learned feature at the requested cutoff."""

from ..data.temporal_integrity import assess_temporal
from ..nlp.topics.data import temporal_evidence


def assessment(row, bundle, *, kind="paper", produced_at=None, cutoff=None):
    evidence = list(row.get("source_temporal_evidence") or temporal_evidence(row))
    if row.get("citation_context_id") and row.get("provenance"):
        evidence = [
            {
                field: source.get(field)
                for field in (
                    "available_at",
                    "publication_date",
                    "publication_year",
                    "citation_observed_at",
                    "feature_available_at",
                    "version_date",
                    "update_date",
                )
            }
            for source in row["provenance"]
        ]
    if produced_at is not None:
        evidence.append({"feature_available_at": produced_at})
    if row.get("citation_context_id") and bundle.intent_manifest:
        evidence.append({"feature_available_at": bundle.intent_manifest.get("_model_created_at")})
    if not evidence:
        evidence = [{}]
    results = [
        assess_temporal(item, cutoff or bundle.cutoff, bundle.policy, kind=kind)
        for item in evidence
    ]
    return {
        "eligible": all(item["eligible"] for item in results),
        "unknown_availability": any(item["unknown_availability"] for item in results),
        "proxy": any(item["proxy"] for item in results),
        "reasons": sorted({r for item in results for r in item["reasons"]}),
        "policy": bundle.policy,
        "cutoff": cutoff or bundle.cutoff,
        "assessments": results,
    }


def split_visibility(row, bundle, config, *, kind="paper", produced_at=None):
    return {
        name: assessment(row, bundle, kind=kind, produced_at=produced_at, cutoff=cutoff)["eligible"]
        for name, cutoff in config.temporal.split_cutoffs.items()
    }
