"""Deterministic, uncalibrated match features with conservative merge decisions."""

from __future__ import annotations

from difflib import SequenceMatcher

from .normalization import AUTHORITATIVE_NAMESPACES, comparison_title, identifier_token

RESOLVER_VERSION = "stage03-conservative-1"
MATCH_CLASSIFICATIONS = (
    "EXACT_VERIFIED",
    "PROBABLE_REVIEW",
    "AMBIGUOUS",
    "CONFLICTED",
    "UNMATCHED",
    "UNRESOLVABLE",
)


def verified_tokens(record: dict) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for identifier in record.get("identifiers", []):
        token = identifier_token(identifier)
        if token and identifier.get("verified"):
            result.setdefault(token[0], set()).add(token[1])
    return result


def authoritative_conflicts(left: dict, right: dict) -> list[dict]:
    """Disjoint verified values of the same authoritative namespace contradict."""
    first, second = verified_tokens(left), verified_tokens(right)
    conflicts = []
    for namespace in sorted(first.keys() & second.keys()):
        if namespace in AUTHORITATIVE_NAMESPACES and first[namespace] != second[namespace]:
            conflicts.append(
                {
                    "namespace": namespace,
                    "left_values": sorted(first[namespace]),
                    "right_values": sorted(second[namespace]),
                }
            )
    for side, tokens in (("left", first), ("right", second)):
        for namespace, values in sorted(tokens.items()):
            if namespace in AUTHORITATIVE_NAMESPACES and len(values) > 1:
                conflicts.append(
                    {"namespace": namespace, "side": side, "multiple_values": sorted(values)}
                )
    return conflicts


def match_features(left: dict, right: dict, *, year_tolerance: int = 1) -> dict:
    first, second = verified_tokens(left), verified_tokens(right)
    shared = [
        {"namespace": namespace, "value": value}
        for namespace in sorted(first.keys() & second.keys())
        for value in sorted(first[namespace] & second[namespace])
    ]
    left_title, right_title = (
        comparison_title(left.get("title")),
        comparison_title(right.get("title")),
    )
    # Bound ranking work on pathological text without modifying the stored title.
    title_similarity = (
        SequenceMatcher(None, left_title[:2048], right_title[:2048]).ratio()
        if left_title and right_title
        else None
    )

    def names(record):
        return {
            comparison_title(author if isinstance(author, str) else author.get("name"))
            for author in record.get("authors", [])[:128]
            if isinstance(author, (str, dict))
        } - {None}

    first_names, second_names = names(left), names(right)
    author_overlap = (
        len(first_names & second_names) / len(first_names | second_names)
        if first_names and second_names
        else None
    )
    left_year, right_year = left.get("publication_year"), right.get("publication_year")
    year_difference = (
        abs(left_year - right_year)
        if isinstance(left_year, int) and isinstance(right_year, int)
        else None
    )
    same_venue = (
        comparison_title(left.get("venue")) == comparison_title(right.get("venue"))
        if left.get("venue") and right.get("venue")
        else None
    )
    return {
        "shared_verified_identifiers": shared,
        "doi_equality": any(item["namespace"] == "doi" for item in shared),
        "arxiv_equality": any(item["namespace"] == "arxiv" for item in shared),
        "verified_external_mapping": any(
            item["namespace"] in AUTHORITATIVE_NAMESPACES - {"doi", "arxiv"} for item in shared
        ),
        "title_similarity": title_similarity,
        "title_comparison_truncated": bool(
            (left_title and len(left_title) > 2048) or (right_title and len(right_title) > 2048)
        ),
        "author_overlap": author_overlap,
        "author_comparison_limit": 128,
        "publication_year_difference": year_difference,
        "publication_year_compatible": (
            year_difference <= year_tolerance if year_difference is not None else None
        ),
        "venue_equal": same_venue,
        "identifier_conflicts": authoritative_conflicts(left, right),
        "source_reliability": {
            "left": left.get("source_reliability"),
            "right": right.get("source_reliability"),
            "status": "source_asserted_or_unknown_not_calibrated",
        },
    }


def score_match(
    left: dict,
    right: dict,
    *,
    year_tolerance: int = 1,
    title_review_threshold: float = 0.9,
    material_title_conflict_threshold: float = 0.35,
) -> dict:
    features = match_features(left, right, year_tolerance=year_tolerance)
    title_similarity = features["title_similarity"]
    material_conflicts = []
    if features["publication_year_compatible"] is False:
        material_conflicts.append("contradictory_publication_years")
    if (
        title_similarity is not None
        and title_similarity < material_title_conflict_threshold
        and min(len(left.get("title") or ""), len(right.get("title") or "")) >= 12
    ):
        material_conflicts.append("contradictory_titles")
    features["material_metadata_conflicts"] = material_conflicts
    if features["identifier_conflicts"] or material_conflicts:
        decision, rule, score = "CONFLICTED", "conflicting_identity_evidence", 0.0
    elif features["shared_verified_identifiers"]:
        decision, rule, score = "EXACT_VERIFIED", "shared_source_verified_identifier", 100.0
    elif title_similarity is not None and title_similarity >= title_review_threshold:
        score = round(
            60 * title_similarity
            + 20 * (features["author_overlap"] or 0)
            + 10 * bool(features["publication_year_compatible"])
            + 5 * bool(features["venue_equal"]),
            6,
        )
        decision, rule = "PROBABLE_REVIEW", "metadata_similarity_requires_review"
    else:
        decision, rule, score = "UNMATCHED", "insufficient_identity_evidence", 0.0
    return {
        "left_record_key": left["record_key"],
        "right_record_key": right["record_key"],
        "decision": decision,
        "rule": rule,
        "features": features,
        "heuristic_score": score,
        "score_interpretation": "uncalibrated_ranking_score_not_probability",
        "resolver_version": RESOLVER_VERSION,
        "review_status": "not_required" if decision == "EXACT_VERIFIED" else "pending",
        "automatic_merge": False,
        "evidence": features["shared_verified_identifiers"],
    }
