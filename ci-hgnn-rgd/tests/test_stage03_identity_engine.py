"""Small synthetic identity cases, including bounded scaling and cluster conflicts."""

import copy

import pytest

from ci_hgnn_rgd.data.deduplication import SourceDeduplicator
from ci_hgnn_rgd.data.entity_resolution import IdentityEngine
from ci_hgnn_rgd.data.match_scoring import score_match
from ci_hgnn_rgd.data.normalization import normalize_identifier


def paper(key, *, dataset="synthetic-a", title="Scholarly α + β = γ", year=2020, ids=()):
    return {
        "record_key": key,
        "source_dataset": dataset,
        "dataset_version": "synthetic-v1",
        "source_record_id": key,
        "title": title,
        "abstract": "Preserved formula x² + y² = z².",
        "publication_year": year,
        "authors": ["A. Researcher", "B. Scientist"],
        "identifiers": [
            normalize_identifier(namespace, value, verified=verified)
            for namespace, value, verified in ids
        ],
        "raw_values": {"id": key, "title": title, "ids": [list(item) for item in ids]},
        "provenance": [{"raw_path": "synthetic-fixture.jsonl", "line_number": 1}],
    }


def resolve(tmp_path, records, **options):
    engine = IdentityEngine(tmp_path / "identity.sqlite", **options)
    for record in records:
        engine.add(record)
    summary = engine.finalize()
    return engine, summary


@pytest.mark.parametrize(
    ("namespace", "left_id", "right_id"),
    [
        ("doi", "https://doi.org/10.1000/AbC", "doi:10.1000/abc"),
        ("arxiv", "2301.01234v1", "2301.01234v3"),
        ("openalex", "W123", "https://openalex.org/W123"),
        ("semantic_scholar", "A" * 40, "a" * 40),
        ("mag", "12345", "12345"),
    ],
)
def test_only_verified_identity_evidence_merges(tmp_path, namespace, left_id, right_id):
    records = [
        paper("a", ids=[(namespace, left_id, True)]),
        paper("b", dataset="synthetic-b", ids=[(namespace, right_id, True)]),
    ]
    with resolve(tmp_path, records)[0] as engine:
        assert engine.lookup("a") == engine.lookup("b")
        merged = list(engine.iter_papers())
        assert len(merged) == 1
        assert len(merged[0]["source_record_ids"]) == 2
        assert len(merged[0]["provenance"]) == 2
        assert merged[0]["raw_values"][0]["values"] == records[0]["raw_values"]
        decision = next(engine.iter_decisions())
        assert decision["decision"] == "EXACT_VERIFIED"
        assert decision["automatic_merge"]
        assert decision["score_interpretation"].endswith("not_probability")
        if namespace == "arxiv":
            assert {item["version"] for item in merged[0]["identifiers"]} == {"v1", "v3"}


def test_title_and_author_similarity_is_review_only(tmp_path):
    with resolve(tmp_path, [paper("a"), paper("b", dataset="synthetic-b")])[0] as engine:
        assert engine.lookup("a") != engine.lookup("b")
        assert next(engine.iter_decisions())["decision"] == "PROBABLE_REVIEW"


def test_unverified_identifier_equality_never_auto_merges(tmp_path):
    records = [paper(key, ids=[("doi", "10.1000/shared", False)]) for key in ("a", "b")]
    with resolve(tmp_path, records)[0] as engine:
        assert engine.lookup("a") != engine.lookup("b")
        assert not next(engine.iter_decisions())["automatic_merge"]


def test_shared_doi_conflicting_external_id_is_not_merged(tmp_path):
    records = [
        paper("a", ids=[("doi", "10.1000/shared", True), ("openalex", "W1", True)]),
        paper("b", ids=[("doi", "10.1000/shared", True), ("openalex", "W2", True)]),
    ]
    with resolve(tmp_path, records)[0] as engine:
        assert engine.lookup("a") != engine.lookup("b")
        decision = next(engine.iter_decisions())
        assert decision["decision"] == "CONFLICTED"
        assert decision["features"]["identifier_conflicts"][0]["namespace"] == "openalex"


def test_authoritative_bridge_cannot_hide_transitive_conflict(tmp_path):
    records = [
        paper("a", ids=[("doi", "10.1000/x", True), ("openalex", "W1", True)]),
        paper("b", ids=[("doi", "10.1000/x", True), ("arxiv", "2301.01234", True)]),
        paper("c", ids=[("arxiv", "2301.01234", True), ("openalex", "W2", True)]),
    ]
    with resolve(tmp_path, records)[0] as engine:
        assert engine.lookup("a") == engine.lookup("b")
        assert engine.lookup("b") != engine.lookup("c")
        decisions = list(engine.iter_decisions())
        assert any(value["rule"] == "cluster_identity_conflict" for value in decisions)
        assert len(list(engine.iter_papers())) == 2


def test_cluster_metadata_conflict_prevents_year_bridge(tmp_path):
    records = [
        paper(key, year=year, ids=[("doi", "10.1000/x", True)])
        for key, year in (("a", 2020), ("b", 2021), ("c", 2022))
    ]
    with resolve(tmp_path, records)[0] as engine:
        assert engine.lookup("a") == engine.lookup("b")
        assert engine.lookup("a") != engine.lookup("c")


def test_multiple_title_candidates_remain_ambiguous(tmp_path):
    with resolve(tmp_path, [paper(key) for key in ("a", "b", "c")])[0] as engine:
        assert len(list(engine.iter_papers())) == 3
        assert {value["decision"] for value in engine.iter_decisions()} == {"AMBIGUOUS"}
        assert engine.finalize()["record_classifications"]["AMBIGUOUS"] == 3


def test_missing_metadata_and_unmatched_records_are_retained(tmp_path):
    records = [paper("a", title=None, year=None), paper("b", title="Unique title")]
    with resolve(tmp_path, records)[0] as engine:
        assert len(list(engine.iter_papers())) == 2
        assert engine.lookup("missing") is None
        counts = engine.finalize()["record_classifications"]
        assert counts["UNRESOLVABLE"] == 1 and counts["UNMATCHED"] == 1


def test_source_field_alternatives_and_multiple_versions_are_retained(tmp_path):
    records = [
        paper("a", title="Graph learning with α", ids=[("arxiv", "2301.01234v1", True)]),
        paper("b", title="Graph learning with β", ids=[("arxiv", "2301.01234v2", True)]),
    ]
    with resolve(tmp_path, records)[0] as engine:
        canonical = next(engine.iter_papers())
        assert len(canonical["field_alternatives"]["title"]) == 2
        assert "alternative_title_values" in canonical["conflict_flags"]
        assert len(canonical["raw_values"]) == 2


def test_material_title_contradiction_is_a_conflict():
    first = paper("a", title="Quantum superconductivity under high magnetic fields")
    second = paper("b", title="Bird migration along northern estuaries")
    first["identifiers"] = second["identifiers"] = [
        normalize_identifier("doi", "10.1000/x", verified=True)
    ]
    assert score_match(first, second)["decision"] == "CONFLICTED"


def test_candidate_generation_determinism_and_id_stability(tmp_path):
    records = [paper(key, ids=[("doi", "10.1000/x", True)]) for key in ("a", "b", "c")]
    observed = []
    for index, ordered in enumerate((records, records[::-1])):
        path = tmp_path / str(index)
        with resolve(path, ordered)[0] as engine:
            observed.append((list(engine.iter_candidates()), list(engine.iter_mappings())))
            assert engine.finalize() == engine.finalize()
    assert observed[0] == observed[1]


def test_reopen_preserves_results_and_changed_policy_recomputes(tmp_path):
    records = [paper(key, ids=[("doi", "10.1000/x", True)]) for key in ("a", "b")]
    with resolve(tmp_path, records)[0] as engine:
        original = list(engine.iter_mappings())
    with IdentityEngine(tmp_path / "identity.sqlite") as engine:
        assert list(engine.iter_mappings()) == original
        assert engine.add(records[0])["already_indexed"]
    with IdentityEngine(tmp_path / "identity.sqlite", allowed_auto_merge_rules=()) as engine:
        assert engine.finalize()["canonical_papers"] == 2


def test_source_keys_cannot_overwrite_values(tmp_path):
    with IdentityEngine(tmp_path / "identity.sqlite") as engine:
        engine.add(paper("a"))
        with pytest.raises(ValueError, match="cannot overwrite"):
            engine.add(paper("a", title="Changed source value"))


def test_reused_source_ids_are_not_identity_proof(tmp_path):
    records = [paper("a", title="First source paper"), paper("b", title="Second source paper")]
    for record in records:
        record["source_record_id"] = "nonunique-source-annotation-id"
    with resolve(tmp_path, records)[0] as engine:
        mappings = list(engine.iter_mappings())
        assert {item["source_record_id"] for item in mappings} == {"nonunique-source-annotation-id"}
        assert len({item["canonical_paper_id"] for item in mappings}) == 2
        assert engine.finalize()["input_records"] == 2


def test_canonical_temporal_assessment_requires_every_contributor(tmp_path):
    records = [paper(key, ids=[("doi", "10.1000/x", True)]) for key in ("a", "b")]
    records[0]["temporal_eligibility"] = {
        "eligible": True,
        "unknown_availability": False,
        "reasons": [],
    }
    records[1]["temporal_eligibility"] = {
        "eligible": False,
        "unknown_availability": False,
        "reasons": ["future_metadata"],
    }
    with resolve(tmp_path, records)[0] as engine:
        canonical = next(engine.iter_papers())
        assert canonical["identity_status"] == "EXACT_VERIFIED"
        assert canonical["record_identity_statuses"] == ["EXACT_VERIFIED"]
        assert not canonical["temporal_eligibility"]["eligible"]
        assert "future_metadata" in canonical["temporal_eligibility"]["reasons"]
        assert len(canonical["temporal_eligibility"]["contributors"]) == 2


def test_missing_source_temporal_assessment_cannot_default_to_eligible(tmp_path):
    with resolve(tmp_path, [paper("a")])[0] as engine:
        assessment = next(engine.iter_papers())["temporal_eligibility"]
        assert not assessment["eligible"] and assessment["unknown_availability"]
        assert "missing_source_temporal_assessment" in assessment["reasons"]


def test_source_duplicates_are_observed_without_deleting_records(tmp_path):
    first = paper("a", title="Repeated exact raw source")
    second = copy.deepcopy(first)
    second["record_key"] = "b"
    second["source_record_id"] = "a"
    with resolve(tmp_path, [first, second])[0] as engine:
        assert engine.finalize()["within_source_duplicate_records"] == 1
        assert engine.finalize()["input_records"] == 2
        assert len(list(engine.iter_papers())) == 2  # Text equality is not identity proof.


def test_context_multiplicity_and_author_name_collisions_are_not_deduped(tmp_path):
    dedup = SourceDeduplicator(tmp_path / "duplicates.sqlite")
    try:
        first = {
            "record_key": "context-a",
            "source_dataset": "synthetic",
            "dataset_version": "v1",
            "citing_paper_id": "p1",
            "cited_paper_id": "p2",
            "original_context": "First context",
        }
        second = {**first, "record_key": "context-b", "original_context": "Second context"}
        assert not dedup.observe(first, kind="citation_context")["duplicate"]
        assert not dedup.observe(second, kind="citation_context")["duplicate"]
        third = {**first, "record_key": "context-c"}
        assert dedup.observe(third, kind="citation_context")["duplicate"]
        for source_id in ("author-1", "author-2"):
            author = {
                "record_key": source_id,
                "source_record_id": source_id,
                "source_dataset": "synthetic",
                "dataset_version": "v1",
                "name": "J. Smith",
            }
            assert not dedup.observe(author, kind="author")["duplicate"]
    finally:
        dedup.close()


def test_bounded_candidates_do_not_claim_coverage_or_merge_omitted_evidence(tmp_path):
    records = [paper(f"p{number:03}", ids=[("doi", "10.1000/x", True)]) for number in range(20)]
    with resolve(tmp_path, records, max_candidates=3)[0] as engine:
        candidates = list(engine.iter_candidates())
        assert all(value["candidate_count"] <= 3 for value in candidates)
        assert all(value["truncated"] for value in candidates)
        assert all(value["candidate_population_exact"] is None for value in candidates)
        assert engine.finalize()["canonical_papers"] == 20
        assert engine.finalize()["records_with_truncated_candidates"] == 20


def test_synthetic_scaling_uses_bounded_indexed_candidates(tmp_path):
    # This checks expansion on 500 fixtures; it makes no million-record speed claim.
    count, limit = 500, 4
    records = [
        paper(f"p{number:05}", title=f"Title block {number // 10}") for number in range(count)
    ]
    with resolve(tmp_path, records, max_candidates=limit, batch_size=31)[0] as engine:
        summary = engine.finalize()
        assert summary["candidate_memberships_returned"] <= count * limit
        assert summary["candidates_examined"] <= count * limit
        assert summary["candidates_examined"] < count * (count - 1) // 20
        assert summary["records_with_truncated_candidates"] == count
        assert summary["input_records"] == summary["canonical_papers"] == count


def test_cluster_resource_limit_leaves_records_separate_for_review(tmp_path):
    records = [paper(key, ids=[("doi", "10.1000/x", True)]) for key in ("a", "b", "c")]
    with resolve(tmp_path, records, max_cluster_size=2)[0] as engine:
        assert engine.finalize()["canonical_papers"] == 2
        assert any(value["rule"] == "cluster_resource_budget" for value in engine.iter_decisions())


def test_oversized_records_and_identifier_lists_are_rejected_recoverably(tmp_path):
    with IdentityEngine(tmp_path / "identity.sqlite", max_identifiers_per_record=1) as engine:
        record = paper("a", ids=[("doi", "10.1000/x", True), ("arxiv", "2301.01234", True)])
        with pytest.raises(ValueError, match="identifier limit"):
            engine.add(record)
        huge = paper("b")
        huge["abstract"] = "x" * (1024 * 1024 + 1)
        with pytest.raises(ValueError, match="payload budget"):
            engine.add(huge)
        assert engine.finalize()["input_records"] == 0
