"""Synthetic contract tests only; the numerical values below are not research results."""

from datetime import timedelta

import pytest
from conftest import CUTOFF, OBSERVED, revised
from pydantic import ValidationError

from ci_hgnn_rgd.graph.schema import NodeType, get_graph_schema
from ci_hgnn_rgd.schemas import (
    CitationIntentPrediction,
    DataCoverage,
    DatasetSnapshot,
    EvidenceReference,
    GraphManifest,
    NodeCount,
    Paper,
    PaperTopicEvidence,
    RelationCount,
    ResearchGapCandidate,
    SciCiteClass,
    Uncertainty,
)


@pytest.fixture
def prediction(evidence):
    return CitationIntentPrediction(
        prediction_id="synthetic:prediction1",
        context_id="synthetic:c1",
        taxonomy_version="synthetic-schema-test",
        label="Background",
        confidence=0.5,
        model_version="synthetic-model-not-trained",
        training_data_cutoff=OBSERVED,
        produced_at=OBSERVED,
        evidence=(evidence,),
    )


@pytest.mark.parametrize("label", ["Background", "Method", "Result"])
def test_supported_scicite_classes(prediction, label):
    assert revised(prediction, label=label).label == label
    assert len(SciCiteClass) == 3


@pytest.mark.parametrize("label", ["Investigates", "Unknown", "background", "Support", " "])
def test_scicite_rejects_other_labels(prediction, label):
    with pytest.raises(ValidationError):
        revised(prediction, label=label)


def test_unknown_intent_is_null_not_a_new_scicite_class(prediction, synthetic_corpus):
    unknown = revised(prediction, label=None, confidence=None)
    assert unknown.label is None
    assert synthetic_corpus.citations[0].intent is None
    with pytest.raises(ValidationError):
        revised(prediction, label=None)


def test_extended_taxonomy_is_explicitly_separate(prediction):
    extended = revised(
        prediction,
        taxonomy="extended",
        extended_taxonomy_id="synthetic-custom",
        extended_classes=("SyntheticRelation",),
        label="SyntheticRelation",
    )
    assert extended.taxonomy == "extended"
    with pytest.raises(ValidationError):
        revised(extended, taxonomy="scicite")
    with pytest.raises(ValidationError):
        revised(extended, extended_taxonomy_id="SciCite-expanded")
    with pytest.raises(ValidationError):
        revised(extended, extended_classes=())


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_confidence_must_be_finite_probability(prediction, value):
    with pytest.raises(ValidationError):
        revised(prediction, confidence=value)


def test_model_cutoff_cannot_follow_inference(prediction):
    with pytest.raises(ValidationError):
        revised(prediction, training_data_cutoff=OBSERVED + timedelta(days=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("payload_sha256", "fake"),
        ("source_url", "file:///not-scholarly"),
        ("retrieved_at", OBSERVED.replace(tzinfo=None)),
        ("locator", ""),
        ("available_at", OBSERVED + timedelta(seconds=1)),
    ],
)
def test_invalid_provenance(evidence, field, value):
    with pytest.raises(ValidationError):
        revised(evidence, **{field: value})


def test_paper_requires_source_evidence(synthetic_corpus):
    with pytest.raises(ValidationError):
        revised(synthetic_corpus.papers[0], evidence=())


def test_unknown_date_is_preserved_and_extra_fields_rejected(evidence):
    values = dict(
        paper_id="synthetic:unknown",
        title="Synthetic unknown date",
        published_on=None,
        evidence=(evidence,),
    )
    assert Paper(**values).published_on is None
    with pytest.raises(ValidationError):
        Paper(**values, invented_ground_truth=True)


def test_contracts_are_frozen_and_json_roundtrip(synthetic_corpus):
    paper = synthetic_corpus.papers[0]
    with pytest.raises(ValidationError):
        paper.title = "mutation"
    assert Paper.model_validate_json(paper.model_dump_json()) == paper


def test_investigation_cannot_be_inferred_from_frequency(synthetic_corpus):
    item = synthetic_corpus.paper_topic_evidence[0]
    with pytest.raises(ValidationError, match="citation frequency"):
        revised(item, relationship="investigates_topic", confidence=0.5, basis="citation_frequency")
    for relationship in ("mentions_topic", "investigates_topic"):
        result = revised(item, relationship=relationship, confidence=0.5, basis="textual_evidence")
        assert result.relationship == relationship


def test_investigation_model_requires_training_cutoff(synthetic_corpus):
    with pytest.raises(ValidationError):
        revised(synthetic_corpus.paper_topic_evidence[0], method="model")


def test_investigation_requires_confidence_when_known(synthetic_corpus):
    with pytest.raises(ValidationError):
        revised(synthetic_corpus.paper_topic_evidence[0], relationship="investigates_topic")


@pytest.fixture
def graph(evidence):
    return GraphManifest(
        graph_id="synthetic:graph",
        built_at=CUTOFF,
        cutoff=CUTOFF,
        node_counts=tuple(
            NodeCount(node_type=node, count=1) for node in get_graph_schema().node_types
        ),
        relation_counts=(RelationCount(source="paper", relation="cites", target="paper", count=1),),
        datasets=(
            DatasetSnapshot(
                dataset_id=evidence.dataset_id,
                version_id=evidence.dataset_version,
                sha256=evidence.payload_sha256,
            ),
        ),
        evidence=(evidence,),
    )


def test_graph_schema_separates_mentions_from_investigations(graph):
    schema = get_graph_schema()
    relations = {relation.relation.value for relation in schema.relations}
    assert {"cites", "authored_by", "mentions_topic", "investigates_topic"} <= relations
    assert schema.citation_intent_optional
    assert NodeType.INSTITUTION not in schema.node_types
    assert NodeType.INSTITUTION in get_graph_schema(True).node_types
    assert graph.schema_version == schema.version


def test_graph_rejects_wrong_endpoints_and_duplicate_types(graph):
    with pytest.raises(ValidationError):
        RelationCount(source="author", relation="cites", target="topic", count=1)
    with pytest.raises(ValidationError):
        revised(graph, node_counts=graph.node_counts + graph.node_counts)
    with pytest.raises(ValidationError):
        revised(graph, relation_counts=graph.relation_counts * 2)
    with pytest.raises(ValidationError):
        revised(graph, datasets=graph.datasets * 2)


def test_graph_rejects_future_features_and_missing_datasets(graph):
    with pytest.raises(ValidationError):
        revised(graph, feature_training_cutoff=CUTOFF + timedelta(seconds=1))
    with pytest.raises(ValidationError):
        revised(graph, datasets=())
    with pytest.raises(ValidationError):
        revised(graph, node_counts=(NodeCount(node_type="paper", count=1),))


@pytest.fixture
def candidate(evidence):
    return ResearchGapCandidate(
        candidate_id="synthetic:candidate",
        topic_id="synthetic:t1",
        attention_signal=1.0,
        investigation_signal=1.0,
        research_gap_score=0.0,
        supporting_evidence=(evidence,),
        uncertainty=Uncertainty(status="unquantified", notes="Synthetic schema test; no estimate."),
        model_version="synthetic-not-trained",
        scoring_version="synthetic-not-a-formula",
        data_coverage=DataCoverage(
            total_papers=2,
            papers_with_topics=1,
            papers_with_contexts=1,
            papers_with_investigation_evidence=0,
            notes="Synthetic fixture only.",
        ),
        cutoff=CUTOFF,
        produced_at=CUTOFF,
    )


def test_candidate_is_an_unvalidated_hypothesis(candidate):
    assert candidate.interpretation == "unvalidated_hypothesis"
    assert candidate.uncertainty.status == "unquantified"
    with pytest.raises(ValidationError):
        revised(candidate, interpretation="proven_gap")
    with pytest.raises(ValidationError):
        revised(candidate, research_gap_score=float("nan"))


def test_candidate_requires_evidence_and_consistent_coverage(candidate):
    with pytest.raises(ValidationError):
        revised(candidate, supporting_evidence=())
    with pytest.raises(ValidationError):
        revised(candidate.data_coverage, papers_with_topics=3)
    with pytest.raises(ValidationError):
        revised(candidate.data_coverage, total_papers=True)


def test_uncertainty_does_not_imply_an_invented_estimate(candidate):
    with pytest.raises(ValidationError):
        Uncertainty(status="estimated", notes="Missing bounds")
    with pytest.raises(ValidationError):
        revised(candidate.uncertainty, lower_bound=0.0, upper_bound=1.0)
    interval = Uncertainty(
        status="estimated",
        notes="Synthetic interval for shape testing only.",
        method="synthetic-test",
        lower_bound=1.0,
        upper_bound=2.0,
    )
    with pytest.raises(ValidationError):
        revised(candidate, uncertainty=interval)


def test_outputs_reject_future_or_unknown_source_observations(candidate, graph, evidence):
    for changed in (
        revised(evidence, available_at=None),
        revised(evidence, retrieved_at=CUTOFF + timedelta(seconds=1)),
    ):
        with pytest.raises(ValidationError):
            revised(candidate, supporting_evidence=(changed,))
        with pytest.raises(ValidationError):
            revised(graph, evidence=(changed,))


def test_all_requested_models_publish_json_schema():
    from ci_hgnn_rgd import schemas

    for name in (
        "Paper",
        "Author",
        "Topic",
        "CitationEdge",
        "CitationContext",
        "CitationIntentPrediction",
        "PaperTopicEvidence",
        "GraphManifest",
        "ResearchGapCandidate",
        "EvidenceReference",
    ):
        assert getattr(schemas, name).model_json_schema()["title"] == name


def test_evidence_requires_explicit_unknown_availability(evidence):
    values = evidence.model_dump()
    values.pop("available_at")
    with pytest.raises(ValidationError):
        EvidenceReference.model_validate(values)
    assert revised(evidence, available_at=None).available_at is None


def test_paper_topic_contract_is_available_without_any_nlp_backend(synthetic_corpus):
    assert isinstance(synthetic_corpus.paper_topic_evidence[0], PaperTopicEvidence)


def test_learned_topics_require_training_provenance(synthetic_corpus):
    topic = synthetic_corpus.topics[0]
    with pytest.raises(ValidationError):
        revised(topic, model_version="synthetic-not-trained")
    with pytest.raises(ValidationError):
        revised(
            topic,
            model_version="synthetic-not-trained",
            training_data_cutoff=CUTOFF,
            produced_at=OBSERVED,
        )
