from datetime import timedelta

import pytest
from conftest import CUTOFF, OBSERVED, revised

from ci_hgnn_rgd.data.validation import CorpusValidationError, validate_corpus
from ci_hgnn_rgd.evaluation.temporal import TemporalWindow, validate_split_windows
from ci_hgnn_rgd.schemas import CitationIntentPrediction


def assert_issue(corpus, code):
    with pytest.raises(CorpusValidationError) as caught:
        validate_corpus(corpus, CUTOFF, allow_synthetic=True)
    assert code in {issue.code for issue in caught.value.issues}


def test_synthetic_inputs_require_explicit_opt_in(synthetic_corpus):
    with pytest.raises(CorpusValidationError, match="synthetic_evidence"):
        validate_corpus(synthetic_corpus, CUTOFF)
    validate_corpus(synthetic_corpus, CUTOFF, allow_synthetic=True)


@pytest.mark.parametrize(
    "field", ["papers", "authors", "topics", "citations", "contexts", "paper_topic_evidence"]
)
def test_duplicate_ids_in_all_record_groups(synthetic_corpus, field):
    assert_issue(
        revised(synthetic_corpus, **{field: getattr(synthetic_corpus, field) * 2}), "duplicate_id"
    )


def test_unknown_and_future_publication_dates(synthetic_corpus):
    for date, code in (
        (None, "unknown_publication_date"),
        ((CUTOFF + timedelta(days=1)).date(), "future_publication"),
    ):
        paper = revised(synthetic_corpus.papers[0], published_on=date)
        assert_issue(revised(synthetic_corpus, papers=(paper, synthetic_corpus.papers[1])), code)


@pytest.mark.parametrize(
    "field", ["papers", "authors", "topics", "citations", "contexts", "paper_topic_evidence"]
)
def test_future_retrieval_rejected_across_all_contracts(synthetic_corpus, evidence, field):
    reference = revised(evidence, retrieved_at=CUTOFF + timedelta(seconds=1))
    records = getattr(synthetic_corpus, field)
    changed = revised(records[0], evidence=(reference,))
    assert_issue(revised(synthetic_corpus, **{field: (changed,) + records[1:]}), "future_retrieval")


def test_unknown_availability_is_not_imputed_from_publication(synthetic_corpus, evidence):
    paper = revised(synthetic_corpus.papers[0], evidence=(revised(evidence, available_at=None),))
    assert_issue(
        revised(synthetic_corpus, papers=(paper, synthetic_corpus.papers[1])),
        "unknown_availability",
    )


def test_cutoff_is_inclusive(synthetic_corpus, evidence):
    reference = revised(
        evidence, reference_id="synthetic:boundary", available_at=CUTOFF, retrieved_at=CUTOFF
    )
    paper = revised(synthetic_corpus.papers[0], evidence=(reference,))
    validate_corpus(
        revised(synthetic_corpus, papers=(paper, synthetic_corpus.papers[1])),
        CUTOFF,
        allow_synthetic=True,
    )


def test_cross_record_references_are_checked(synthetic_corpus):
    assert_issue(revised(synthetic_corpus, authors=()), "missing_author")
    assert_issue(revised(synthetic_corpus, papers=()), "missing_paper")
    assert_issue(revised(synthetic_corpus, topics=()), "missing_topic_endpoint")
    assert_issue(revised(synthetic_corpus, contexts=()), "invalid_context")
    assert_issue(revised(synthetic_corpus, citations=()), "missing_edge")


def test_conflicting_reference_payloads_are_rejected(synthetic_corpus, evidence):
    other = revised(evidence, locator="different synthetic locator")
    paper = revised(synthetic_corpus.papers[0], evidence=(other,))
    assert_issue(
        revised(synthetic_corpus, papers=(paper, synthetic_corpus.papers[1])),
        "conflicting_provenance",
    )


def test_model_training_and_context_linkage_are_checked(synthetic_corpus, evidence):
    future = CUTOFF + timedelta(days=1)
    prediction = CitationIntentPrediction(
        prediction_id="synthetic:pred",
        context_id="synthetic:c1",
        taxonomy_version="synthetic-v1",
        label="Method",
        confidence=0.5,
        model_version="synthetic-not-trained",
        training_data_cutoff=future,
        produced_at=future,
        evidence=(evidence,),
    )
    edge = revised(synthetic_corpus.citations[0], intent=prediction)
    assert_issue(revised(synthetic_corpus, citations=(edge,)), "future_training_data")
    prediction = revised(prediction, training_data_cutoff=OBSERVED, context_id="unresolved")
    assert_issue(
        revised(synthetic_corpus, citations=(revised(edge, intent=prediction),)),
        "invalid_prediction_context",
    )


def test_split_windows_are_chronological_disjoint_and_half_open():
    points = [CUTOFF + timedelta(days=i) for i in range(4)]
    train, validation, test = [
        TemporalWindow(start=a, end=b) for a, b in zip(points, points[1:], strict=False)
    ]
    validate_split_windows(train, validation, test)
    with pytest.raises(ValueError):
        validate_split_windows(validation, train, test)
    overlap = TemporalWindow(start=points[0], end=points[2])
    with pytest.raises(ValueError):
        validate_split_windows(overlap, validation, test)
    with pytest.raises(ValueError):
        TemporalWindow(start=CUTOFF, end=CUTOFF)


def test_learned_topics_cannot_use_future_training_data(synthetic_corpus):
    future = CUTOFF + timedelta(days=1)
    topic = revised(
        synthetic_corpus.topics[0],
        model_version="synthetic-not-trained",
        training_data_cutoff=future,
        produced_at=future,
    )
    assert_issue(revised(synthetic_corpus, topics=(topic,)), "future_training_data")
