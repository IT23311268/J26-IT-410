"""Synthetic investigation examples exercise code; they are not human supervision."""

import pytest
from stage05_helpers import POLICY, annotation, configuration, papers, source_row

from ci_hgnn_rgd.nlp.topics.contracts import (
    LABEL_MAPPING,
    InvestigationAnnotation,
    PaperTopicEvidence,
)
from ci_hgnn_rgd.nlp.topics.data import prepare_paper
from ci_hgnn_rgd.nlp.topics.investigation import (
    adjudicate,
    agreement_report,
    annotation_sample,
    candidate_evidence,
    validate_annotations,
    validated_training_splits,
)
from ci_hgnn_rgd.nlp.topics.investigation_model import investigation_metrics
from ci_hgnn_rgd.nlp.topics.models import fit_topics


def context(tmp_path):
    config = configuration(tmp_path)
    corpus = papers(config, count=12)
    model = fit_topics(
        corpus, config.topic_model, algorithm="nmf", version="fixture", snapshot="snapshot", seed=1
    )
    return config, corpus, model


def test_investigation_taxonomy_and_required_grounded_annotation(tmp_path):
    _, corpus, model = context(tmp_path)
    assert LABEL_MAPPING == {"INVESTIGATES_TOPIC": 0, "MENTIONS_TOPIC": 1, "UNKNOWN": 2}
    row = annotation(corpus[0], model.catalog[0], "synthetic-annotator")
    assert (
        InvestigationAnnotation.model_validate(row).relevant_text_span
        == corpus[0].original_abstract
    )
    for change in (
        {"label": "result"},
        {"relevant_text_span": "invented"},
        {"start_offset": -1},
        {"relevant_text_span": None},
    ):
        with pytest.raises(ValueError):
            InvestigationAnnotation.model_validate({**row, **change})


@pytest.mark.parametrize(
    "text,expected",
    [
        ("We investigate physics using measurements.", "INVESTIGATES_TOPIC"),
        ("Prior work investigates physics in a different context.", "MENTIONS_TOPIC"),
        ("We do not investigate physics; this is future work.", "MENTIONS_TOPIC"),
        ("We investigate unrelated entities.", None),
    ],
)
def test_topic_aware_rules_only_emit_review_candidates(tmp_path, text, expected):
    config, _, model = context(tmp_path)
    topic = next(t for t in model.catalog if "physics" in t.keywords)
    paper = prepare_paper(source_row(abstract=text), config, policy=POLICY, cutoff=None)
    evidence = candidate_evidence(paper, topic)
    assert evidence.evidence_label == "UNKNOWN"
    assert evidence.candidate_label == expected
    if evidence.evidence_text_span is not None:
        original = getattr(paper, evidence.text_field)
        assert (
            original[evidence.evidence_start_offset : evidence.evidence_end_offset]
            == evidence.evidence_text_span
        )


def test_membership_and_scicite_result_never_establish_investigation(tmp_path):
    _, corpus, model = context(tmp_path)
    result_only = corpus[0].model_dump()
    result_only["original_abstract"] = "A citation is labelled Result."
    result_only["original_title"] = "Generic paper"
    paper = type(corpus[0]).model_validate(result_only)
    evidence = candidate_evidence(paper, model.catalog[0], ranking_score=0.99)
    assert evidence.evidence_label == "UNKNOWN"
    assert evidence.evidence_source == "ranking_candidate"
    assert evidence.source_provenance == paper.provenance
    with pytest.raises(ValueError, match="candidate signals"):
        PaperTopicEvidence.model_validate(
            {**evidence.model_dump(), "evidence_label": "INVESTIGATES_TOPIC"}
        )


def test_annotation_sampling_is_stratified_and_contains_no_completed_labels(tmp_path):
    config, corpus, model = context(tmp_path)
    tasks = annotation_sample(
        corpus, model.catalog, model.assignments, size=6, seed=42, guideline_version="stage05-v1"
    )
    assert len(tasks) == 6
    assert all(task["label"] is None and task["annotations"] == [] for task in tasks)
    assert {task["required_independent_annotators"] for task in tasks} == {2}
    assert len({task["topic_id"] for task in tasks}) == 2


def test_disagreement_kappa_and_adjudication_require_independent_people(tmp_path):
    _, corpus, model = context(tmp_path)
    annotations = []
    for index in range(3):
        topic = model.catalog[0]
        annotations.extend(
            [
                annotation(corpus[index], topic, "synthetic-A"),
                annotation(
                    corpus[index],
                    topic,
                    "synthetic-B",
                    label="MENTIONS_TOPIC" if index == 1 else "INVESTIGATES_TOPIC",
                ),
            ]
        )
    rows = validate_annotations(annotations, corpus, model.catalog)
    report = agreement_report(rows)
    assert report["status"] == "MEASURED" and report["synthetic_fixture"] is True
    assert len(report["disagreement_pairs"]) == 1
    assert len(adjudicate(rows)) == 2
    resolution = {
        "paper_key": corpus[1].paper_key,
        "topic_id": topic.topic_id,
        "final_label": "MENTIONS_TOPIC",
        "adjudicator_id": "synthetic-C",
        "notes": "fixture-only",
    }
    assert len(adjudicate(rows, adjudications=[resolution])) == 3
    with pytest.raises(ValueError, match="independent adjudicator"):
        adjudicate(rows, adjudications=[{**resolution, "adjudicator_id": "synthetic-A"}])
    assert agreement_report(())["status"] == "NOT_EVALUABLE"


def test_annotation_source_text_and_duplicate_responses_rejected(tmp_path):
    _, corpus, model = context(tmp_path)
    row = annotation(corpus[0], model.catalog[0], "fixture-A")
    with pytest.raises(ValueError, match="duplicate annotator"):
        validate_annotations([row, row], corpus, model.catalog)
    with pytest.raises(ValueError, match="source text"):
        validate_annotations([{**row, "title": "tampered"}], corpus, model.catalog)


def finals_fixture(tmp_path):
    config, corpus, model = context(tmp_path)
    rows = []
    for index, paper in enumerate(corpus):
        label = list(LABEL_MAPPING)[index % 3]
        split = "train" if index < 6 else "dev" if index < 9 else "test"
        rows.extend(
            annotation(paper, model.catalog[0], person, label=label, split=split)
            for person in ("synthetic-A", "synthetic-B")
        )
    parsed = validate_annotations(rows, corpus, model.catalog)
    return config, adjudicate(parsed)


def test_supervised_training_blocked_without_validated_real_annotations(tmp_path):
    with pytest.raises(ValueError, match="NOT_TRAINED"):
        validated_training_splits([], min_train=3)
    _, finals = finals_fixture(tmp_path)
    with pytest.raises(ValueError, match="synthetic fixtures"):
        validated_training_splits(finals, min_train=3)
    groups = validated_training_splits(finals, min_train=3, allow_synthetic=True)
    assert [len(groups[name]) for name in ("train", "dev", "test")] == [6, 3, 3]


def test_paper_groups_cannot_leak_across_investigation_splits(tmp_path):
    _, finals = finals_fixture(tmp_path)
    first_id = finals[0]["annotation"]["canonical_paper_id"]
    target = next(row for row in finals if row["annotation"]["split"] == "dev")
    for row in target["annotation_records"]:
        row["canonical_paper_id"] = first_id
    target["annotation"]["canonical_paper_id"] = first_id
    with pytest.raises(ValueError, match="paper-group leakage"):
        validated_training_splits(finals, min_train=3, allow_synthetic=True)


def test_supervised_tiny_local_model_smoke_is_explicitly_synthetic(tmp_path):
    pytest.importorskip("torch")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from ci_hgnn_rgd.nlp.topics.investigation_model import train_investigation

    config, finals = finals_fixture(tmp_path)
    values = config.model_dump()
    values["investigation"].update({"min_train_annotations": 3, "epochs": 1, "batch_size": 2})
    values["embedding"]["device"] = "cpu"
    config = type(config).model_validate(values)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nphysics\nbiology\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=7,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
            num_labels=3,
        )
    )
    result = train_investigation(
        finals, tmp_path / "run", config, model=model, tokenizer=tokenizer, allow_synthetic=True
    )
    assert result["status"] == "TRAINED_SYNTHETIC_FIXTURE"
    assert result["test_evaluated"] is False
    assert (tmp_path / "run/model/model.safetensors").exists()


def test_investigation_macro_f1_and_per_class_names():
    metrics = investigation_metrics([0, 1, 2], [0, 1, 0])
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1) / 3)
    assert set(metrics["per_class"]) == set(LABEL_MAPPING)


def test_scientific_span_ranking_is_candidate_evidence_with_exact_offsets(tmp_path):
    import numpy as np

    from ci_hgnn_rgd.nlp.topics.investigation import rank_evidence_spans

    config, _, model = context(tmp_path)
    paper = prepare_paper(
        source_row(abstract="Unrelated first sentence. Physics second sentence."),
        config,
        policy=POLICY,
        cutoff=None,
    )

    class Encoder:
        def encode(self, rows):
            return np.array([[0, 1], [1, 0]], dtype="float32"), [False, False]

    evidence = rank_evidence_spans(paper, model.catalog[0], np.array([1, 0]), Encoder())
    assert evidence.evidence_label == "UNKNOWN" and evidence.evidence_source == "ranking_candidate"
    assert evidence.evidence_score == 1
    assert (
        paper.original_abstract[evidence.evidence_start_offset : evidence.evidence_end_offset]
        == evidence.evidence_text_span
    )
    assert "Physics second" in evidence.evidence_text_span
