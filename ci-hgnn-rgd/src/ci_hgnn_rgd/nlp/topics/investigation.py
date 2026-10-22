"""Topic-aware candidate spans, independent annotation, agreement and adjudication."""

import json
import re
from collections import defaultdict
from itertools import combinations

from ...pipeline import fingerprint
from .contracts import InvestigationAnnotation, PaperTopicEvidence

CUES = re.compile(
    r"\b(?:we\s+(?:investigate|evaluate|examine|study|analy[sz]e|experimentally assess)"
    r"|this study\s+(?:examines|investigates|evaluates)|our objective is)\b",
    re.I,
)
CONTEXT = re.compile(
    r"\b(?:previous|prior|related work|background|future work|could|might"
    r"|not\s+(?:investigate|evaluate|examine)|do not)\b",
    re.I,
)


def candidate_evidence(paper, topic, *, ranking_score=None):
    """Rules require topic alignment in the same sentence; every candidate stays UNKNOWN."""
    base = dict(
        paper_key=paper.paper_key,
        canonical_paper_id=paper.canonical_paper_id,
        topic_id=topic.topic_id,
        evidence_label="UNKNOWN",
        evidence_source="unknown",
        evidence_method="topic_context_rules_v1",
        score_semantics="unvalidated_candidate_signal",
        review_status="insufficient_text",
        annotator_or_model_id="stage05_rules_v1",
        model_version="rules-v1",
        topic_model_version=topic.model_version,
        source_provenance=paper.provenance,
        temporal_snapshot_id=topic.temporal_snapshot_id,
        uncertainty_status="insufficient_evidence",
    )
    generic_cues = {
        "we",
        "investigate",
        "investigates",
        "evaluate",
        "evaluates",
        "examine",
        "examines",
        "study",
        "studies",
        "research",
        "paper",
        "objective",
        "sep",
    }
    terms = [
        word for word in topic.keywords if len(word) >= 3 and word.casefold() not in generic_cues
    ]
    for field in ("original_abstract", "original_title"):
        text = getattr(paper, field) or ""
        for sentence in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
            span = sentence.group()
            aligned = any(
                re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", span, re.I) for term in terms
            )
            if not aligned:
                continue
            cue = CUES.search(span)
            mention = CONTEXT.search(span)
            if cue or mention:
                label = "MENTIONS_TOPIC" if mention else "INVESTIGATES_TOPIC"
                return PaperTopicEvidence(
                    **{
                        **base,
                        "evidence_source": "rule_candidate",
                        "evidence_text_span": span,
                        "evidence_start_offset": sentence.start(),
                        "evidence_end_offset": sentence.end(),
                        "text_field": field,
                        "review_status": "unreviewed",
                        "candidate_label": label,
                        "uncertainty_status": "candidate_requires_independent_review",
                    }
                )
    if ranking_score is not None:
        base.update(
            evidence_source="ranking_candidate",
            evidence_method="paper_topic_cosine",
            evidence_score=float(ranking_score),
            score_semantics="cosine_similarity_not_investigation_probability",
            review_status="unreviewed",
            uncertainty_status="semantic_relatedness_only",
        )
    return PaperTopicEvidence(**base)


def rank_evidence_spans(paper, topic, topic_vector, encoder, *, max_sentences=16):
    """Rank actual source spans against a fitted scientific topic centroid.

    Paper title + one abstract sentence is encoded for each candidate. This is
    descriptive evidence retrieval, never a trained investigation classifier.
    """
    import numpy as np

    spans = list(re.finditer(r"[^.!?]+(?:[.!?]+|$)", paper.original_abstract or ""))[:max_sentences]
    if not spans:
        return candidate_evidence(paper, topic)
    rows = []
    for span in spans:
        values = paper.model_dump(mode="python")
        values.update(
            abstract=span.group(),
            model_text=paper.title + " [SEP] " + span.group(),
            text_sha256=fingerprint([paper.text_sha256, span.start(), span.end()]),
        )
        rows.append(type(paper).model_validate(values))
    vectors, _ = encoder.encode(rows)
    vector = np.asarray(topic_vector, dtype="float32")
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(vector)
    if not np.isfinite(vectors).all() or (denominator <= 1e-12).any():
        raise ValueError("invalid semantic evidence-ranking vectors")
    scores = (vectors @ vector) / denominator
    index = int(scores.argmax())
    span = spans[index]
    base = candidate_evidence(paper, topic, ranking_score=float(scores[index])).model_dump(
        mode="python"
    )
    base.update(
        evidence_source="ranking_candidate",
        evidence_label="UNKNOWN",
        candidate_label=None,
        evidence_text_span=span.group(),
        evidence_start_offset=span.start(),
        evidence_end_offset=span.end(),
        text_field="original_abstract",
        evidence_method="scientific_span_to_frozen_topic_centroid_cosine",
        evidence_score=float(scores[index]),
        score_semantics="cosine_similarity_not_investigation_probability",
        review_status="unreviewed",
        uncertainty_status="semantic_relatedness_only",
    )
    return PaperTopicEvidence.model_validate(base)


def annotation_sample(papers, topics, assignments, *, size, seed, guideline_version):
    """Round-robin strata include topic/year/category/source/completeness/score bin."""
    paper_map, topic_map = {p.paper_key: p for p in papers}, {t.topic_id: t for t in topics}
    strata = defaultdict(list)
    for assignment in assignments:
        if assignment.topic_id is None:
            continue
        paper, topic = paper_map[assignment.paper_key], topic_map[assignment.topic_id]
        score = assignment.assignment_score
        level = (
            "unknown"
            if score is None
            else "low"
            if score < 0.3
            else "mid"
            if score < 0.7
            else "high"
        )
        key = (
            topic.topic_id,
            str(paper.publication_year),
            str(paper.categories),
            paper.source_dataset,
            paper.text_mode,
            level,
        )
        task = {
            "paper_key": paper.paper_key,
            "canonical_paper_id": paper.canonical_paper_id,
            "topic_id": topic.topic_id,
            "topic_definition": topic.provisional_label,
            "title": paper.original_title,
            "abstract": paper.original_abstract,
            "source_provenance": paper.provenance,
            "stratum": list(key),
            "guideline_version": guideline_version,
            "required_independent_annotators": 2,
            "annotations": [],
            "label": None,
            "annotator_id": None,
        }
        strata[key].append(task)
    for key in strata:
        strata[key].sort(key=lambda task: fingerprint([seed, task["paper_key"], task["topic_id"]]))
    result = []
    while len(result) < size and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(result) < size:
                result.append(strata[key].pop(0))
    return result


def validate_annotations(rows, papers, topics, *, guideline_version="stage05-v1"):
    paper_map, topic_map = {p.paper_key: p for p in papers}, {t.topic_id: t for t in topics}
    result, seen = [], set()
    for row in rows:
        annotation = InvestigationAnnotation.model_validate(row)
        pair = (annotation.paper_key, annotation.topic_id)
        unique = (*pair, annotation.annotator_id)
        if unique in seen:
            raise ValueError("duplicate annotator response for paper-topic pair")
        seen.add(unique)
        paper = paper_map.get(annotation.paper_key)
        topic = topic_map.get(annotation.topic_id)
        if paper is None or topic is None:
            raise ValueError("annotation references an unknown paper/topic")
        if annotation.canonical_paper_id != paper.canonical_paper_id:
            raise ValueError("annotation canonical paper identity differs from source")
        if (
            annotation.title != paper.original_title
            or annotation.abstract != paper.original_abstract
        ):
            raise ValueError("annotation source text differs from frozen corpus")
        if annotation.topic_definition != topic.provisional_label:
            raise ValueError("annotation topic definition differs from frozen catalog")
        if annotation.guideline_version != guideline_version:
            raise ValueError("annotation guideline version mismatch")
        result.append(annotation)
    return tuple(result)


def agreement_report(annotations):
    from sklearn.metrics import cohen_kappa_score

    by_person = defaultdict(dict)
    pair_labels = defaultdict(list)
    for annotation in annotations:
        pair = (annotation.paper_key, annotation.topic_id)
        by_person[annotation.annotator_id][pair] = annotation.label
        pair_labels[pair].append(annotation.label)
    comparisons = []
    for left, right in combinations(sorted(by_person), 2):
        pairs = sorted(by_person[left].keys() & by_person[right].keys())
        if not pairs:
            continue
        a, b = [by_person[left][p] for p in pairs], [by_person[right][p] for p in pairs]
        # Kappa is undefined with constant identical annotations; don't emit NaN.
        kappa = float(cohen_kappa_score(a, b)) if len(set(a + b)) > 1 else None
        comparisons.append(
            {
                "annotators": [left, right],
                "independently_shared_pairs": len(pairs),
                "cohen_kappa": kappa,
                "raw_agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(pairs),
            }
        )
    return {
        "status": "MEASURED" if comparisons else "NOT_EVALUABLE",
        "synthetic_fixture": bool(annotations) and all(a.synthetic for a in annotations),
        "comparisons": comparisons,
        "disagreement_pairs": [
            list(pair) for pair, labels in pair_labels.items() if len(set(labels)) > 1
        ],
        "annotations": len(annotations),
        "annotators": len(by_person),
        "supervised_status": "NOT_TRAINED",
    }


def adjudicate(annotations, *, adjudications=()):
    """Unanimity requires >=2 distinct annotators; conflicts require a new adjudicator."""
    groups = defaultdict(list)
    for row in annotations:
        groups[row.paper_key, row.topic_id].append(row)
    resolutions = {(row["paper_key"], row["topic_id"]): row for row in adjudications}
    finals = []
    for pair, rows in sorted(groups.items()):
        if len({row.annotator_id for row in rows}) < 2:
            continue
        labels = {row.label for row in rows}
        splits = {row.split for row in rows}
        if len(splits) != 1 or None in splits:
            raise ValueError(
                "annotations require a consistent explicit train/dev/test split per pair"
            )
        if len(labels) == 1:
            chosen = rows[0]
            adjudicator = None
        else:
            decision = resolutions.get(pair)
            if not decision:
                continue
            adjudicator = decision.get("adjudicator_id")
            if (
                not adjudicator
                or adjudicator in {row.annotator_id for row in rows}
                or not decision.get("notes")
            ):
                raise ValueError(
                    "disagreement resolution requires an independent adjudicator and notes"
                )
            candidates = [row for row in rows if row.label == decision.get("final_label")]
            if not candidates:
                raise ValueError("adjudicated label needs a grounded supporting annotation")
            chosen = candidates[0]
        finals.append(
            {
                "annotation": chosen.model_dump(mode="json"),
                "validated": True,
                "independent_annotator_ids": sorted({row.annotator_id for row in rows}),
                "adjudicator_id": adjudicator,
                "annotation_records": [row.model_dump(mode="json") for row in rows],
                "annotation_sha256": fingerprint([row.model_dump(mode="json") for row in rows]),
            }
        )
    return finals


def validated_training_splits(finals, *, min_train=30, allow_synthetic=False):
    groups, paper_splits = {"train": [], "dev": [], "test": []}, defaultdict(set)
    for final in finals:
        if not final.get("validated") or len(set(final.get("independent_annotator_ids", []))) < 2:
            raise ValueError("NOT_TRAINED: validated independent annotations are required")
        records = [
            InvestigationAnnotation.model_validate(row)
            for row in final.get("annotation_records", [])
        ]
        chosen = InvestigationAnnotation.model_validate(final["annotation"])
        if len({row.annotator_id for row in records}) < 2 or chosen not in records:
            raise ValueError("NOT_TRAINED: annotation validation records are missing")
        if any(row.synthetic for row in records) and not allow_synthetic:
            raise ValueError("synthetic fixtures cannot train a research investigation classifier")
        if any(
            (row.paper_key, row.topic_id, row.title, row.abstract, row.split)
            != (chosen.paper_key, chosen.topic_id, chosen.title, chosen.abstract, chosen.split)
            for row in records
        ):
            raise ValueError("validated annotation records disagree on paper/topic/text/split")
        if len({row.label for row in records}) > 1 and not final.get("adjudicator_id"):
            raise ValueError("NOT_TRAINED: conflicting annotations lack adjudication")
        if chosen.split is None:
            raise ValueError("validated annotations require explicit splits")
        # Canonical identity is authoritative for split grouping even if a source
        # record happens to be represented by more than one local key.
        paper_splits[chosen.canonical_paper_id or chosen.paper_key].add(chosen.split)
        groups[chosen.split].append(chosen)
    if any(len(splits) > 1 for splits in paper_splits.values()):
        raise ValueError("paper-group leakage across investigation splits")
    if len(groups["train"]) < min_train or not groups["dev"] or not groups["test"]:
        raise ValueError("NOT_TRAINED: insufficient validated train/dev/test annotation coverage")
    if {row.label for row in groups["train"]} != {
        "INVESTIGATES_TOPIC",
        "MENTIONS_TOPIC",
        "UNKNOWN",
    }:
        raise ValueError("NOT_TRAINED: all three investigation classes required in train")
    return {
        name: tuple(sorted(rows, key=lambda a: (a.paper_key, a.topic_id)))
        for name, rows in groups.items()
    }


def read_annotation_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
