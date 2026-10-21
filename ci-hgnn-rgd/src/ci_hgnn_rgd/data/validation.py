"""Cross-record integrity and strict observation-cutoff validation."""

from dataclasses import dataclass
from datetime import UTC, datetime

from ..schemas import Corpus, EvidenceReference


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    record_id: str
    message: str


class CorpusValidationError(ValueError):
    def __init__(self, issues: tuple[ValidationIssue, ...]):
        self.issues = issues
        super().__init__("; ".join(f"{i.code} [{i.record_id}]: {i.message}" for i in issues))


def validate_corpus(corpus: Corpus, cutoff: datetime, *, allow_synthetic: bool = False) -> None:
    if cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")
    issues = []
    references: dict[str, EvidenceReference] = {}

    def issue(code, record_id, message):
        issues.append(ValidationIssue(code, record_id, message))

    def evidence_checks(evidence, record_id):
        for reference in evidence:
            if reference.evidence_kind == "synthetic" and not allow_synthetic:
                issue("synthetic_evidence", record_id, "synthetic evidence is disabled by default")
            previous = references.setdefault(reference.reference_id, reference)
            if previous != reference:
                issue(
                    "conflicting_provenance", record_id, "reference ID maps to different provenance"
                )
            if reference.available_at is None:
                issue("unknown_availability", record_id, "source availability must be verified")
            elif reference.available_at > cutoff:
                issue(
                    "future_availability", record_id, "source version became available after cutoff"
                )
            if reference.retrieved_at > cutoff:
                issue(
                    "future_retrieval", record_id, "source was retrieved after observation cutoff"
                )

    groups = (
        (corpus.papers, "paper_id"),
        (corpus.authors, "author_id"),
        (corpus.topics, "topic_id"),
        (corpus.citations, "edge_id"),
        (corpus.contexts, "context_id"),
        (corpus.paper_topic_evidence, "evidence_id"),
    )
    for records, field in groups:
        seen = set()
        for record in records:
            record_id = getattr(record, field)
            if record_id in seen:
                issue("duplicate_id", record_id, f"duplicate {field}")
            seen.add(record_id)
            evidence_checks(record.evidence, record_id)
    papers = {p.paper_id for p in corpus.papers}
    authors = {a.author_id for a in corpus.authors}
    topics = {t.topic_id for t in corpus.topics}
    edges = {c.edge_id: c for c in corpus.citations}
    contexts = {c.context_id: c for c in corpus.contexts}
    predictions = set()
    for topic in corpus.topics:
        if topic.training_data_cutoff is not None and topic.training_data_cutoff > cutoff:
            issue("future_training_data", topic.topic_id, "topic model training follows cutoff")
    for paper in corpus.papers:
        if paper.published_on is None:
            issue("unknown_publication_date", paper.paper_id, "publication date is unknown")
        elif paper.published_on > cutoff.astimezone(UTC).date():
            issue("future_publication", paper.paper_id, "publication follows cutoff date in UTC")
        if any(author not in authors for author in paper.author_ids):
            issue("missing_author", paper.paper_id, "unresolved author ID")
    for edge in corpus.citations:
        if edge.citing_paper_id not in papers or edge.cited_paper_id not in papers:
            issue("missing_paper", edge.edge_id, "unresolved citation endpoint")
        for context_id in edge.context_ids:
            if context_id not in contexts or contexts[context_id].edge_id != edge.edge_id:
                issue("invalid_context", edge.edge_id, "context must exist and reference this edge")
        if edge.intent is not None:
            prediction = edge.intent
            if prediction.prediction_id in predictions:
                issue("duplicate_prediction", edge.edge_id, "prediction identifiers must be unique")
            predictions.add(prediction.prediction_id)
            if prediction.context_id not in edge.context_ids:
                issue(
                    "invalid_prediction_context", edge.edge_id, "intent context must belong to edge"
                )
            if prediction.training_data_cutoff > cutoff:
                issue("future_training_data", edge.edge_id, "intent model training follows cutoff")
            evidence_checks(prediction.evidence, prediction.prediction_id)
    for context in corpus.contexts:
        if (
            context.edge_id not in edges
            or context.context_id not in edges[context.edge_id].context_ids
        ):
            issue("missing_edge", context.context_id, "context requires a reciprocal citation edge")
    for item in corpus.paper_topic_evidence:
        if item.paper_id not in papers or item.topic_id not in topics:
            issue("missing_topic_endpoint", item.evidence_id, "unresolved paper/topic endpoint")
        if item.training_data_cutoff is not None and item.training_data_cutoff > cutoff:
            issue(
                "future_training_data",
                item.evidence_id,
                "investigation model training follows cutoff",
            )
    if issues:
        raise CorpusValidationError(tuple(issues))
