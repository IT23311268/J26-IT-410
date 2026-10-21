"""Version 1.0 contracts. Structural validity does not establish scientific truth."""

from datetime import UTC, date
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    HttpUrl,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from .graph.schema import NodeType, RelationType, get_graph_schema

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FiniteNumber = Annotated[FiniteFloat, Field(strict=True)]
Probability = Annotated[FiniteNumber, Field(ge=0, le=1)]
NonNegative = Annotated[FiniteNumber, Field(ge=0)]
Count = Annotated[StrictInt, Field(ge=0)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, allow_inf_nan=False
    )


class SciCiteClass(StrEnum):
    BACKGROUND = "Background"
    METHOD = "Method"
    RESULT = "Result"


# Raw lowercase labels are mapped explicitly by future adapters, never expanded.
SCICITE_RAW_LABELS = {label.value.lower(): label for label in SciCiteClass}


class EvidenceReference(FrozenModel):
    reference_id: NonEmpty
    dataset_id: NonEmpty
    dataset_version: NonEmpty
    source_record_id: NonEmpty
    source_url: HttpUrl
    payload_sha256: SHA256
    locator: NonEmpty
    available_at: AwareDatetime | None
    retrieved_at: AwareDatetime
    evidence_text: NonEmpty | None = None
    evidence_kind: Literal["scholarly", "synthetic"] = "scholarly"

    @field_validator("available_at", "retrieved_at")
    @classmethod
    def normalize_utc(cls, value):
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def availability_order(self):
        if self.available_at is not None and self.available_at > self.retrieved_at:
            raise ValueError("source availability cannot follow retrieval")
        return self


Evidence = Annotated[tuple[EvidenceReference, ...], Field(min_length=1)]


class ExternalIdentifier(FrozenModel):
    namespace: NonEmpty
    value: NonEmpty


class Paper(FrozenModel):
    paper_id: NonEmpty
    title: NonEmpty
    abstract: NonEmpty | None = None
    published_on: date | None
    author_ids: tuple[NonEmpty, ...] = ()
    identifiers: tuple[ExternalIdentifier, ...] = ()
    evidence: Evidence


class Author(FrozenModel):
    author_id: NonEmpty
    name: NonEmpty
    identifiers: tuple[ExternalIdentifier, ...] = ()
    evidence: Evidence


class Topic(FrozenModel):
    topic_id: NonEmpty
    label: NonEmpty
    description: NonEmpty | None = None
    model_version: NonEmpty | None = None
    training_data_cutoff: AwareDatetime | None = None
    produced_at: AwareDatetime | None = None
    evidence: Evidence

    @model_validator(mode="after")
    def learned_topic_provenance(self):
        if self.model_version is not None:
            if self.training_data_cutoff is None or self.produced_at is None:
                raise ValueError("learned topics require training cutoff and production time")
            if self.training_data_cutoff > self.produced_at:
                raise ValueError("topic model training cutoff cannot follow production time")
        elif self.training_data_cutoff is not None:
            raise ValueError("topic training cutoff requires a model version")
        return self


class CitationIntentPrediction(FrozenModel):
    prediction_id: NonEmpty
    context_id: NonEmpty
    taxonomy: Literal["scicite", "extended"] = "scicite"
    taxonomy_version: NonEmpty
    extended_taxonomy_id: NonEmpty | None = None
    extended_classes: tuple[NonEmpty, ...] = ()
    label: NonEmpty | None
    confidence: Probability | None
    model_version: NonEmpty
    training_data_cutoff: AwareDatetime
    produced_at: AwareDatetime
    evidence: Evidence

    @model_validator(mode="after")
    def valid_prediction(self):
        if self.taxonomy == "scicite":
            if self.extended_taxonomy_id is not None or self.extended_classes:
                raise ValueError("SciCite cannot include an extended taxonomy")
            classes = {item.value for item in SciCiteClass}
        else:
            if not self.extended_taxonomy_id or not self.extended_classes:
                raise ValueError("extended taxonomy requires a separate identifier and class list")
            if self.extended_taxonomy_id.casefold().startswith("scicite"):
                raise ValueError("extended taxonomy must not be presented as SciCite")
            if len(self.extended_classes) != len(set(self.extended_classes)):
                raise ValueError("extended classes must be unique")
            classes = set(self.extended_classes)
        if self.label is not None and self.label not in classes:
            raise ValueError("label is not in the declared taxonomy")
        if (self.label is None) != (self.confidence is None):
            raise ValueError("unknown intent has null label/confidence; known intent requires both")
        if self.training_data_cutoff > self.produced_at:
            raise ValueError("training cutoff cannot follow prediction time")
        return self


class CitationEdge(FrozenModel):
    edge_id: NonEmpty
    citing_paper_id: NonEmpty
    cited_paper_id: NonEmpty
    context_ids: tuple[NonEmpty, ...] = ()
    intent: CitationIntentPrediction | None = None
    evidence: Evidence


class CitationContext(FrozenModel):
    context_id: NonEmpty
    edge_id: NonEmpty
    text: NonEmpty
    section: NonEmpty | None = None
    evidence: Evidence


class PaperTopicEvidence(FrozenModel):
    evidence_id: NonEmpty
    paper_id: NonEmpty
    topic_id: NonEmpty
    relationship: Literal["mentions_topic", "investigates_topic", "unknown"]
    confidence: Probability | None
    method: Literal["human", "model", "source"]
    method_version: NonEmpty
    basis: Literal["textual_evidence", "structured_source", "citation_frequency"]
    training_data_cutoff: AwareDatetime | None = None
    produced_at: AwareDatetime
    evidence: Evidence

    @model_validator(mode="after")
    def independent_investigation(self):
        if self.relationship == "investigates_topic" and self.basis == "citation_frequency":
            raise ValueError("citation frequency alone cannot establish direct investigation")
        if self.relationship != "unknown" and self.confidence is None:
            raise ValueError("known paper-topic evidence requires confidence")
        if self.method == "model" and self.training_data_cutoff is None:
            raise ValueError("model evidence requires a training-data cutoff")
        if self.training_data_cutoff is not None:
            if self.method != "model" or self.training_data_cutoff > self.produced_at:
                raise ValueError("invalid method/training-data cutoff combination")
        return self


class NodeCount(FrozenModel):
    node_type: NodeType
    count: Count


class RelationCount(FrozenModel):
    source: NodeType
    relation: RelationType
    target: NodeType
    count: Count

    @model_validator(mode="after")
    def typed_relation(self):
        supported = {(r.source, r.relation, r.target) for r in get_graph_schema(True).relations}
        if (self.source, self.relation, self.target) not in supported:
            raise ValueError("unsupported relation endpoints")
        return self


class DatasetSnapshot(FrozenModel):
    dataset_id: NonEmpty
    version_id: NonEmpty
    sha256: SHA256


def _require_observed_evidence(evidence: Evidence, cutoff: AwareDatetime) -> None:
    for reference in evidence:
        if reference.available_at is None:
            raise ValueError("output evidence requires verified source availability")
        if reference.available_at > cutoff or reference.retrieved_at > cutoff:
            raise ValueError("output evidence must have been available and retrieved by cutoff")


class GraphManifest(FrozenModel):
    graph_id: NonEmpty
    schema_version: Literal["1.0"] = "1.0"
    built_at: AwareDatetime
    cutoff: AwareDatetime
    node_counts: tuple[NodeCount, ...]
    relation_counts: tuple[RelationCount, ...]
    datasets: Annotated[tuple[DatasetSnapshot, ...], Field(min_length=1)]
    feature_training_cutoff: AwareDatetime | None = None
    evidence: Evidence

    @model_validator(mode="after")
    def consistent_graph(self):
        _require_observed_evidence(self.evidence, self.cutoff)
        snapshots = {(item.dataset_id, item.version_id) for item in self.datasets}
        if len(snapshots) != len(self.datasets):
            raise ValueError("duplicate dataset snapshots")
        if any((ref.dataset_id, ref.dataset_version) not in snapshots for ref in self.evidence):
            raise ValueError("graph provenance must refer to a listed dataset snapshot")
        nodes = {item.node_type: item.count for item in self.node_counts}
        if len(nodes) != len(self.node_counts):
            raise ValueError("duplicate node types")
        if not {NodeType.PAPER, NodeType.AUTHOR, NodeType.TOPIC} <= nodes.keys():
            raise ValueError("paper, author, and topic node counts are required, including zeros")
        keys = [(r.source, r.relation, r.target) for r in self.relation_counts]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate relation counts")
        for relation in self.relation_counts:
            if relation.source not in nodes or relation.target not in nodes:
                raise ValueError("relation endpoints must have node counts")
            if relation.count and (not nodes[relation.source] or not nodes[relation.target]):
                raise ValueError("nonempty relations require nonempty endpoint node types")
        if self.feature_training_cutoff and self.feature_training_cutoff > self.cutoff:
            raise ValueError("graph features cannot be fitted after the graph cutoff")
        if self.built_at < self.cutoff:
            raise ValueError("graph build time cannot precede its observation cutoff")
        return self


class DataCoverage(FrozenModel):
    total_papers: Count
    papers_with_topics: Count
    papers_with_contexts: Count
    papers_with_investigation_evidence: Count
    notes: NonEmpty

    @model_validator(mode="after")
    def valid_counts(self):
        for field in (
            "papers_with_topics",
            "papers_with_contexts",
            "papers_with_investigation_evidence",
        ):
            if getattr(self, field) > self.total_papers:
                raise ValueError(f"{field} cannot exceed total_papers")
        return self


class Uncertainty(FrozenModel):
    status: Literal["unquantified", "estimated"]
    notes: NonEmpty
    method: NonEmpty | None = None
    lower_bound: FiniteNumber | None = None
    upper_bound: FiniteNumber | None = None

    @model_validator(mode="after")
    def valid_interval(self):
        if self.status == "estimated":
            if self.method is None or self.lower_bound is None or self.upper_bound is None:
                raise ValueError("estimated uncertainty requires method and both bounds")
            if self.lower_bound > self.upper_bound:
                raise ValueError("uncertainty lower bound exceeds upper bound")
        elif any(v is not None for v in (self.method, self.lower_bound, self.upper_bound)):
            raise ValueError("unquantified uncertainty cannot claim an estimated interval")
        return self


class ResearchGapCandidate(FrozenModel):
    candidate_id: NonEmpty
    topic_id: NonEmpty
    attention_signal: NonNegative
    investigation_signal: NonNegative
    research_gap_score: FiniteNumber
    supporting_evidence: Evidence
    uncertainty: Uncertainty
    model_version: NonEmpty
    scoring_version: NonEmpty
    data_coverage: DataCoverage
    cutoff: AwareDatetime
    produced_at: AwareDatetime
    interpretation: Literal["unvalidated_hypothesis"] = "unvalidated_hypothesis"

    @model_validator(mode="after")
    def candidate_time(self):
        _require_observed_evidence(self.supporting_evidence, self.cutoff)
        if self.produced_at < self.cutoff:
            raise ValueError("candidate production time cannot precede its observation cutoff")
        if self.uncertainty.status == "estimated" and not (
            self.uncertainty.lower_bound <= self.research_gap_score <= self.uncertainty.upper_bound
        ):
            raise ValueError(
                "research-gap score must lie within its estimated uncertainty interval"
            )
        return self


class Corpus(FrozenModel):
    papers: tuple[Paper, ...] = ()
    authors: tuple[Author, ...] = ()
    topics: tuple[Topic, ...] = ()
    citations: tuple[CitationEdge, ...] = ()
    contexts: tuple[CitationContext, ...] = ()
    paper_topic_evidence: tuple[PaperTopicEvidence, ...] = ()
