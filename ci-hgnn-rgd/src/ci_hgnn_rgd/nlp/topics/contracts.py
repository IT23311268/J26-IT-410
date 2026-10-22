"""Versioned Stage 05 contracts. A local paper key is never a fabricated canonical ID."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, ConfigDict, Field, StrictInt, model_validator

from ...schemas import FrozenModel, NonEmpty

Label = Literal["INVESTIGATES_TOPIC", "MENTIONS_TOPIC", "UNKNOWN"]
LABELS = ("INVESTIGATES_TOPIC", "MENTIONS_TOPIC", "UNKNOWN")
LABEL_MAPPING = {label: index for index, label in enumerate(LABELS)}
Score = Annotated[float, Field(ge=0, le=1)]


class TextModel(FrozenModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=False, allow_inf_nan=False
    )


class PreparedPaper(TextModel):
    paper_key: NonEmpty
    canonical_paper_id: NonEmpty | None = None
    source_dataset: NonEmpty
    source_record_id: str | int | None = None
    dataset_version: str | None = None
    title: str
    abstract: str | None = None
    model_text: NonEmpty
    original_title: str
    original_abstract: str | None = None
    text_mode: Literal["title_abstract", "title_only", "abstract_only"]
    publication_year: int | None = None
    publication_date: str | None = None
    date_precision: str = "unknown"
    categories: tuple[str, ...] = ()
    language: str | None = None
    authors: tuple[Any, ...] = ()
    external_ids: Any = None
    identity_resolution_status: str = "SOURCE_LOCAL_UNVERIFIED"
    provenance: tuple[dict, ...]
    temporal_assessment: dict
    source_temporal_evidence: tuple[dict, ...] = ()
    text_sha256: NonEmpty


class TopicRecord(FrozenModel):
    topic_id: NonEmpty
    local_topic_id: int
    model_version: NonEmpty
    temporal_snapshot_id: NonEmpty
    keywords: tuple[str, ...]
    provisional_label: NonEmpty
    label_status: Literal["provisional"] = "provisional"
    representative_paper_keys: tuple[str, ...] = ()
    paper_count: Annotated[StrictInt, Field(ge=0)]
    corpus_coverage: Score
    publication_year_distribution: dict[str, int] = Field(default_factory=dict)
    category_distribution: dict[str, int] = Field(default_factory=dict)
    representation_method: NonEmpty
    provenance: tuple[dict, ...] = ()


class TopicAssignment(FrozenModel):
    paper_key: NonEmpty
    canonical_paper_id: NonEmpty | None = None
    topic_id: NonEmpty | None
    topic_model_version: NonEmpty
    assignment_method: NonEmpty
    assignment_score: float | None
    score_semantics: NonEmpty
    outlier_status: bool
    source_provenance: tuple[dict, ...]
    temporal_snapshot_id: NonEmpty
    uncertainty_status: Literal["unvalidated_score", "unassigned"] = "unvalidated_score"

    @model_validator(mode="after")
    def outlier_has_no_topic(self):
        if self.outlier_status != (self.topic_id is None):
            raise ValueError("outliers must remain unassigned")
        if self.topic_id and not self.topic_id.startswith(self.temporal_snapshot_id + ":"):
            raise ValueError("topic ID must belong to the temporal snapshot")
        return self


class PaperTopicEvidence(TextModel):
    paper_key: NonEmpty
    canonical_paper_id: NonEmpty | None = None
    topic_id: NonEmpty
    evidence_label: Label = "UNKNOWN"
    evidence_source: Literal["human", "model", "rule_candidate", "ranking_candidate", "unknown"]
    evidence_text_span: str | None = None
    evidence_start_offset: int | None = None
    evidence_end_offset: int | None = None
    text_field: Literal["original_title", "original_abstract"] | None = None
    evidence_method: NonEmpty
    evidence_score: float | None = None
    score_semantics: NonEmpty
    review_status: Literal["unreviewed", "human_verified", "model_predicted", "insufficient_text"]
    annotator_or_model_id: NonEmpty
    model_version: NonEmpty
    topic_model_version: NonEmpty
    source_provenance: tuple[dict, ...]
    temporal_snapshot_id: NonEmpty
    uncertainty_status: NonEmpty
    candidate_label: Label | None = None

    @model_validator(mode="after")
    def candidates_are_not_validated_labels(self):
        if self.evidence_source in {"rule_candidate", "ranking_candidate", "unknown"}:
            if self.evidence_label != "UNKNOWN":
                raise ValueError("candidate signals cannot establish an investigation label")
        offsets = (self.evidence_start_offset, self.evidence_end_offset)
        if self.evidence_text_span is not None:
            if (
                any(value is None for value in offsets)
                or offsets[0] < 0
                or offsets[1] <= offsets[0]
            ):
                raise ValueError("evidence spans require ordered offsets")
            if offsets[1] - offsets[0] != len(self.evidence_text_span) or self.text_field is None:
                raise ValueError("evidence span length and source field must match offsets")
        elif any(value is not None for value in offsets):
            raise ValueError("offsets require a source text span")
        return self


class InvestigationAnnotation(TextModel):
    paper_key: NonEmpty
    canonical_paper_id: NonEmpty | None = None
    topic_id: NonEmpty
    topic_definition: NonEmpty
    title: str
    abstract: str | None
    label: Label
    annotator_id: NonEmpty
    annotation_timestamp: AwareDatetime
    guideline_version: NonEmpty = "stage05-v1"
    confidence: Score
    text_field: Literal["title", "abstract"] | None = None
    relevant_text_span: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    notes: str = ""
    disagreement_status: Literal["not_assessed", "agreement", "disagreement", "adjudicated"] = (
        "not_assessed"
    )
    split: Literal["train", "dev", "test"] | None = None
    synthetic: bool = False

    @model_validator(mode="after")
    def grounded_annotation(self):
        if self.label != "UNKNOWN" and self.relevant_text_span is None:
            raise ValueError("known annotation labels require a supporting text span")
        if self.relevant_text_span is not None:
            text = self.title if self.text_field == "title" else self.abstract
            if not isinstance(text, str) or self.start_offset is None or self.end_offset is None:
                raise ValueError("annotation requires source text and offsets")
            if not 0 <= self.start_offset < self.end_offset <= len(text):
                raise ValueError("annotation span offsets are invalid")
            if text[self.start_offset : self.end_offset] != self.relevant_text_span:
                raise ValueError("annotation span does not match original source text")
        return self


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
