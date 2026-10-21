"""Loss-aware Stage 03 staging contracts, separate from resolved graph schemas.

Missing scholarly identities remain null. These records permit additional audit fields
so source adapters can preserve observed metadata without changing Stage 01 contracts.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CanonicalRecord(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, allow_inf_nan=False)

    record_key: str
    source_dataset: str
    dataset_version: str | None = None
    source_record_id: str | int | None = None
    raw_values: dict[str, Any] = Field(default_factory=dict)
    normalized_fields: dict[str, Any] = Field(default_factory=dict)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    transformations: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    quarantine: bool = False
    source_split: str | None = None


class CanonicalPaper(CanonicalRecord):
    kind: Literal["paper"] = "paper"
    canonical_paper_id: str | None = None
    title: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    doi: str | None = None
    arxiv_id: str | None = None
    identifiers: list[dict[str, Any]] = Field(default_factory=list)
    external_ids: list[dict[str, Any]] | dict[str, Any] = Field(default_factory=dict)
    source_record_ids: list[dict[str, Any]] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    completeness: dict[str, bool] = Field(default_factory=dict)
    conflict_flags: list[str] = Field(default_factory=list)


class CanonicalAuthor(CanonicalRecord):
    kind: Literal["author"] = "author"
    canonical_author_id: str | None = None
    display_name: str | None = None
    source_author_ids: list[dict[str, Any]] = Field(default_factory=list)
    verified_external_ids: list[dict[str, Any]] = Field(default_factory=list)
    affiliation_data: list[Any] = Field(default_factory=list)
    identity_status: str = "SOURCE_LOCAL_UNVERIFIED"


class CanonicalCitation(CanonicalRecord):
    kind: Literal["citation_edge", "relation"] = "citation_edge"
    source_paper_id: str | None = None
    target_paper_id: str | None = None
    original_source_id: str | int | None = None
    original_target_id: str | int | None = None
    identifier_namespace: str | None = None
    source_identifier: dict[str, Any] | None = None
    target_identifier: dict[str, Any] | None = None
    edge_provenance: list[dict[str, Any]] = Field(default_factory=list)
    citation_context_ids: list[str] = Field(default_factory=list)
    citation_intent_status: str = "UNOBSERVED"
    temporal_status: str = "NOT_ASSESSED"


class CanonicalCitationContext(CanonicalRecord):
    kind: Literal["citation_context"] = "citation_context"
    context_id: str | None = None
    citing_paper_id: str | None = None
    cited_paper_id: str | None = None
    original_context: str | None = None
    normalized_context: str | None = None
    original_intent_label: str | None = None
    normalized_intent_label: str | None = None
    citing_identifier: dict[str, Any] | None = None
    cited_identifier: dict[str, Any] | None = None
    identity_resolution_status: str = "UNRESOLVED"


class CanonicalField(CanonicalRecord):
    kind: Literal["source_field"] = "source_field"
    source_field_id: str | int | None = None
    field_name: str | None = None
    field_taxonomy: str | None = None


def validate_canonical(record: dict[str, Any]) -> CanonicalRecord:
    """Validate a staging envelope without asserting an identity or graph eligibility."""
    model = {
        "paper": CanonicalPaper,
        "author": CanonicalAuthor,
        "citation_context": CanonicalCitationContext,
        "citation_edge": CanonicalCitation,
        "relation": CanonicalCitation,
        "source_field": CanonicalField,
    }.get(record.get("kind"), CanonicalRecord)
    return model.model_validate(record)
