"""Loss-aware staging records. Missing source IDs/dates are never manufactured."""

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from ..schemas import SHA256, ExternalIdentifier


class StagingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceProvenance(StagingModel):
    source_dataset: str
    source_url: str
    dataset_version: str | None
    version_status: Literal["known", "unknown"]
    source_file: str
    locator: str
    record_sha256: SHA256
    inspected_at: AwareDatetime
    acquired_at: AwareDatetime | None = None
    available_at: AwareDatetime | None = None
    synthetic: bool = False


class StagedPaper(StagingModel):
    kind: Literal["paper"] = "paper"
    source_dataset: str
    source_record_id: str | None
    title: str | None
    abstract: str | None
    authors: tuple[str, ...] = ()
    publication_year: int | None = None
    publication_date: date | None = None
    update_date: date | None = None
    doi_raw: str | None = None
    doi_normalized: str | None = None
    arxiv_id_raw: str | None = None
    arxiv_id_normalized: str | None = None
    arxiv_version: str | None = None
    versions: tuple[dict, ...] = ()
    external_ids: tuple[ExternalIdentifier, ...] = ()
    subject_categories: tuple[str, ...] = ()
    source_provenance: SourceProvenance
    metadata_completeness: dict[str, bool] = Field(default_factory=dict)


class StagedCitationContext(StagingModel):
    kind: Literal["citation_context"] = "citation_context"
    source_dataset: str
    source_record_id: str | None
    citation_text: str | None
    original_intent_label: str | None
    normalized_intent_label: str | None
    taxonomy: Literal["scicite"] = "scicite"
    citing_paper_identifier: ExternalIdentifier | None = None
    cited_paper_identifier: ExternalIdentifier | None = None
    identifier_provenance: SourceProvenance
    source_split: str | None
    text_availability: Literal["present", "missing"]
    validation_status: Literal["valid", "invalid"]


class StagedCitationEdge(StagingModel):
    kind: Literal["citation_edge"] = "citation_edge"
    source_dataset: str
    source_node_id: int | str
    target_node_id: int | str
    identifier_namespace: str
    edge_type: str = "cites"
    citation_intent: str | None = None
    publication_year: int | None = None
    provenance: SourceProvenance
