"""Version-aware dataset catalogue, not a downloader or an assertion of acquired data."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import HttpUrl, TypeAdapter, model_validator

from ..schemas import SHA256, EvidenceReference, FrozenModel, NonEmpty
from .manifests import DatasetState, DatasetStatus, InspectionRegistry

__all__ = [
    "DatasetMetadata",
    "DatasetRegistry",
    "DatasetState",
    "DatasetStatus",
    "InspectionRegistry",
    "builtin_registry",
    "verify_checksum",
]


class DatasetMetadata(FrozenModel):
    dataset_id: NonEmpty
    name: NonEmpty
    homepage: HttpUrl
    description: NonEmpty
    licence_notes: NonEmpty
    identifier_namespace: NonEmpty
    schema_expectations: tuple[NonEmpty, ...]
    schema_verified: bool = False
    citation_contexts: Literal["expected", "not_assumed", "unverified"] = "unverified"
    version_id: NonEmpty | None = None
    sha256: SHA256 | None = None
    provenance: tuple[EvidenceReference, ...] = ()

    @model_validator(mode="after")
    def version_integrity(self):
        if self.sha256 is not None and self.version_id is None:
            raise ValueError("a dataset checksum must identify an explicit version")
        for reference in self.provenance:
            if (
                reference.dataset_id != self.dataset_id
                or reference.dataset_version != self.version_id
            ):
                raise ValueError("dataset provenance must match dataset identifier and version")
        return self

    def require_ready(self) -> None:
        if (
            not self.version_id
            or not self.sha256
            or not self.provenance
            or not self.schema_verified
        ):
            raise ValueError(
                "dataset requires a pinned version, checksum, provenance, and verified schema"
            )


class DatasetRegistry:
    def __init__(self, datasets: tuple[DatasetMetadata, ...] = ()):
        self._datasets: dict[tuple[str, str | None], DatasetMetadata] = {}
        for dataset in datasets:
            self.register(dataset)

    def register(self, dataset: DatasetMetadata) -> None:
        key = (dataset.dataset_id, dataset.version_id)
        if key in self._datasets:
            raise ValueError(f"dataset/version already registered: {key}")
        self._datasets[key] = dataset

    def get(self, dataset_id: str, version_id: str | None = None) -> DatasetMetadata:
        try:
            return self._datasets[(dataset_id, version_id)]
        except KeyError:
            raise KeyError(f"unknown dataset/version: {dataset_id}, {version_id}") from None

    def list(self) -> tuple[DatasetMetadata, ...]:
        return tuple(
            sorted(
                self._datasets.values(), key=lambda item: (item.dataset_id, item.version_id or "")
            )
        )


def verify_checksum(path: str | Path, expected: str) -> None:
    """Check caller-supplied local bytes against an independently obtained SHA-256."""
    expected = TypeAdapter(SHA256).validate_python(expected)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {Path(path).name}")


def builtin_registry() -> DatasetRegistry:
    """Only planned metadata: versions, archive hashes, and snapshot schemas are unknown."""
    entries = (
        dict(
            dataset_id="scicite",
            name="SciCite",
            homepage="https://github.com/allenai/scicite",
            description="Citation-context classification source; only Background, Method, Result.",
            licence_notes=(
                "Repository declares Apache-2.0; verify archive-specific terms and attribution."
            ),
            identifier_namespace="scicite-source-paper-id",
            schema_expectations=(
                "citation context text (verify string/context key)",
                "label",
                "citingPaperId",
                "citedPaperId",
            ),
            citation_contexts="expected",
        ),
        dict(
            dataset_id="arxiv-metadata",
            name="Cornell arXiv metadata",
            homepage="https://info.arxiv.org/help/bulk_data.html",
            description=(
                "Planned scholarly titles, abstracts, authors, version dates and categories."
            ),
            licence_notes="Verify selected metadata snapshot terms; article licences vary.",
            identifier_namespace="arxiv",
            schema_expectations=(
                "source identifier",
                "title",
                "abstract",
                "authors",
                "dates",
                "categories",
            ),
            citation_contexts="not_assumed",
        ),
        dict(
            dataset_id="mag-year-h2gb",
            name="MAG-Year / H2GB (schema pending)",
            homepage="https://github.com/junhongmit/H2GB",
            description="Planned heterogeneous graph source; exact variant and schema unverified.",
            licence_notes="Verify dataset-specific terms separately from H2GB library licensing.",
            identifier_namespace="unverified-mag-year-source-id",
            schema_expectations=(
                "verify node types",
                "verify relation types",
                "verify temporal fields",
                "verify identifier mappings",
            ),
        ),
        dict(
            dataset_id="ogbn-arxiv",
            name="OGBN-ArXiv",
            homepage="https://ogb.stanford.edu/docs/nodeprop/",
            description=(
                "Homogeneous paper citation graph for prototypes; no intent labels assumed."
            ),
            licence_notes=(
                "Official dataset page lists ODC-BY; retain attribution and verify release terms."
            ),
            identifier_namespace="ogbn-arxiv-node-index",
            schema_expectations=(
                "paper nodes",
                "directed citations",
                "node features",
                "publication year",
                "official splits",
            ),
            citation_contexts="not_assumed",
        ),
        dict(
            dataset_id="openalex",
            name="OpenAlex (optional)",
            homepage="https://openalex.org/",
            description="Optional metadata/identifier resolution; no merge is assumed.",
            licence_notes="Review chosen API/snapshot terms and access policy before acquisition.",
            identifier_namespace="openalex",
            schema_expectations=(
                "verify work IDs",
                "verify identifiers",
                "verify version timestamps",
            ),
        ),
        dict(
            dataset_id="semantic-scholar",
            name="Semantic Scholar (optional)",
            homepage="https://api.semanticscholar.org/api-docs/",
            description=(
                "Optional metadata/identifier resolution; no cross-source identity assumed."
            ),
            licence_notes="Review API and dataset release terms before acquisition.",
            identifier_namespace="semantic-scholar",
            schema_expectations=(
                "verify paper IDs",
                "verify external identifiers",
                "verify API version",
            ),
        ),
    )
    return DatasetRegistry(tuple(DatasetMetadata(**entry) for entry in entries))
