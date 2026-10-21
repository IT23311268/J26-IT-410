import hashlib

import pytest
from pydantic import ValidationError

from ci_hgnn_rgd.data.registry import (
    DatasetMetadata,
    DatasetRegistry,
    builtin_registry,
    verify_checksum,
)


def test_catalogue_contains_planned_sources_without_fake_versions():
    registry = builtin_registry()
    assert {d.dataset_id for d in registry.list()} == {
        "scicite",
        "arxiv-metadata",
        "mag-year-h2gb",
        "ogbn-arxiv",
        "openalex",
        "semantic-scholar",
    }
    for item in registry.list():
        assert item.version_id is None
        assert item.sha256 is None
        assert item.provenance == ()
        assert not item.schema_verified
        assert item.licence_notes and item.schema_expectations and item.homepage
        with pytest.raises(ValueError, match="pinned version"):
            item.require_ready()


def test_registry_never_assumes_shared_identifiers_or_contexts():
    registry = builtin_registry()
    scicite = registry.get("scicite")
    ogb = registry.get("ogbn-arxiv")
    assert scicite.identifier_namespace != ogb.identifier_namespace
    assert ogb.citation_contexts == "not_assumed"
    assert registry.get("mag-year-h2gb").schema_verified is False


def test_unknown_dataset_and_version_fail_explicitly():
    for key in (("unknown-source", None), ("scicite", "invented-version")):
        with pytest.raises(KeyError):
            builtin_registry().get(*key)


def test_duplicate_registration_is_rejected():
    registry = builtin_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("scicite"))


def test_explicit_version_registration_preserves_provenance(evidence):
    metadata = DatasetMetadata(
        dataset_id=evidence.dataset_id,
        name="Synthetic software fixture",
        homepage=evidence.source_url,
        description="Not a scholarly dataset",
        licence_notes="Synthetic test bytes only",
        identifier_namespace="synthetic-test",
        schema_expectations=("synthetic-test-field",),
        schema_verified=True,
        version_id=evidence.dataset_version,
        sha256=evidence.payload_sha256,
        provenance=(evidence,),
    )
    registry = DatasetRegistry((metadata,))
    registry.get(evidence.dataset_id, evidence.dataset_version).require_ready()
    with pytest.raises(KeyError):
        registry.get(evidence.dataset_id)  # No silent 'latest' version.
    data = metadata.model_dump()
    data["version_id"] = "different-synthetic-version"
    with pytest.raises(ValidationError):
        DatasetMetadata.model_validate(data)


def test_local_checksum_support_detects_modified_bytes(tmp_path):
    path = tmp_path / "synthetic-fixture.txt"
    payload = b"synthetic software-test fixture"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    verify_checksum(path, expected)
    path.write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checksum(path, expected)
    with pytest.raises(ValidationError):
        verify_checksum(path, "invalid-checksum")
