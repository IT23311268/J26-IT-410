"""All records below are SYNTHETIC software-test fixtures, never empirical evidence."""

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ci_hgnn_rgd.config import load_config
from ci_hgnn_rgd.schemas import (
    Author,
    CitationContext,
    CitationEdge,
    Corpus,
    EvidenceReference,
    Paper,
    PaperTopicEvidence,
    Topic,
)

PROJECT = Path(__file__).resolve().parents[1]
OBSERVED = datetime(2024, 1, 1, tzinfo=UTC)
CUTOFF = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.fixture
def evidence():
    payload = b"SYNTHETIC unit-test fixture; not a scholarly source or expert judgment."
    return EvidenceReference(
        reference_id="synthetic:ref1",
        dataset_id="synthetic-fixture",
        dataset_version="test-v1",
        source_record_id="synthetic:record1",
        source_url="https://example.org/synthetic-fixture",
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        locator="synthetic fixture record 1",
        available_at=OBSERVED,
        retrieved_at=OBSERVED,
        evidence_text=payload.decode(),
        evidence_kind="synthetic",
    )


@pytest.fixture
def synthetic_corpus(evidence):
    refs = (evidence,)
    return Corpus(
        papers=(
            Paper(
                paper_id="synthetic:p1",
                title="Synthetic paper one",
                published_on=date(2023, 1, 1),
                author_ids=("synthetic:a1",),
                evidence=refs,
            ),
            Paper(
                paper_id="synthetic:p2",
                title="Synthetic paper two",
                published_on=date(2023, 1, 1),
                evidence=refs,
            ),
        ),
        authors=(Author(author_id="synthetic:a1", name="Synthetic fixture author", evidence=refs),),
        topics=(Topic(topic_id="synthetic:t1", label="Synthetic fixture topic", evidence=refs),),
        citations=(
            CitationEdge(
                edge_id="synthetic:e1",
                citing_paper_id="synthetic:p1",
                cited_paper_id="synthetic:p2",
                context_ids=("synthetic:c1",),
                evidence=refs,
            ),
        ),
        contexts=(
            CitationContext(
                context_id="synthetic:c1",
                edge_id="synthetic:e1",
                text="Synthetic citation context, not copied from a paper.",
                evidence=refs,
            ),
        ),
        paper_topic_evidence=(
            PaperTopicEvidence(
                evidence_id="synthetic:pt1",
                paper_id="synthetic:p1",
                topic_id="synthetic:t1",
                relationship="unknown",
                confidence=None,
                method="source",
                method_version="synthetic-test-v1",
                basis="structured_source",
                produced_at=OBSERVED,
                evidence=refs,
            ),
        ),
    )


@pytest.fixture
def config():
    return load_config(PROJECT / "configs/base.yaml")


def revised(model, **changes):
    """Revalidate updates; Pydantic model_copy(update=...) intentionally skips validation."""
    values = model.model_dump()
    values.update(changes)
    return type(model).model_validate(values)
