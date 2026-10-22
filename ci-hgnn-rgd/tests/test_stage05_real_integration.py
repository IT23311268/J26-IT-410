"""Real corpus/encoder/BERTopic tests are opt-in and never substitute fake data."""

import os
from pathlib import Path

import pytest

from ci_hgnn_rgd.nlp.topics.config import Stage05Config
from ci_hgnn_rgd.nlp.topics.data import prepare_corpus

pytestmark = pytest.mark.integration


def test_genuine_stage03_paper_view():
    source = os.environ.get("CI_HGNN_STAGE05_STAGE03_RUN")
    if not source:
        pytest.skip("CI_HGNN_STAGE05_STAGE03_RUN needs genuine Stage 03 scholarly papers")
    config = Stage05Config.model_validate(
        {
            "dataset": {"stage03_run": Path(source)},
            "temporal": {"policy": "EXPLORATORY_NO_TEMPORAL_CLAIM"},
        }
    )
    view = prepare_corpus(config)
    assert view.papers, view.profile
    assert all(not ref.get("synthetic") for p in view.papers for ref in p.provenance)


@pytest.mark.parametrize("backend", ["specter2", "scibert"])
def test_genuine_encoder_and_bertopic(tmp_path, backend):
    source = os.environ.get("CI_HGNN_STAGE05_STAGE03_RUN")
    if not source or os.environ.get("CI_HGNN_STAGE05_REAL_MODELS") != "1":
        pytest.skip(
            "requires genuine papers, cached pinned models, the topics stack and explicit opt-in"
        )
    import numpy as np

    from ci_hgnn_rgd.nlp.topics.embeddings import embed_corpus
    from ci_hgnn_rgd.nlp.topics.models import fit_topics

    config = Stage05Config.model_validate(
        {
            "dataset": {"stage03_run": Path(source), "max_records": 100},
            "temporal": {"policy": "EXPLORATORY_NO_TEMPORAL_CLAIM"},
            "embedding": {"backend": backend},
            "topic_model": {"min_topic_size": 5, "min_df": 1},
        }
    )
    view = prepare_corpus(config)
    embed_corpus(
        view.papers, tmp_path / "vectors.npy", config.embedding, tmp_path / "cache", offline=True
    )
    matrix = np.load(tmp_path / "vectors.npy", allow_pickle=False, mmap_mode="r")
    model = fit_topics(
        view.papers,
        config.topic_model,
        algorithm="bertopic",
        version="integration",
        snapshot=view.temporal_snapshot["snapshot_id"],
        seed=42,
        embeddings=matrix,
    )
    assert len(model.assignments) == len(view.papers)
