"""Offline numerical baselines, frozen inference, BERTopic glue and quality tests."""

from unittest.mock import patch

import numpy as np
import pytest
from stage05_helpers import configuration, papers

from ci_hgnn_rgd.nlp.topics.contracts import TopicAssignment
from ci_hgnn_rgd.nlp.topics.models import FittedTopics, fit_topics
from ci_hgnn_rgd.nlp.topics.quality import (
    align_topics,
    embedding_neighbors,
    expert_review_template,
    redundancy,
    topic_diversity,
    topic_quality,
)


@pytest.mark.parametrize("algorithm", ["nmf", "lda"])
def test_real_baseline_implementations_fit_and_frozen_reload(tmp_path, algorithm):
    config = configuration(tmp_path)
    corpus = papers(config)
    model = fit_topics(
        corpus,
        config.topic_model,
        algorithm=algorithm,
        version="fixture-run",
        snapshot="fixture-snapshot",
        seed=17,
    )
    assert len(model.catalog) == 2
    assert len(model.assignments) == len(corpus)
    assert all(a.topic_id.startswith("fixture-snapshot:fixture-run:") for a in model.assignments)
    model.save(tmp_path / "model")
    assert not list((tmp_path / "model").glob("*.pkl"))
    loaded = FittedTopics.load(tmp_path / "model")
    with (
        patch("sklearn.decomposition.NMF.fit", side_effect=AssertionError("inference refit")),
        patch(
            "sklearn.decomposition.LatentDirichletAllocation.fit",
            side_effect=AssertionError("refit"),
        ),
    ):
        output = loaded.assign(corpus)
    assert len(output) == len(corpus)
    assert all(a.assignment_method == "frozen_components" for a in output)
    assert all("not_" in a.score_semantics for a in output)
    words = {word for topic in model.catalog for word in topic.keywords}
    assert words and all(topic.representative_paper_keys for topic in model.catalog)
    quality = topic_quality(model, corpus)
    assert quality["coherence"]["reference_documents"] == 8
    assert quality["topic_count"] == 2
    assert 0 <= quality["topic_diversity"] <= 1
    if algorithm == "lda":
        assert model.metadata["lda_input"].startswith("raw_counts")


def test_vectorizer_training_does_not_include_new_paper_vocabulary(tmp_path):
    config = configuration(tmp_path)
    model = fit_topics(
        papers(config), config.topic_model, algorithm="nmf", version="v1", snapshot="s1", seed=1
    )
    unseen = papers(config, count=1)[0].model_dump()
    unseen["model_text"] = "unseenvocabularyword"
    from ci_hgnn_rgd.nlp.topics.contracts import PreparedPaper

    output = model.assign([PreparedPaper.model_validate(unseen)])
    assert "unseenvocabularyword" not in model.metadata["vocabulary"]
    assert output[0].outlier_status and output[0].topic_id is None


class FakeBERTopic:
    """Only an API fixture: no real BERTopic or scientific clustering claim."""

    def fit_transform(self, docs, embeddings):
        assert embeddings.shape == (len(docs), 3)
        return [0, 0, 1, 1, -1], np.array([0.8, 0.7, 0.9, 0.8, 0])

    def get_topic(self, topic):
        return [("physics" if topic == 0 else "biology", 1.0), ("experiment", 0.3)]


def test_bertopic_contract_outliers_and_safe_approximate_prediction(tmp_path):
    config = configuration(tmp_path)
    corpus = papers(config, count=5)
    vectors = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    model = fit_topics(
        corpus,
        config.topic_model,
        algorithm="bertopic",
        version="v1",
        snapshot="s1",
        seed=42,
        embeddings=vectors,
        bertopic_model=FakeBERTopic(),
    )
    assert model.assignments[-1].outlier_status and model.assignments[-1].topic_id is None
    assert model.catalog[0].representation_method == "c-TF-IDF"
    model.save(tmp_path / "bertopic")
    restored = FittedTopics.load(tmp_path / "bertopic")
    assigned = restored.assign(corpus, embeddings=vectors)
    assert assigned[-1].outlier_status
    assert assigned[0].assignment_method == "frozen_centroid_approximation"
    assert assigned[0].score_semantics == "cosine_similarity_not_probability"
    assert topic_quality(model, corpus)["outlier_rate"] == 0.2


@pytest.mark.integration
def test_installed_bertopic_stack_with_synthetic_vectors(tmp_path, monkeypatch):
    """Exercise actual UMAP/HDBSCAN/c-TF-IDF; these vectors are software fixtures only."""
    pytest.importorskip("bertopic")

    def no_download(*args, **kwargs):
        raise AssertionError("precomputed vectors must not trigger model downloads")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    config = configuration(tmp_path)
    corpus = papers(config, count=24)
    rng = np.random.default_rng(42)
    vectors = np.array([[1, 0, 0] if i % 2 == 0 else [0, 1, 0] for i in range(24)])
    vectors = (vectors + rng.normal(0, 0.02, vectors.shape)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    model = fit_topics(
        corpus,
        config.topic_model,
        algorithm="bertopic",
        version="synthetic-installed-stack",
        snapshot="synthetic-snapshot",
        seed=42,
        embeddings=vectors,
    )
    assert model.catalog
    assert len(model.assignments) == len(corpus)
    assert model.metadata["training_clustering"] == "UMAP_HDBSCAN_cTFIDF"
    model.save(tmp_path / "installed-stack")
    reloaded = FittedTopics.load(tmp_path / "installed-stack")
    assert len(reloaded.assign(corpus, embeddings=vectors)) == len(corpus)


def test_snapshot_topic_ids_cannot_be_interchanged(tmp_path):
    config = configuration(tmp_path)
    first = fit_topics(
        papers(config), config.topic_model, algorithm="nmf", version="v1", snapshot="s1", seed=1
    )
    second = fit_topics(
        papers(config), config.topic_model, algorithm="nmf", version="v2", snapshot="s2", seed=1
    )
    assert first.catalog[0].topic_id != second.catalog[0].topic_id
    invalid = first.assignments[0].model_dump()
    invalid["temporal_snapshot_id"] = "different"
    with pytest.raises(ValueError, match="snapshot"):
        TopicAssignment.model_validate(invalid)


def test_topic_stability_aligns_before_comparing_and_reports_unmatched(tmp_path):
    config = configuration(tmp_path)
    model = fit_topics(
        papers(config), config.topic_model, algorithm="nmf", version="v1", snapshot="s1", seed=1
    )
    alignment = align_topics(model.catalog, tuple(reversed(model.catalog)))
    assert alignment["mean_matched_jaccard"] == 1
    assert len(alignment["matches"]) == 2
    assert align_topics((), model.catalog)["status"] == "NOT_EVALUABLE"
    assert align_topics(model.catalog, model.catalog[:1])["unmatched_left"] == 1
    assert topic_diversity(()) is None


def test_quality_review_and_nearest_neighbor_exports_are_unassessed(tmp_path):
    config = configuration(tmp_path)
    corpus = papers(config)
    model = fit_topics(
        corpus, config.topic_model, algorithm="nmf", version="v1", snapshot="s1", seed=1
    )
    form = expert_review_template(model.catalog)
    assert form[0]["interpretability"] is None and form[0]["reviewer_id"] is None
    neighbors = embedding_neighbors(np.eye(8, dtype="float32"), corpus, max_queries=2)
    assert len(neighbors["examples"]) == 2
    assert redundancy(model.catalog)["method"] == "keyword_Jaccard"


def test_versioned_publication_checksums_and_safe_table_roundtrip(tmp_path):
    from ci_hgnn_rgd.nlp.topics.artifacts import RunBuilder, iter_rows, resolve_run, write_rows

    config = configuration(tmp_path)
    builder = RunBuilder(config.output_root, config, kind="fixture", project_root=tmp_path)
    write_rows(builder.pending, "papers", papers(config))
    manifest = builder.publish({"fixture": True})
    path, loaded = resolve_run(config.output_root, manifest["run_id"])
    records = list(iter_rows(path / "papers.parquet"))
    assert records[0]["provenance"][0]["synthetic"] is True
    assert not config.model_root.exists()  # no model was fitted
    second = RunBuilder(config.output_root, config, kind="fixture", project_root=tmp_path)
    assert second.run_id != manifest["run_id"]
    (path / "papers.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        resolve_run(config.output_root, manifest["run_id"])
