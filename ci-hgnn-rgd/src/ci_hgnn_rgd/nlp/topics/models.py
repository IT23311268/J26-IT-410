"""Train-only topic models with non-executable component exports and frozen inference."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ...data.manifests import atomic_json
from .contracts import TopicAssignment, TopicRecord


def _vectorizer(config, *, tfidf):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    cls = TfidfVectorizer if tfidf else CountVectorizer
    return cls(
        max_features=config.max_features,
        min_df=config.min_df,
        stop_words=config.keyword_stop_words,
        ngram_range=(1, config.ngram_max),
    )


def _topic_id(snapshot: str, version: str, local: int):
    return f"{snapshot}:{version}:topic:{local}"


@dataclass
class FittedTopics:
    algorithm: str
    version: str
    snapshot: str
    catalog: tuple[TopicRecord, ...]
    assignments: tuple[TopicAssignment, ...]
    metadata: dict
    arrays: dict

    def save(self, path: Path):
        import numpy as np

        path.mkdir(parents=True, exist_ok=False)
        atomic_json(path / "components.json", self.metadata)
        np.savez_compressed(path / "components.npz", **self.arrays)
        atomic_json(
            path / "topic_catalog.json", [row.model_dump(mode="json") for row in self.catalog]
        )

    @classmethod
    def load(cls, path: Path):
        import numpy as np

        metadata = json.loads((path / "components.json").read_text(encoding="utf-8"))
        if metadata["algorithm"] not in {"nmf", "lda", "bertopic"}:
            raise ValueError("unsupported safe topic artifact")
        with np.load(path / "components.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if any(not np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("non-finite topic model components")
        catalog = tuple(
            TopicRecord.model_validate(row)
            for row in json.loads((path / "topic_catalog.json").read_text(encoding="utf-8"))
        )
        return cls(
            metadata["algorithm"],
            metadata["version"],
            metadata["snapshot"],
            catalog,
            (),
            metadata,
            arrays,
        )

    def assign(self, papers, *, embeddings=None):
        """Freeze keywords and components; never call fit or update c-TF-IDF here."""
        import numpy as np

        if self.algorithm == "bertopic":
            if embeddings is None or embeddings.shape != (len(papers), self.metadata["dimension"]):
                raise ValueError("BERTopic assignment requires aligned scientific embeddings")
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if not np.isfinite(embeddings).all() or (norm <= 1e-12).any():
                raise ValueError("invalid inference embedding vectors")
            vectors = embeddings / norm
            ids = self.metadata["local_topic_ids"]
            outputs = []
            # Per-paper scores avoid a corpus-by-corpus or unbounded similarity matrix.
            for paper, vector in zip(papers, vectors, strict=True):
                scores = self.arrays["centroids"] @ vector
                offset = int(scores.argmax()) if len(scores) else None
                accepted = (
                    offset is not None
                    and scores[offset] >= self.arrays["rejection_thresholds"][offset]
                )
                outputs.append(
                    self._assignment(
                        paper,
                        ids[offset] if accepted else -1,
                        float(scores[offset]) if offset is not None else None,
                        "frozen_centroid_approximation",
                        "cosine_similarity_not_probability",
                    )
                )
            return tuple(outputs)
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

        vocabulary = self.metadata["vocabulary"]
        vectorizer = CountVectorizer(
            vocabulary=vocabulary,
            ngram_range=tuple(self.metadata["ngram_range"]),
            stop_words=self.metadata["stop_words"],
        )
        matrix = vectorizer.transform([p.model_text for p in papers])
        if self.algorithm == "nmf":
            transformer = TfidfTransformer()
            transformer.idf_ = self.arrays["idf"]
            matrix = transformer.transform(matrix)
        output = []
        for paper, row in zip(papers, matrix, strict=True):
            values = row.toarray().ravel()
            if not values.any():
                output.append(
                    self._assignment(paper, -1, None, "frozen_components", "no_known_vocabulary")
                )
                continue
            if self.algorithm == "nmf":
                from scipy.optimize import nnls

                weights, _ = nnls(self.arrays["components"].T, values)
                semantics = "nonnegative_factor_weight_not_probability"
            else:
                from scipy.special import digamma

                components = self.arrays["components"]
                exp_beta = np.exp(digamma(components) - digamma(components.sum(axis=1))[:, None])
                active = np.flatnonzero(values)
                gamma = np.ones(len(components))
                for _ in range(100):
                    exp_theta = np.exp(digamma(gamma) - digamma(gamma.sum()))
                    norm = exp_theta @ exp_beta[:, active] + 1e-100
                    revised = self.metadata["doc_topic_prior"] + exp_theta * (
                        exp_beta[:, active] @ (values[active] / norm)
                    )
                    if np.mean(abs(revised - gamma)) < 1e-6:
                        gamma = revised
                        break
                    gamma = revised
                weights = gamma / gamma.sum()
                semantics = "variational_topic_weight_not_calibrated_membership"
            local = int(weights.argmax()) if weights.sum() else -1
            output.append(
                self._assignment(paper, local, float(weights.max()), "frozen_components", semantics)
            )
        return tuple(output)

    def _assignment(self, paper, local, score, method, semantics):
        return TopicAssignment(
            paper_key=paper.paper_key,
            canonical_paper_id=paper.canonical_paper_id,
            topic_id=_topic_id(self.snapshot, self.version, local) if local >= 0 else None,
            topic_model_version=self.version,
            assignment_method=method,
            assignment_score=score,
            score_semantics=semantics,
            outlier_status=local < 0,
            source_provenance=paper.provenance,
            temporal_snapshot_id=self.snapshot,
            uncertainty_status="unassigned" if local < 0 else "unvalidated_score",
        )


def fit_topics(
    papers,
    settings,
    *,
    algorithm: str,
    version: str,
    snapshot: str,
    seed: int,
    embeddings=None,
    bertopic_model=None,
) -> FittedTopics:
    import numpy as np

    if len(papers) < 3:
        raise ValueError("topic discovery requires at least three eligible papers")
    # Keyword preprocessing is independent of transformer separator/tokenization.
    documents = [paper.title + " " + (paper.abstract or "") for paper in papers]
    metadata = {
        "algorithm": algorithm,
        "version": version,
        "snapshot": snapshot,
        "settings": settings.model_dump(mode="json"),
        "seed": seed,
        "training_paper_keys": [paper.paper_key for paper in papers],
        "serialization": (
            "JSON and numeric NPZ; no pickle; BERTopic inference is an "
            "explicit frozen centroid approximation"
        ),
    }
    if algorithm in {"nmf", "lda"}:
        from sklearn.decomposition import NMF, LatentDirichletAllocation

        vectorizer = _vectorizer(settings, tfidf=algorithm == "nmf")
        features = vectorizer.fit_transform(documents)
        if settings.n_topics > min(features.shape):
            raise ValueError(
                "n_topics exceeds the corpus/vocabulary size; configure a smaller value"
            )
        estimator = (
            NMF(
                n_components=settings.n_topics,
                random_state=seed,
                init="nndsvda",
                max_iter=settings.max_iter,
            )
            if algorithm == "nmf"
            else LatentDirichletAllocation(
                n_components=settings.n_topics,
                random_state=seed,
                learning_method="batch",
                max_iter=settings.max_iter,
                n_jobs=1,
            )
        )
        weights = estimator.fit_transform(features)
        assignments = [
            int(row.argmax()) if features[index].nnz else -1 for index, row in enumerate(weights)
        ]
        scores = [float(row.max()) for row in weights]
        vocabulary = {key: int(value) for key, value in vectorizer.vocabulary_.items()}
        terms = vectorizer.get_feature_names_out()
        keywords = {
            local: tuple(
                terms[index] for index in component.argsort()[::-1][: settings.top_n_words]
            )
            for local, component in enumerate(estimator.components_)
        }
        arrays = {"components": estimator.components_.astype("float64")}
        if algorithm == "nmf":
            arrays["idf"] = vectorizer.idf_.astype("float64")
        metadata.update(
            vocabulary=vocabulary,
            ngram_range=vectorizer.ngram_range,
            stop_words=settings.keyword_stop_words,
            doc_topic_prior=float(estimator.doc_topic_prior_) if algorithm == "lda" else None,
            lda_input="raw_counts; TF-IDF is unsuitable for the generative count likelihood"
            if algorithm == "lda"
            else None,
        )
        semantics = (
            "nonnegative_factor_weight_not_probability"
            if algorithm == "nmf"
            else "variational_topic_weight_not_calibrated_membership"
        )
    elif algorithm == "bertopic":
        if embeddings is None or len(embeddings) != len(papers):
            raise ValueError("BERTopic requires aligned externally generated scientific embeddings")
        if bertopic_model is None:
            try:
                from bertopic import BERTopic
                from bertopic.backend import BaseEmbedder
                from hdbscan import HDBSCAN
                from umap import UMAP
            except ImportError as error:
                raise RuntimeError(
                    "BERTopic needs the optional topics extra (BERTopic, UMAP, HDBSCAN); "
                    "no alternative was silently fitted"
                ) from error
            if len(papers) < settings.min_topic_size:
                raise ValueError("corpus is smaller than min_topic_size")
            reducer = UMAP(
                n_neighbors=min(settings.umap_neighbors, len(papers) - 1),
                n_components=min(settings.umap_dimensions, len(papers) - 2),
                min_dist=settings.umap_min_dist,
                metric=settings.umap_metric,
                random_state=seed,
                transform_seed=seed,
                n_jobs=1,
                init="random",
            )
            clusterer = HDBSCAN(
                min_cluster_size=settings.min_topic_size,
                min_samples=settings.hdbscan_min_samples,
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
            )
            bertopic_model = BERTopic(
                embedding_model=BaseEmbedder(),
                umap_model=reducer,
                hdbscan_model=clusterer,
                vectorizer_model=_vectorizer(settings, tfidf=False),
                min_topic_size=settings.min_topic_size,
                top_n_words=settings.top_n_words,
                calculate_probabilities=False,
                verbose=False,
            )
        labels, strengths = bertopic_model.fit_transform(documents, embeddings=embeddings)
        assignments = [int(value) for value in labels]
        scores = (
            [float(value) for value in strengths]
            if strengths is not None and np.ndim(strengths) == 1
            else [None] * len(papers)
        )
        ids = sorted(set(assignments) - {-1})
        keywords = {
            local: tuple(
                word
                for word, score in (bertopic_model.get_topic(local) or [])
                if word and score > 0
            )[: settings.top_n_words]
            for local in ids
        }
        centroids, thresholds = [], []
        for local in ids:
            vectors = np.asarray(embeddings)[np.asarray(assignments) == local].astype("float32")
            centroid = vectors.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            similarities = (
                vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            ) @ centroid
            centroids.append(centroid)
            thresholds.append(
                float(np.quantile(similarities, settings.centroid_rejection_quantile))
            )
        arrays = {
            "centroids": np.asarray(centroids, dtype="float32").reshape(
                len(ids), embeddings.shape[1]
            ),
            "rejection_thresholds": np.asarray(thresholds, dtype="float32"),
        }
        metadata.update(
            dimension=embeddings.shape[1],
            local_topic_ids=ids,
            inference="frozen_centroid_with_train_only_rejection_threshold",
            training_clustering="UMAP_HDBSCAN_cTFIDF",
        )
        semantics = "hdbscan_membership_strength_not_calibrated_probability"
    else:
        raise ValueError("unsupported topic algorithm")
    fitted = FittedTopics(algorithm, version, snapshot, (), (), metadata, arrays)
    fitted.assignments = tuple(
        fitted._assignment(paper, local, score, algorithm + "_training_fit", semantics)
        for paper, local, score in zip(papers, assignments, scores, strict=True)
    )
    catalog = []
    for local in sorted(keywords):
        indices = [i for i, label in enumerate(assignments) if label == local]
        pool = indices or list(range(len(papers)))
        ranked = sorted(
            pool,
            key=lambda i: (
                -float(weights[i, local]) if algorithm in {"nmf", "lda"} else -(scores[i] or 0),
                papers[i].paper_key,
            ),
        )
        years = Counter(
            str(papers[i].publication_year) if papers[i].publication_year else "unknown"
            for i in indices
        )
        categories = Counter(category for i in indices for category in papers[i].categories)
        words = keywords[local]
        catalog.append(
            TopicRecord(
                topic_id=_topic_id(snapshot, version, local),
                local_topic_id=local,
                model_version=version,
                temporal_snapshot_id=snapshot,
                keywords=words,
                provisional_label=" / ".join(words[:3]) or f"Provisional topic {local}",
                representative_paper_keys=tuple(
                    papers[i].paper_key for i in ranked[: settings.representative_papers]
                ),
                paper_count=len(indices),
                corpus_coverage=len(indices) / len(papers),
                publication_year_distribution=dict(years),
                category_distribution=dict(categories),
                representation_method="c-TF-IDF"
                if algorithm == "bertopic"
                else algorithm + "_component_weights",
                provenance=tuple(
                    ref
                    for i in ranked[: settings.representative_papers]
                    for ref in papers[i].provenance
                ),
            )
        )
    fitted.catalog = tuple(catalog)
    return fitted
