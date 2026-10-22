"""Comparable keyword diagnostics; document-level NPMI uses a declared reference corpus."""

import math
from collections import Counter
from itertools import combinations

from ...pipeline import fingerprint


def topic_diversity(topics) -> float | None:
    words = [word for topic in topics for word in topic.keywords]
    return len(set(words)) / len(words) if words else None


def topic_quality(model, papers) -> dict:
    from sklearn.feature_extraction.text import CountVectorizer

    settings = model.metadata["settings"]
    analyzer = CountVectorizer(
        stop_words=settings["keyword_stop_words"], ngram_range=(1, settings["ngram_max"])
    ).build_analyzer()
    vocabulary = {word for topic in model.catalog for word in topic.keywords}
    counts, pairs = Counter(), Counter()
    for paper in papers:
        observed = sorted(set(analyzer(paper.model_text)) & vocabulary)
        counts.update(observed)
        pairs.update(combinations(observed, 2))
    per_topic = {}
    for topic in model.catalog:
        values = []
        for left, right in combinations(sorted(set(topic.keywords)), 2):
            together = pairs[left, right]
            if not together:
                values.append(-1.0)
            elif together == len(papers):
                # The limiting value when both words occur in every document.
                values.append(0.0)
            else:
                pxy = together / len(papers)
                values.append(
                    math.log(pxy / ((counts[left] / len(papers)) * (counts[right] / len(papers))))
                    / -math.log(pxy)
                )
        per_topic[topic.topic_id] = sum(values) / len(values) if values else None
    available = [value for value in per_topic.values() if value is not None]
    outliers = sum(a.outlier_status for a in model.assignments)
    return {
        "topic_count": len(model.catalog),
        "topic_diversity": topic_diversity(model.catalog),
        "coherence": {
            "method": "document_boolean_NPMI_no_smoothing",
            "zero_cooccurrence": -1,
            "universal_cooccurrence": 0,
            "reference_corpus_sha256": fingerprint([(p.paper_key, p.text_sha256) for p in papers]),
            "reference_documents": len(papers),
            "tokenization": {
                "stop_words": settings["keyword_stop_words"],
                "ngram_max": settings["ngram_max"],
            },
            "mean": sum(available) / len(available) if available else None,
            "per_topic": per_topic,
        },
        "topic_sizes": {topic.topic_id: topic.paper_count for topic in model.catalog},
        "outlier_count": outliers,
        "outlier_rate": outliers / len(papers) if papers else None,
        "interpretation": (
            "Keyword/corpus diagnostics only; no expert interpretability assessment has occurred."
        ),
    }


def align_topics(left, right) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not left or not right:
        return {"status": "NOT_EVALUABLE", "reason": "one run has no assigned topics"}
    if max(len(left), len(right)) > 500:
        raise ValueError(
            "stability alignment capped at 500 topics; select a bounded topic comparison"
        )
    overlaps = np.zeros((len(left), len(right)))
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            x, y = set(a.keywords), set(b.keywords)
            overlaps[i, j] = len(x & y) / len(x | y) if x | y else 0
    rows, columns = linear_sum_assignment(-overlaps)
    matches = [
        {
            "left": left[i].topic_id,
            "right": right[j].topic_id,
            "keyword_jaccard": float(overlaps[i, j]),
        }
        for i, j in zip(rows, columns, strict=True)
    ]
    return {
        "status": "MEASURED",
        "method": "Hungarian_max_keyword_Jaccard",
        "matches": matches,
        "mean_matched_jaccard": sum(m["keyword_jaccard"] for m in matches) / len(matches),
        "unmatched_left": len(left) - len(matches),
        "unmatched_right": len(right) - len(matches),
        "potential_splits": sum((overlaps[i] >= 0.25).sum() > 1 for i in range(len(left))).item(),
        "potential_merges": sum(
            (overlaps[:, j] >= 0.25).sum() > 1 for j in range(len(right))
        ).item(),
        "interpretation": (
            "Keyword matches are uncertain cross-run correspondences, not identical topic IDs."
        ),
    }


def redundancy(topics, threshold=0.5) -> dict:
    if len(topics) > 500:
        return {
            "status": "NOT_EVALUABLE",
            "reason": "more than 500 topics; bounded comparison required",
        }
    pairs = []
    for left, right in combinations(topics, 2):
        a, b = set(left.keywords), set(right.keywords)
        similarity = len(a & b) / len(a | b) if a | b else 0
        if similarity >= threshold:
            pairs.append(
                {"left": left.topic_id, "right": right.topic_id, "keyword_jaccard": similarity}
            )
    return {"method": "keyword_Jaccard", "threshold": threshold, "pairs": pairs}


def embedding_neighbors(matrix, papers, *, max_queries=5, neighbors=3):
    import numpy as np

    examples = []
    for i in range(min(max_queries, len(papers))):
        candidates = []
        query = matrix[i] / max(float(np.linalg.norm(matrix[i])), 1e-12)
        for start in range(0, len(papers), 256):
            block = matrix[start : start + 256]
            norms = np.maximum(np.linalg.norm(block, axis=1), 1e-12)
            values = (block @ query) / norms
            candidates.extend(
                (float(value), start + j) for j, value in enumerate(values) if start + j != i
            )
            candidates = sorted(candidates, reverse=True)[:neighbors]
        examples.append(
            {
                "paper_key": papers[i].paper_key,
                "neighbors": [
                    {
                        "paper_key": papers[j].paper_key,
                        "cosine_similarity": value,
                        "shared_category": bool(
                            set(papers[i].categories) & set(papers[j].categories)
                        ),
                    }
                    for value, j in candidates
                ],
            }
        )
    return {
        "examples": examples,
        "interpretation": (
            "Category agreement is a descriptive diagnostic, not validated embedding quality."
        ),
    }


def expert_review_template(topics):
    return [
        {
            "topic_id": topic.topic_id,
            "provisional_label": topic.provisional_label,
            "representative_paper_keys": topic.representative_paper_keys,
            "reviewer_id": None,
            "interpretability": None,
            "semantic_coherence": None,
            "specificity": None,
            "representative_paper_relevance": None,
            "notes": None,
        }
        for topic in topics
    ]
