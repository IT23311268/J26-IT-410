"""Train-only topic fitting boundary, independent of the eventual NLP backend."""

from datetime import datetime

from ..schemas import Paper, Topic


class TopicDiscoverer:
    def __init__(self, *, settings=None, algorithm="nmf"):
        from .topics.config import ModelSettings

        self.settings = settings or ModelSettings()
        self.algorithm = algorithm

    def fit(self, papers: tuple[Paper, ...], training_cutoff: datetime) -> tuple[Topic, ...]:
        from datetime import UTC

        from ..pipeline import fingerprint
        from .topics.config import Stage05Config
        from .topics.data import prepare_paper
        from .topics.models import fit_topics

        if not papers or training_cutoff is None:
            raise ValueError("topic fitting needs papers and an explicit historical cutoff")
        from ..data.validation import validate_corpus
        from ..schemas import Corpus

        validate_corpus(Corpus(papers=papers), training_cutoff)
        config = Stage05Config()
        prepared = tuple(
            prepare_paper(
                {
                    "canonical_paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": p.abstract,
                    "publication_date": p.published_on.isoformat() if p.published_on else None,
                    "source_dataset": "stage01",
                    "source_record_id": p.paper_id,
                    "provenance": [r.model_dump(mode="json") for r in p.evidence],
                    "available_at": max(
                        (r.available_at for r in p.evidence if r.available_at), default=None
                    ),
                },
                config,
                policy="STRICT_POINT_IN_TIME",
                cutoff=training_cutoff,
            )
            for p in papers
        )
        version = (
            "stage01-adapter:" + fingerprint([(p.paper_key, p.text_sha256) for p in prepared])[:16]
        )
        fitted = fit_topics(
            prepared,
            self.settings,
            algorithm=self.algorithm,
            version=version,
            snapshot="snapshot:" + fingerprint(str(training_cutoff))[:16],
            seed=42,
        )
        source_refs = tuple(ref for paper in papers for ref in paper.evidence)
        return tuple(
            Topic(
                topic_id=t.topic_id,
                label=t.provisional_label,
                model_version=version,
                training_data_cutoff=training_cutoff,
                produced_at=datetime.now(UTC),
                evidence=source_refs,
            )
            for t in fitted.catalog
        )
