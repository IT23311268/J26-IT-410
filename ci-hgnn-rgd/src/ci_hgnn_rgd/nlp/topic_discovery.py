"""Train-only topic fitting boundary, independent of the eventual NLP backend."""

from datetime import datetime

from ..schemas import Paper, Topic


class TopicDiscoverer:
    def fit(self, papers: tuple[Paper, ...], training_cutoff: datetime) -> tuple[Topic, ...]:
        raise NotImplementedError(
            "Stage 05: topic discovery and train-only fitting are not implemented"
        )
