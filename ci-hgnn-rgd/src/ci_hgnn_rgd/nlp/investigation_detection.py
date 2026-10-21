"""Investigation requires independent text/source evidence, not citation frequency."""

from ..schemas import Paper, PaperTopicEvidence, Topic


class InvestigationDetector:
    def detect(
        self, papers: tuple[Paper, ...], topics: tuple[Topic, ...]
    ) -> tuple[PaperTopicEvidence, ...]:
        raise NotImplementedError(
            "Stage 06: direct-investigation evidence extraction is not implemented"
        )
