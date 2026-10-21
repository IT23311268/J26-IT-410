"""High attention/low investigation is a hypothesis requiring independent validation."""

from ..schemas import PaperTopicEvidence, ResearchGapCandidate, Topic


class CitationAsymmetryScorer:
    def score(self, topic: Topic, evidence: tuple[PaperTopicEvidence, ...]) -> ResearchGapCandidate:
        raise NotImplementedError(
            "Stage 10: citation-asymmetry scoring is not implemented or validated"
        )
