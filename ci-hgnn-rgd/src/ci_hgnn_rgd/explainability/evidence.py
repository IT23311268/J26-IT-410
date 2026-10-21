"""Future traceable explanations, never invented expert review."""

from ..schemas import EvidenceReference, ResearchGapCandidate


class EvidenceExplainer:
    def explain(self, candidate: ResearchGapCandidate) -> tuple[EvidenceReference, ...]:
        raise NotImplementedError("Stage 12: source-grounded explanations are not implemented")
