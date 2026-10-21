"""Future research report interface."""

from pathlib import Path

from ..schemas import ResearchGapCandidate


class ReportGenerator:
    def generate(self, candidates: tuple[ResearchGapCandidate, ...], destination: Path) -> Path:
        raise NotImplementedError(
            "Stage 12: report generation requires actual evidence and evaluations"
        )
