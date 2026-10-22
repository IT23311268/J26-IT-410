"""Investigation requires independent text/source evidence, not citation frequency."""

from ..schemas import Paper, PaperTopicEvidence, Topic


class InvestigationDetector:
    def detect(
        self, papers: tuple[Paper, ...], topics: tuple[Topic, ...]
    ) -> tuple[PaperTopicEvidence, ...]:
        from datetime import UTC, datetime

        from ..pipeline import fingerprint

        if not papers or not topics:
            raise ValueError(
                "investigation evidence requires papers and independently defined topics"
            )
        # The Stage 01 facade has no annotation/model provenance argument. It
        # therefore preserves UNKNOWN; the Stage 05 API exposes grounded review
        # candidates and explicitly trained/human-verified evidence separately.
        return tuple(
            PaperTopicEvidence(
                evidence_id="unknown:" + fingerprint([paper.paper_id, topic.topic_id]),
                paper_id=paper.paper_id,
                topic_id=topic.topic_id,
                relationship="unknown",
                confidence=None,
                method="source",
                method_version="stage05-unknown-v1",
                basis="textual_evidence",
                produced_at=datetime.now(UTC),
                evidence=paper.evidence,
            )
            for paper in papers
            for topic in topics
        )
