"""Backend-neutral graph builder boundary; no graph is constructed in Stage 01."""

from datetime import datetime

from ..config import GraphSettings
from ..schemas import Corpus, GraphManifest


class GraphBuilder:
    def build(self, corpus: Corpus, settings: GraphSettings, cutoff: datetime) -> GraphManifest:
        raise NotImplementedError(
            "Stage 07: evidence-preserving heterogeneous graph construction is not implemented"
        )
