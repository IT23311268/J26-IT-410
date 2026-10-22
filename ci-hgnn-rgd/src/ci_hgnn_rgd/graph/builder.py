"""Backend-neutral graph builder boundary; no graph is constructed in Stage 01."""

from datetime import datetime

from ..config import GraphSettings
from ..schemas import Corpus, GraphManifest


class GraphBuilder:
    def build(self, corpus: Corpus, settings: GraphSettings, cutoff: datetime) -> GraphManifest:
        raise NotImplementedError(
            "The frozen Corpus graph boundary is not executable; use Stage06GraphBuilder "
            "with the explicit Stage06Config API"
        )


class Stage06GraphBuilder:
    """Explicit graph pipeline; the frozen Stage 01 Corpus API remains unchanged."""

    def build(self, config, *, graph_version=None, dry_run=False):
        from .workflow import build_graph

        return build_graph(config, graph_version=graph_version, dry_run=dry_run)
