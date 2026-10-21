"""Future heterogeneous graph representation-learning interface."""

from pathlib import Path

from ..config import GNNSettings
from ..schemas import GraphManifest


class HeteroGNN:
    def fit(self, graph: GraphManifest, settings: GNNSettings) -> Path:
        raise NotImplementedError(
            "Stage 08: no GNN implementation, fitted weights, or embeddings exist"
        )
