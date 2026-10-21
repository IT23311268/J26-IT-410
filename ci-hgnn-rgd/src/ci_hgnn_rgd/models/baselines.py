"""Future common-split baseline interface; no scores are synthesized."""

from ..schemas import Corpus


class BaselineRunner:
    def evaluate(self, corpus: Corpus) -> dict[str, float]:
        raise NotImplementedError("Stage 09: baselines and ablation evaluation are not implemented")
