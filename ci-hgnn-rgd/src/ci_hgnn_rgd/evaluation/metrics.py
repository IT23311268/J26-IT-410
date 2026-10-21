"""Future task metrics; no performance can be reported from Stage 01."""

from collections.abc import Mapping

from ..schemas import CitationIntentPrediction, SciCiteClass


def evaluate_intent(
    predictions: tuple[CitationIntentPrediction, ...], ground_truth: Mapping[str, SciCiteClass]
) -> dict[str, float]:
    raise NotImplementedError(
        "Stage 11: evaluation requires verified independent labels and a split protocol"
    )
