"""Public Stage 04 citation-intent inference facade.

It intentionally returns the provenance-rich Stage 04 evidence contract instead
of constructing a Stage 01 graph-edge prediction. Citation contexts without
verified graph endpoints remain unlinked records for a later graph stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import CitationContext
from .intent_contracts import CitationIntentEvidence
from .intent_inference import load_predictor, predict_contexts


class CitationIntentClassifier:
    """A loaded local Stage 04 model with no import-time model download.

    Use :meth:`from_run` to load a completed local SciBERT or selected baseline
    run. The classifier cannot create a graph edge or infer direct investigation
    from a citation-intent prediction.
    """

    def __init__(self, predictor: Any, model_info: dict[str, Any], *, batch_size: int = 32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._predictor = predictor
        self._model_info = dict(model_info)
        self._batch_size = batch_size

    @classmethod
    def from_run(
        cls,
        run_dir: Path,
        *,
        model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ) -> CitationIntentClassifier:
        """Load a published local model run without contacting a model hub."""
        predictor, model_info = load_predictor(Path(run_dir), model_name=model_name, device=device)
        return cls(predictor, model_info, batch_size=batch_size)

    def predict(self, contexts: tuple[CitationContext, ...]) -> tuple[CitationIntentEvidence, ...]:
        """Classify one or more Stage 01 context objects in their supplied order."""
        rows = [
            {
                "citation_context_id": context.context_id,
                "source_dataset": "stage01_context",
                "source_record_id": context.context_id,
                "text": context.text,
                "provenance": [item.model_dump(mode="json") for item in context.evidence],
                "identity_resolution_status": "UNRESOLVED",
            }
            for context in contexts
        ]
        return predict_contexts(
            rows,
            self._predictor,
            self._model_info,
            batch_size=self._batch_size,
            low_confidence_threshold=0.5,
        )
