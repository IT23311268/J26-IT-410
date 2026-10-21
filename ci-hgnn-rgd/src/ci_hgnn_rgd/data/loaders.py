"""Stage 02 staging streams and inspections; strict resolved Corpus creation stays deferred."""

from pathlib import Path

from ..schemas import Corpus
from .registry import DatasetMetadata


class DatasetLoader:
    def iter_staging(self, dataset: DatasetMetadata, local_path: Path, *, synthetic=False):
        from .sources import arxiv, scicite
        from .streaming import events

        adapters = {"scicite": scicite, "arxiv-metadata": arxiv}
        if dataset.dataset_id not in adapters:
            raise ValueError("graph sources use their official source-specific adapters")
        for event in events(local_path, dataset.dataset_id):
            if event.record is None:
                yield {
                    "record": None,
                    "issues": [event.error],
                    "source": event.source,
                    "line": event.line,
                }
            else:
                record, issues = adapters[dataset.dataset_id].adapt(
                    event, version=dataset.version_id, synthetic=synthetic
                )
                yield {
                    "record": record,
                    "issues": ([event.error] if event.error else []) + issues,
                    "source": event.source,
                    "line": event.line,
                }

    def inspect(self, dataset: DatasetMetadata, local_path: Path, output_dir: Path, **options):
        from .inspection import inspect_text

        return inspect_text(dataset.dataset_id, local_path, output_dir, **options)

    def load(self, dataset: DatasetMetadata, local_path: Path) -> Corpus:
        raise NotImplementedError(
            "Strict resolved Corpus construction requires Stage 03 entity resolution; use "
            "iter_staging or inspect"
        )
