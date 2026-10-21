"""Stage 02 orchestration using existing project config, catalogue and logging boundaries."""

import json
from pathlib import Path

from .acquisition import acquire
from .download_manager import AcquisitionBlocked
from .inspection import inspect_text
from .manifests import DatasetState, DatasetStatus, InspectionRegistry, atomic_json, now
from .registry import builtin_registry
from .sources.mag_year import MAGYearAdapter
from .sources.ogbn_arxiv import OGBNArxivAdapter

ALIASES = {"arxiv": "arxiv-metadata", "mag_year": "mag-year-h2gb", "ogbn_arxiv": "ogbn-arxiv"}
ORGANIZATIONS = {
    "scicite": "Allen Institute for AI",
    "arxiv-metadata": "Cornell University / arXiv",
    "mag-year-h2gb": "H2GB maintainers; supplied SpaceLearner URL requires verification",
    "ogbn-arxiv": "Open Graph Benchmark",
    "openalex": "OpenAlex",
    "semantic-scholar": "Semantic Scholar / AI2",
}


class DataService:
    def __init__(self, config, *, data_dir=None, output_dir=None):
        self.config = config
        self.data_dir = Path(data_dir or config.paths.raw).resolve()
        self.output_dir = Path(output_dir or config.paths.reports).resolve()
        self.registry = InspectionRegistry(self.output_dir / "dataset_registry.json")
        for metadata in builtin_registry().list():
            if metadata.dataset_id not in self.registry.states:
                dataset = metadata.dataset_id
                source = str(metadata.homepage)
                if dataset == "arxiv-metadata":
                    source = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
                if dataset == "mag-year-h2gb":
                    source = "https://github.com/SpaceLearner/H2GB"
                state = DatasetState(
                    dataset_id=dataset,
                    source_url=source,
                    source_organization=ORGANIZATIONS[dataset],
                    purpose=metadata.description,
                    access_method="official_source_or_manual_local_file",
                    licence_notes=metadata.licence_notes,
                    citation_requirements=(
                        (
                            "Cite the maintainer's dataset paper/release and retain source "
                            "attribution; "
                        )
                        + (
                            "SciCite: Structural Scaffolds for Citation Intent Classification "
                            "in Scientific Publications (NAACL 2019)."
                        )
                        if dataset == "scicite"
                        else (
                            "Retain dataset-specific release citation and upstream "
                            "attribution; verify before redistribution."
                        )
                    ),
                )
                self.registry.put(state, action="catalogue_initialized")

    def canonical_id(self, dataset):
        dataset = ALIASES.get(dataset, dataset)
        if dataset not in self.registry.states:
            raise ValueError(f"unknown dataset: {dataset}")
        return dataset

    def update(self, dataset, action, **updates):
        values = self.registry.states[dataset].model_dump()
        values.update(updates)
        return self.registry.put(DatasetState.model_validate(values), action=action)

    def acquire(self, dataset, *, path=None, offline=False, expected_sha256=None):
        dataset = self.canonical_id(dataset)
        try:
            if path is not None:
                local = Path(path).resolve()
                if not local.exists():
                    raise FileNotFoundError(f"local dataset not found: {local}")
                return self.update(
                    dataset,
                    "local_file_registered",
                    status=DatasetStatus.AVAILABLE,
                    local_path=str(local),
                    errors=(),
                )
            local, metadata = acquire(
                dataset,
                self.data_dir,
                offline=offline,
                max_bytes=self.config.data.max_download_bytes,
                expected_sha256=expected_sha256,
            )
            return self.update(
                dataset,
                "official_source_acquired",
                status=DatasetStatus.DOWNLOADED,
                local_path=str(local.resolve()),
                downloaded_at=metadata["downloaded_at"],
                checksum=metadata["sha256"],
                checksum_trust=metadata["checksum_trust"],
                provenance=(metadata,),
                original_format=local.name,
                errors=(),
            )
        except (OSError, ValueError) as error:
            self.update(
                dataset,
                "acquisition_failed_or_blocked",
                status=DatasetStatus.BLOCKED
                if isinstance(error, AcquisitionBlocked)
                else DatasetStatus.FAILED,
                errors=(str(error),),
            )
            raise

    def local_path(self, dataset):
        state = self.registry.states[dataset]
        if state.local_path:
            return Path(state.local_path)
        candidates = (self.data_dir / dataset,)
        if dataset == "scicite":
            candidates = (self.data_dir / "scicite/scicite.tar.gz", self.data_dir / "scicite")
        elif dataset == "arxiv-metadata":
            candidates = (
                self.data_dir / "arxiv-metadata/arxiv-metadata-oai-snapshot.json",
                self.data_dir / "arxiv-metadata/arxiv.zip",
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def inspect(
        self,
        dataset,
        *,
        path=None,
        max_records=None,
        sample_size=None,
        seed=None,
        version=None,
        synthetic=False,
        offline=True,
        trusted_cache=False,
        max_edges=100000,
    ):
        dataset = self.canonical_id(dataset)
        try:
            local = Path(path).resolve() if path else self.local_path(dataset)
            if dataset in {"scicite", "arxiv-metadata"}:
                if local is None:
                    raise AcquisitionBlocked(
                        "no local source files; supply --path or explicitly run data acquire"
                    )
                result = inspect_text(
                    dataset,
                    local,
                    self.output_dir,
                    max_records=max_records,
                    sample_size=sample_size,
                    seed=self.config.run.seed if seed is None else seed,
                    version=version,
                    synthetic=synthetic,
                    short_text=self.config.data.short_text_threshold,
                    long_text=self.config.data.long_text_threshold,
                    max_line_bytes=self.config.data.max_line_bytes,
                )
                quality = result["quality"]
                self.update(
                    dataset,
                    "source_records_inspected",
                    status=result["status"],
                    local_path=str(local),
                    version_id=version,
                    verified_at=result["verified_at"],
                    checksum=result["source_sha256"],
                    original_format="archive JSONL" if local.suffix in {".gz", ".zip"} else "JSONL",
                    raw_record_count=quality["raw_record_count"],
                    valid_record_count=quality["valid_record_count"],
                    invalid_record_count=quality["invalid_record_count"],
                    detected_schema=result["observed_schema"],
                    quality=quality,
                    labels=quality["label_distribution_by_split"],
                    identifiers=quality["identifier_namespaces"],
                    limitations=tuple(result["limitations"]),
                    inspection_path=result["inspection_path"],
                    staging_path=result["staging_path"],
                    index_path=result["index_path"],
                    errors=(),
                )
                return result
            if dataset not in {"ogbn-arxiv", "mag-year-h2gb"}:
                raise AcquisitionBlocked(
                    "optional enrichment is library opt-in; no dataset was configured"
                )
            root = local or self.data_dir / dataset
            adapter = OGBNArxivAdapter() if dataset == "ogbn-arxiv" else MAGYearAdapter()
            loaded = adapter.load(root, offline=offline, trusted_cache=trusted_cache)
            result = adapter.inspect(loaded, max_edges=max_edges)
            if dataset == "ogbn-arxiv":
                result["identifier_mapping"] = adapter.inspect_mapping(root)
            report_path = (
                self.output_dir / "inspections" / dataset / (now().replace(":", "-") + ".json")
            )
            result.update(
                {
                    "verified_at": now(),
                    "inspection_config": {"max_edges": max_edges, "offline": offline},
                    "source_path": str(root),
                    "real_dataset": True,
                }
            )
            atomic_json(report_path, result)
            self.update(
                dataset,
                "official_graph_inspected",
                status=DatasetStatus.PARTIALLY_VALIDATED
                if result["partial"] or result["issues"]
                else DatasetStatus.VALIDATED,
                local_path=str(root),
                verified_at=result["verified_at"],
                detected_schema=result,
                node_types=tuple(result["node_types"]),
                edge_types=tuple(str(t) for t in result["edge_types"]),
                quality={"issues": result["issues"]},
                inspection_path=str(report_path.resolve()),
                errors=(),
            )
            return result
        except (OSError, ValueError, ImportError) as error:
            self.update(
                dataset,
                "inspection_failed_or_blocked",
                status=DatasetStatus.BLOCKED
                if isinstance(error, (AcquisitionBlocked, ImportError))
                else DatasetStatus.FAILED,
                errors=(str(error),),
            )
            raise

    def inspection_reports(self):
        reports = {}
        for dataset, state in self.registry.states.items():
            if (
                state.status in {DatasetStatus.VALIDATED, DatasetStatus.PARTIALLY_VALIDATED}
                and state.inspection_path
                and Path(state.inspection_path).exists()
            ):
                reports[dataset] = json.loads(
                    Path(state.inspection_path).read_text(encoding="utf-8")
                )
        return reports
