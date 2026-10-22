"""Atomic versioned Stage 05 data runs and safe table transport."""

import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from ...data.manifests import atomic_json
from ...data.provenance import file_sha256
from ...pipeline import canonical_json, fingerprint
from ..intent_artifacts import environment_snapshot
from .contracts import utc_now


def _publish_directory(pending: Path, final: Path):
    """Keep atomic rename while tolerating short-lived Windows scanner locks."""
    for attempt in range(8):
        try:
            os.rename(pending, final)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.5))


def write_rows(root: Path, name: str, rows) -> int:
    """Portable JSONL and Parquet audit envelopes, with bounded writer batches."""
    from importlib.util import find_spec

    if find_spec("pyarrow") is None:
        root.mkdir(parents=True, exist_ok=True)
        count = 0
        with (root / (name + ".jsonl")).open("x", encoding="utf-8") as stream:
            for row in rows:
                value = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
                stream.write(canonical_json(value) + "\n")
                count += 1
        return count
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            (key, pa.string())
            for key in ("paper_key", "canonical_paper_id", "topic_id", "record_json")
        ]
    )
    count = 0
    with (
        (root / (name + ".jsonl")).open("x", encoding="utf-8") as stream,
        pq.ParquetWriter(root / (name + ".parquet"), schema, compression="zstd") as writer,
    ):
        buffer = []
        for row in rows:
            value = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            encoded = canonical_json(value)
            stream.write(encoded + "\n")
            buffer.append(
                {
                    "paper_key": value.get("paper_key"),
                    "canonical_paper_id": value.get("canonical_paper_id"),
                    "topic_id": value.get("topic_id"),
                    "record_json": encoded,
                }
            )
            count += 1
            if len(buffer) == 256:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer = []
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    return count


def iter_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                yield json.loads(row["record_json"]) if "record_json" in row else row
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("input JSONL rows must be objects")
                    yield row
    else:
        raise ValueError("Stage 05 table input must be JSONL or Parquet")


class RunBuilder:
    def __init__(self, root: Path, config, *, kind: str, project_root: Path):
        self.root, self.config, self.kind = root.resolve(), config, kind
        self.project_root = project_root.resolve()
        if not self.root.is_relative_to(self.project_root):
            raise ValueError("Stage 05 publication root must stay inside the project")
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = "run-" + utc_now().replace(":", "-") + "-" + uuid4().hex[:8]
        self.pending = self.root / (".pending-" + self.run_id)
        self.final = self.root / self.run_id
        self.pending.mkdir(exist_ok=False)
        atomic_json(self.pending / "config.json", config.model_dump(mode="json"))

    def publish(self, details: dict, *, fitted_topic_model=False, fitted_investigation_model=False):
        profile_path = self.pending / "input_data_profile.json"
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["full_exclusions_path"] = str(self.final / "excluded_records.jsonl")
            atomic_json(profile_path, profile)
        environment = environment_snapshot(self.project_root)
        from importlib import metadata

        for package in ("adapters", "bertopic", "umap-learn", "hdbscan", "scipy"):
            try:
                environment["packages"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                environment["packages"][package] = None
        atomic_json(self.pending / "environment.json", environment)
        if fitted_topic_model or fitted_investigation_model:
            model_root = (
                self.config.model_root
                if fitted_topic_model
                else self.config.investigation_model_root
            )
            destination = model_root / self.run_id
            if destination.exists():
                raise ValueError("model artifact run already exists")
            temporary = model_root / (".pending-" + self.run_id)
            temporary.mkdir(parents=True, exist_ok=False)
            names = (
                (
                    "models",
                    "embedding_config.json",
                    "embedding_model_manifest.json",
                    "topic_catalog.jsonl",
                    "topic_assignment_config.json",
                    "training_corpus_manifest.json",
                    "temporal_snapshot.json",
                    "config.json",
                    "environment.json",
                )
                if fitted_topic_model
                else (
                    "model",
                    "label_mapping.json",
                    "annotation_manifest.json",
                    "training_history.json",
                    "validation_metrics.json",
                    "config.json",
                    "environment.json",
                )
            )
            for name in names:
                source = self.pending / name
                if source.is_dir():
                    shutil.copytree(source, temporary / name)
                elif source.is_file():
                    shutil.copy2(source, temporary / name)
            atomic_json(
                temporary / "run_manifest.json",
                {
                    "stage": 5,
                    "status": "COMPLETE",
                    "run_id": self.run_id,
                    "data_run": str(self.final),
                    "actually_fitted": True,
                    "artifact_checksums": {
                        str(p.relative_to(temporary)): file_sha256(p)
                        for p in temporary.rglob("*")
                        if p.is_file()
                    },
                },
            )
            _publish_directory(temporary, destination)
            details["model_artifact_dir"] = str(destination)
        manifest = {
            "schema_version": "1.0",
            "stage": 5,
            "run_id": self.run_id,
            "kind": self.kind,
            "status": "COMPLETE",
            "created_at": utc_now(),
            "config_sha256": fingerprint(self.config),
            **details,
            "artifact_checksums": {
                str(p.relative_to(self.pending)): file_sha256(p)
                for p in self.pending.rglob("*")
                if p.is_file()
            },
        }
        atomic_json(self.pending / "stage05_manifest.json", manifest)
        _publish_directory(self.pending, self.final)
        return {"run_id": self.run_id, "run_dir": str(self.final), **manifest}

    def fail(self, error):
        if self.pending.exists():
            atomic_json(self.pending / "failure.json", {"status": "FAILED", "error": str(error)})


def latest_topic_run_id(root: Path, *, model_name: str | None = None) -> str:
    """Select a published fitted model, ignoring newer inspections and downstream exports."""
    for file in sorted(root.glob("run-*/stage05_manifest.json"), reverse=True):
        manifest = json.loads(file.read_text(encoding="utf-8"))
        if (
            manifest.get("stage") != 5
            or manifest.get("status") != "COMPLETE"
            or manifest.get("execution_status") != "TOPIC_MODELS_FITTED"
            or manifest.get("kind") not in {"train", "train-baselines"}
        ):
            continue
        algorithms = {
            item["model"].split(":", 1)[0] for item in manifest.get("model_comparison", [])
        }
        if model_name and model_name not in algorithms:
            continue
        # The caller loads this exact run through resolve_run, which verifies
        # every checksum. Never fall back to older weights on a corrupt run.
        return file.parent.name
    selection = f" for {model_name}" if model_name else ""
    raise ValueError(
        f"No completed fitted topic model{selection} is available. "
        "Provide Stage 03 paper titles/abstracts, then run 'topics train --model bertopic' "
        "or 'topics train-baselines'; --run-id can select an existing fitted run explicitly."
    )


def resolve_run(root: Path, run_id: str):
    path = (root / run_id).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Stage 05 run ID escapes output root")
    if path.name.startswith(".pending-") or (path / "failure.json").is_file():
        raise ValueError("Stage 05 run is not completely published")
    file = path / "stage05_manifest.json"
    if not file.is_file():
        raise ValueError(f"Stage 05 run not found: {run_id}")
    manifest = json.loads(file.read_text(encoding="utf-8"))
    if manifest.get("stage") != 5 or manifest.get("status") != "COMPLETE":
        raise ValueError("Stage 05 run is not completely published")
    for name, expected in manifest["artifact_checksums"].items():
        target = (path / name).resolve()
        if (
            not target.is_relative_to(path)
            or not target.is_file()
            or file_sha256(target) != expected
        ):
            raise ValueError("Stage 05 artifact checksum mismatch")
    return path, manifest
