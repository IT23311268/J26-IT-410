"""Atomic, versioned Stage 04 model artifact publication."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from uuid import uuid4

from .. import __version__
from ..data.manifests import atomic_json, now
from ..data.provenance import file_sha256
from ..pipeline import fingerprint


class IntentArtifactError(ValueError):
    """Raised when a Stage 04 artifact cannot be safely published or resolved."""


def environment_snapshot(project_root: Path) -> dict:
    """Capture installed package metadata without importing optional ML libraries."""
    versions = {}
    for package in (
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
        "pyarrow",
        "matplotlib",
        "pydantic",
        "PyYAML",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    git_revision = None
    git_dirty = None
    try:
        git_revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            ).stdout.strip()
            or None
        )
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            ).stdout.strip()
        )
    except OSError:
        pass
    return {
        "component_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
        "git_revision": git_revision,
        "git_worktree_dirty": git_dirty,
    }


class IntentRunPublication:
    """Publish a new run directory without overwriting an existing model run."""

    def __init__(self, root: Path, *, config: dict, project_root: Path):
        self.root = Path(root).resolve()
        self.project_root = Path(project_root).resolve()
        if not self.root.is_relative_to(self.project_root):
            raise IntentArtifactError("Stage 04 model root must remain in the project")
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = "run-" + now().replace(":", "-") + "-" + uuid4().hex[:8]
        self.pending = self.root / (".pending-" + self.run_id)
        self.final = self.root / self.run_id
        self.pending.mkdir(exist_ok=False)
        self.config = config

    def write_json(self, relative: str, value: dict | list) -> Path:
        path = (self.pending / relative).resolve()
        if not path.is_relative_to(self.pending):
            raise IntentArtifactError("artifact path escapes the pending run")
        atomic_json(path, value)
        return path

    def publish(self, manifest: dict) -> Path:
        if (
            not self.pending.exists()
            or self.final.exists()
            or not self.pending.is_relative_to(self.root)
        ):
            raise IntentArtifactError("run publication must use a new pending directory")
        manifest = {
            **manifest,
            "run_id": self.run_id,
            "stage": 4,
            "status": "COMPLETE",
            "created_at": manifest.get("created_at", now()),
            "completed_at": now(),
            "config": self.config,
            "config_sha256": fingerprint(self.config),
            "environment": environment_snapshot(self.project_root),
        }
        manifest["artifact_checksums"] = {
            str(path.relative_to(self.pending)): file_sha256(path)
            for path in sorted(self.pending.rglob("*"))
            if path.is_file()
        }
        atomic_json(self.pending / "run_manifest.json", manifest)
        os.rename(self.pending, self.final)
        return self.final

    def fail(self, error: Exception) -> None:
        if self.pending.exists():
            atomic_json(
                self.pending / "failure.json",
                {
                    "stage": 4,
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "config_sha256": fingerprint(self.config),
                    "failed_at": now(),
                },
            )


def resolve_intent_run(root: Path, run_id: str) -> Path:
    candidate = (Path(root).resolve() / run_id).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise IntentArtifactError("run identifier resolves outside the Stage 04 model root")
    manifest = candidate / "run_manifest.json"
    if not manifest.is_file():
        raise IntentArtifactError(f"Stage 04 run not found: {run_id}")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntentArtifactError(
            f"cannot read Stage 04 run manifest: {type(error).__name__}"
        ) from None
    if document.get("status") != "COMPLETE" or document.get("stage") != 4:
        raise IntentArtifactError("Stage 04 run is not completely published")
    return candidate


def update_run_manifest(run_dir: Path, updates: dict) -> dict:
    """Record an explicit post-training artifact update without replacing run identity."""
    run_dir = Path(run_dir).resolve()
    path = run_dir / "run_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.setdefault("updates", []).append({"at": now(), **updates})
    if updates.get("test_split_used") is True:
        document["test_evaluated"] = True
    # A completed run may gain an explicit final evaluation, calibration, or
    # prediction export.  Refresh checksums for those immutable payloads while
    # excluding the self-referential manifest itself.
    document["artifact_checksums"] = {
        str(item.relative_to(run_dir)): file_sha256(item)
        for item in sorted(run_dir.rglob("*"))
        if item.is_file() and item != path
    }
    atomic_json(path, document)
    return document
