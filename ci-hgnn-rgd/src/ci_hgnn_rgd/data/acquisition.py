"""Explicit source-specific acquisition. Inspection itself never initiates a download."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from .download_manager import AcquisitionBlocked, DownloadManager, validate_container
from .manifests import atomic_json, now
from .provenance import file_sha256
from .sources.scicite import ARCHIVE_URL


def acquire(
    dataset, data_dir: Path, *, offline=False, max_bytes=64 * 1024**2, expected_sha256=None
):
    if dataset == "scicite":
        path = data_dir / "scicite/scicite.tar.gz"
        return path, DownloadManager(offline=offline, max_bytes=max_bytes).acquire(
            ARCHIVE_URL, path, expected_sha256=expected_sha256
        )
    if dataset != "arxiv-metadata":
        raise AcquisitionBlocked(
            "graph acquisition is deferred to a reviewed resource-bounded official "
            "package environment; no dataset substituted"
        )
    cached = data_dir / "arxiv-metadata/arxiv.zip"
    sidecar = cached.with_name(cached.name + ".source.json")
    if cached.exists() and sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if file_sha256(cached) != metadata["sha256"]:
            raise ValueError("cached arXiv archive checksum mismatch")
        return cached, metadata
    if offline:
        raise AcquisitionBlocked("offline mode: arXiv archive absent; supply a local --path")
    executable = shutil.which("kaggle")
    if not executable:
        raise AcquisitionBlocked("Kaggle CLI is not installed; manually download and supply --path")
    config_root = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    if not (
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (config_root / "kaggle.json").exists()
    ):
        raise AcquisitionBlocked("Kaggle credentials are not configured")
    # Public file metadata preflight; the source CLI owns authentication and the actual download.
    from .sources.api_client import MetadataClient

    client = MetadataClient(
        data_dir / "source_audits/kaggle", enabled=True, offline=False, max_calls=3
    )
    listing = client.get("https://www.kaggle.com/api/v1/datasets/list/Cornell-University/arxiv")
    files = listing["data"].get("datasetFiles", [])
    required = sum(int(item["totalBytes"]) for item in files)
    if not files or not required or listing["data"].get("hasNextPageToken"):
        raise AcquisitionBlocked(
            "complete Kaggle size metadata unavailable; no unbounded download attempted"
        )
    if required > max_bytes:
        raise AcquisitionBlocked(
            f"Kaggle source advertises {required} bytes; budget is {max_bytes}"
        )
    if shutil.disk_usage(data_dir).free < required * 3:
        raise AcquisitionBlocked("insufficient disk for archive, raw data and staging")
    work = data_dir / "arxiv-metadata" / ("acquisition-" + str(uuid4()))
    work.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        [executable, "datasets", "download", "Cornell-University/arxiv", "--path", str(work)],
        capture_output=True,
        timeout=3600,
        check=False,
    )
    if result.returncode:
        raise AcquisitionBlocked(
            f"Kaggle CLI failed with exit code {result.returncode}; output withheld to "
            f"protect credentials"
        )
    archives = list(work.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError("Kaggle did not produce exactly one expected ZIP archive")
    path = archives[0]
    if path.stat().st_size > max_bytes:
        raise ValueError("downloaded archive exceeds byte budget")
    validate_container(path, path.name, max_expanded_bytes=max_bytes)
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Kaggle archive fails trusted checksum")
    if cached.exists():
        raise ValueError("existing arXiv dataset is never overwritten")
    os.link(path, cached)
    metadata = {
        "url": "https://www.kaggle.com/datasets/Cornell-University/arxiv",
        "downloaded_at": now(),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "source_size_metadata": listing["provenance"],
        "checksum_trust": "trusted_expected_sha256" if expected_sha256 else "locally_computed_only",
    }
    atomic_json(sidecar, metadata)
    return cached, metadata
