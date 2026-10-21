"""Content identity and observed provenance, separate from unverified historical availability."""

import hashlib
from pathlib import Path

from .manifests import now
from .staging import SourceProvenance


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(dataset_id, source_url, version, event, *, synthetic=False, acquired_at=None):
    return SourceProvenance(
        source_dataset=dataset_id,
        source_url=source_url,
        dataset_version=version,
        version_status="known" if version else "unknown",
        source_file=event.source,
        locator=f"{event.source}:line:{event.line}",
        record_sha256=event.sha256,
        inspected_at=now(),
        acquired_at=acquired_at,
        synthetic=synthetic,
    )
