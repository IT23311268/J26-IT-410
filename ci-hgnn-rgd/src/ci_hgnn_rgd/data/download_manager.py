"""Bounded streaming HTTP acquisition, resumable owned partials, atomic finalization."""

import gzip
import json
import os
import re
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .manifests import atomic_json, now
from .provenance import file_sha256


class AcquisitionBlocked(ValueError):
    pass


def validate_container(path: Path, name: str, *, max_expanded_bytes=1024**3):
    if name.endswith(".gz"):
        total = 0
        with gzip.open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(block)
                if total > max_expanded_bytes:
                    raise ValueError("expanded gzip exceeds configured size limit")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > max_expanded_bytes:
                raise ValueError("expanded ZIP exceeds configured size limit")
            if archive.testzip() is not None:
                raise ValueError("corrupt ZIP member")


class DownloadManager:
    def __init__(
        self,
        *,
        offline=False,
        max_bytes=64 * 1024**2,
        timeout=30,
        retries=3,
        opener=urlopen,
        sleeper=time.sleep,
    ):
        self.offline = offline
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.retries = retries
        self.opener = opener
        self.sleeper = sleeper

    def _open(self, request):
        for attempt in range(self.retries + 1):
            try:
                return self.opener(request, timeout=self.timeout)
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise AcquisitionBlocked(f"HTTP {error.code} from source") from None
                retry_after = error.headers.get("Retry-After", "")
                delay = (
                    min(float(retry_after), 60) if retry_after.isdigit() else min(2**attempt, 30)
                )
                self.sleeper(delay)
            except (URLError, OSError):
                if attempt == self.retries:
                    raise AcquisitionBlocked("source network connection failed") from None
                self.sleeper(min(2**attempt, 30))

    def acquire(self, url: str, target: Path, *, expected_sha256: str | None = None) -> dict:
        if urlsplit(url).scheme != "https":
            raise ValueError("dataset acquisition requires HTTPS")
        if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("invalid expected SHA-256")
        metadata_path = target.with_name(target.name + ".source.json")
        if target.exists():
            if not metadata_path.exists():
                raise ValueError(
                    "existing user file has no acquisition metadata; use local ingestion"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = file_sha256(target)
            if metadata.get("url") != url or metadata.get("sha256") != digest:
                raise ValueError(
                    "existing file conflicts with acquisition metadata; not overwritten"
                )
            if expected_sha256 and digest != expected_sha256:
                raise ValueError("existing dataset fails trusted checksum")
            return metadata
        if self.offline:
            raise AcquisitionBlocked("offline mode: requested dataset is not cached")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial_meta = target.with_name(target.name + ".part.json")
        offset, etag = 0, None
        if partial.exists():
            if not partial_meta.exists():
                raise ValueError("unowned partial file exists; refusing to overwrite it")
            previous = json.loads(partial_meta.read_text(encoding="utf-8"))
            if previous.get("url") != url:
                raise ValueError("partial file belongs to another source")
            offset, etag = partial.stat().st_size, previous.get("etag")
        headers = {
            "User-Agent": "CI-HGNN-RGD/0.1 scientific dataset inspection",
            "Accept-Encoding": "identity",
        }
        if offset and etag:
            headers.update({"Range": f"bytes={offset}-", "If-Range": etag})
        with self._open(Request(url, headers=headers)) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
            if content_type in {"text/html", "application/xhtml+xml"}:
                raise ValueError("source returned an HTML page instead of a dataset")
            append = response.status == 206 and offset > 0 and etag is not None
            if response.status == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if not append or match is None or int(match[1]) != offset:
                    raise ValueError("invalid resumable Content-Range")
                expected_total = int(match[3])
            else:
                offset = 0
                length = response.headers.get("Content-Length")
                expected_total = int(length) if length else None
            if expected_total and expected_total > self.max_bytes:
                raise AcquisitionBlocked(
                    f"source requires {expected_total} bytes; budget is {self.max_bytes}"
                )
            atomic_json(partial_meta, {"url": url, "etag": response.headers.get("ETag")})
            size = offset
            with partial.open("ab" if append else "wb") as stream:
                while block := response.read(min(1024 * 1024, self.max_bytes - size + 1)):
                    size += len(block)
                    if size > self.max_bytes:
                        raise AcquisitionBlocked("download exceeded configured byte budget")
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if expected_total is not None and size != expected_total:
                raise ValueError("incomplete download; owned partial retained for resumption")
            validate_container(partial, target.name)
            digest = file_sha256(partial)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError("download fails trusted SHA-256; partial retained")
            # Hard-link publication fails if another actor created the final path; no overwrite.
            os.link(partial, target)
            metadata = {
                "url": url,
                "final_url": response.url,
                "downloaded_at": now(),
                "bytes": size,
                "sha256": digest,
                "content_type": content_type,
                "etag": response.headers.get("ETag"),
                "checksum_trust": (
                    "trusted_expected_sha256" if expected_sha256 else "locally_computed_only"
                ),
            }
            atomic_json(metadata_path, metadata)
            partial.unlink()
            partial_meta.unlink()
            return metadata
