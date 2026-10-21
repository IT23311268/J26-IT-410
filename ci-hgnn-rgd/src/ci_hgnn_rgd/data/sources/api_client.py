"""Opt-in bounded GET client with private credentials, cache provenance and request accounting."""

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request

from ..download_manager import AcquisitionBlocked, DownloadManager
from ..manifests import atomic_json, now


class MetadataClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        enabled=False,
        offline=True,
        max_calls=10,
        minimum_interval=1.0,
        opener=None,
        sleeper=time.sleep,
    ):
        self.cache_dir, self.enabled, self.offline = cache_dir, enabled, offline
        self.max_calls, self.minimum_interval = max_calls, minimum_interval
        self.calls = 0
        self._last = 0.0
        self.sleeper = sleeper
        kwargs = {"offline": offline, "sleeper": sleeper}
        if opener is not None:
            kwargs["opener"] = opener
        self.transport = DownloadManager(**kwargs)

    def get(self, endpoint, *, params=None, private_params=None, headers=None):
        params = params or {}
        public_url = endpoint + ("?" + urlencode(params) if params else "")
        key = hashlib.sha256(public_url.encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        if not self.enabled:
            raise AcquisitionBlocked("metadata enrichment is not explicitly enabled")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise AcquisitionBlocked("offline metadata response is not cached")
        request_headers = {
            "User-Agent": "CI-HGNN-RGD/0.2 research metadata lookup",
            **(headers or {}),
        }
        full_params = {**params, **(private_params or {})}
        url = endpoint + ("?" + urlencode(full_params) if full_params else "")
        delay = self.minimum_interval - (time.monotonic() - self._last)
        if delay > 0:
            self.sleeper(delay)
        if self.calls >= self.max_calls:
            raise AcquisitionBlocked("metadata API request budget exhausted")
        self.calls += 1
        # Transport retries consume attempts too; keep a bounded ledger via the opener wrapper.
        original = self.transport.opener
        attempts = 0

        def budgeted_opener(request, **kwargs):
            nonlocal attempts
            if attempts:
                if self.calls >= self.max_calls:
                    raise AcquisitionBlocked("metadata API request budget exhausted during retries")
                self.calls += 1
            attempts += 1
            self._last = time.monotonic()
            return original(request, **kwargs)

        self.transport.opener = budgeted_opener
        try:
            with self.transport._open(Request(url, headers=request_headers)) as response:
                content_type = response.headers.get("Content-Type", "")
                if "json" not in content_type:
                    raise ValueError("metadata endpoint did not return JSON")
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ValueError("metadata response exceeds 2 MB limit")
                data = json.loads(payload)
                result = {
                    "data": data,
                    "provenance": {
                        "url": public_url,
                        "retrieved_at": now(),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "http_status": response.status,
                        "attempts": attempts,
                        "external_resolution": "source_response_only",
                    },
                }
                atomic_json(path, result)
                return result
        finally:
            self.transport.opener = original
