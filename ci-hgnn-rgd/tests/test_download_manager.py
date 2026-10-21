import hashlib
import io
import json
from urllib.error import HTTPError

import pytest

from ci_hgnn_rgd.data.download_manager import AcquisitionBlocked, DownloadManager


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(payload)),
            **(headers or {}),
        }
        self.url = "https://example.org/synthetic.bin"


def test_offline_acquisition_never_calls_network(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("network attempted")

    with pytest.raises(AcquisitionBlocked, match="offline"):
        DownloadManager(offline=True, opener=forbidden).acquire(
            "https://example.org/synthetic", tmp_path / "fixture.bin"
        )


def test_cached_download_is_reused_and_checksum_checked(tmp_path):
    payload = b"SYNTHETIC dataset bytes for software test"
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        return Response(payload)

    manager = DownloadManager(opener=opener)
    path = tmp_path / "fixture.bin"
    expected = hashlib.sha256(payload).hexdigest()
    first = manager.acquire("https://example.org/synthetic.bin", path, expected_sha256=expected)
    second = manager.acquire("https://example.org/synthetic.bin", path, expected_sha256=expected)
    assert first == second and len(calls) == 1
    assert first["checksum_trust"] == "trusted_expected_sha256"
    path.write_bytes(b"modified synthetic bytes")
    with pytest.raises(ValueError, match="conflicts"):
        manager.acquire("https://example.org/synthetic.bin", path)


def test_resume_owned_partial_with_range(tmp_path):
    path = tmp_path / "fixture.bin"
    path.with_name("fixture.bin.part").write_bytes(b"1234")
    path.with_name("fixture.bin.part.json").write_text(
        json.dumps({"url": "https://example.org/synthetic.bin", "etag": "fixed"})
    )

    def opener(request, **kwargs):
        assert request.get_header("Range") == "bytes=4-"
        return Response(
            b"567890", status=206, headers={"Content-Range": "bytes 4-9/10", "ETag": "fixed"}
        )

    DownloadManager(opener=opener).acquire("https://example.org/synthetic.bin", path)
    assert path.read_bytes() == b"1234567890"


@pytest.mark.parametrize(
    "headers,expected_error",
    [({"Content-Type": "text/html"}, "HTML"), ({"Content-Length": "999999"}, "budget")],
)
def test_html_and_excessive_downloads_rejected(tmp_path, headers, expected_error):
    manager = DownloadManager(
        max_bytes=100, opener=lambda *a, **k: Response(b"fixture", headers=headers)
    )
    with pytest.raises(ValueError, match=expected_error):
        manager.acquire("https://example.org/synthetic.bin", tmp_path / "fixture.bin")


def test_incomplete_download_never_publishes_final_file(tmp_path):
    path = tmp_path / "fixture.bin"
    manager = DownloadManager(
        opener=lambda *a, **k: Response(b"partial", headers={"Content-Length": "20"})
    )
    with pytest.raises(ValueError, match="incomplete"):
        manager.acquire("https://example.org/synthetic.bin", path)
    assert not path.exists()
    assert path.with_name("fixture.bin.part").exists()


def test_corrupt_gzip_is_not_published(tmp_path):
    path = tmp_path / "fixture.gz"
    with pytest.raises(OSError):
        DownloadManager(opener=lambda *a, **k: Response(b"not gzip")).acquire(
            "https://example.org/fixture.gz", path
        )
    assert not path.exists()


def test_existing_user_file_is_never_overwritten(tmp_path):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"user-owned synthetic fixture")
    with pytest.raises(ValueError, match="user file"):
        DownloadManager(offline=True).acquire("https://example.org/fixture.bin", path)
    assert path.read_bytes() == b"user-owned synthetic fixture"


def test_bounded_retry_and_backoff_for_429(tmp_path):
    attempts = []
    delays = []

    def opener(request, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise HTTPError(
                request.full_url, 429, "synthetic throttling", {"Retry-After": "1"}, None
            )
        return Response(b"synthetic")

    DownloadManager(opener=opener, sleeper=delays.append).acquire(
        "https://example.org/fixture.bin", tmp_path / "fixture.bin"
    )
    assert len(attempts) == 3 and delays == [1, 1]
