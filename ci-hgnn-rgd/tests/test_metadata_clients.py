import io
import json

import pytest

from ci_hgnn_rgd.data.download_manager import AcquisitionBlocked
from ci_hgnn_rgd.data.sources.api_client import MetadataClient
from ci_hgnn_rgd.data.sources.openalex import OpenAlexClient
from ci_hgnn_rgd.data.sources.semantic_scholar import SemanticScholarClient


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self):
        super().__init__(b'{"synthetic":true,"externalIds":{}}')


def test_client_is_opt_in_and_offline_by_default(tmp_path):
    with pytest.raises(AcquisitionBlocked, match="enabled"):
        MetadataClient(tmp_path).get("https://example.org/synthetic")
    with pytest.raises(AcquisitionBlocked, match="offline"):
        MetadataClient(tmp_path, enabled=True).get("https://example.org/synthetic")


def test_private_keys_do_not_enter_cache_or_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "synthetic-private-token")
    monkeypatch.setenv("OPENALEX_CONTACT_EMAIL", "synthetic@example.org")
    requests = []

    def opener(request, **kwargs):
        requests.append(request.full_url)
        return Response()

    client = OpenAlexClient(
        tmp_path, enabled=True, offline=False, opener=opener, minimum_interval=0
    )
    result = client.lookup_doi("10.1234/SYNTHETIC")
    client.lookup_doi("10.1234/SYNTHETIC")
    assert client.calls == 1 and len(requests) == 1
    assert "synthetic-private-token" in requests[0]
    assert "synthetic-private-token" not in json.dumps(result)
    assert all("synthetic-private-token" not in path.read_text() for path in tmp_path.iterdir())


def test_semantic_scholar_lookup_preserves_namespace_and_request_budget(tmp_path):
    client = SemanticScholarClient(
        tmp_path,
        enabled=True,
        offline=False,
        max_calls=1,
        minimum_interval=0,
        opener=lambda *a, **k: Response(),
    )
    report = client.lookup("arXiv:2301.12345v2", namespace="arxiv")
    assert "ARXIV%3A2301.12345" in report["provenance"]["url"]
    with pytest.raises(AcquisitionBlocked, match="budget"):
        client.lookup("10.1234/SYNTHETIC", namespace="doi")
