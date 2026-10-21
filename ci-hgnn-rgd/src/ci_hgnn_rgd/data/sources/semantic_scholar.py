"""Optional Graph API metadata only. Complete citation-context availability is not assumed."""

import os
from urllib.parse import quote

from ..identifier_audit import normalize_arxiv, normalize_doi
from .api_client import MetadataClient


class SemanticScholarClient(MetadataClient):
    def lookup(self, value, *, namespace="doi", citations=False):
        if namespace == "doi":
            normalized = normalize_doi(value)
            prefix = "DOI:"
        elif namespace == "arxiv":
            identifier = normalize_arxiv(value)
            normalized = identifier.base if identifier else None
            prefix = "ARXIV:"
        else:
            raise ValueError("lookup namespace must be doi or arxiv")
        if normalized is None:
            raise ValueError("invalid lookup identifier")
        endpoint = "https://api.semanticscholar.org/graph/v1/paper/" + quote(
            prefix + normalized, safe=""
        )
        if citations:
            endpoint += "/citations"
            params = {
                "fields": "citingPaper.paperId,citingPaper.externalIds,citingPaper.title",
                "limit": 100,
            }
        else:
            params = {"fields": "paperId,externalIds,title,year,authors,citationCount"}
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        return self.get(endpoint, params=params, headers={"x-api-key": key} if key else {})
