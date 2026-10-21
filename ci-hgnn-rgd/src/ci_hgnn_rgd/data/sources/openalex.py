"""Optional DOI metadata lookup; no entity merging, spending, or automatic paging."""

import os
from urllib.parse import quote

from ..download_manager import AcquisitionBlocked
from ..identifier_audit import normalize_doi
from .api_client import MetadataClient


class OpenAlexClient(MetadataClient):
    def lookup_doi(self, value):
        doi = normalize_doi(value)
        if doi is None:
            raise ValueError("invalid DOI")
        key = os.environ.get("OPENALEX_API_KEY")
        contact = os.environ.get("OPENALEX_CONTACT_EMAIL")
        if not contact:
            raise AcquisitionBlocked(
                "set OPENALEX_CONTACT_EMAIL for identifiable optional requests"
            )
        return self.get(
            "https://api.openalex.org/works/" + quote("https://doi.org/" + doi, safe=""),
            private_params={"api_key": key} if key else {},
            headers={"User-Agent": f"CI-HGNN-RGD/0.2 (mailto:{contact})"},
        )

    def citation_metadata(self, work_id):
        if not work_id.startswith("W") or not work_id[1:].isdigit():
            raise ValueError("expected an OpenAlex W identifier")
        key = os.environ.get("OPENALEX_API_KEY")
        return self.get(
            f"https://api.openalex.org/works/{work_id}",
            params={"select": "id,ids,cited_by_count,referenced_works"},
            private_params={"api_key": key} if key else {},
        )
