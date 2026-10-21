"""Loss-aware scholarly identifier normalization; syntax is not an identity assertion.

``verified`` records the caller's source-backed identifier evidence. It never means
that this module contacted an authority. Internal indices require dataset, version,
and node type, and are never interchangeable with external scholarly identifiers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from urllib.parse import unquote, urlsplit

from .identifier_audit import normalize_arxiv, normalize_doi

AUTHORITATIVE_NAMESPACES = frozenset(
    {"doi", "arxiv", "openalex", "semantic_scholar", "mag", "semantic_scholar_corpus"}
)

_ALIASES = {
    "arxiv_id": "arxiv",
    "arxiv-id": "arxiv",
    "openalex_work": "openalex",
    "openalex-work-id": "openalex",
    "openalex_work_id": "openalex",
    "semantic-scholar": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "semantic-scholar-paper-id": "semantic_scholar",
    "semantic_scholar_paper_id": "semantic_scholar",
    "s2": "semantic_scholar",
    "s2-corpus-id": "semantic_scholar_corpus",
    "corpusid": "semantic_scholar_corpus",
    "mag-paper-id": "mag",
    "mag_paper_id": "mag",
    "mag-paper": "mag",
    "ogbn-arxiv-node-index": "internal_graph",
    "ogbn_arxiv_node_index": "internal_graph",
    "h2gb-node-index": "internal_graph",
    "h2gb-mag-year-node-index": "internal_graph",
    "mag-year-node-index": "internal_graph",
    "internal-node-index": "internal_graph",
    "internal_graph_index": "internal_graph",
    "dataset-specific": "source_local",
    "dataset_specific": "source_local",
    "dataset-specific-paper-id": "source_local",
    "source-local": "source_local",
}


def comparison_title(value: object) -> str | None:
    """A separate comparison key; never used to replace stored scientific text."""
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_identifier(
    namespace: str,
    value: object,
    *,
    dataset: str | None = None,
    version: str | None = None,
    node_type: str | None = None,
    verified: bool = False,
) -> dict:
    """Return raw value, comparison key, scope and validation evidence explicitly.

    DOI punctuation is preserved. arXiv versions share a work comparison key but
    retain their version suffix. Unknown/source-specific namespaces are scoped to
    their dataset and version instead of being guessed from their value's shape.
    """
    original_namespace = namespace
    namespace = _ALIASES.get(namespace.strip().casefold(), namespace.strip().casefold())
    raw = value
    result = {
        "namespace": namespace,
        "original_namespace": original_namespace,
        "raw": raw,
        "normalized": None,
        "valid": False,
        "verified": False,
        "version": None,
        "scope": None,
        "dataset": dataset,
        "dataset_version": version,
        "node_type": node_type,
        "reason": None,
    }
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        result["reason"] = "unexpected_identifier_type"
        return result
    text = str(value).strip()
    if len(text) > 4096:
        result["reason"] = "identifier_exceeds_4096_character_matching_limit"
        return result
    normalized = None
    if namespace == "doi":
        normalized = normalize_doi(text)
    elif namespace == "arxiv":
        arxiv = normalize_arxiv(text)
        if arxiv:
            normalized = arxiv.base
            result["version"] = arxiv.version
    elif namespace == "openalex":
        if text.lower().startswith(("https://openalex.org/", "http://openalex.org/")):
            parsed = urlsplit(text)
            text = unquote(parsed.path.lstrip("/"))
            if parsed.query or parsed.fragment:
                text = ""
        text = re.sub(r"^openalex\s*:\s*", "", text, flags=re.I)
        if re.fullmatch(r"W[1-9]\d*", text, re.I):
            normalized = text.upper()
    elif namespace == "semantic_scholar":
        text = re.sub(r"^(?:s2|semanticscholar)\s*:\s*", "", text, flags=re.I)
        if text.lower().startswith("https://www.semanticscholar.org/paper/"):
            parsed = urlsplit(text)
            text = parsed.path.rstrip("/").rsplit("/", 1)[-1]
            if parsed.query or parsed.fragment:
                text = ""
        if re.fullmatch(r"[0-9a-f]{40}", text, re.I):
            normalized = text.casefold()
    elif namespace in {"mag", "semantic_scholar_corpus"}:
        text = re.sub(r"^(?:mag|corpusid)\s*:\s*", "", text, flags=re.I)
        if re.fullmatch(r"[1-9]\d*", text):
            normalized = str(int(text))
    else:
        internal = namespace == "internal_graph"
        unknown_scopes = {"unknown", "unverified", "unspecified", "not_reported", "latest"}
        if (
            not dataset
            or not version
            or version.strip().casefold() in unknown_scopes
            or (internal and not node_type)
        ):
            result["reason"] = "missing_dataset_version_or_node_type_scope"
            return result
        if internal:
            if re.fullmatch(r"0|[1-9]\d*", text):
                normalized = str(int(text))
        elif text and not any(ord(character) < 32 for character in text):
            normalized = text
        result["scope"] = {
            "dataset": dataset,
            "version": version,
            "node_type": node_type or "paper",
        }
    result["normalized"] = normalized
    result["valid"] = normalized is not None
    result["verified"] = bool(verified and normalized is not None)
    if normalized is None:
        result["reason"] = "invalid_identifier_syntax"
    return result


def identifier_token(identifier: dict) -> tuple[str, str] | None:
    """Return namespace-safe key including internal/source-local scope."""
    if not identifier.get("valid", identifier.get("normalized") is not None):
        return None
    value = identifier.get("normalized")
    if value is None:
        return None
    namespace = identifier["namespace"]
    scope = identifier.get("scope")
    if scope:
        namespace += ":" + json.dumps(
            [scope["dataset"], scope["version"], scope.get("node_type", "paper")],
            separators=(",", ":"),
            ensure_ascii=False,
        )
    elif namespace not in AUTHORITATIVE_NAMESPACES:
        return None
    return namespace, str(value)
