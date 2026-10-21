"""Deterministic, conservative cleaning that keeps source values recoverable.

Only NFC and whitespace normalization are applied to separate scientific-text fields.
No case folding, punctuation removal, mathematical rewriting, or encoding guesses occur.
"""

import hashlib
import json
import math
import re
import unicodedata
from datetime import date
from typing import Any

from .canonical_records import validate_canonical

_NULLS = frozenset({"", "null", "none", "n/a", "na", "nan"})
_TEXT_FIELDS = ("title", "abstract", "display_name", "field_name", "venue")


def json_safe(value: Any) -> Any:
    """Encode source non-finite numbers explicitly; source bytes remain authoritative."""
    if isinstance(value, float) and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return {"__source_nonfinite__": label}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def clean_text(value: str) -> str:
    """Normalize Unicode composition and whitespace, retaining scientific symbols."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _record_key(record: dict) -> str:
    evidence = {
        "source_dataset": record.get("source_dataset"),
        "dataset_version": record.get("dataset_version"),
        "kind": record.get("kind"),
        "source_record_id": record.get("source_record_id"),
        "raw_values": record.get("raw_values"),
        "locators": [
            (item.get("locator"), item.get("record_sha256"))
            for item in record.get("provenance", [])
        ],
    }
    encoded = json.dumps(evidence, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "record:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def clean_record(
    record: dict[str, Any], *, kind: str | None = None, source_dataset: str | None = None
) -> dict[str, Any]:
    """Clean an adapter dictionary without mutating input or replacing source text.

    Source adapters establish identifier meaning; this function validates the already
    namespaced normalized fields. Missing dates are recorded without inventing dates.
    Quarantines retain all raw values and provenance for later explicit review.
    """
    result = json_safe(record)
    result["kind"] = kind or result.get("kind", "paper")
    result["source_dataset"] = source_dataset or result.get("source_dataset", "unknown")
    result.setdefault("raw_values", json_safe(record))
    result.setdefault("dataset_version", None)
    result.setdefault("source_record_id", None)
    result.setdefault("source_split", None)
    result.setdefault("provenance", [])
    result.setdefault("normalized_fields", {})
    result.setdefault("transformations", [])
    result.setdefault("issues", [])
    result.setdefault("missing_fields", [])
    result.setdefault("quarantine", False)
    issues = result["issues"]
    missing = result["missing_fields"]
    transformations = result["transformations"]

    def text(field: str, value: Any, *, required=False):
        if value is None:
            if required:
                missing.append(field)
            return None
        if not isinstance(value, str):
            issues.append(f"unexpected_type:{field}")
            result["quarantine"] = True
            return None
        normalized = clean_text(value)
        if normalized.casefold() in _NULLS:
            normalized = None
            missing.append(field)
        if normalized != value:
            operation = {"field": field, "operation": "NFC_whitespace_or_null", "raw_field": field}
            if operation not in transformations:
                transformations.append(operation)
        if "\ufffd" in value or any(marker in value for marker in ("Ãƒ", "â€", "Ã©")):
            issues.append(f"suspected_encoding:{field}")
        result["normalized_fields"][field] = normalized
        return normalized

    for field in _TEXT_FIELDS:
        if field in result or field == "title" and result["kind"] == "paper":
            result[field] = text(field, result.get(field), required=field == "title")
    if result["kind"] == "paper":
        for field in ("authors", "categories"):
            original = result.get(field, [])
            if not isinstance(original, list):
                issues.append(f"unexpected_type:{field}")
                result["quarantine"] = True
                original = []
            values = [text(f"{field}[{index}]", value) for index, value in enumerate(original)]
            result[field] = [value for value in values if value is not None]
        result.setdefault("publication_year", None)
        result.setdefault("publication_date", None)
        result.setdefault("date_precision", "unknown")
        if result["publication_date"] is not None:
            raw_date = result["publication_date"]
            try:
                if not isinstance(raw_date, str):
                    raise ValueError("date must be text")
                if re.fullmatch(r"\d{4}", raw_date):
                    year = int(raw_date)
                    date(year, 1, 1)
                    precision = "year"
                elif re.fullmatch(r"\d{4}-\d{2}", raw_date):
                    parsed = date.fromisoformat(raw_date + "-01")
                    year, precision = parsed.year, "month"
                else:
                    parsed = date.fromisoformat(raw_date)
                    year, precision = parsed.year, "day"
                result["date_precision"] = precision
                if result["publication_year"] is None:
                    result["publication_year"] = year
                elif result["publication_year"] != year:
                    issues.append("conflicting_publication_year_date")
            except (ValueError, TypeError, OverflowError):
                issues.append("invalid_publication_date")
                result["publication_date"] = None
                result["date_precision"] = "unknown"
        year = result["publication_year"]
        if year is not None and (
            isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999
        ):
            issues.append("invalid_publication_year")
            result["publication_year"] = None
        if result["publication_date"] is None:
            missing.append("publication_date")
            if result["publication_year"] is not None:
                result["date_precision"] = "year"
        result.setdefault("identifiers", [])
        for identifier in result["identifiers"]:
            if not isinstance(identifier, dict) or not identifier.get("valid", False):
                issues.append("invalid_identifier")
        for namespace, field in (("doi", "doi"), ("arxiv", "arxiv_id")):
            result.setdefault(
                field,
                next(
                    (
                        identifier.get("normalized")
                        for identifier in result["identifiers"]
                        if isinstance(identifier, dict)
                        and identifier.get("namespace") == namespace
                        and identifier.get("valid")
                    ),
                    None,
                ),
            )
        result["completeness"] = {
            field: bool(result.get(field))
            for field in ("title", "abstract", "authors", "categories", "publication_date")
        }
    elif result["kind"] == "citation_context":
        original = result.get("original_context", result.get("citation_text"))
        result["original_context"] = original if isinstance(original, str) else None
        result["normalized_context"] = text("original_context", original, required=True)
        if result["normalized_context"] is None:
            result["quarantine"] = True
            issues.append("missing_citation_text")
        result.setdefault("original_intent_label", None)
        result.setdefault("normalized_intent_label", None)
        if result["normalized_intent_label"] not in {"Background", "Method", "Result", None}:
            issues.append("unsupported_scicite_label")
            result["normalized_intent_label"] = None
        if result["normalized_intent_label"] is None:
            result["quarantine"] = True
            issues.append("missing_or_unsupported_intent_label")
    elif result["kind"] in {"citation_edge", "relation"}:
        result.setdefault("edge_provenance", result["provenance"])
    result["issues"] = sorted(set(issues))
    result["missing_fields"] = sorted(set(missing))
    result.setdefault("record_key", _record_key(result))
    if result["kind"] == "citation_context":
        result.setdefault("context_id", result["record_key"])
    # Models do not strip strings; in particular original_context and raw_values survive.
    return validate_canonical(result).model_dump(mode="json")
