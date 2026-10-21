"""Synthetic, offline cleaning/contract tests; no scholarly results are asserted."""

import copy
import json

import pytest

from ci_hgnn_rgd.data.canonical_records import (
    CanonicalAuthor,
    CanonicalCitation,
    CanonicalCitationContext,
    CanonicalField,
    CanonicalPaper,
    validate_canonical,
)
from ci_hgnn_rgd.data.cleaning import clean_record, clean_text, json_safe


def synthetic_paper(**changes):
    return {
        "kind": "paper",
        "source_dataset": "synthetic",
        "title": "Synthetic paper",
        "authors": ["Ada A", "B. B"],
        "categories": ["synthetic-field"],
        **changes,
    }


def test_unicode_cleaning_preserves_scientific_symbols_and_original_case():
    original = "  Cafe\u0301:\tE = mc²; α + β ≤ 10⁻³\n$\\sum_i x_i$  "
    cleaned = clean_record(synthetic_paper(title=original))
    assert cleaned["title"] == "Café: E = mc²; α + β ≤ 10⁻³ $\\sum_i x_i$"
    assert cleaned["raw_values"]["title"] == original
    assert cleaned["normalized_fields"]["title"] == cleaned["title"]


def test_cleaning_does_not_mutate_source_or_author_order():
    source = synthetic_paper(authors=["  Zed Z ", " Ada\nA "])
    before = copy.deepcopy(source)
    cleaned = clean_record(source)
    assert source == before
    assert cleaned["authors"] == ["Zed Z", "Ada A"]
    assert cleaned["raw_values"]["authors"] == ["  Zed Z ", " Ada\nA "]


def test_citation_original_text_label_markers_and_split_remain_exact():
    original = "  Synthetic\nformula: $a_i$ [12]; α².  "
    source = {
        "kind": "citation_context",
        "source_dataset": "scicite",
        "original_context": original,
        "original_intent_label": "method",
        "normalized_intent_label": "Method",
        "source_split": "dev",
    }
    cleaned = clean_record(source)
    assert cleaned["original_context"] == original
    assert cleaned["normalized_context"] == "Synthetic formula: $a_i$ [12]; α²."
    assert cleaned["original_intent_label"] == "method"
    assert cleaned["source_split"] == "dev"
    assert not cleaned["quarantine"]


def test_cleaning_is_deterministic_and_idempotent():
    source = synthetic_paper(title="  Synthetic\nTitle  ", abstract="  αβ  ")
    first = clean_record(source)
    assert first == clean_record(source)
    assert first == clean_record(first)


@pytest.mark.parametrize(
    "value,precision,year",
    [
        ("2020", "year", 2020),
        ("2020-02", "month", 2020),
        ("2020-02-29", "day", 2020),
    ],
)
def test_publication_precision_is_not_filled_with_invented_days(value, precision, year):
    cleaned = clean_record(synthetic_paper(publication_date=value))
    assert cleaned["publication_date"] == value
    assert cleaned["date_precision"] == precision
    assert cleaned["publication_year"] == year


def test_invalid_date_is_reported_but_raw_is_recoverable():
    cleaned = clean_record(synthetic_paper(publication_date="2021-02-29"))
    assert cleaned["publication_date"] is None
    assert cleaned["raw_values"]["publication_date"] == "2021-02-29"
    assert "invalid_publication_date" in cleaned["issues"]


def test_missing_publication_date_is_unknown_not_imputed():
    cleaned = clean_record(synthetic_paper())
    assert cleaned["publication_date"] is None
    assert cleaned["publication_year"] is None
    assert cleaned["date_precision"] == "unknown"
    assert "publication_date" in cleaned["missing_fields"]


def test_year_alone_has_year_precision_without_date_fabrication():
    cleaned = clean_record(synthetic_paper(publication_year=2018))
    assert cleaned["publication_date"] is None
    assert cleaned["date_precision"] == "year"


def test_contradicting_year_and_date_are_preserved_for_review():
    cleaned = clean_record(synthetic_paper(publication_year=2018, publication_date="2020-01-01"))
    assert "conflicting_publication_year_date" in cleaned["issues"]
    assert cleaned["publication_year"] == 2018
    assert cleaned["publication_date"] == "2020-01-01"


def test_unexpected_text_type_is_quarantined_without_coercion():
    cleaned = clean_record(synthetic_paper(title={"unexpected": "object"}))
    assert cleaned["title"] is None
    assert cleaned["quarantine"]
    assert cleaned["raw_values"]["title"] == {"unexpected": "object"}
    assert "unexpected_type:title" in cleaned["issues"]


def test_suspect_encoding_is_reported_not_rewritten():
    cleaned = clean_record(synthetic_paper(title="Synthetic \ufffd text"))
    assert cleaned["title"] == "Synthetic \ufffd text"
    assert "suspected_encoding:title" in cleaned["issues"]
    assert not cleaned["quarantine"]


def test_nonfinite_source_values_are_losslessly_diagnosed_with_json_safe_markers():
    safe = json_safe({"x": float("nan"), "y": [float("inf"), float("-inf")]})
    assert safe == {
        "x": {"__source_nonfinite__": "NaN"},
        "y": [{"__source_nonfinite__": "Infinity"}, {"__source_nonfinite__": "-Infinity"}],
    }
    json.dumps(safe, allow_nan=False)


def test_null_like_optional_text_is_preserved_raw_and_marked_missing():
    cleaned = clean_record(synthetic_paper(abstract=" N/A "))
    assert cleaned["abstract"] is None
    assert "abstract" in cleaned["missing_fields"]
    assert cleaned["raw_values"]["abstract"] == " N/A "


def test_source_categories_are_retained_without_topic_interpretation():
    cleaned = clean_record(synthetic_paper(categories=["cs.AI", "math.AG"]))
    assert cleaned["categories"] == ["cs.AI", "math.AG"]
    assert "investigates_topic" not in cleaned


@pytest.mark.parametrize(
    "kind,model",
    [
        ("paper", CanonicalPaper),
        ("author", CanonicalAuthor),
        ("citation_edge", CanonicalCitation),
        ("citation_context", CanonicalCitationContext),
        ("source_field", CanonicalField),
    ],
)
def test_typed_staging_contracts_preserve_raw_string_whitespace(kind, model):
    record = {
        "record_key": "synthetic:1",
        "kind": kind,
        "source_dataset": "synthetic",
        "raw_values": {"title": "  Keep  "},
    }
    validated = validate_canonical(record)
    assert isinstance(validated, model)
    assert validated.raw_values["title"] == "  Keep  "


def test_clean_text_does_not_lowercase_or_remove_citation_punctuation():
    assert clean_text("  CNNs & RNNs: [1] $\\beta$?  ") == "CNNs & RNNs: [1] $\\beta$?"
