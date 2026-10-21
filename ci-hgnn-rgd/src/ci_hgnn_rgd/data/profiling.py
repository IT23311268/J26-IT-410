"""One-pass profile with disk-backed exact duplicates and split-overlap checks."""

import hashlib
import json
from collections import Counter, defaultdict

from .identifier_audit import IdentifierIndex
from .schema_inspector import SchemaInspector


class Profiler:
    def __init__(self, index_path, *, short_text=20, long_text=10000):
        self.schema = SchemaInspector()
        self.index = IdentifierIndex(index_path)
        self.index.db.execute("CREATE TABLE lengths(value INTEGER)")
        self.index.db.execute("CREATE TABLE source_keys(field TEXT, value TEXT)")
        self.issues = Counter()
        self.missing = Counter()
        self.labels = defaultdict(Counter)
        self.categories, self.years = Counter(), Counter()
        self.count = self.valid = self.invalid = self.malformed = 0
        self.canonical_count = 0
        self.short_text, self.long_text = short_text, long_text
        self.text_flags = Counter()
        self.identifier_patterns = Counter()

    def observe(self, event, canonical, issues):
        self.count += 1
        self.valid += not bool(issues)
        self.invalid += bool(issues)
        self.issues.update(issues)
        key = f"{event.source}:line:{event.line}"
        if event.record is not None:
            self.schema.observe(event.record)
            for field in ("id", "unique_id"):
                value = event.record.get(field)
                if isinstance(value, str):
                    self.index.db.execute("INSERT INTO source_keys VALUES (?,?)", (field, value))
        else:
            self.malformed += 1
        text, source_id = None, None
        if canonical is not None:
            self.canonical_count += 1
            source_id = canonical.source_record_id
            if canonical.kind == "citation_context":
                text = canonical.citation_text
                label = canonical.original_intent_label
                self.labels[event.split or "unknown"][label or "<missing>"] += 1
                self.missing["citation_text"] += not bool(text and text.strip())
                self.missing["citation_label"] += label is None
                for role in ("citing", "cited"):
                    identifier = getattr(canonical, f"{role}_paper_identifier")
                    self.missing[f"{role}_paper_identifier"] += identifier is None
                    self.index.add_entity(f"{key}:{role}", (identifier,) if identifier else ())
                    if identifier:
                        value = identifier.value
                        pattern = (
                            "hex40"
                            if len(value) == 40
                            and all(c in "0123456789abcdefABCDEF" for c in value)
                            else "other_source_string"
                        )
                        self.identifier_patterns[f"{identifier.namespace}:{pattern}"] += 1
            else:
                text = canonical.abstract
                self.index.add_entity(key, canonical.external_ids, canonical.title)
                self.missing.update(
                    {
                        field: not present
                        for field, present in canonical.metadata_completeness.items()
                    }
                )
                if canonical.publication_year is not None:
                    self.years[canonical.publication_year] += 1
                for category in canonical.subject_categories:
                    if category in self.categories or len(self.categories) < 2000:
                        self.categories[category] += 1
                    else:
                        self.issues["category_cardinality_limit"] += 1
            self.missing["source_record_id"] += source_id is None
        if text is not None:
            length = len(text)
            self.index.db.execute("INSERT INTO lengths VALUES (?)", (length,))
            self.text_flags["short"] += length < self.short_text
            self.text_flags["long"] += length > self.long_text
            self.text_flags["replacement_character"] += "\ufffd" in text
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        record_hash = (
            hashlib.sha256(
                json.dumps(event.record, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if event.record is not None
            else event.sha256
        )
        self.index.db.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?)",
            (key, source_id, record_hash, digest, event.split, not bool(issues)),
        )

    def report(self):
        summary = self.index.summary()
        candidate_keys = {}
        for field, count in self.index.db.execute(
            "SELECT field,COUNT(*) FROM source_keys GROUP BY field"
        ):
            distinct = self.index.db.execute(
                "SELECT COUNT(DISTINCT value) FROM source_keys WHERE field=?", (field,)
            ).fetchone()[0]
            candidate_keys[field] = {
                "present": count,
                "distinct": distinct,
                "repeated_occurrences": count - distinct,
            }
        overlap_examples = [
            {"text_sha256": key, "splits": splits.split(",")}
            for key, splits in self.index.db.execute(
                "SELECT text_hash,GROUP_CONCAT(DISTINCT split) FROM records WHERE "
                "text_hash IS NOT NULL GROUP BY text_hash HAVING COUNT(DISTINCT split)>1 "
                "LIMIT 20"
            )
        ]
        summary["candidate_source_keys"] = candidate_keys
        summary["cross_split_overlap_examples"] = overlap_examples
        lengths = self.index.db.execute(
            "SELECT COUNT(*),MIN(value),MAX(value),AVG(value) FROM lengths"
        ).fetchone()
        quantiles = {}
        if lengths[0]:
            for percentile in (50, 95, 99):
                quantiles[f"p{percentile}"] = self.index.db.execute(
                    "SELECT value FROM lengths ORDER BY value LIMIT 1 OFFSET ?",
                    (int((lengths[0] - 1) * percentile / 100),),
                ).fetchone()[0]
        return {
            "raw_record_count": self.count,
            "valid_record_count": self.valid,
            "invalid_record_count": self.invalid,
            "malformed_records": self.malformed,
            "issues": dict(self.issues),
            "missing_counts": dict(self.missing),
            "completeness_denominator": self.canonical_count,
            "missing_rates": {
                key: count / self.canonical_count if self.canonical_count else None
                for key, count in self.missing.items()
            },
            "label_distribution_by_split": {key: dict(value) for key, value in self.labels.items()},
            "category_distribution": dict(self.categories),
            "year_distribution": dict(self.years),
            "text_length": {
                "count": lengths[0],
                "min": lengths[1],
                "max": lengths[2],
                "mean": lengths[3],
                **quantiles,
            },
            "text_flags": dict(self.text_flags),
            "text_thresholds": {"short": self.short_text, "long": self.long_text},
            "identifier_patterns": dict(self.identifier_patterns),
            "language_information": "not inferred; inspect observed raw fields",
            **summary,
        }

    def close(self):
        self.index.close()
