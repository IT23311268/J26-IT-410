"""Conservative, disk-backed scholarly identity decisions with recoverable inputs.

Canonical IDs hash sorted cluster source keys. Identical input keys and settings
produce identical IDs independent of ingest order. Incremental rebuilds may change
an ID when its cluster membership changes; this is not a persistent identity
registry and callers must retain each run's mappings instead of assuming otherwise.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from ..schemas import Evidence, ExternalIdentifier, FrozenModel, NonEmpty, Paper, Probability
from .candidate_generation import CandidateGenerator
from .deduplication import SourceDeduplicator, deterministic_json
from .match_scoring import MATCH_CLASSIFICATIONS, RESOLVER_VERSION, score_match, verified_tokens
from .normalization import (
    AUTHORITATIVE_NAMESPACES,
    comparison_title,
    identifier_token,
    normalize_identifier,
)


class EntityResolution(FrozenModel):
    source: ExternalIdentifier
    canonical_paper_id: NonEmpty
    confidence: Probability
    method_version: NonEmpty
    evidence: Evidence


class EntityResolver:
    def resolve(self, papers: tuple[Paper, ...]) -> tuple[EntityResolution, ...]:
        raise NotImplementedError(
            "Legacy Paper records omit the source-level identity evidence required for Stage 03; "
            "use IdentityEngine with provenance-bearing normalized staging records."
        )

    def resolve_staging(self, records, *, index_path: str | Path, **options) -> dict:
        """New compatible entry point; the legacy return contract is unchanged."""
        with IdentityEngine(index_path, **options) as engine:
            for record in records:
                engine.add(record)
            return engine.finalize()


class IdentityEngine:
    """SQLite index, bounded candidates, explicit decisions and conflict-safe clusters.

    Only compatible shared source-verified identifiers can create a merge. Source
    supplied identifiers are evidence assertions, not online verification. Names
    and titles are review signals. Truncated candidate neighborhoods never create
    automatic merges. Raw normalized records are retained in ``identity_records``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_candidates: int = 50,
        memory_budget_mb: int = 64,
        batch_size: int = 1000,
        year_tolerance: int = 1,
        title_review_threshold: float = 0.9,
        material_title_conflict_threshold: float = 0.35,
        max_cluster_size: int = 256,
        max_identifiers_per_record: int = 64,
        candidate_methods: tuple[str, ...] = ("identifier", "title_year"),
        allowed_auto_merge_rules: tuple[str, ...] = ("shared_source_verified_identifier",),
    ):
        if memory_budget_mb < 8 or batch_size < 1 or max_cluster_size < 1:
            raise ValueError(
                "Memory budget must be >=8 MiB; batch and cluster sizes must be positive"
            )
        if max_identifiers_per_record < 1:
            raise ValueError("Identifier limit must be positive")
        if not 0 <= material_title_conflict_threshold <= title_review_threshold <= 1:
            raise ValueError("Title thresholds must satisfy 0 <= conflict <= review <= 1")
        if not set(allowed_auto_merge_rules) <= {"shared_source_verified_identifier"}:
            raise ValueError("Only verified identifier merge rules are supported")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute(f"PRAGMA cache_size=-{max(1024, memory_budget_mb * 1024 // 4)}")
        self.batch_size = batch_size
        self.memory_budget_mb = memory_budget_mb
        self.max_cluster_size = max_cluster_size
        self.max_identifiers_per_record = max_identifiers_per_record
        self.max_cluster_payload_bytes = memory_budget_mb * 1024 * 1024 // 8
        self.max_record_payload_bytes = min(1024 * 1024, self.max_cluster_payload_bytes)
        self.year_tolerance = year_tolerance
        self.title_review_threshold = title_review_threshold
        self.material_title_conflict_threshold = material_title_conflict_threshold
        self.allowed_auto_merge_rules = allowed_auto_merge_rules
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS identity_records(
              record_key TEXT PRIMARY KEY,title_key TEXT,publication_year INTEGER,
              source_dataset TEXT,dataset_version TEXT,source_record_id TEXT,
              payload_json TEXT,payload_bytes INTEGER,duplicate_json TEXT);
            CREATE INDEX IF NOT EXISTS identity_title_year_lookup
              ON identity_records(title_key,publication_year,record_key);
            CREATE INDEX IF NOT EXISTS identity_title_record_lookup
              ON identity_records(title_key,record_key);
            CREATE TABLE IF NOT EXISTS identity_ids(
              record_key TEXT,namespace TEXT,normalized TEXT,verified INTEGER,
              PRIMARY KEY(record_key,namespace,normalized));
            CREATE INDEX IF NOT EXISTS identity_token_lookup
              ON identity_ids(namespace,normalized,record_key);
            CREATE TABLE IF NOT EXISTS identity_candidates(
              record_key TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE IF NOT EXISTS identity_candidate_pairs(
              left_key TEXT,right_key TEXT,PRIMARY KEY(left_key,right_key));
            CREATE TABLE IF NOT EXISTS identity_decisions(
              left_key TEXT,right_key TEXT,decision TEXT,payload TEXT,
              PRIMARY KEY(left_key,right_key));
            CREATE INDEX IF NOT EXISTS identity_decisions_left
              ON identity_decisions(left_key,decision);
            CREATE INDEX IF NOT EXISTS identity_decisions_right
              ON identity_decisions(right_key,decision);
            CREATE TABLE IF NOT EXISTS identity_clusters(
              record_key TEXT PRIMARY KEY,cluster_key TEXT,canonical_paper_id TEXT,status TEXT);
            CREATE INDEX IF NOT EXISTS identity_cluster_members ON identity_clusters(cluster_key);
            CREATE TABLE IF NOT EXISTS identity_meta(key TEXT PRIMARY KEY,value TEXT);
        """)
        self.deduplicator = SourceDeduplicator(self.db)
        self.generator = CandidateGenerator(
            self.db,
            max_candidates=max_candidates,
            year_tolerance=year_tolerance,
            methods=candidate_methods,
        )
        self._prepared = False
        self._closed = False
        self._finalized = self._get_meta("dirty") == "0"
        settings = deterministic_json(
            {
                "max_candidates": max_candidates,
                "memory_budget_mb": memory_budget_mb,
                "batch_size": batch_size,
                "year_tolerance": year_tolerance,
                "title_review_threshold": title_review_threshold,
                "material_title_conflict_threshold": material_title_conflict_threshold,
                "max_cluster_size": max_cluster_size,
                "max_identifiers_per_record": max_identifiers_per_record,
                "candidate_methods": candidate_methods,
                "allowed_auto_merge_rules": allowed_auto_merge_rules,
                "resolver_version": RESOLVER_VERSION,
            }
        )
        if self._get_meta("settings") != settings:
            self._set_meta("settings", settings)
            self._set_meta("dirty", "1")
            self._finalized = False

    def _get_meta(self, key):
        row = self.db.execute("SELECT value FROM identity_meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO identity_meta VALUES (?,?)", (key, value))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self.db.commit()
            self.db.close()
            self._closed = True

    def add(self, record: dict):
        """Retain one bounded normalized record; duplicate keys must be identical."""
        record = dict(record)
        if not isinstance(record.get("record_key"), str) or not record["record_key"]:
            raise ValueError("record_key is required")
        if len(record["record_key"]) > 512:
            raise ValueError("record_key exceeds 512 characters; keep raw source ID separately")
        if not isinstance(record.get("source_dataset"), str) or not record["source_dataset"]:
            raise ValueError("source_dataset is required")
        identifiers = record.get("identifiers", [])
        if len(identifiers) > self.max_identifiers_per_record:
            raise ValueError("Record exceeds configured identifier limit; retain it in quarantine")
        normalized = []
        for identifier in identifiers:
            scope = identifier.get("scope") or {}
            normalized_identifier = normalize_identifier(
                identifier["namespace"],
                identifier.get("value", identifier.get("raw", identifier.get("normalized"))),
                dataset=identifier.get("dataset")
                or scope.get("dataset")
                or record["source_dataset"],
                version=(
                    identifier.get("dataset_version")
                    or scope.get("version")
                    or record.get("dataset_version")
                ),
                node_type=identifier.get("node_type") or scope.get("node_type"),
                verified=identifier.get("verified", False),
            )
            if normalized_identifier["version"] is None and identifier.get("version"):
                normalized_identifier["version"] = identifier["version"]
            normalized_identifier["original_namespace"] = identifier.get(
                "original_namespace", normalized_identifier["original_namespace"]
            )
            normalized.append({**identifier, **normalized_identifier})
        record["identifiers"] = normalized
        payload = deterministic_json(record)
        payload_bytes = len(payload.encode("utf-8"))
        if payload_bytes > self.max_record_payload_bytes:
            raise ValueError("Record exceeds configured payload budget; retain it in quarantine")
        existing = self.db.execute(
            "SELECT payload_json FROM identity_records WHERE record_key=?", (record["record_key"],)
        ).fetchone()
        if existing:
            if existing[0] != payload:
                raise ValueError("A source record_key cannot overwrite different source values")
            return {"record_key": record["record_key"], "already_indexed": True}
        duplicate = self.deduplicator.observe(record)
        self.db.execute(
            "INSERT INTO identity_records VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record["record_key"],
                comparison_title(record.get("title")),
                record.get("publication_year"),
                record["source_dataset"],
                record.get("dataset_version"),
                record.get("source_record_id"),
                payload,
                payload_bytes,
                deterministic_json(duplicate),
            ),
        )
        for identifier in normalized:
            token = identifier_token(identifier)
            if token:
                self.db.execute(
                    "INSERT INTO identity_ids VALUES (?,?,?,?) ON CONFLICT(record_key,namespace,"
                    "normalized) DO UPDATE SET verified=MAX(verified,excluded.verified)",
                    (record["record_key"], *token, int(identifier.get("verified", False))),
                )
        self._prepared = False
        self._finalized = False
        self._set_meta("dirty", "1")
        return duplicate

    def _record(self, key: str) -> dict:
        row = self.db.execute(
            "SELECT payload_json FROM identity_records WHERE record_key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def iter_candidates(self):
        """Generate and persist candidates without resolving or merging identities."""
        if not self._prepared:
            self.generator.prepare()
            self.db.execute("DELETE FROM identity_candidates")
            self.db.execute("DELETE FROM identity_candidate_pairs")
            self._prepared = True
        for number, result in enumerate(
            self.generator.iter_records(batch_size=self.batch_size), start=1
        ):
            key = result["record_key"]
            self.db.execute(
                "INSERT OR REPLACE INTO identity_candidates VALUES (?,?)",
                (key, deterministic_json(result)),
            )
            for candidate in result["candidates"]:
                left, right = sorted((key, candidate["record_key"]))
                self.db.execute(
                    "INSERT OR IGNORE INTO identity_candidate_pairs VALUES (?,?)", (left, right)
                )
            yield result
            if number % self.batch_size == 0:
                self.db.commit()
        self.db.commit()

    def _save_decision(self, decision):
        self.db.execute(
            "INSERT OR REPLACE INTO identity_decisions VALUES (?,?,?,?)",
            (
                decision["left_record_key"],
                decision["right_record_key"],
                decision["decision"],
                deterministic_json(decision),
            ),
        )

    def _cluster(self, key):
        return self.db.execute(
            "SELECT cluster_key FROM identity_clusters WHERE record_key=?", (key,)
        ).fetchone()[0]

    def _cluster_compatibility(self, first: str, second: str) -> list[dict]:
        counts = self.db.execute(
            "SELECT COUNT(*),SUM(r.payload_bytes) FROM identity_clusters c "
            "JOIN identity_records r USING(record_key) WHERE cluster_key IN (?,?)",
            (first, second),
        ).fetchone()
        if counts[0] > self.max_cluster_size or counts[1] > self.max_cluster_payload_bytes:
            return [{"reason": "cluster_resource_budget", "records": counts[0]}]
        # Cluster expansion is bounded above. Check every cross-cluster metadata
        # pair so a bridge cannot conceal a contradiction elsewhere in a cluster.
        conflicts = []
        for (left_key,) in self.db.execute(
            "SELECT record_key FROM identity_clusters WHERE cluster_key=? ORDER BY record_key",
            (first,),
        ).fetchall():
            left_record = self._record(left_key)
            for (right_key,) in self.db.execute(
                "SELECT record_key FROM identity_clusters WHERE cluster_key=? ORDER BY record_key",
                (second,),
            ).fetchall():
                result = score_match(
                    left_record,
                    self._record(right_key),
                    year_tolerance=self.year_tolerance,
                    title_review_threshold=self.title_review_threshold,
                    material_title_conflict_threshold=self.material_title_conflict_threshold,
                )
                if result["decision"] == "CONFLICTED":
                    conflicts.append(
                        {
                            "left_record_key": left_key,
                            "right_record_key": right_key,
                            "identifier_conflicts": result["features"]["identifier_conflicts"],
                            "material_metadata_conflicts": result["features"][
                                "material_metadata_conflicts"
                            ],
                        }
                    )
                    return conflicts
        return conflicts

    def finalize(self) -> dict:
        if self._finalized:
            return json.loads(self._get_meta("summary"))
        for _ in self.iter_candidates():
            pass
        self.db.execute("DELETE FROM identity_decisions")
        self.db.execute("DELETE FROM identity_clusters")
        self.db.execute(
            "INSERT INTO identity_clusters(record_key,cluster_key) "
            "SELECT record_key,record_key FROM identity_records"
        )
        for left_key, right_key in self.db.execute(
            "SELECT left_key,right_key FROM identity_candidate_pairs ORDER BY left_key,right_key"
        ):
            left, right = self._record(left_key), self._record(right_key)
            decision = score_match(
                left,
                right,
                year_tolerance=self.year_tolerance,
                title_review_threshold=self.title_review_threshold,
                material_title_conflict_threshold=self.material_title_conflict_threshold,
            )
            decision["source_records"] = [self._source_ref(left), self._source_ref(right)]
            if decision["decision"] == "EXACT_VERIFIED":
                truncated = any(
                    json.loads(
                        self.db.execute(
                            "SELECT payload FROM identity_candidates WHERE record_key=?", (key,)
                        ).fetchone()[0]
                    )["truncated"]
                    for key in (left_key, right_key)
                )
                if truncated:
                    decision.update(
                        decision="AMBIGUOUS",
                        rule="candidate_bound_requires_review",
                        review_status="pending",
                    )
            self._save_decision(decision)
        self.db.executescript("""
            DROP TABLE IF EXISTS identity_probable_counts;
            CREATE TABLE identity_probable_counts AS
              SELECT record_key,COUNT(*) n FROM (
                SELECT left_key record_key FROM identity_decisions WHERE decision='PROBABLE_REVIEW'
                UNION ALL
                SELECT right_key record_key FROM identity_decisions WHERE decision='PROBABLE_REVIEW'
              ) GROUP BY record_key;
            CREATE UNIQUE INDEX identity_probable_record ON identity_probable_counts(record_key);
        """)
        for left_key, right_key, payload in self.db.execute(
            "SELECT left_key,right_key,payload FROM identity_decisions "
            "WHERE decision='PROBABLE_REVIEW' ORDER BY left_key,right_key"
        ):
            if self.db.execute(
                "SELECT 1 FROM identity_probable_counts WHERE record_key IN (?,?) AND n>1 LIMIT 1",
                (left_key, right_key),
            ).fetchone():
                decision = json.loads(payload)
                decision.update(decision="AMBIGUOUS", rule="multiple_metadata_candidates")
                self._save_decision(decision)
        merges = 0
        for left_key, right_key, payload in self.db.execute(
            "SELECT left_key,right_key,payload FROM identity_decisions "
            "WHERE decision='EXACT_VERIFIED' ORDER BY left_key,right_key"
        ):
            decision = json.loads(payload)
            first, second = self._cluster(left_key), self._cluster(right_key)
            conflicts = self._cluster_compatibility(first, second) if first != second else []
            if conflicts:
                resource_only = all(
                    item.get("reason") == "cluster_resource_budget" for item in conflicts
                )
                decision.update(
                    decision="AMBIGUOUS" if resource_only else "CONFLICTED",
                    rule="cluster_resource_budget"
                    if resource_only
                    else "cluster_identity_conflict",
                    cluster_conflicts=conflicts,
                    review_status="pending",
                )
            elif decision["rule"] in self.allowed_auto_merge_rules:
                if first != second:
                    chosen, other = sorted((first, second))
                    self.db.execute(
                        "UPDATE identity_clusters SET cluster_key=? WHERE cluster_key=?",
                        (chosen, other),
                    )
                    merges += 1
                decision["automatic_merge"] = True
                decision["new_cluster_merge"] = first != second
                decision["cluster_compatibility_checked"] = True
            self._save_decision(decision)
        for (cluster,) in self.db.execute(
            "SELECT DISTINCT cluster_key FROM identity_clusters ORDER BY cluster_key"
        ):
            keys = [
                key
                for (key,) in self.db.execute(
                    "SELECT record_key FROM identity_clusters "
                    "WHERE cluster_key=? ORDER BY record_key",
                    (cluster,),
                )
            ]
            canonical_id = (
                "paper_" + hashlib.sha256(deterministic_json(keys).encode()).hexdigest()[:32]
            )
            self.db.execute(
                "UPDATE identity_clusters SET canonical_paper_id=? WHERE cluster_key=?",
                (canonical_id, cluster),
            )
        for (key,) in self.db.execute(
            "SELECT record_key FROM identity_records ORDER BY record_key"
        ):
            observed = {
                value
                for (value,) in self.db.execute(
                    "SELECT decision FROM identity_decisions WHERE left_key=? OR right_key=?",
                    (key, key),
                )
            }
            record = self._record(key)
            status = next(
                (
                    value
                    for value in ("CONFLICTED", "AMBIGUOUS", "EXACT_VERIFIED", "PROBABLE_REVIEW")
                    if value in observed
                ),
                "UNMATCHED"
                if comparison_title(record.get("title"))
                or any(identifier_token(identifier) for identifier in record["identifiers"])
                else "UNRESOLVABLE",
            )
            internal_conflict = any(
                len(values) > 1
                for namespace, values in verified_tokens(record).items()
                if namespace in AUTHORITATIVE_NAMESPACES
            )
            if internal_conflict:
                status = "CONFLICTED"
            self.db.execute(
                "UPDATE identity_clusters SET status=? WHERE record_key=?", (status, key)
            )
        pair_counts = Counter(
            dict(
                self.db.execute(
                    "SELECT decision,COUNT(*) FROM identity_decisions GROUP BY decision"
                ).fetchall()
            )
        )
        record_counts = Counter(
            dict(
                self.db.execute(
                    "SELECT status,COUNT(*) FROM identity_clusters GROUP BY status"
                ).fetchall()
            )
        )
        counts = self.db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT canonical_paper_id) FROM identity_clusters"
        ).fetchone()
        truncated_count = 0
        candidate_memberships = 0
        for (payload,) in self.db.execute("SELECT payload FROM identity_candidates"):
            value = json.loads(payload)
            truncated_count += int(value["truncated"])
            candidate_memberships += value["candidate_count"]
        duplicate_count = sum(
            json.loads(payload)["duplicate"]
            for (payload,) in self.db.execute("SELECT duplicate_json FROM identity_records")
        )
        summary = {
            "resolver_version": RESOLVER_VERSION,
            "input_records": counts[0],
            "canonical_papers": counts[1],
            "automatic_cluster_merges": merges,
            "within_source_duplicate_records": duplicate_count,
            "pair_classifications": {label: pair_counts[label] for label in MATCH_CLASSIFICATIONS},
            "record_classifications": {
                label: record_counts[label] for label in MATCH_CLASSIFICATIONS
            },
            "candidates_examined": sum(pair_counts.values()),
            "candidate_memberships_returned": candidate_memberships,
            "records_with_truncated_candidates": truncated_count,
            "candidate_limit": self.generator.max_candidates,
            "candidate_methods": list(self.generator.methods),
            "automatic_merge_rules": list(self.allowed_auto_merge_rules),
            "score_calibration": "not_calibrated_not_probability",
            "canonical_id_stability": (
                "Sorted source-key membership hash; stable for identical inputs and configuration. "
                "Incremental membership changes may change IDs; retain versioned run mappings."
            ),
            "resource_limits": {
                "memory_budget_mb": self.memory_budget_mb,
                "sqlite_cache_kib": max(1024, self.memory_budget_mb * 1024 // 4),
                "max_cluster_size": self.max_cluster_size,
                "max_cluster_payload_bytes": self.max_cluster_payload_bytes,
                "max_record_payload_bytes": self.max_record_payload_bytes,
                "max_identifiers_per_record": self.max_identifiers_per_record,
                "max_record_key_characters": 512,
                "memory_budget_interpretation": (
                    "Bounds SQLite cache and retained payload/candidate expansion; "
                    "not an operating-system process RSS limit."
                ),
            },
        }
        self._set_meta("summary", deterministic_json(summary))
        self._set_meta("dirty", "0")
        self.db.commit()
        self._finalized = True
        return summary

    @staticmethod
    def _source_ref(record):
        return {
            "record_key": record["record_key"],
            "source_dataset": record["source_dataset"],
            "dataset_version": record.get("dataset_version"),
            "source_record_id": record.get("source_record_id"),
        }

    def lookup(self, record_key: str) -> str | None:
        self.finalize()
        row = self.db.execute(
            "SELECT canonical_paper_id FROM identity_clusters WHERE record_key=?", (record_key,)
        ).fetchone()
        return row[0] if row else None

    def iter_mappings(self):
        self.finalize()
        for key, canonical_id, status in self.db.execute(
            "SELECT record_key,canonical_paper_id,status FROM identity_clusters ORDER BY record_key"
        ):
            record = self._record(key)
            yield {
                **self._source_ref(record),
                "canonical_paper_id": canonical_id,
                "identity_status": status,
                "identifiers": record["identifiers"],
                "resolver_version": RESOLVER_VERSION,
            }

    def iter_decisions(self):
        self.finalize()
        for (payload,) in self.db.execute(
            "SELECT payload FROM identity_decisions ORDER BY left_key,right_key"
        ):
            yield json.loads(payload)
        for key, status in self.db.execute(
            "SELECT record_key,status FROM identity_clusters c WHERE NOT EXISTS "
            "(SELECT 1 FROM identity_decisions d WHERE d.left_key=c.record_key "
            "OR d.right_key=c.record_key) "
            "ORDER BY record_key"
        ):
            record = self._record(key)
            yield {
                "left_record_key": key,
                "right_record_key": None,
                "decision": status,
                "rule": "no_acceptable_indexed_candidate",
                "features": {},
                "evidence": [],
                "heuristic_score": 0.0,
                "score_interpretation": "uncalibrated_ranking_score_not_probability",
                "automatic_merge": False,
                "source_records": [self._source_ref(record)],
                "resolver_version": RESOLVER_VERSION,
                "review_status": "pending",
            }

    def iter_papers(self):
        self.finalize()
        for cluster, canonical_id in self.db.execute(
            "SELECT DISTINCT cluster_key,canonical_paper_id FROM identity_clusters "
            "ORDER BY cluster_key"
        ):
            records = [
                self._record(key)
                for (key,) in self.db.execute(
                    "SELECT record_key FROM identity_clusters "
                    "WHERE cluster_key=? ORDER BY record_key",
                    (cluster,),
                )
            ]

            def first(field, records=records):
                return next(
                    (record[field] for record in records if record.get(field) is not None), None
                )

            identifiers = []
            seen_ids = set()
            for record in records:
                for identifier in record["identifiers"]:
                    key = deterministic_json(identifier)
                    if key not in seen_ids:
                        seen_ids.add(key)
                        identifiers.append(identifier)
            provenance = []
            alternatives = {}
            conflict_flags = []
            for field in (
                "title",
                "abstract",
                "publication_year",
                "publication_date",
                "venue",
                "authors",
                "categories",
            ):
                values = []
                seen_values = set()
                for record in records:
                    if record.get(field) is not None:
                        value_key = deterministic_json(record[field])
                        if value_key not in seen_values:
                            seen_values.add(value_key)
                            values.append(
                                {"value": record[field], "source": self._source_ref(record)}
                            )
                alternatives[field] = values
                if len(values) > 1:
                    conflict_flags.append(f"alternative_{field}_values")
            for record in records:
                for item in record.get("provenance", []):
                    provenance.append({**self._source_ref(record), **item})
            statuses = sorted(
                {
                    status
                    for (status,) in self.db.execute(
                        "SELECT status FROM identity_clusters WHERE cluster_key=?", (cluster,)
                    )
                }
            )
            if "CONFLICTED" in statuses:
                conflict_flags.append("conflicting_identity_candidate_evidence")
            dominant_status = next(
                status
                for status in (
                    "CONFLICTED",
                    "AMBIGUOUS",
                    "EXACT_VERIFIED",
                    "PROBABLE_REVIEW",
                    "UNMATCHED",
                    "UNRESOLVABLE",
                )
                if status in statuses
            )
            contributors = [
                {
                    "source": self._source_ref(record),
                    "eligibility": record.get("temporal_eligibility"),
                }
                for record in records
            ]
            eligible = all(
                isinstance(item["eligibility"], dict)
                and item["eligibility"].get("eligible") is True
                for item in contributors
            )
            unknown = any(
                not isinstance(item["eligibility"], dict)
                or item["eligibility"].get("unknown_availability", True)
                for item in contributors
            )
            temporal_reasons = sorted(
                {
                    reason
                    for item in contributors
                    if isinstance(item["eligibility"], dict)
                    for reason in item["eligibility"].get("reasons", [])
                }
            )
            if any(item["eligibility"] is None for item in contributors):
                temporal_reasons.append("missing_source_temporal_assessment")
            temporal = {
                "eligible": eligible,
                "unknown_availability": unknown,
                "reasons": sorted(set(temporal_reasons)),
                "aggregation_rule": "all_contributing_source_records_must_be_eligible",
                "contributors": contributors,
                "status": "ELIGIBLE_STAGING" if eligible else "EXCLUDED_FROM_HISTORICAL_FEATURES",
            }

            def external(namespace, identifiers=identifiers):
                return next(
                    (
                        item["normalized"]
                        for item in identifiers
                        if item["namespace"] == namespace and item.get("valid")
                    ),
                    None,
                )

            yield {
                "canonical_paper_id": canonical_id,
                "title": first("title"),
                "abstract": first("abstract"),
                "publication_year": first("publication_year"),
                "publication_date": first("publication_date"),
                "date_precision": first("date_precision"),
                "doi": external("doi"),
                "arxiv_id": external("arxiv"),
                "external_ids": identifiers,
                "identifiers": identifiers,
                "source_record_ids": [self._source_ref(record) for record in records],
                "authors": first("authors") or [],
                "categories": first("categories") or [],
                "provenance": provenance,
                "completeness": {
                    field: first(field) is not None
                    for field in ("title", "abstract", "publication_year", "publication_date")
                },
                "conflict_flags": sorted(set(conflict_flags)),
                "field_alternatives": alternatives,
                "raw_values": [
                    {"source": self._source_ref(record), "values": record.get("raw_values", {})}
                    for record in records
                ],
                "identity_status": dominant_status,
                "record_identity_statuses": statuses,
                "temporal_eligibility": temporal,
                "resolver_version": RESOLVER_VERSION,
            }
