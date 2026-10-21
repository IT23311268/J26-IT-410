"""Indexed, bounded candidate retrieval; no all-pairs paper comparisons.

Each block retrieves at most ``max_candidates + 1`` rows. Block membership counts
are computed once in SQLite; overlapping block populations are reported as bounds,
not invented exact counts. Highly repetitive blocks cannot expand Python memory.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3


class CandidateGenerator:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        max_candidates: int = 50,
        year_tolerance: int = 1,
        methods: tuple[str, ...] = ("identifier", "title_year"),
    ):
        if max_candidates < 1 or not 0 <= year_tolerance <= 10:
            raise ValueError("Candidate limit must be positive and year tolerance between 0 and 10")
        if not set(methods) <= {"identifier", "title_year"}:
            raise ValueError("Unsupported candidate method; use identifier and/or title_year")
        self.db = connection
        self.max_candidates = max_candidates
        self.year_tolerance = year_tolerance
        self.methods = methods

    def prepare(self):
        self.db.executescript("""
            DROP TABLE IF EXISTS identity_id_block_counts;
            CREATE TABLE identity_id_block_counts AS
              SELECT namespace,normalized,COUNT(DISTINCT record_key) members
              FROM identity_ids GROUP BY namespace,normalized;
            CREATE UNIQUE INDEX identity_id_block_count_lookup
              ON identity_id_block_counts(namespace,normalized);
            DROP TABLE IF EXISTS identity_title_block_counts;
            CREATE TABLE identity_title_block_counts AS
              SELECT title_key,COALESCE(publication_year,-1) publication_year,COUNT(*) members
              FROM identity_records WHERE title_key IS NOT NULL
              GROUP BY title_key,COALESCE(publication_year,-1);
            CREATE UNIQUE INDEX identity_title_block_count_lookup
              ON identity_title_block_counts(title_key,publication_year);
        """)

    def for_record(self, record_key: str) -> dict:
        row = self.db.execute(
            "SELECT title_key,publication_year FROM identity_records WHERE record_key=?",
            (record_key,),
        ).fetchone()
        if row is None:
            raise KeyError(record_key)
        candidates: dict[str, dict] = {}
        blocks = []

        def retrieve(
            method, query, parameters, membership, evidence, priority, *, contains_self=True
        ):
            count = max(membership - int(contains_self), 0)
            returned = (
                self.db.execute(
                    query + " ORDER BY record_key LIMIT ?", (*parameters, self.max_candidates + 1)
                ).fetchall()
                if count
                else []
            )
            blocks.append(
                {
                    "method": method,
                    "other_record_memberships": count,
                    "retrieved_memberships": min(len(returned), self.max_candidates),
                    "truncated": len(returned) > self.max_candidates,
                    "evidence": evidence,
                }
            )
            for (candidate_key,) in returned[: self.max_candidates]:
                candidate = candidates.setdefault(
                    candidate_key,
                    {"record_key": candidate_key, "methods": [], "evidence": [], "priority": 9},
                )
                candidate["methods"].append(method)
                candidate["evidence"].append(evidence)
                candidate["priority"] = min(candidate["priority"], priority)

        if "identifier" in self.methods:
            for namespace, value, verified, members in self.db.execute(
                "SELECT i.namespace,i.normalized,MAX(i.verified),b.members FROM identity_ids i "
                "JOIN identity_id_block_counts b USING(namespace,normalized) "
                "WHERE i.record_key=? GROUP BY i.namespace,i.normalized "
                "ORDER BY MAX(i.verified) DESC,i.namespace,i.normalized",
                (record_key,),
            ).fetchall():
                retrieve(
                    "exact_identifier",
                    "SELECT DISTINCT record_key FROM identity_ids WHERE namespace=? "
                    "AND normalized=? AND record_key<>?",
                    (namespace, value, record_key),
                    members,
                    {
                        "namespace": namespace,
                        "normalized": value if len(value) <= 256 else None,
                        "normalized_preview": value[:256],
                        "normalized_sha256": hashlib.sha256(value.encode()).hexdigest(),
                        "verified_claim": bool(verified),
                    },
                    0 if verified else 1,
                )
        title, year = row
        if "title_year" in self.methods and title:
            evidence = {
                "normalized_title_preview": title[:256],
                "normalized_title_sha256": hashlib.sha256(title.encode()).hexdigest(),
                "publication_year": year,
            }
            if year is None:
                membership = self.db.execute(
                    "SELECT COALESCE(SUM(members),0) FROM identity_title_block_counts "
                    "WHERE title_key=?",
                    (title,),
                ).fetchone()[0]
                retrieve(
                    "normalized_title_year_block",
                    "SELECT record_key FROM identity_records "
                    "INDEXED BY identity_title_record_lookup "
                    "WHERE title_key=? AND record_key<>?",
                    (title, record_key),
                    membership,
                    evidence,
                    2,
                )
            else:
                # Query each exact year separately so the composite index can both
                # filter and return ordered keys without sorting an enormous block.
                years = [None, *range(year - self.year_tolerance, year + self.year_tolerance + 1)]
                for blocked_year in years:
                    membership_row = self.db.execute(
                        "SELECT members FROM identity_title_block_counts "
                        "WHERE title_key=? AND publication_year=?",
                        (title, blocked_year if blocked_year is not None else -1),
                    ).fetchone()
                    retrieve(
                        "normalized_title_year_block",
                        "SELECT record_key FROM identity_records WHERE title_key=? "
                        "AND publication_year IS ? AND record_key<>?",
                        (title, blocked_year, record_key),
                        membership_row[0] if membership_row else 0,
                        {**evidence, "blocked_publication_year": blocked_year},
                        2,
                        contains_self=blocked_year == year,
                    )
        ordered = sorted(
            candidates.values(), key=lambda item: (item["priority"], item["record_key"])
        )
        truncated = (
            any(block["truncated"] for block in blocks) or len(ordered) > self.max_candidates
        )
        lower_bound = max([block["other_record_memberships"] for block in blocks] + [0])
        upper_bound = sum(block["other_record_memberships"] for block in blocks)
        return {
            "record_key": record_key,
            "candidates": ordered[: self.max_candidates],
            "candidate_count": min(len(ordered), self.max_candidates),
            "candidate_population_lower_bound": lower_bound,
            "candidate_population_upper_bound": upper_bound,
            "candidate_population_exact": len(ordered) if not truncated else None,
            "truncated": truncated,
            "blocks": blocks,
            "max_candidates": self.max_candidates,
            "coverage_note": (
                "Bounded retrieval: omitted candidates unscored; population bounds may overlap."
                if truncated
                else "All candidates in the enabled exact blocks were retrieved."
            ),
        }

    def iter_batches(self, *, batch_size: int = 1000):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        cursor = self.db.execute("SELECT record_key FROM identity_records ORDER BY record_key")
        while batch := cursor.fetchmany(batch_size):
            yield [self.for_record(key) for (key,) in batch]

    def iter_records(self, *, batch_size: int = 1000):
        """Stream one candidate envelope at a time; only source keys are batched."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        cursor = self.db.execute("SELECT record_key FROM identity_records ORDER BY record_key")
        while batch := cursor.fetchmany(batch_size):
            for (key,) in batch:
                yield self.for_record(key)

    @staticmethod
    def to_json(value: dict) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
