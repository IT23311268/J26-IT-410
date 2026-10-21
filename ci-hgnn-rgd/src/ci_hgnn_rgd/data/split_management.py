"""Disk-backed, bounded leakage audits that never modify official assignments."""

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher
from pathlib import Path

from .temporal_integrity import parse_cutoff


def normalized_context(value: str) -> str:
    """An audit key only: raw scientific text remains untouched."""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _blocks(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text)
    shingles = (" ".join(tokens[i : i + 2]) for i in range(max(1, len(tokens) - 1)))
    # Four deterministic token-shingle minhash blocks, not exhaustive fuzzy matching.
    return sorted({_digest(item) for item in shingles})[:4]


def _source_local_paper_key(identifier: dict, row: dict) -> str | None:
    """Audit source ID reuse without promoting it to a verified global identity.

    Missing release identifiers require a shared input archive/file scope. Archive
    members share their unchanged archive path; separate arbitrary input files do
    not become the same release merely because both omit their version.
    """
    if not isinstance(identifier, dict):
        return None
    provenances = row.get("provenance") or []
    provenance = next((item for item in provenances if isinstance(item, dict)), {})
    dataset = (
        row.get("source_dataset") or identifier.get("dataset") or provenance.get("source_dataset")
    )
    version = (
        row.get("dataset_version")
        or identifier.get("dataset_version")
        or provenance.get("dataset_version")
    )
    namespace = identifier.get("original_namespace") or identifier.get("namespace")
    value = identifier.get("raw", identifier.get("value", identifier.get("normalized")))
    if not dataset or not namespace or isinstance(value, bool) or not isinstance(value, str | int):
        return None
    if isinstance(value, str) and not value:
        return None
    source_scope = None
    if version in {None, "unknown"}:
        source_file = provenance.get("source_file")
        source_scope = row.get("source_collection_id")
        if not source_scope and isinstance(source_file, str):
            source_scope = source_file.split("!", 1)[0]
        if not source_scope:
            return None
    return json.dumps(
        [
            dataset,
            version or "unknown",
            source_scope,
            namespace,
            value,
            identifier.get("node_type", "paper"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


class SplitAuditor:
    """Streaming SQLite audit with explicit fuzzy-comparison and candidate caps.

    Shared paper IDs are risks dependent on experiment design, not an automatic
    leakage verdict. Near-duplicate discovery is approximate; capped blocks and
    long contexts skipped for fuzzy comparison are reported, never called clean.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        near_threshold: float = 0.94,
        max_candidates_per_context: int = 20,
        max_comparisons: int = 100_000,
        max_near_text_chars: int = 16_384,
    ):
        if (
            not 0 < near_threshold <= 1
            or min(max_candidates_per_context, max_comparisons, max_near_text_chars) < 1
        ):
            raise ValueError("threshold must be in (0,1] and limits must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.near_threshold = near_threshold
        self.max_candidates = max_candidates_per_context
        self.max_comparisons = max_comparisons
        self.max_chars = max_near_text_chars
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA temp_store=FILE;
            CREATE TABLE IF NOT EXISTS split_contexts(
                seq INTEGER PRIMARY KEY, context_id TEXT, split TEXT, raw_hash TEXT,
                norm_hash TEXT, norm_text TEXT, record_hash TEXT);
            CREATE TABLE IF NOT EXISTS split_blocks(block TEXT, seq INTEGER);
            CREATE INDEX IF NOT EXISTS split_block_idx ON split_blocks(block,seq);
            CREATE INDEX IF NOT EXISTS split_raw_idx ON split_contexts(raw_hash,split);
            CREATE INDEX IF NOT EXISTS split_norm_idx ON split_contexts(norm_hash,split);
            CREATE INDEX IF NOT EXISTS split_record_idx ON split_contexts(record_hash,split);
            CREATE TABLE IF NOT EXISTS split_papers(seq INTEGER, paper_key TEXT, split TEXT);
            CREATE INDEX IF NOT EXISTS split_paper_idx ON split_papers(paper_key,split);
            CREATE TABLE IF NOT EXISTS split_source_papers(
                seq INTEGER,paper_key TEXT,split TEXT);
            CREATE INDEX IF NOT EXISTS split_source_paper_idx
                ON split_source_papers(paper_key,split);
            CREATE TABLE IF NOT EXISTS split_near(left_seq INTEGER, right_seq INTEGER,
                similarity REAL, PRIMARY KEY(left_seq,right_seq));
            CREATE TABLE IF NOT EXISTS split_stats(name TEXT PRIMARY KEY,value INTEGER);
            CREATE TABLE IF NOT EXISTS official_nodes(dataset TEXT, version TEXT,
                namespace TEXT,node_id TEXT,split TEXT);
            CREATE INDEX IF NOT EXISTS official_idx
                ON official_nodes(dataset,version,namespace,node_id,split);
        """)

    def _increment(self, name: str, amount: int = 1):
        self.db.execute(
            "INSERT INTO split_stats VALUES (?,?) ON CONFLICT(name) DO UPDATE "
            "SET value=value+excluded.value",
            (name, amount),
        )

    def _stat(self, name: str) -> int:
        row = self.db.execute("SELECT value FROM split_stats WHERE name=?", (name,)).fetchone()
        return row[0] if row else 0

    def add_context(self, row: dict) -> None:
        raw = row.get("original_context", row.get("citation_text"))
        text = raw if isinstance(raw, str) else ""
        norm = row.get("normalized_context")
        norm = normalized_context(norm if isinstance(norm, str) else text)
        split = row.get("source_split")
        split = str(split) if split is not None else None
        cursor = self.db.execute(
            "INSERT INTO split_contexts(context_id,split,raw_hash,norm_hash,norm_text,"
            "record_hash) VALUES (?,?,?,?,?,?)",
            (
                row.get("context_id"),
                split,
                _digest(text) if text else None,
                _digest(norm) if norm else None,
                norm if len(norm) <= self.max_chars else None,
                row.get("source_record_sha256"),
            ),
        )
        seq = cursor.lastrowid
        if split is None:
            self._increment("missing_split")
        if not text:
            self._increment("missing_text")
        for field in ("citing_paper_id", "cited_paper_id"):
            identity = row.get(field)
            if identity is not None:
                # Canonical IDs are global. Unresolved source IDs require a namespace.
                namespace = row.get("paper_identifier_namespace", "canonical")
                self.db.execute(
                    "INSERT INTO split_papers VALUES (?,?,?)",
                    (seq, json.dumps([namespace, str(identity)]), split),
                )
            else:
                identifier_field = (
                    "citing_identifier" if field == "citing_paper_id" else "cited_identifier"
                )
                identifier = row.get(identifier_field)
                local_key = _source_local_paper_key(identifier, row)
                if local_key is not None:
                    self.db.execute(
                        "INSERT INTO split_source_papers VALUES (?,?,?)", (seq, local_key, split)
                    )
                elif identifier is not None:
                    self._increment("source_local_identifiers_missing_scope")
        if not norm or split is None:
            return
        if len(norm) > self.max_chars:
            self._increment("near_long_text_skipped")
            return
        blocks = _blocks(norm)
        if not blocks:
            return
        if self._stat("near_comparisons") >= self.max_comparisons:
            self._increment("near_budget_skipped_contexts")
        else:
            placeholders = ",".join("?" for _ in blocks)
            candidates = self.db.execute(
                "SELECT DISTINCT c.seq,c.norm_text FROM split_contexts c "
                "JOIN split_blocks b ON b.seq=c.seq WHERE b.block IN ("
                + placeholders
                + ") AND c.split<>? AND c.norm_hash<>? "
                "AND c.norm_text IS NOT NULL ORDER BY c.seq LIMIT ?",
                (*blocks, split, _digest(norm), self.max_candidates + 1),
            ).fetchall()
            if len(candidates) > self.max_candidates:
                self._increment("near_candidate_capped_contexts")
            for candidate_seq, candidate_text in candidates[: self.max_candidates]:
                if self._stat("near_comparisons") >= self.max_comparisons:
                    self._increment("near_budget_skipped_contexts")
                    break
                self._increment("near_comparisons")
                # quick_ratio is a valid upper bound; avoid expensive impossible pairs.
                matcher = SequenceMatcher(None, candidate_text, norm, autojunk=False)
                if matcher.real_quick_ratio() < self.near_threshold:
                    continue
                similarity = matcher.ratio()
                if similarity >= self.near_threshold:
                    self.db.execute(
                        "INSERT OR IGNORE INTO split_near VALUES (?,?,?)",
                        (candidate_seq, seq, similarity),
                    )
        self.db.executemany("INSERT INTO split_blocks VALUES (?,?)", ((b, seq) for b in blocks))

    def add_official_index(
        self, dataset: str, version: str | None, namespace: str, node_id: int | str, split: str
    ) -> None:
        self.db.execute(
            "INSERT INTO official_nodes VALUES (?,?,?,?,?)",
            (dataset, version or "unknown", namespace, str(node_id), split),
        )

    def report(self) -> dict:
        self.db.commit()

        def overlap(column):
            return self.db.execute(
                "SELECT COUNT(*) FROM (SELECT " + column + " FROM split_contexts "
                "WHERE " + column + " IS NOT NULL AND split IS NOT NULL "
                "GROUP BY " + column + " HAVING COUNT(DISTINCT split)>1)"
            ).fetchone()[0]

        paper_groups = self.db.execute(
            "SELECT COUNT(*) FROM (SELECT paper_key FROM split_papers WHERE split IS NOT NULL "
            "GROUP BY paper_key HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        source_local_groups = self.db.execute(
            "SELECT COUNT(*) FROM (SELECT paper_key FROM split_source_papers "
            "WHERE split IS NOT NULL GROUP BY paper_key HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        near_count = self.db.execute("SELECT COUNT(*) FROM split_near").fetchone()[0]
        examples = [
            {
                "left_context_id": left,
                "right_context_id": right,
                "left_split": ls,
                "right_split": rs,
                "similarity": similarity,
            }
            for left, right, ls, rs, similarity in self.db.execute(
                "SELECT l.context_id,r.context_id,l.split,r.split,n.similarity "
                "FROM split_near n JOIN split_contexts l ON l.seq=n.left_seq "
                "JOIN split_contexts r ON r.seq=n.right_seq "
                "ORDER BY n.left_seq,n.right_seq LIMIT 20"
            )
        ]
        official_overlap = self.db.execute(
            "SELECT COUNT(*) FROM (SELECT dataset,version,namespace,node_id FROM official_nodes "
            "GROUP BY dataset,version,namespace,node_id HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        official_counts = [
            {
                "dataset": dataset,
                "dataset_version": version,
                "namespace": namespace,
                "source_split": split,
                "records": count,
            }
            for dataset, version, namespace, split, count in self.db.execute(
                "SELECT dataset,version,namespace,split,COUNT(*) FROM official_nodes "
                "GROUP BY dataset,version,namespace,split ORDER BY dataset,version,namespace,split"
            )
        ]
        return {
            "construction_method": "official_source_assignments_preserved",
            "official_splits_modified": False,
            "contexts_inspected": self.db.execute("SELECT COUNT(*) FROM split_contexts").fetchone()[
                0
            ],
            "split_counts": dict(
                self.db.execute(
                    "SELECT COALESCE(split,'unknown'),COUNT(*) FROM split_contexts GROUP BY split"
                )
            ),
            "exact_record_overlap_groups": overlap("record_hash"),
            "exact_text_overlap_groups": overlap("raw_hash"),
            "normalized_text_overlap_groups": overlap("norm_hash"),
            "near_duplicate_cross_split_pairs": near_count,
            "near_duplicate_examples": examples,
            "paper_identity_overlap_groups": paper_groups,
            "resolved_paper_identity_overlap_groups": paper_groups,
            "source_local_paper_overlap_groups": source_local_groups,
            "source_local_paper_identifier_occurrences": self.db.execute(
                "SELECT COUNT(*) FROM split_source_papers"
            ).fetchone()[0],
            "source_local_identifiers_missing_scope": self._stat(
                "source_local_identifiers_missing_scope"
            ),
            "source_local_identity_interpretation": (
                "within_source_release_or_archive_id_reuse_not_verified_global_paper_identity"
            ),
            "paper_overlap_interpretation": (
                "risk_dependent_on_experiment_design_not_automatic_leakage"
            ),
            "missing_splits": self._stat("missing_split"),
            "missing_context_text": self._stat("missing_text"),
            "near_audit": {
                "method": "four_token_shingle_minhash_blocks_then_sequence_matcher",
                "threshold": self.near_threshold,
                "exhaustive": False,
                "comparisons": self._stat("near_comparisons"),
                "max_comparisons": self.max_comparisons,
                "max_candidates_per_context": self.max_candidates,
                "candidate_capped_contexts": self._stat("near_candidate_capped_contexts"),
                "budget_skipped_contexts": self._stat("near_budget_skipped_contexts"),
                "long_text_skipped": self._stat("near_long_text_skipped"),
            },
            "official_node_splits": official_counts,
            "official_node_overlap_groups": official_overlap,
        }

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def group_aware_splits(
    rows: Iterable[dict],
    *,
    group_field: str,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Iterator[dict]:
    """Opt-in experiment assignments; caller supplies a verified group identifier.

    Source splits are preserved. This is group-aware, not a claim of paper-disjoint
    splits unless groups were built from all connected paper identities beforehand.
    Hash allocation is deterministic and approximate in size, not stratified.
    """
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("three nonnegative split ratios must sum to one")
    for row in rows:
        group = row.get(group_field)
        if group is None:
            raise ValueError(f"missing required grouping field: {group_field}")
        draw = int(_digest(json.dumps([seed, group], sort_keys=True))[:16], 16) / 2**64
        split = "train" if draw < ratios[0] else "dev" if draw < sum(ratios[:2]) else "test"
        yield {**row, "experiment_split": split, "split_construction": "group_hash_v1"}


def temporal_split(
    available_at,
    *,
    training_cutoff,
    validation_end,
    test_end,
) -> str | None:
    """Disjoint ordered availability windows; unknown availability is unassigned."""
    train, validation, test = map(parse_cutoff, (training_cutoff, validation_end, test_end))
    if not train < validation < test:
        raise ValueError("temporal windows must satisfy train < validation < test")
    if available_at is None:
        return None
    observed = parse_cutoff(available_at)
    if observed <= train:
        return "train"
    if observed <= validation:
        return "dev"
    if observed <= test:
        return "test"
    return None
