"""Deterministic comparison keys and indexed overlap audits; no fuzzy entity resolution."""

import itertools
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


def normalize_doi(value):
    if not isinstance(value, str):
        return None
    key = value.strip()
    key = re.sub(r"^doi\s*:\s*", "", key, flags=re.I)
    if key.lower().startswith(
        ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/")
    ):
        key = unquote(urlsplit(key).path.lstrip("/"))
    key = key.strip().casefold()
    return key if re.fullmatch(r"10\.\d{4,9}/[^\s]+", key) else None


@dataclass(frozen=True)
class ArxivIdentifier:
    base: str
    version: str | None


def normalize_arxiv(value):
    if not isinstance(value, str):
        return None
    key = re.sub(r"^arxiv\s*:\s*", "", value.strip(), flags=re.I)
    if key.lower().startswith(
        ("https://arxiv.org/", "http://arxiv.org/", "https://export.arxiv.org/")
    ):
        path = urlsplit(key).path
        if not path.startswith(("/abs/", "/pdf/")):
            return None
        key = path[5:].removesuffix(".pdf")
    modern = re.fullmatch(r"(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(v[1-9]\d*)?", key, re.I)
    legacy = re.fullmatch(r"([a-z][a-z.\-]*/\d{2}(?:0[1-9]|1[0-2])\d{3})(v[1-9]\d*)?", key, re.I)
    match = modern or legacy
    return (
        ArxivIdentifier(match[1].casefold(), match[2].casefold() if match[2] else None)
        if match
        else None
    )


def title_key(value):
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split()) or None


class IdentifierIndex:
    """Disk-backed IDs, duplicate keys and text hashes; memory does not grow with corpus size."""

    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            CREATE TABLE IF NOT EXISTS records(record_key TEXT PRIMARY KEY, source_id TEXT,
                record_hash TEXT, text_hash TEXT, split TEXT, valid INTEGER);
            CREATE TABLE IF NOT EXISTS entities(record_key TEXT PRIMARY KEY, title_key TEXT);
            CREATE TABLE IF NOT EXISTS ids(record_key TEXT, namespace TEXT, value TEXT);
            CREATE INDEX IF NOT EXISTS ids_lookup ON ids(namespace,value);
            CREATE INDEX IF NOT EXISTS ids_record ON ids(record_key);
            CREATE INDEX IF NOT EXISTS text_lookup ON records(text_hash,split);
            CREATE INDEX IF NOT EXISTS source_lookup ON records(source_id);
            CREATE INDEX IF NOT EXISTS title_lookup ON entities(title_key);
        """)

    def add_entity(self, key, identifiers=(), title=None):
        self.db.execute("INSERT OR IGNORE INTO entities VALUES (?,?)", (key, title_key(title)))
        for identifier in identifiers:
            self.db.execute(
                "INSERT INTO ids VALUES (?,?,?)", (key, identifier.namespace, identifier.value)
            )

    def summary(self):
        self.db.commit()

        def duplicates(column):
            return self.db.execute(
                f"SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records WHERE "
                f"{column} IS NOT NULL GROUP BY {column} HAVING COUNT(*)>1)"
            ).fetchone()[0]

        overlap = self.db.execute(
            "SELECT COUNT(*) FROM (SELECT text_hash FROM records WHERE text_hash IS NOT "
            "NULL AND split IS NOT NULL GROUP BY text_hash HAVING COUNT(DISTINCT "
            "split)>1)"
        ).fetchone()[0]
        return {
            "duplicate_source_ids": duplicates("source_id"),
            "duplicate_records": duplicates("record_hash"),
            "duplicate_contexts": duplicates("text_hash"),
            "cross_split_exact_text_overlap_groups": overlap,
            "identifier_namespaces": {
                namespace: count
                for namespace, count in self.db.execute(
                    "SELECT namespace,COUNT(*) FROM ids GROUP BY namespace"
                )
            },
            "entities_indexed": self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
            "entities_with_identifiers": self.db.execute(
                "SELECT COUNT(DISTINCT record_key) FROM ids"
            ).fetchone()[0],
        }

    def close(self):
        self.db.commit()
        self.db.close()


def audit_indices(paths: dict[str, Path], output_db: Path, *, candidate_limit=20):
    """Aggregate keys before pairing; never construct an all-pairs record join."""
    db = sqlite3.connect(output_db)
    try:
        db.executescript("""
            PRAGMA cache_size=-4096;
            CREATE TABLE ids(dataset TEXT, record_key TEXT, namespace TEXT, value TEXT);
            CREATE TABLE entities(dataset TEXT, record_key TEXT, title_key TEXT);
            CREATE INDEX common_ids ON ids(dataset,namespace,value);
            CREATE INDEX entity_ids ON ids(dataset,record_key);
            CREATE INDEX common_titles ON entities(dataset,title_key);
        """)
        for dataset, path in paths.items():
            source = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            try:
                for query, table, count in (
                    ("SELECT record_key,namespace,value FROM ids", "ids", 4),
                    ("SELECT record_key,title_key FROM entities", "entities", 3),
                ):
                    cursor = source.execute(query)
                    while rows := cursor.fetchmany(1000):
                        marks = ",".join("?" for _ in range(count))
                        db.executemany(
                            f"INSERT INTO {table} VALUES ({marks})",
                            ((dataset, *row) for row in rows),
                        )
            finally:
                source.close()
        db.executescript("""
            CREATE TABLE tokens AS SELECT dataset,namespace,value,
                COUNT(DISTINCT record_key) n FROM ids GROUP BY dataset,namespace,value;
            CREATE INDEX token_keys ON tokens(dataset,namespace,value);
            CREATE TABLE titles AS SELECT dataset,title_key,COUNT(*) n
                FROM entities WHERE title_key IS NOT NULL GROUP BY dataset,title_key;
            CREATE INDEX title_keys ON titles(dataset,title_key);
        """)
        pairs = []
        strong = "('doi','arxiv','semantic_scholar','openalex','mag')"
        for left, right in itertools.combinations(sorted(paths), 2):
            shared = f"""SELECT a.namespace,a.value,a.n,b.n FROM tokens a JOIN tokens b
                ON a.namespace=b.namespace AND a.value=b.value
                WHERE a.dataset=? AND b.dataset=? AND a.namespace IN {strong}"""
            exact, ambiguous, namespaces, examples = 0, 0, {}, []
            for namespace, value, n_left, n_right in db.execute(shared, (left, right)):
                status = "EXACT_VERIFIED" if n_left == n_right == 1 else "AMBIGUOUS"
                exact += status == "EXACT_VERIFIED"
                ambiguous += status == "AMBIGUOUS"
                namespaces[namespace] = namespaces.get(namespace, 0) + 1
                if len(examples) < candidate_limit:
                    examples.append(
                        dict(
                            namespace=namespace,
                            value=value,
                            status=status,
                            left_records=n_left,
                            right_records=n_right,
                        )
                    )
            title_query = """SELECT a.title_key,a.n,b.n FROM titles a JOIN titles b
                ON a.title_key=b.title_key WHERE a.dataset=? AND b.dataset=?"""
            candidates, title_ambiguous, title_examples = 0, 0, []
            for key, n_left, n_right in db.execute(title_query, (left, right)):
                status = "CANDIDATE" if n_left == n_right == 1 else "AMBIGUOUS"
                candidates += status == "CANDIDATE"
                title_ambiguous += status == "AMBIGUOUS"
                if len(title_examples) < candidate_limit:
                    title_examples.append(dict(key=key, status=status))
            coverage = {}
            for dataset, other in ((left, right), (right, left)):
                classifications = f"""SELECT status,COUNT(*) FROM (
                    SELECT e.record_key, CASE
                    WHEN EXISTS (
                        SELECT 1 FROM ids i JOIN tokens a
                        ON a.dataset=i.dataset AND a.namespace=i.namespace AND a.value=i.value
                        JOIN tokens b ON b.namespace=i.namespace AND b.value=i.value
                        WHERE i.dataset=e.dataset AND i.record_key=e.record_key
                        AND b.dataset=:other AND a.n=1 AND b.n=1 AND i.namespace IN {strong}
                    ) THEN 'EXACT_VERIFIED'
                    WHEN EXISTS (
                        SELECT 1 FROM ids i JOIN tokens b
                        ON b.namespace=i.namespace AND b.value=i.value
                        WHERE i.dataset=e.dataset AND i.record_key=e.record_key
                        AND b.dataset=:other AND i.namespace IN {strong}
                    ) THEN 'AMBIGUOUS'
                    WHEN EXISTS (
                        SELECT 1 FROM titles a JOIN titles b ON a.title_key=b.title_key
                        WHERE a.dataset=e.dataset AND b.dataset=:other
                        AND a.title_key=e.title_key AND (a.n>1 OR b.n>1)
                    ) THEN 'AMBIGUOUS'
                    WHEN EXISTS (
                        SELECT 1 FROM titles b WHERE b.dataset=:other
                        AND b.title_key=e.title_key
                    ) THEN 'CANDIDATE'
                    WHEN e.title_key IS NULL AND NOT EXISTS (
                        SELECT 1 FROM ids i WHERE i.dataset=e.dataset
                        AND i.record_key=e.record_key
                    ) THEN 'UNRESOLVABLE'
                    ELSE 'UNMATCHED' END status FROM entities e WHERE e.dataset=:dataset
                ) GROUP BY status"""
                counts = dict.fromkeys(
                    ("EXACT_VERIFIED", "CANDIDATE", "AMBIGUOUS", "UNMATCHED", "UNRESOLVABLE"), 0
                )
                counts.update(dict(db.execute(classifications, dict(dataset=dataset, other=other))))
                total = sum(counts.values())
                covered = counts["EXACT_VERIFIED"]
                coverage[dataset] = {
                    "indexed_entities": total,
                    "exact_identifier_covered": covered,
                    "estimated_integration_coverage": covered / total if total else None,
                    "record_classifications": counts,
                    "UNRESOLVABLE": counts["UNRESOLVABLE"],
                    "not_exactly_matched": total - covered,
                }
            pairs.append(
                dict(
                    left=left,
                    right=right,
                    EXACT_VERIFIED=exact,
                    AMBIGUOUS=ambiguous,
                    CANDIDATE_title_keys=candidates,
                    ambiguous_title_keys=title_ambiguous,
                    common_namespaces=namespaces,
                    identifier_examples=examples,
                    title_examples=title_examples,
                    coverage=coverage,
                )
            )
        return dict(
            pairs=pairs,
            datasets_audited=sorted(paths),
            scope="inspected record occurrences only; no whole-dataset extrapolation",
            exact_meaning=(
                "unique shared reliable normalized identifier; "
                "not online resolution or an entity merge"
            ),
            limitations=[
                "Title equality is only a candidate; no fuzzy matching or merging.",
                "Dataset-specific integer namespaces cannot establish identity.",
                "Absent datasets/mappings cannot yield measured compatibility.",
            ],
        )
    finally:
        db.close()
