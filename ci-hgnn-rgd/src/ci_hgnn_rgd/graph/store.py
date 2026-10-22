"""Disk-backed deterministic graph assembly, lossless evidence and bounded table writes."""

import json
import sqlite3
from pathlib import Path

from ..pipeline import canonical_json, fingerprint

NODE_TYPES = {"paper", "author", "topic", "research_field", "institution", "citation_context"}
RELATIONS = {
    "cites": ("paper", "paper"),
    "authored_by": ("paper", "author"),
    "associated_with_topic": ("paper", "topic"),
    "mentions_topic": ("paper", "topic"),
    "investigates_topic": ("paper", "topic"),
    "predicted_mentions_topic": ("paper", "topic"),
    "predicted_investigates_topic": ("paper", "topic"),
    "candidate_mentions_topic": ("paper", "topic"),
    "candidate_investigates_topic": ("paper", "topic"),
    "classified_in": ("paper", "research_field"),
    "affiliated_with": ("author", "institution"),
    "originates_from": ("citation_context", "paper"),
    "refers_to": ("citation_context", "paper"),
    "expresses_intent_about": ("citation_context", "paper"),
}
REVERSES = {"authored_by": "author_of", "associated_with_topic": "associated_with_paper"}


class GraphStore:
    def __init__(self, path: Path, config):
        self.path, self.config = path, config
        self.edge_count = 0
        self.duplicate_count = 0
        self.db = sqlite3.connect(path)
        self.db.execute(f"PRAGMA cache_size=-{config.sqlite_cache_mib * 1024}")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
            CREATE TABLE nodes(kind TEXT,key TEXT,idx INTEGER,payload TEXT,
                               PRIMARY KEY(kind,key));
            CREATE TABLE edges(id TEXT PRIMARY KEY,sk TEXT,rel TEXT,tk TEXT,s TEXT,t TEXT,
                               derived INTEGER,payload TEXT);
            CREATE INDEX edge_pair ON edges(rel,s,t);
            CREATE TABLE evidence(kind TEXT,id TEXT,payload TEXT,PRIMARY KEY(kind,id));
            CREATE TABLE auxiliary(kind TEXT,key TEXT,payload TEXT,PRIMARY KEY(kind,key));
            CREATE TABLE aliases(kind TEXT,scope TEXT,id TEXT,key TEXT,
                                 PRIMARY KEY(kind,scope,id));
            CREATE TABLE features(key TEXT PRIMARY KEY, row_number INTEGER, text_hash TEXT,
                                  payload TEXT);
            CREATE UNIQUE INDEX feature_row ON features(row_number);
            CREATE TABLE intent_contributions(edge_id TEXT,context_id TEXT,p0 REAL,p1 REAL,p2 REAL,
                                               PRIMARY KEY(edge_id,context_id));
        """)

    def close(self):
        self.db.commit()
        self.db.close()

    def node(self, kind, key, payload):
        if kind not in NODE_TYPES or not isinstance(key, str) or not key:
            raise ValueError("invalid graph node type or stable ID")
        if not payload.get("provenance"):
            raise ValueError(f"node lacks source provenance: {key}")
        value = {**payload, "node_type": kind, "stable_id": key}
        encoded = canonical_json(value)
        old = self.db.execute(
            "SELECT payload FROM nodes WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        if old and old[0] != encoded:
            raise ValueError(f"conflicting node identity/metadata: {kind}/{key}")
        self.db.execute("INSERT OR IGNORE INTO nodes VALUES(?,?,NULL,?)", (kind, key, encoded))

    def get_node(self, kind, key):
        row = self.db.execute(
            "SELECT payload FROM nodes WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def alias(self, kind, scope, identifier, key):
        old = self.db.execute(
            "SELECT key FROM aliases WHERE kind=? AND scope=? AND id=?",
            (kind, scope, str(identifier)),
        ).fetchone()
        if old and old[0] != key:
            raise ValueError("ambiguous source-local identity mapping")
        self.db.execute(
            "INSERT OR IGNORE INTO aliases VALUES(?,?,?,?)", (kind, scope, str(identifier), key)
        )

    def resolve(self, kind, scope, identifier):
        row = self.db.execute(
            "SELECT key FROM aliases WHERE kind=? AND scope=? AND id=?",
            (kind, scope, str(identifier)),
        ).fetchone()
        return row[0] if row else None

    def auxiliary(self, kind, key, payload):
        encoded = canonical_json(payload)
        old = self.db.execute(
            "SELECT payload FROM auxiliary WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        if old and old[0] != encoded:
            raise ValueError(f"conflicting source record: {kind}/{key}")
        self.db.execute("INSERT OR IGNORE INTO auxiliary VALUES(?,?,?)", (kind, key, encoded))

    def aux(self, kind, key):
        row = self.db.execute(
            "SELECT payload FROM auxiliary WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def evidence(self, kind, identifier, payload):
        encoded = canonical_json(payload)
        old = self.db.execute(
            "SELECT payload FROM evidence WHERE kind=? AND id=?", (kind, identifier)
        ).fetchone()
        if old and old[0] != encoded:
            raise ValueError(f"conflicting evidence identity: {kind}/{identifier}")
        self.db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?)", (kind, identifier, encoded))

    def exclude(self, kind, key, reason, payload):
        identifier = fingerprint([kind, key, reason, payload])
        self.evidence(
            "excluded_records",
            identifier,
            {"kind": kind, "source_id": key, "reason": reason, "original": payload},
        )

    def edge(
        self, relation, source, target, payload, *, identifier=None, derived=False, node_types=None
    ):
        sk, tk = node_types or RELATIONS[relation]
        if not self.get_node(sk, source) or not self.get_node(tk, target):
            raise ValueError(f"missing graph endpoint: {relation}/{source}/{target}")
        if not payload.get("provenance"):
            raise ValueError("graph edge lacks provenance")
        if not derived and relation not in RELATIONS:
            raise ValueError("unknown scholarly relation")
        identifier = identifier or "edge:" + fingerprint([relation, source, target, payload])
        value = {
            **payload,
            "edge_id": identifier,
            "source_id": source,
            "target_id": target,
            "source_type": sk,
            "target_type": tk,
            "relation": relation,
            "derived": derived,
        }
        encoded = canonical_json(value)
        old = self.db.execute("SELECT payload FROM edges WHERE id=?", (identifier,)).fetchone()
        if old and old[0] != encoded:
            raise ValueError("conflicting edge ID across source observations")
        if old:
            self.duplicate_count += 1
            self.evidence(
                "duplicate_source_observations",
                str(self.duplicate_count),
                {
                    "duplicate_of_edge_id": identifier,
                    "original": value,
                    "policy": "identical_source_edge_retained_in_audit_not_double_counted",
                },
            )
        if not old:
            if self.edge_count >= self.config.max_edges:
                raise ValueError("graph edge budget exceeded; filter input or increase max_edges")
            self.edge_count += 1
        self.db.execute(
            "INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?,?)",
            (identifier, sk, relation, tk, source, target, int(derived), encoded),
        )
        return identifier

    def index_nodes(self):
        # Materialize ranks once. A correlated window subquery rescans ranks per
        # node in SQLite and would turn this otherwise indexed join quadratic.
        self.db.executescript("""
            CREATE TEMP TABLE ranked_nodes(kind TEXT,key TEXT,idx INTEGER,
                                           PRIMARY KEY(kind,key));
            INSERT INTO ranked_nodes SELECT kind,key,
                ROW_NUMBER() OVER (PARTITION BY kind ORDER BY key)-1 FROM nodes;
            UPDATE nodes SET idx=(SELECT idx FROM ranked_nodes ranked
                WHERE ranked.kind=nodes.kind AND ranked.key=nodes.key);
            DROP TABLE ranked_nodes;
        """)
        self.db.execute("CREATE UNIQUE INDEX node_indices ON nodes(kind,idx)")
        self.db.commit()

    def node_rows(self, kind):
        for key, idx, payload in self.db.execute(
            "SELECT key,idx,payload FROM nodes WHERE kind=? ORDER BY idx", (kind,)
        ):
            yield {**json.loads(payload), "graph_index": idx, "stable_id": key}

    def edge_rows(self, relation):
        query = """SELECT e.payload,s.idx,t.idx FROM edges e
            JOIN nodes s ON s.kind=e.sk AND s.key=e.s
            JOIN nodes t ON t.kind=e.tk AND t.key=e.t WHERE e.rel=? ORDER BY e.id"""
        for payload, si, ti in self.db.execute(query, (relation,)):
            yield {**json.loads(payload), "source_index": si, "target_index": ti}

    def evidence_rows(self, kind):
        for (payload,) in self.db.execute(
            "SELECT payload FROM evidence WHERE kind=? ORDER BY id", (kind,)
        ):
            yield json.loads(payload)


def write_sidecar(path: Path, values, batch_size=256):
    """Queryable IDs plus a lossless record_json envelope; bounded Parquet row groups."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            (key, pa.string())
            for key in ("stable_id", "edge_id", "source_id", "target_id", "record_json")
        ]
    )
    buffer, count = [], 0
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for value in values:
            row = {
                key: str(value[key]) if value.get(key) is not None else None
                for key in schema.names[:-1]
            }
            row["record_json"] = canonical_json(value)
            buffer.append(row)
            count += 1
            if len(buffer) >= batch_size:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    return count
