"""pgvector indexes: `chunks` (one per document row) and `node_names` (one per graph node)."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from langchain_core.documents import Document as LCDocument
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from neo4j import Driver
from sqlalchemy import Engine, create_engine

from kgrag.config import get_settings
from kgrag.observability import traced
from kgrag.retrieve.embeddings import get_embeddings

EMBED_BATCH = 128
CHUNKS = "chunks"
NODE_NAMES = "node_names"


class EmbeddingsModelMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionStats:
    added: int
    updated: int
    deleted: int
    unchanged: int

    @property
    def total(self) -> int:
        return self.added + self.updated + self.unchanged


@dataclass(frozen=True)
class IndexStats:
    chunks: CollectionStats
    node_names: CollectionStats


def pgvector_connection(dsn: str | None = None) -> str:
    """The configured Postgres DSN with the SQLAlchemy psycopg3 scheme (ruling R2)."""
    dsn = dsn or get_settings().pg_dsn
    scheme, sep, rest = dsn.partition("://")
    if not sep or scheme not in ("postgresql", "postgres", "postgresql+psycopg"):
        raise ValueError(f"unsupported Postgres DSN scheme: {scheme!r}")
    return f"postgresql+psycopg://{rest}"


def store_connection(connection: str | None) -> str | Engine:
    """What PGVector connects with: a SQLAlchemy URL, or in cloudsql mode an engine whose
    `creator` is `db.connect` (the Cloud SQL connector), so psycopg and SQLAlchemy share one path."""
    if connection is None and get_settings().db_mode == "cloudsql":
        from kgrag.store import db

        return create_engine("postgresql+psycopg://", creator=lambda: db.connect())
    return pgvector_connection(connection)


def _store(collection: str, connection: str | None) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection,
        connection=store_connection(connection),
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )


@contextmanager
def open_stores(connection: str | None = None) -> Iterator[tuple[PGVector, PGVector]]:
    """(chunks, node_names) stores; `connection` defaults to settings.pg_dsn. Closes pools on exit."""
    pair = _store(CHUNKS, connection), _store(NODE_NAMES, connection)
    try:
        yield pair
    finally:
        for store in pair:
            store._engine.dispose()


def _existing(conn: psycopg.Connection, collection: str, key: str) -> dict[str, tuple[str, str, str]]:
    """key value -> (vector id, text hash, model) for what is already indexed."""
    rows = conn.execute(
        "select e.cmetadata ->> %s, e.id, e.cmetadata ->> 'text_sha256', e.cmetadata ->> 'model' "
        "from langchain_pg_embedding e "
        "join langchain_pg_collection c on c.uuid = e.collection_id where c.name = %s",
        (key, collection),
    ).fetchall()
    conn.commit()  # end the read transaction so later reads see other connections' commits
    return {r[0]: (r[1], r[2], r[3]) for r in rows if r[0] is not None}


def _add(store: PGVector, items: list[tuple[str, str, dict]]) -> None:
    """items: (unique id, text to embed, metadata); embedded and stored in batches."""
    for i in range(0, len(items), EMBED_BATCH):
        batch = items[i : i + EMBED_BATCH]
        store.add_documents(
            [LCDocument(page_content=text, metadata=meta) for _, text, meta in batch],
            ids=[id_ for id_, _, _ in batch],
        )


def _sync(
    conn: psycopg.Connection, store: PGVector, collection: str, key: str,
    wanted: list[tuple[str, str, dict]], model: str,
) -> CollectionStats:
    """Bring one collection in line with `wanted` [(key value, text, extra metadata)]."""
    have = _existing(conn, collection, key)
    prefix = "" if collection == CHUNKS else "node:"
    todo, added, updated, unchanged = [], 0, 0, 0
    for k, text, extra in wanted:
        digest = hashlib.sha256(text.encode()).hexdigest()
        cur = have.get(k)
        if cur is not None and cur[1] == digest and cur[2] == model:
            unchanged += 1
            continue
        added, updated = (added + 1, updated) if cur is None else (added, updated + 1)
        todo.append((prefix + k, text, {key: k, **extra, "text_sha256": digest, "model": model}))
    _add(store, todo)  # same id -> upsert, so changed items are replaced in place
    gone = [v[0] for k, v in have.items() if k not in {w[0] for w in wanted}]
    if gone:
        store.delete(ids=gone)
    return CollectionStats(added, updated, len(gone), unchanged)


def graph_nodes(driver: Driver) -> list[tuple[str, str, str]]:
    """(id, label, name) of every named non-Email graph node: what `node_names` indexes."""
    with driver.session() as s:
        rows = s.run(
            "MATCH (n) WHERE NOT n:GraphMeta AND NOT n:Email AND n.name IS NOT NULL "
            "RETURN n.id AS id, labels(n)[0] AS label, n.name AS name ORDER BY id"
        )
        return [(r["id"], r["label"], r["name"]) for r in rows]


@traced("index.build")
def build_index(
    conn: psycopg.Connection, driver: Driver, connection: str | None = None,
    model: str | None = None,
) -> IndexStats:
    """Sync both collections: embed new/changed items (text hash or model differs), drop stale ones.

    Refuses to run when the embeddings object reports a different model than the one recorded
    (e.g. RecordedEmbeddings captured with an older model): its vectors would be mislabelled.
    """
    model = model or get_settings().models.embed
    actual = getattr(get_embeddings(), "model", None)
    if actual is not None and actual != model:
        raise EmbeddingsModelMismatchError(
            f"the embeddings in use are {actual!r} but the index would record {model!r}; "
            "re-record them or set models.embed to match")
    docs = conn.execute("select doc_id, text from documents order by doc_id").fetchall()
    conn.commit()
    with open_stores(connection) as (chunks, node_names):
        return IndexStats(
            chunks=_sync(conn, chunks, CHUNKS, "doc_id", [(d, t, {}) for d, t in docs], model),
            node_names=_sync(conn, node_names, NODE_NAMES, "node_id", [
                (nid, name, {"label": label, "name": name})
                for nid, label, name in graph_nodes(driver)
            ], model),
        )
