"""pgvector indexes: `chunks` (one per document row) and `node_names` (one per graph node)."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from langchain_core.documents import Document as LCDocument
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from neo4j import Driver

from kgrag.config import get_settings
from kgrag.observability import traced
from kgrag.retrieve.embeddings import get_embeddings

EMBED_BATCH = 128
CHUNKS = "chunks"
NODE_NAMES = "node_names"


@dataclass(frozen=True)
class IndexStats:
    chunks: int
    node_names: int


def pgvector_connection(dsn: str | None = None) -> str:
    """The configured Postgres DSN with the SQLAlchemy psycopg3 scheme (ruling R2)."""
    dsn = dsn or get_settings().pg_dsn
    scheme, sep, rest = dsn.partition("://")
    if not sep or scheme not in ("postgresql", "postgres", "postgresql+psycopg"):
        raise ValueError(f"unsupported Postgres DSN scheme: {scheme!r}")
    return f"postgresql+psycopg://{rest}"


def _store(collection: str, connection: str | None) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection,
        connection=pgvector_connection(connection),
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


def _existing(conn: psycopg.Connection, collection: str, key: str) -> set[str]:
    rows = conn.execute(
        "select e.cmetadata ->> %s from langchain_pg_embedding e "
        "join langchain_pg_collection c on c.uuid = e.collection_id where c.name = %s",
        (key, collection),
    ).fetchall()
    conn.commit()  # end the read transaction so later reads see other connections' commits
    return {r[0] for r in rows if r[0] is not None}


def _add(store: PGVector, items: list[tuple[str, str, dict]]) -> None:
    """items: (unique id, text to embed, metadata); embedded and stored in batches."""
    for i in range(0, len(items), EMBED_BATCH):
        batch = items[i : i + EMBED_BATCH]
        store.add_documents(
            [LCDocument(page_content=text, metadata=meta) for _, text, meta in batch],
            ids=[id_ for id_, _, _ in batch],
        )


def _graph_nodes(driver: Driver) -> list[tuple[str, str, str]]:
    with driver.session() as s:
        rows = s.run(
            "MATCH (n) WHERE NOT n:GraphMeta AND NOT n:Email AND n.name IS NOT NULL "
            "RETURN n.id AS id, labels(n)[0] AS label, n.name AS name ORDER BY id"
        )
        return [(r["id"], r["label"], r["name"]) for r in rows]


@traced("index.build")
def build_index(
    conn: psycopg.Connection, driver: Driver, connection: str | None = None
) -> IndexStats:
    """Embed documents and graph node names not yet indexed; return the collection sizes."""
    with open_stores(connection) as (chunks, node_names):
        have = _existing(conn, CHUNKS, "doc_id")
        docs = conn.execute("select doc_id, text from documents order by doc_id").fetchall()
        conn.commit()
        _add(chunks, [(d, t, {"doc_id": d}) for d, t in docs if d not in have])

        have = _existing(conn, NODE_NAMES, "node_id")
        _add(node_names, [
            (f"node:{nid}", name, {"node_id": nid, "label": label, "name": name})
            for nid, label, name in _graph_nodes(driver) if nid not in have
        ])
    return IndexStats(
        chunks=len(_existing(conn, CHUNKS, "doc_id")),
        node_names=len(_existing(conn, NODE_NAMES, "node_id")),
    )
