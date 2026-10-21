"""KnowledgeGraph: the small toolkit a graph retriever is built from.

Wraps the Neo4j graph, the Postgres `documents` table and the two pgvector collections
(`chunks`, `node_names`). It contains no retrieval strategy; that is `retrieve/graph_retriever.py`.
"""

from datetime import date, datetime
from typing import Literal

import psycopg
from langchain_postgres import PGVector
from neo4j import Driver
from neo4j.time import DateTime
from rapidfuzz import fuzz, process

from kgrag.graph.types import Edge, Node
from kgrag.retrieve.types import Evidence
from kgrag.store.repo import Document, get_documents

FUZZY_CUTOFF = 80
_NOT_ENTITY = "NOT n:GraphMeta AND NOT n:Email"


def _plain(value):
    """Neo4j temporal values -> native datetimes, so props are ordinary Python."""
    if isinstance(value, DateTime):
        return value.to_native()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _node(n) -> Node:
    props = {k: _plain(v) for k, v in dict(n).items()}
    return Node(id=props["id"], label=next(iter(n.labels)), name=props.get("name") or "", props=props)


def _edge(r, source_id: str, target_id: str) -> Edge:
    return Edge(type=r.type, source_id=source_id, target_id=target_id,
                props={k: _plain(v) for k, v in dict(r).items()})


class KnowledgeGraph:
    def __init__(self, driver: Driver, conn: psycopg.Connection,
                 chunk_store: PGVector, node_store: PGVector):
        self.driver, self.conn = driver, conn
        self.chunk_store, self.node_store = chunk_store, node_store
        self._names: list[tuple[str, str, str]] | None = None  # (id, label, name), entities only

    def _name_cache(self) -> list[tuple[str, str, str]]:
        if self._names is None:
            with self.driver.session() as s:
                self._names = [(r["id"], r["label"], r["name"]) for r in s.run(
                    f"MATCH (n) WHERE {_NOT_ENTITY} AND n.name IS NOT NULL "
                    "RETURN n.id AS id, labels(n)[0] AS label, n.name AS name ORDER BY id")]
        return self._names

    def _fetch_nodes(self, ids: list[str]) -> dict[str, Node]:
        with self.driver.session() as s:
            return {n["id"]: _node(n) for n in
                    (r["n"] for r in s.run("MATCH (n) WHERE n.id IN $ids RETURN n", ids=ids))}

    def find_nodes(self, name: str, label: str | None = None, fuzzy: bool = False,
                   limit: int = 10) -> list[Node]:
        """Nodes whose name or id equals `name` (case-insensitive); fuzzy: name similarity >= 80."""
        if not fuzzy:
            with self.driver.session() as s:
                rows = s.run(
                    "MATCH (n) WHERE NOT n:GraphMeta AND ($label IS NOT NULL OR NOT n:Email) "
                    "AND ($label IS NULL OR $label IN labels(n)) "
                    "AND (toLower(n.name) = toLower($q) OR toLower(n.id) = toLower($q)) "
                    "RETURN n ORDER BY n.id LIMIT $limit", q=name, label=label, limit=limit)
                return [_node(r["n"]) for r in rows]
        pool = {i: n for i, lab, n in self._name_cache() if label is None or lab == label}
        hits = process.extract(name, pool, scorer=fuzz.WRatio, score_cutoff=FUZZY_CUTOFF, limit=limit)
        nodes = self._fetch_nodes([i for _, _, i in hits])
        return [nodes[i] for _, _, i in hits if i in nodes]

    def neighbors(self, node_id: str, edge_types: list[str] | None = None,
                  direction: Literal["out", "in", "both"] = "both",
                  since: date | None = None, until: date | None = None) -> list[tuple[Edge, Node]]:
        """(edge, other node) pairs; `since`/`until` bound the edge `date` (undated edges are dropped)."""
        pattern = {"out": "(a)-[r]->(b)", "in": "(a)<-[r]-(b)", "both": "(a)-[r]-(b)"}[direction]
        with self.driver.session() as s:
            rows = s.run(
                f"MATCH {pattern} WHERE a.id = $id AND ($types IS NULL OR type(r) IN $types) "
                "RETURN r, b, startNode(r).id AS src, endNode(r).id AS dst ORDER BY b.id, type(r)", id=node_id, types=edge_types)
            pairs = [(_edge(r["r"], r["src"], r["dst"]), _node(r["b"])) for r in rows]
        if since is None and until is None:
            return pairs

        def keep(e: Edge) -> bool:
            d = e.props.get("date")
            if d is None:
                return False
            d = d.date() if isinstance(d, datetime) else d
            return (since is None or d >= since) and (until is None or d <= until)

        return [p for p in pairs if keep(p[0])]

    def source_docs(self, edge: Edge) -> list[str]:
        """The `documents` ids that justify this edge."""
        return list(edge.props.get("source_doc_ids") or [])

    def search_nodes(self, query: str, k: int = 10) -> list[tuple[Node, float]]:
        """Entities whose name embeds closest to `query`: (node, similarity), best first."""
        hits = self.node_store.similarity_search_with_score(query, k=k)
        nodes = self._fetch_nodes([d.metadata["node_id"] for d, _ in hits])
        return [(nodes[d.metadata["node_id"]], 1.0 - dist) for d, dist in hits
                if d.metadata["node_id"] in nodes]

    def search_chunks(self, query: str, k: int = 20) -> list[Evidence]:
        """Top-k chunks by vector similarity, with their text from `documents`."""
        hits = self.chunk_store.similarity_search_with_score(query, k=k)
        ids = [d.metadata["doc_id"] for d, _ in hits]
        text = {d.doc_id: d.text for d in get_documents(self.conn, ids)}
        self.conn.commit()
        return [Evidence(doc_id=i, text=text[i], score=1.0 - dist)
                for i, (_, dist) in zip(ids, hits, strict=True) if i in text]

    def documents(self, doc_ids: list[str]) -> list[Document]:
        docs = get_documents(self.conn, doc_ids)
        self.conn.commit()
        return docs

    def cypher(self, query: str, params: dict | None = None) -> list[dict]:
        """Run read-only Cypher (a read transaction: writes raise) and return rows as dicts."""
        with self.driver.session() as s:
            return s.execute_read(lambda tx: [dict(r) for r in tx.run(query, params or {})])
