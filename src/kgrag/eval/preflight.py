"""Freshness preflight: refuse to evaluate against a stale index or graph.

Run before any question is answered (and before `kgrag ablate`). Each failure names the
command that fixes it. The checks:

  * every gold doc id of the selected questions is a `documents` row;
  * the `chunks` vectors were embedded with `models.embed` and there is one per document;
  * graph runs: the stored resolution is the one the graph was built from (GraphMeta holds
    its hash), and the `node_names` vectors are exactly the current graph's named nodes,
    embedded with `models.embed`.
"""

from collections.abc import Sequence
from typing import Any

import psycopg

from kgrag.store.repo import QuestionRow, resolution_hash

INDEX_CMD = "run `kgrag index`"


class EvalPreconditionError(RuntimeError):
    """The run cannot start (wrong graph, Neo4j down, stale index, test split already spent)."""


def _sample(ids: Sequence[str], n: int = 5) -> str:
    ids = sorted(ids)
    return ", ".join(ids[:n]) + (f" (+{len(ids) - n} more)" if len(ids) > n else "")


def check_gold_docs(conn: psycopg.Connection, questions: Sequence[QuestionRow]) -> None:
    wanted = sorted({d for q in questions for d in q.gold_doc_ids})
    if not wanted:
        return
    missing = [r[0] for r in conn.execute(
        "select g from unnest(%s::text[]) g where not exists "
        "(select 1 from documents d where d.doc_id = g)", (wanted,))]
    conn.commit()
    if missing:
        raise EvalPreconditionError(
            f"{len(missing)} gold doc id(s) of the selected questions are not in `documents`: "
            f"{_sample(missing)}. The documents table no longer matches the question set "
            "(re-ingested or re-chunked?); restore the ingest the questions were drafted from "
            "(same months and chunk_tokens), then run `kgrag index`.")


def _index_hint() -> str:
    from kgrag.retrieve.embeddings import embeddings_configured

    if embeddings_configured():
        return INDEX_CMD
    return ("configure embeddings (models.embed in config.yaml, VOYAGE_API_KEY in .env), "
            f"then {INDEX_CMD}")


def _collection(conn: psycopg.Connection, name: str, key: str) -> dict[str, str | None] | None:
    """{key value: embedding model} of a pgvector collection; None when it does not exist."""
    if conn.execute("select to_regclass('langchain_pg_embedding')").fetchone()[0] is None:
        conn.commit()
        return None
    exists = conn.execute("select 1 from langchain_pg_collection where name = %s",
                          (name,)).fetchone()
    rows = conn.execute(
        "select e.cmetadata ->> %s, e.cmetadata ->> 'model' from langchain_pg_embedding e "
        "join langchain_pg_collection c on c.uuid = e.collection_id where c.name = %s",
        (key, name)).fetchall()
    conn.commit()
    return None if exists is None else {k: m for k, m in rows if k is not None}


def _check_models(name: str, have: dict[str, str | None], embed_model: str) -> None:
    other = sorted({str(m) for m in have.values() if m != embed_model})
    if other:
        raise EvalPreconditionError(
            f"the `{name}` vectors were embedded with {', '.join(other)}, but models.embed is "
            f"{embed_model}; {_index_hint()} to re-embed them.")


def check_chunks(conn: psycopg.Connection, embed_model: str) -> None:
    have = _collection(conn, "chunks", "doc_id")
    if not have:
        raise EvalPreconditionError(f"no `chunks` vector index found; {_index_hint()}.")
    _check_models("chunks", have, embed_model)
    n_docs = conn.execute("select count(*) from documents").fetchone()[0]
    conn.commit()
    if len(have) != n_docs:
        raise EvalPreconditionError(
            f"the `chunks` index holds {len(have)} vectors for {n_docs} documents; "
            f"{_index_hint()}.")


def check_resolution(conn: psycopg.Connection, meta: Any, resolver: str) -> None:
    fix = f"run `kgrag build --resolver {resolver}` then `kgrag index`"
    current = resolution_hash(conn, resolver)
    conn.commit()
    if meta.resolution_hash is None:
        raise EvalPreconditionError(
            "the graph's GraphMeta records no resolution hash (built by an older kgrag), so its "
            f"resolution cannot be verified; {fix}.")
    if current != meta.resolution_hash:
        raise EvalPreconditionError(
            f"the stored '{resolver}' resolution changed since the graph was built "
            f"(`kgrag resolve` re-run?); {fix}.")


def check_node_names(conn: psycopg.Connection, driver: Any, embed_model: str) -> None:
    from kgrag.retrieve.index import graph_nodes

    have = _collection(conn, "node_names", "node_id")
    graph = {nid for nid, _, _ in graph_nodes(driver)}
    if have is None:
        if graph:
            raise EvalPreconditionError(f"no `node_names` vector index found; {_index_hint()}.")
        return
    _check_models("node_names", have, embed_model)
    stale, missing = set(have) - graph, graph - set(have)
    if stale or missing:
        raise EvalPreconditionError(
            "the `node_names` index does not match the current graph "
            f"({len(stale)} indexed node(s) not in the graph, e.g. {_sample(stale, 3) or '-'}; "
            f"{len(missing)} graph node(s) not indexed, e.g. {_sample(missing, 3) or '-'}). "
            f"The graph was rebuilt after the last index; {_index_hint()}.")


def check_freshness(
    conn: psycopg.Connection, driver: Any, questions: Sequence[QuestionRow], *, system: str,
    resolver: str | None, meta: Any = None, embed_model: str | None = None,
) -> None:
    """Raise EvalPreconditionError unless the index (and graph) match the data. See module doc."""
    from kgrag.config import get_settings

    embed_model = embed_model or get_settings().models.embed
    check_gold_docs(conn, questions)
    if system == "graph":
        if meta is None:
            from kgrag.graph.build import read_graph_meta

            meta = read_graph_meta(driver)
        check_resolution(conn, meta, resolver or "")
    check_chunks(conn, embed_model)
    if system == "graph":
        check_node_names(conn, driver, embed_model)
