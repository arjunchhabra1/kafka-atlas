"""`kgrag ablate`: gold-source recall of named graph-retriever configs on the dev split.

Retrieval only (no LLM calls). Questions come from the `questions` table (kept/edited, the
given split). Never runs on the test split: the ablation exists to tune the retriever.
"""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

from kgrag.eval.run import RESULTS_DIR, EvalPreconditionError, _check_graph, load_questions
from kgrag.retrieve.graph_ablation import ablate
from kgrag.retrieve.graph_config import GraphRetrieverConfig


@contextmanager
def _default_kg(conn: psycopg.Connection, driver: Any) -> Iterator[tuple[Any, Any]]:
    from kgrag.graph.kg import KnowledgeGraph
    from kgrag.retrieve.index import open_stores
    from kgrag.retrieve.vector import VectorRetriever

    with open_stores() as (chunks, nodes):
        yield KnowledgeGraph(driver, conn, chunks, nodes), VectorRetriever(chunks, conn).retrieve


def run_ablation(
    conn: psycopg.Connection, driver: Any, *, split: str, resolver: str,
    configs: Mapping[str, GraphRetrieverConfig], budget: int, vector_baseline: bool = True,
    results_dir: Path | None = None, kg: Any = None,
) -> tuple[Path, dict]:
    """Run the ablation and write results/ablation-{timestamp}.json; returns (path, results).

    `kg` (with `.chunk_store` and `.conn`) replaces the default KnowledgeGraph in tests.
    """
    if split == "test":
        raise EvalPreconditionError("the ablation tunes the retriever; it never runs on the "
                                    "test split (use --split dev)")
    if not configs:
        raise EvalPreconditionError("no graph configs to ablate")
    meta = _check_graph(driver, resolver)
    questions = load_questions(conn, split)
    rows = [{"question": q.question, "type": q.type, "gold_doc_ids": q.gold_doc_ids}
            for q in questions]
    if not any(r["gold_doc_ids"] for r in rows):
        raise EvalPreconditionError(f"no kept/edited questions with gold docs in split '{split}'")

    started = datetime.now(UTC)

    @contextmanager
    def graph() -> Iterator[tuple[Any, Any]]:
        if kg is not None:
            from kgrag.retrieve.vector import VectorRetriever

            yield kg, VectorRetriever(kg.chunk_store, kg.conn).retrieve
        else:
            with _default_kg(conn, driver) as pair:
                yield pair

    with graph() as (the_kg, vector):
        results = ablate(the_kg, rows, configs, budget,
                         baselines={"vector": vector} if vector_baseline else None)

    out_dir = results_dir if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ablation-{started:%Y%m%dT%H%M%SZ}.json"
    header = {
        "kind": "ablation", "split": split, "resolver": resolver, "budget_tokens": budget,
        "started_at": started.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
        "n_questions": sum(bool(r["gold_doc_ids"]) for r in rows),
        "configs": {name: cfg.fingerprint() for name, cfg in configs.items()},
        "vector_baseline": vector_baseline,
        "graph_meta": {"resolver": meta.resolver, "built_at": meta.built_at.isoformat()},
    }
    path.write_text(json.dumps({"header": header, "results": results}, indent=2))
    return path, results
