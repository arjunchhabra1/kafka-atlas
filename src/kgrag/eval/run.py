"""Eval runner: retrieve, answer and judge every question of a split, then write one results file."""

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import psycopg

from kgrag.answer.chain import Answer, answer
from kgrag.answer.prompt import ANSWER_PROMPT_VERSION, SYSTEM_PROMPT
from kgrag.config import config_hash, get_settings
from kgrag.eval.judge import JUDGE_PROMPT_VERSION, judge
from kgrag.eval.metrics import full_evidence, source_recall
from kgrag.graph.build import read_graph_meta
from kgrag.observability import tag_current_trace, traced
from kgrag.retrieve.types import RetrievalResult
from kgrag.store.repo import QuestionRow
from kgrag.tokens import count_tokens

RESULTS_DIR = Path("results")
DEGRADED_ERROR_RATE = 0.10  # a run with more than this fraction of errored questions is degraded
log = logging.getLogger(__name__)

Retriever = Callable[[str, int], RetrievalResult]


class EvalPreconditionError(RuntimeError):
    """The run cannot start (wrong graph, Neo4j down, test split already spent)."""


def load_questions(conn: psycopg.Connection, split: str) -> list[QuestionRow]:
    cur = conn.execute(
        f"select {', '.join(QuestionRow.model_fields)} from questions "
        "where status in ('kept','edited') and split = %s order by id",
        (split,),
    )
    rows = [QuestionRow(**dict(zip(QuestionRow.model_fields, r, strict=True)))
            for r in cur.fetchall()]
    conn.commit()
    return rows


def _check_one_shot(results_dir: Path, system: str, resolver: str | None) -> None:
    for path in sorted(results_dir.glob("*.json")):
        try:
            header = json.loads(path.read_text()).get("header", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        if (header.get("split") == "test" and header.get("system") == system
                and header.get("resolver") == resolver):
            raise EvalPreconditionError(
                f"a test-split run for ({system}, {resolver}) already exists ({path.name}); "
                "the test split is one-shot. Pass force to run it again."
            )


def _check_graph(driver: Any, resolver: str | None) -> Any:
    if not resolver:
        raise EvalPreconditionError("the graph system needs a resolver (exact or user)")
    try:
        meta = read_graph_meta(driver)
    except Exception as e:  # noqa: BLE001 - any driver failure means Neo4j is unusable
        raise EvalPreconditionError(f"Neo4j is not reachable: {e}") from e
    if meta is None:
        raise EvalPreconditionError("Neo4j has no GraphMeta; build the graph first (kgrag build)")
    if meta.resolver != resolver:
        raise EvalPreconditionError(
            f"the graph was built with resolver '{meta.resolver}', not '{resolver}'; "
            f"rebuild it (kgrag build --resolver {resolver})"
        )
    return meta


@contextmanager
def _default_retriever(system: str, conn: psycopg.Connection, driver: Any) -> Iterator[Retriever]:
    from kgrag.retrieve.index import open_stores

    with open_stores() as (chunks, nodes):
        if system == "vector":
            from kgrag.retrieve.vector import VectorRetriever

            yield VectorRetriever(chunks, conn).retrieve
        else:
            from kgrag.graph.kg import KnowledgeGraph
            from kgrag.retrieve import graph_retriever

            kg = KnowledgeGraph(driver, conn, chunks, nodes)
            yield lambda q, budget: graph_retriever.retrieve(q, kg, budget)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _rollback(conn: psycopg.Connection) -> None:
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        log.warning("rollback failed", exc_info=True)


@traced("eval.question")
def _eval_question(
    q: QuestionRow, *, system: str, run_id: str, retrieve: Retriever, conn: psycopg.Connection,
    budget: int, answer_llm: Any, judge_llm: Any,
) -> dict:
    tag_current_trace(system=system, question_id=q.id, run_id=run_id)
    rec: dict[str, Any] = {
        "question_id": q.id, "type": q.type, "question": q.question, "gold_answer": q.answer,
        "gold_doc_ids": q.gold_doc_ids, "retrieved_doc_ids": [], "recall": None,
        "full_evidence": None, "answer": None, "verdict": "error", "latency_ms": None,
        "tokens_in": None, "tokens_out": None, "error": None, "retrieval_trace": {},
    }
    start = time.perf_counter()
    stage = "retrieve"
    try:
        result = retrieve(q.question, budget)
        rec["retrieved_doc_ids"] = [e.doc_id for e in result.evidence]
        rec["retrieval_trace"] = _json_safe(result.trace)
        rec["recall"] = source_recall(rec["retrieved_doc_ids"], q.gold_doc_ids)
        rec["full_evidence"] = full_evidence(rec["retrieved_doc_ids"], q.gold_doc_ids)
        stage = "answer"
        ans: Answer = answer(q.question, result, llm=answer_llm, conn=conn)
        rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        rec["answer"] = ans.model_dump()
        # Approximate token usage with the project counter (the chain does not expose usage).
        rec["tokens_in"] = (count_tokens(SYSTEM_PROMPT) + count_tokens(q.question)
                            + sum(count_tokens(e.text) for e in result.evidence))
        rec["tokens_out"] = count_tokens(ans.text)
        stage = "judge"
        rec["verdict"] = judge(q.question, q.answer, ans, judge_llm, is_null=q.type == "null")
    except NotImplementedError:
        raise  # an unimplemented retriever is not a per-question failure
    except Exception as e:  # noqa: BLE001 - one bad question must not sink the run
        log.warning("question %s failed at %s: %s", q.id, stage, e)
        rec["verdict"] = "error"
        rec["error"] = f"{stage}: {type(e).__name__}: {e}"
        _rollback(conn)
    if rec["latency_ms"] is None:
        rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return rec


def run_eval(
    conn: psycopg.Connection,
    driver: Any,
    *,
    system: Literal["vector", "graph"],
    split: Literal["dev", "test"],
    resolver: str | None,
    force: bool = False,
    results_dir: Path | None = None,
    retriever: Retriever | None = None,
    answer_llm: Any = None,
    judge_llm: Any = None,
) -> Path:
    """Run every kept question of `split` through `system`; returns the results file path.

    Preconditions are checked before any question runs. `retriever`, `answer_llm` and
    `judge_llm` default to the real ones and exist so tests can inject fakes.
    """
    results_dir = results_dir if results_dir is not None else RESULTS_DIR
    settings = get_settings()
    if system == "vector":
        resolver = None
    if split == "test" and not force:
        _check_one_shot(results_dir, system, resolver)
    meta = _check_graph(driver, resolver) if system == "graph" else None

    started = datetime.now(UTC)
    run_id = f"{system}-{resolver or 'na'}-{split}-{started:%Y%m%dT%H%M%SZ}"
    questions = load_questions(conn, split)
    if not questions:
        raise EvalPreconditionError(f"no kept/edited questions in split '{split}'")

    @contextmanager
    def retrievers() -> Iterator[Retriever]:
        if retriever is not None:
            yield retriever
        else:
            with _default_retriever(system, conn, driver) as r:
                yield r

    with retrievers() as retrieve:
        records = [
            _eval_question(
                q, system=system, run_id=run_id, retrieve=retrieve, conn=conn,
                budget=settings.budget_tokens, answer_llm=answer_llm, judge_llm=judge_llm,
            )
            for q in questions
        ]

    n_errors = sum(r["verdict"] == "error" for r in records)
    header = {
        "run_id": run_id, "system": system, "resolver": resolver, "split": split,
        "models": settings.models.model_dump(),
        "prompt_versions": {"answer": ANSWER_PROMPT_VERSION, "judge": JUDGE_PROMPT_VERSION,
                            **settings.prompt_versions},
        "config_hash": config_hash(settings),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "n_questions": len(records), "n_errors": n_errors,
        "degraded": n_errors / len(records) > DEGRADED_ERROR_RATE,
        "graph_meta": None if meta is None else {
            "resolver": meta.resolver, "built_at": meta.built_at.isoformat(),
            "config_hash": meta.config_hash},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_id}.json"
    path.write_text(json.dumps({"header": header, "questions": records}, indent=2))
    return path

