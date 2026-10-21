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
from kgrag.config import CONFIG_PATH, build_hash, get_settings, run_hash
from kgrag.eval.judge import JUDGE_PROMPT_VERSION, grade
from kgrag.eval.metrics import full_evidence, source_recall
from kgrag.eval.preflight import EvalPreconditionError, check_freshness
from kgrag.graph.build import read_graph_meta
from kgrag.observability import tag_current_trace, traced
from kgrag.retrieve.graph_config import (
    DEFAULT_CONFIG_NAME,
    GraphRetrieverConfig,
    UnknownGraphConfigError,
    get_graph_config,
)
from kgrag.retrieve.types import RetrievalResult
from kgrag.store.repo import QuestionRow
from kgrag.tokens import count_tokens

RESULTS_DIR = CONFIG_PATH.parent / "results"  # anchored to the project, never the cwd
DEGRADED_ERROR_RATE = 0.10  # a run with more than this fraction of errored questions is degraded
log = logging.getLogger(__name__)

Retriever = Callable[[str, int], RetrievalResult]


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


def run_identity(header: dict) -> tuple[str, str | None, str | None]:
    """(system, resolver, graph_config) of a run header. Graph runs from before graph configs
    were recorded used the default config."""
    system = header.get("system")
    if system != "graph":
        return system, None, None
    return system, header.get("resolver"), header.get("graph_config") or DEFAULT_CONFIG_NAME


def _read_partial_header(path: Path) -> dict | None:
    try:
        with path.open() as f:
            first = json.loads(f.readline() or "null")
    except (OSError, json.JSONDecodeError):
        return None
    return first.get("header") if isinstance(first, dict) else None


def _check_one_shot(results_dir: Path, identity: tuple[str, str | None, str | None]) -> None:
    label = describe(*identity)
    for path in sorted(results_dir.glob("*.json")):
        try:
            header = json.loads(path.read_text()).get("header", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        if header.get("degraded"):
            continue  # a degraded run does not consume the one-shot
        if header.get("split") == "test" and run_identity(header) == identity:
            raise EvalPreconditionError(
                f"a test-split run for {label} already exists ({path.name}); "
                "the test split is one-shot (degraded runs do not count). Pass --force to run it again."
            )
    for path in sorted(results_dir.glob("*.partial.jsonl")):
        header = _read_partial_header(path)
        if header and header.get("split") == "test" and run_identity(header) == identity:
            raise EvalPreconditionError(
                f"an interrupted test-split run for {label} exists ({path.name}); its answers "
                "were already produced (and possibly seen), so the test split counts as spent for "
                "it. Pass --force to run it again, and report that you did."
            )


def describe(system: str, resolver: str | None, graph_config: str | None) -> str:
    """Human label of a run identity, e.g. `vector`, `graph-user`, `graph-user (graph-only)`."""
    if system != "graph":
        return system
    base = f"graph-{resolver}"
    return base if graph_config in (None, DEFAULT_CONFIG_NAME) else f"{base} ({graph_config})"


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
def _default_retriever(system: str, conn: psycopg.Connection, driver: Any,
                       config: GraphRetrieverConfig | None) -> Iterator[Retriever]:
    from kgrag.retrieve.index import open_stores

    with open_stores() as (chunks, nodes):
        if system == "vector":
            from kgrag.retrieve.vector import VectorRetriever

            yield VectorRetriever(chunks, conn).retrieve
        else:
            from kgrag.graph.kg import KnowledgeGraph
            from kgrag.retrieve import graph_retriever

            kg = KnowledgeGraph(driver, conn, chunks, nodes)
            yield lambda q, budget: graph_retriever.retrieve(q, kg, budget, config)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _rollback(conn: psycopg.Connection) -> None:
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        log.warning("rollback failed", exc_info=True)


def _record_error(rec: dict, conn: psycopg.Connection, q: QuestionRow, stage: str, e: Any) -> None:
    log.warning("question %s failed at %s: %s", q.id, stage, e)
    rec["verdict"] = "error"
    rec["error"] = f"{stage}: {type(e).__name__}: {e}"
    _rollback(conn)


@traced("eval.question")
def _eval_question(
    q: QuestionRow, *, system: str, run_id: str, retrieve: Retriever, conn: psycopg.Connection,
    budget: int, answer_llm: Any, judge_llm: Any,
) -> dict:
    tag_current_trace(system=system, question_id=q.id, run_id=run_id)
    rec: dict[str, Any] = {
        "question_id": q.id, "type": q.type, "question": q.question, "gold_answer": q.answer,
        "gold_doc_ids": q.gold_doc_ids, "retrieved_doc_ids": [], "recall": None,
        "full_evidence": None, "answer": None, "verdict": "error", "judge_rationale": None, "latency_ms": None,
        "tokens_in": None, "tokens_out": None, "error": None, "retrieval_trace": {},
    }
    start = time.perf_counter()
    stage = "retrieve"
    try:
        try:
            result = retrieve(q.question, budget)
        except Exception:
            # A failed retrieval counts against the system: nothing retrieved, so recall 0.
            rec["recall"] = source_recall([], q.gold_doc_ids)
            rec["full_evidence"] = full_evidence([], q.gold_doc_ids)
            raise
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
        verdict = grade(q.question, q.answer, ans, judge_llm, is_null=q.type == "null")
        rec["verdict"], rec["judge_rationale"] = verdict.verdict, verdict.rationale
    except Exception as e:  # noqa: BLE001 - one bad question must not sink the run
        _record_error(rec, conn, q, stage, e)
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
    graph_config: str | None = None,
    force: bool = False,
    results_dir: Path | None = None,
    retriever: Retriever | None = None,
    answer_llm: Any = None,
    judge_llm: Any = None,
) -> Path:
    """Run every kept question of `split` through `system`; returns the results file path.

    A run is identified by (system, resolver, graph_config); `graph_config` names an entry of
    GRAPH_CONFIGS (default "hybrid") and is None for the vector system. Preconditions (the
    one-shot rule, the graph's resolver, and the freshness preflight in eval/preflight.py)
    are checked before any question runs. `retriever`, `answer_llm` and `judge_llm` default to the
    real ones and exist so tests can inject fakes.
    """
    results_dir = results_dir if results_dir is not None else RESULTS_DIR
    settings = get_settings()
    if system == "vector":
        resolver, graph_config, config = None, None, None
    else:
        graph_config = graph_config or DEFAULT_CONFIG_NAME
        try:
            config = get_graph_config(graph_config)
        except UnknownGraphConfigError as e:
            raise EvalPreconditionError(str(e)) from None
    identity = (system, resolver, graph_config)
    if split == "test" and not force:
        _check_one_shot(results_dir, identity)
    meta = _check_graph(driver, resolver) if system == "graph" else None

    started = datetime.now(UTC)
    cfg_part = f"-{graph_config}" if graph_config not in (None, DEFAULT_CONFIG_NAME) else ""
    run_id = f"{system}-{resolver or 'na'}{cfg_part}-{split}-{started:%Y%m%dT%H%M%SZ}"
    questions = load_questions(conn, split)
    # stale documents/index/graph would silently skew every number: refuse before any question
    check_freshness(conn, driver, questions, system=system, resolver=resolver, meta=meta)
    if not questions:
        raise EvalPreconditionError(f"no kept/edited questions in split '{split}'")

    @contextmanager
    def retrievers() -> Iterator[Retriever]:
        if retriever is not None:
            yield retriever
        else:
            with _default_retriever(system, conn, driver, config) as r:
                yield r

    results_dir.mkdir(parents=True, exist_ok=True)
    partial = results_dir / f"{run_id}.partial.jsonl"
    records = []
    with retrievers() as retrieve, partial.open("w") as sink:
        # first line: the identity, so an interrupted test run still counts as spent
        sink.write(json.dumps({"header": {"run_id": run_id, "system": system, "resolver": resolver,
                                          "graph_config": graph_config, "split": split}}) + "\n")
        sink.flush()
        for q in questions:
            rec = _eval_question(
                q, system=system, run_id=run_id, retrieve=retrieve, conn=conn,
                budget=settings.budget_tokens, answer_llm=answer_llm, judge_llm=judge_llm,
            )
            records.append(rec)
            sink.write(json.dumps(rec) + "\n")
            sink.flush()  # a crash leaves everything done so far for inspection

    n_errors = sum(r["verdict"] == "error" for r in records)
    header = {
        "run_id": run_id, "system": system, "resolver": resolver, "split": split,
        "graph_config": graph_config,
        "graph_config_hash": None if config is None else config.fingerprint(),
        "models": settings.models.model_dump(),
        "prompt_versions": {"answer": ANSWER_PROMPT_VERSION, "judge": JUDGE_PROMPT_VERSION,
                            **settings.prompt_versions},
        # run_hash: everything the numbers depend on besides the data (incl. the graph config);
        # base_run_hash: the system-independent part, compared across systems by the report;
        # build_hash: the extraction/chunking settings, compared with the graph's GraphMeta.
        "run_hash": run_hash(settings, config),
        "base_run_hash": run_hash(settings, None),
        "build_hash": build_hash(settings),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "n_questions": len(records), "n_errors": n_errors,
        "degraded": n_errors / len(records) > DEGRADED_ERROR_RATE,
        "graph_meta": None if meta is None else {
            "resolver": meta.resolver, "built_at": meta.built_at.isoformat(),
            "build_hash": meta.build_hash, "resolution_hash": meta.resolution_hash},
    }
    path = results_dir / f"{run_id}.json"
    path.write_text(json.dumps({"header": header, "questions": records}, indent=2))
    partial.unlink()
    return path

