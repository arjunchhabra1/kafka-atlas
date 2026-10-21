"""End-to-end smoke test, offline: dev fixture -> resolve -> build -> index -> eval -> report.

No network and no paid API: recorded extractions, deterministic (or recorded) embeddings, and fake
answer/judge chat models. It checks that the stages are wired together, not retrieval quality.
"""

import json
import time

import pytest
from langchain_core.runnables import RunnableLambda

from kgrag.answer.chain import AnswerSchema
from kgrag.eval.judge import Verdict
from kgrag.eval.report import render_report
from kgrag.eval.run import run_eval
from kgrag.graph.build import build_graph
from kgrag.graph.kg import KnowledgeGraph
from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all
from kgrag.resolve.resolver import UserResolver
from kgrag.retrieve import graph_retriever
from kgrag.retrieve.embeddings import HashEmbeddings, RecordedEmbeddings, set_embeddings_for_tests
from kgrag.retrieve.index import build_index, open_stores
from kgrag.retrieve.vector import VectorRetriever
from kgrag.store.repo import (
    QuestionRow,
    insert_question,
    load_anchor_hints,
    load_mentions,
    save_resolution,
)
from tests.conftest import FIXTURES, make_fixture_module

pytestmark = pytest.mark.integration


class FakeLLM:
    """Answers every question with no citations; judges every answer correct."""

    def with_structured_output(self, schema, **kwargs):
        def run(messages, config=None):
            if schema is Verdict:
                return Verdict(verdict="correct")
            return AnswerSchema(answer="an answer", cited_doc_ids=[], insufficient=False)

        return RunnableLambda(run)


def test_pipeline_end_to_end(pg_conn, neo4j_driver, tmp_path):
    started = time.perf_counter()
    set_embeddings_for_tests(RecordedEmbeddings(FIXTURES / "recorded_embeddings.json")
                             if (FIXTURES / "recorded_embeddings.json").exists()
                             else HashEmbeddings(dim=64))
    mf = make_fixture_module()
    i = pg_conn.info
    admin = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{i.dbname}"
    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    try:
        with mf.fresh_database(admin, "kgrag_smoke") as conn:
            dsn = f"postgresql://{i.user}:{i.password}@{i.host}:{i.port}/{conn.info.dbname}"
            # ingest dev_slice.mbox + recorded extractions, then resolve with both resolvers
            mf.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
            for name, resolver in (("exact", ExactMatchResolver()), ("user", UserResolver())):
                save_resolution(conn, name, resolve_all(
                    resolver, load_mentions(conn), load_anchor_hints(conn)))
            for q in json.loads((FIXTURES / "dev_questions.json").read_text()):
                insert_question(conn, QuestionRow(
                    id=q["id"], type=q["type"], question=q["question"], answer=q["answer"],
                    gold_doc_ids=q["gold_doc_ids"], status="kept", split="dev"))
            conn.commit()

            runs = []
            for resolver in ("exact", "user"):  # the graph holds one resolver at a time
                stats = build_graph(conn, neo4j_driver, resolver)
                assert sum(stats.nodes.values()) > 0 and sum(stats.rels.values()) > 0
                build_index(conn, neo4j_driver, connection=dsn)
                with open_stores(dsn) as (chunks, nodes):
                    kg = KnowledgeGraph(neo4j_driver, conn, chunks, nodes)
                    if resolver == "exact":
                        runs.append(run_eval(
                            conn, neo4j_driver, system="vector", split="dev", resolver=None,
                            results_dir=tmp_path, retriever=VectorRetriever(chunks, conn).retrieve,
                            answer_llm=FakeLLM(), judge_llm=FakeLLM()))
                    runs.append(run_eval(
                        conn, neo4j_driver, system="graph", split="dev", resolver=resolver,
                        results_dir=tmp_path,
                        retriever=lambda q, b, kg=kg: graph_retriever.retrieve(q, kg, b),
                        answer_llm=FakeLLM(), judge_llm=FakeLLM()))

            labels = []
            for path in runs:
                data = json.loads(path.read_text())
                labels.append(data["header"]["system"] + (data["header"]["resolver"] or ""))
                assert len(data["questions"]) == 10
                assert data["header"]["n_errors"] == 0 and not data["header"]["degraded"]
                assert all(q["retrieved_doc_ids"] for q in data["questions"])
            assert labels == ["vector", "graphexact", "graphuser"]

            report = render_report(runs, split="dev")
            assert "Multi-hop" in report
            assert "Paired differences vs the vector baseline" in report
            assert "graph-exact minus vector" in report and "graph-user minus vector" in report
    finally:
        set_embeddings_for_tests(None)
    assert time.perf_counter() - started < 60, "smoke test must stay under 60 s"
