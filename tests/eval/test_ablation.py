"""`kgrag ablate`: dev-split recall per graph config, retrieval only."""

import json

import pytest
from typer.testing import CliRunner

from kgrag.cli import app
from kgrag.eval.ablation import run_ablation
from kgrag.eval.run import EvalPreconditionError
from kgrag.retrieve.graph_config import DEFAULT_CONFIG, GRAPH_ONLY_CONFIG
from kgrag.store.repo import QuestionRow, insert_question
from tests.conftest import FIXTURES


def test_cli_refuses_the_test_split():
    r = CliRunner().invoke(app, ["ablate", "--split", "test", "--resolver", "user"])
    assert r.exit_code == 2 and "refusing --split test" in r.output


def test_cli_rejects_unknown_config():
    r = CliRunner().invoke(app, ["ablate", "--resolver", "user", "--configs", "hybrid,nope"])
    assert r.exit_code == 2 and "unknown graph config" in r.output


def test_run_ablation_refuses_test_split(tmp_path):
    with pytest.raises(EvalPreconditionError, match="test split"):
        run_ablation(None, None, split="test", resolver="exact", configs={"hybrid": DEFAULT_CONFIG},
                     budget=4000, results_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def dev_questions(fixture_kg):
    conn = fixture_kg.conn
    qs = json.loads((FIXTURES / "dev_questions.json").read_text())
    for q in qs:
        insert_question(conn, QuestionRow(
            id=f"abl-{q['id']}", type=q["type"], question=q["question"], answer=q["answer"],
            gold_doc_ids=q["gold_doc_ids"], status="kept", split="dev"))
    insert_question(conn, QuestionRow(  # test split: never read by the ablation
        id="abl-test", type="single_hop", question="KIP-1043?", answer="a",
        gold_doc_ids=["nope"], status="kept", split="test"))
    conn.commit()
    yield qs
    conn.execute("delete from questions where id like 'abl-%'")
    conn.commit()


@pytest.mark.integration
def test_run_ablation_writes_results(fixture_kg, dev_questions, tmp_path):
    path, results = run_ablation(
        fixture_kg.conn, fixture_kg.driver, split="dev", resolver="exact",
        configs={"hybrid": DEFAULT_CONFIG, "graph-only": GRAPH_ONLY_CONFIG}, budget=4000,
        results_dir=tmp_path, kg=fixture_kg)
    assert path.name.startswith("ablation-") and path.parent == tmp_path
    data = json.loads(path.read_text())
    h = data["header"]
    assert h["split"] == "dev" and h["resolver"] == "exact" and h["budget_tokens"] == 4000
    assert h["configs"] == {"hybrid": DEFAULT_CONFIG.fingerprint(),
                            "graph-only": GRAPH_ONLY_CONFIG.fingerprint()}
    assert set(data["results"]) == {"hybrid", "graph-only", "vector"}
    n_gold = sum(bool(q["gold_doc_ids"]) for q in dev_questions)
    assert h["n_questions"] == n_gold and results["hybrid"]["n"]["all"] == n_gold
    assert results["graph-only"]["unlinked_rate"] is not None
    assert results["vector"]["unlinked_rate"] is None
    assert results["hybrid"]["recall"]["multi-hop"] >= results["vector"]["recall"]["multi-hop"]


@pytest.mark.integration
def test_run_ablation_needs_matching_graph(fixture_kg, dev_questions, tmp_path):
    with pytest.raises(EvalPreconditionError, match="resolver 'exact', not 'user'"):
        run_ablation(fixture_kg.conn, fixture_kg.driver, split="dev", resolver="user",
                     configs={"hybrid": DEFAULT_CONFIG}, budget=4000, results_dir=tmp_path,
                     kg=fixture_kg)
