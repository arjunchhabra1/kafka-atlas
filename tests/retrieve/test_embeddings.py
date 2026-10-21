import json
from pathlib import Path

import pytest

from kgrag.retrieve.embeddings import RecordedEmbeddings


def test_recorded_documents_and_queries_are_separate(tmp_path):
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({
        "model": "m",
        "documents": {"doc text": [1.0, 0.0]},
        "queries": {"a question": [0.0, 1.0], "doc text": [0.5, 0.5]},
    }))
    e = RecordedEmbeddings(path)
    assert e.embed_documents(["doc text"]) == [[1.0, 0.0]]
    assert e.embed_query("a question") == [0.0, 1.0]
    assert e.embed_query("doc text") == [0.5, 0.5]
    with pytest.raises(KeyError):
        e.embed_documents(["a question"])
    with pytest.raises(KeyError):
        e.embed_query("unknown")


def test_recorded_embeddings_cover_every_test_query():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    recorded = fixtures / "recorded_embeddings.json"
    if not recorded.exists():
        pytest.skip("recorded_embeddings.json not recorded yet")
    queries = json.loads(recorded.read_text())["queries"]
    wanted = list(json.loads((fixtures / "test_queries.json").read_text()).values())
    wanted += [q["question"] for q in json.loads((fixtures / "dev_questions.json").read_text())]
    assert [q for q in wanted if q not in queries] == []
