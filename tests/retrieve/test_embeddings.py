import json
from pathlib import Path

import pytest

from kgrag.retrieve.embeddings import (
    HashEmbeddings,
    RecordedEmbeddings,
    offline_embeddings,
    set_embeddings_for_tests,
)
from kgrag.store.repo import first_document_text


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


def test_recorded_embeddings_cover_every_test_query(slice_conn):
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    recorded = fixtures / "recorded_embeddings.json"
    if not recorded.exists():
        pytest.skip("recorded_embeddings.json not recorded yet")
    if not isinstance(offline_embeddings(recorded), RecordedEmbeddings):
        pytest.skip("recorded_embeddings.json was recorded with another model (stale)")
    queries = json.loads(recorded.read_text())["queries"]
    wanted = list(json.loads((fixtures / "test_queries.json").read_text()).values())
    wanted += [q["question"] for q in json.loads((fixtures / "dev_questions.json").read_text())]
    wanted.append(first_document_text(slice_conn))  # test_vector uses it as a query
    assert [q for q in wanted if q not in queries] == []


def _recording(tmp_path, model):
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({"model": model, "documents": {"d": [1.0, 0.0]},
                                "queries": {"q": [0.0, 1.0]}}))
    return path


def test_recorded_embeddings_expose_model_and_stale_ones_are_not_used(tmp_path):
    path = _recording(tmp_path, "voyage-a")
    assert RecordedEmbeddings(path).model == "voyage-a"
    assert isinstance(offline_embeddings(path, model="voyage-a"), RecordedEmbeddings)
    assert isinstance(offline_embeddings(path, model="voyage-b"), HashEmbeddings)  # stale
    assert isinstance(offline_embeddings(tmp_path / "missing.json", "voyage-a"), HashEmbeddings)


def test_build_index_refuses_embeddings_of_another_model(tmp_path):
    from kgrag.retrieve.index import EmbeddingsModelMismatchError, build_index

    set_embeddings_for_tests(RecordedEmbeddings(_recording(tmp_path, "voyage-a")))
    try:
        with pytest.raises(EmbeddingsModelMismatchError, match="voyage-a.*voyage-b"):
            build_index(None, None, model="voyage-b")  # refuses before touching any database
    finally:
        set_embeddings_for_tests(None)
