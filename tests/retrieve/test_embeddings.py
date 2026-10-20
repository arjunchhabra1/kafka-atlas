import json

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
