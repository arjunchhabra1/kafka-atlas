from datetime import datetime

import pytest
from langchain_core.runnables import RunnableLambda

from kgrag.answer.chain import Answer, AnswerSchema, answer
from kgrag.answer.prompt import ANSWER_PROMPT_VERSION, SYSTEM_PROMPT, format_sources
from kgrag.retrieve.types import Evidence, RetrievalResult
from kgrag.store.repo import Document


def _doc(doc_id: str, subject: str = "Subj") -> Document:
    return Document(
        doc_id=doc_id, source_kind="email", source_id="s", chunk_index=0, text="full text",
        token_count=2, date=datetime(2024, 3, 5, 10, 0), sender="alice@x.org", subject=subject,
    )


class StubLLM:
    def __init__(self, parsed: AnswerSchema):
        self.parsed = parsed
        self.method = None
        self.schema = None
        self.messages = None
        self.config = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.method = kwargs.get("method")

        def run(messages, config=None):
            self.messages = messages
            return self.parsed

        return RunnableLambda(run)


EVIDENCE = [Evidence(doc_id="d1", text="trimmed one", score=0.9),
            Evidence(doc_id="d2", text="trimmed two", score=0.5)]
DOCS = {"d1": _doc("d1", "First"), "d2": _doc("d2", "Second")}


def test_prompt_version():
    assert ANSWER_PROMPT_VERSION == "answer-v1"


def test_format_sources_layout():
    out = format_sources(EVIDENCE, DOCS)
    assert out == (
        "[d1] 2024-03-05 · alice@x.org · First\ntrimmed one\n\n"
        "[d2] 2024-03-05 · alice@x.org · Second\ntrimmed two"
    )


def test_format_sources_missing_doc_uses_unknown():
    out = format_sources([Evidence(doc_id="zz", text="t", score=0.1)], {})
    assert out == "[zz] unknown · unknown · unknown\nt"


def test_answer_parses_citations_and_sends_messages():
    llm = StubLLM(AnswerSchema(answer="It is X.", cited_doc_ids=["d2", "d1"], insufficient=False))
    res = answer("What is X?", RetrievalResult(evidence=EVIDENCE), docs=DOCS, llm=llm)
    assert res == Answer(text="It is X.", cited_doc_ids=["d2", "d1"], insufficient=False)
    assert llm.method == "json_schema"
    assert llm.schema is AnswerSchema
    system, human = llm.messages
    assert system.type == "system" and system.content == SYSTEM_PROMPT
    assert SYSTEM_PROMPT == (
        "Answer the question using only the sources below. Cite the doc_id of every source you "
        "use. If the sources do not contain enough information to answer, set insufficient to "
        "true and say so."
    )
    assert human.type == "human"
    assert human.content == f"Sources:\n\n{format_sources(EVIDENCE, DOCS)}\n\nQuestion: What is X?"


def test_citations_filtered_to_given_sources():
    llm = StubLLM(AnswerSchema(answer="a", cited_doc_ids=["d1", "bogus", "d1", "d2"],
                               insufficient=False))
    res = answer("q", RetrievalResult(evidence=EVIDENCE), docs=DOCS, llm=llm)
    assert res.cited_doc_ids == ["d1", "d2"]


def test_empty_evidence_short_circuits():
    class Boom:
        def with_structured_output(self, *a, **k):
            raise AssertionError("llm must not be touched")

    res = answer("q", RetrievalResult(evidence=[]), llm=Boom())
    assert res == Answer(text="Insufficient information: no sources were retrieved.",
                         cited_doc_ids=[], insufficient=True)


def test_docs_or_conn_required():
    llm = StubLLM(AnswerSchema(answer="a", cited_doc_ids=[], insufficient=False))
    with pytest.raises(ValueError):
        answer("q", RetrievalResult(evidence=EVIDENCE), llm=llm)


def test_fetches_docs_via_conn(monkeypatch):
    seen = {}

    def fake_get(conn, ids):
        seen["args"] = (conn, ids)
        return list(DOCS.values())

    monkeypatch.setattr("kgrag.answer.chain.repo.get_documents", fake_get)
    llm = StubLLM(AnswerSchema(answer="a", cited_doc_ids=["d1"], insufficient=False))
    res = answer("q", RetrievalResult(evidence=EVIDENCE), conn="CONN", llm=llm)
    assert seen["args"] == ("CONN", ["d1", "d2"])
    assert res.cited_doc_ids == ["d1"]
    assert "[d1] 2024-03-05 · alice@x.org · First" in llm.messages[1].content
