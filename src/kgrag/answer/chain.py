"""Shared answer chain: both retrieval systems feed evidence through this one function."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from kgrag.answer.prompt import SYSTEM_PROMPT, format_sources
from kgrag.config import get_settings
from kgrag.observability import langchain_callbacks
from kgrag.retrieve.types import RetrievalResult
from kgrag.store import repo
from kgrag.store.repo import Document

NO_SOURCES_TEXT = "Insufficient information: no sources were retrieved."


class AnswerSchema(BaseModel):
    """Structured output requested from the model."""

    answer: str
    cited_doc_ids: list[str] = Field(default_factory=list)
    insufficient: bool


class Answer(BaseModel):
    text: str
    cited_doc_ids: list[str]
    insufficient: bool


def answer(
    question: str,
    result: RetrievalResult,
    *,
    docs: dict[str, Document] | None = None,
    llm: BaseChatModel | None = None,
    conn=None,
) -> Answer:
    if not result.evidence:
        return Answer(text=NO_SOURCES_TEXT, cited_doc_ids=[], insufficient=True)
    if docs is None:
        if conn is None:
            raise ValueError("answer() needs either docs or conn")
        ids = [e.doc_id for e in result.evidence]
        docs = {d.doc_id: d for d in repo.get_documents(conn, ids)}
    if llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=get_settings().models.answer, max_tokens=4000)
    chain = llm.with_structured_output(AnswerSchema, method="json_schema")
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"Sources:\n\n{format_sources(result.evidence, docs)}\n\nQuestion: {question}"
        ),
    ]
    parsed = chain.invoke(messages, config={"callbacks": langchain_callbacks()})
    allowed = {e.doc_id for e in result.evidence}
    cited = [i for i in dict.fromkeys(parsed.cited_doc_ids) if i in allowed]
    return Answer(text=parsed.answer, cited_doc_ids=cited, insufficient=parsed.insufficient)
