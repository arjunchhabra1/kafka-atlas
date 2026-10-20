import psycopg
from langchain_postgres import PGVector

from kgrag.observability import traced
from kgrag.retrieve.packing import pack_to_budget
from kgrag.retrieve.types import Evidence, RetrievalResult
from kgrag.store.repo import get_documents

K = 50


class VectorRetriever:
    """Baseline: top-k chunks by cosine similarity, packed to the token budget."""

    def __init__(self, store: PGVector, conn: psycopg.Connection):
        self.store, self.conn = store, conn

    @traced("retrieve.vector")
    def retrieve(self, question: str, budget_tokens: int) -> RetrievalResult:
        hits = self.store.similarity_search_with_score(question, k=K)
        ids = [d.metadata["doc_id"] for d, _ in hits]
        text = {d.doc_id: d.text for d in get_documents(self.conn, ids)}
        self.conn.commit()
        candidates = [
            Evidence(doc_id=i, text=text[i], score=1.0 - dist)
            for i, (_, dist) in zip(ids, hits, strict=True) if i in text
        ]
        return RetrievalResult(
            evidence=pack_to_budget(candidates, budget_tokens),
            trace={"k": K, "considered": len(candidates)},
        )
