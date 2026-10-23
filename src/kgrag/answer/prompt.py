from kgrag.retrieve.types import Evidence
from kgrag.store.repo import Document

ANSWER_PROMPT_VERSION = "answer-v1"

SYSTEM_PROMPT = (
    "Answer the question using only the sources below. Cite the doc_id of every source you use. "
    "If the sources do not contain enough information to answer, set insufficient to true and "
    "say so."
)


def format_sources(evidence: list[Evidence], docs: dict[str, Document]) -> str:
    """One block per source, in evidence order: a metadata header line, then the evidence text."""
    blocks = []
    for ev in evidence:
        doc = docs.get(ev.doc_id)
        date = f"{doc.date:%Y-%m-%d}" if doc else "unknown"
        sender = doc.sender if doc else "unknown"
        subject = doc.subject if doc else "unknown"
        blocks.append(f"[{ev.doc_id}] {date} · {sender} · {subject}\n{ev.text}")
    return "\n\n".join(blocks)
