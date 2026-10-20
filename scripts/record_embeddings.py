"""Record embeddings for the dev fixture: document texts, graph node names and dev questions.

Run once the embedding model is pinned (`models.embed`) and VOYAGE_API_KEY is set:
    uv run python scripts/record_embeddings.py
Idempotent: texts already in tests/fixtures/recorded_embeddings.json are not re-embedded.
Output: {"model", "documents": {text: vec}, "queries": {text: vec}}; documents and node
names use embed_documents, questions use embed_query (as production does).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures"
OUT = FIXTURES / "recorded_embeddings.json"
BATCH = 128


def fixture_texts() -> tuple[list[str], list[str]]:
    """(document texts + node names, questions) the retrieval tests embed; stable, deduped."""
    import make_fixture
    from kgrag.config import get_settings
    from kgrag.graph.build import _assemble
    from kgrag.resolve.exact_match import ExactMatchResolver
    from kgrag.resolve.interface import resolve_all
    from kgrag.store.repo import load_anchor_hints, load_mentions

    recorded = json.loads((FIXTURES / "recorded_extractions.json").read_text())
    with make_fixture.fresh_database(get_settings().pg_dsn, "kgrag_record_embeddings") as conn:
        make_fixture.load_slice(conn, FIXTURES / "dev_slice.mbox", recorded)
        docs = [r[0] for r in conn.execute("select text from documents order by doc_id")]
        resolution = resolve_all(ExactMatchResolver(), load_mentions(conn), load_anchor_hints(conn))
        graph = _assemble(conn, resolution)
    names = [
        n["name"] for label, nodes in sorted(graph.nodes.items()) if label != "Email"
        for n in nodes.values() if n.get("name")
    ]
    questions = [q["question"] for q in json.loads((FIXTURES / "dev_questions.json").read_text())]
    return list(dict.fromkeys(docs + names)), list(dict.fromkeys(questions))


def main() -> None:
    from kgrag.config import get_settings
    from kgrag.retrieve.embeddings import EmbeddingsNotConfiguredError, get_embeddings

    try:
        emb = get_embeddings()
    except EmbeddingsNotConfiguredError as e:
        sys.exit(str(e))
    model = get_settings().models.embed
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if data and data.get("model") != model:
        sys.exit(f"{OUT.name} was recorded with {data.get('model')!r}, not {model!r}; delete it first")
    data = {"model": model, "documents": data.get("documents", {}), "queries": data.get("queries", {})}
    docs, queries = fixture_texts()
    new = 0
    for section, texts, embed in (
        ("documents", docs, emb.embed_documents),
        ("queries", queries, lambda b: [emb.embed_query(t) for t in b]),
    ):
        todo = [t for t in texts if t not in data[section]]
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            data[section].update(zip(batch, embed(batch), strict=True))
            print(f"{section}: embedded {min(i + BATCH, len(todo))}/{len(todo)}")
        new += len(todo)
    OUT.write_text(json.dumps(data))
    print(f"{len(data['documents'])} documents, {len(data['queries'])} queries ({new} new) -> {OUT}")


if __name__ == "__main__":
    main()
