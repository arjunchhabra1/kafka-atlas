"""Record embeddings for the dev fixture: document texts, graph node names and dev questions.

Run once the embedding model is pinned (`models.embed`) and VOYAGE_API_KEY is set:
    uv run python scripts/record_embeddings.py
Idempotent: texts already in tests/fixtures/recorded_embeddings.json are not re-embedded.
Output: {text: vector}.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures"
OUT = FIXTURES / "recorded_embeddings.json"
BATCH = 128


def fixture_texts() -> list[str]:
    """Every text the retrieval tests will embed, in a stable order, without duplicates."""
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
    return list(dict.fromkeys(docs + names + questions))


def main() -> None:
    from kgrag.retrieve.embeddings import EmbeddingsNotConfiguredError, get_embeddings

    try:
        emb = get_embeddings()
    except EmbeddingsNotConfiguredError as e:
        sys.exit(str(e))
    recorded: dict[str, list[float]] = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [t for t in fixture_texts() if t not in recorded]
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        recorded.update(zip(batch, emb.embed_documents(batch), strict=True))
        print(f"embedded {min(i + BATCH, len(todo))}/{len(todo)}")
    OUT.write_text(json.dumps(recorded))
    print(f"{len(recorded)} texts recorded ({len(todo)} new) -> {OUT}")


if __name__ == "__main__":
    main()
