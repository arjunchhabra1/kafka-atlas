"""YOUR TASK: graph retrieval. Find the evidence for a question by walking the knowledge graph.

The problem
-----------
The vector baseline embeds the question and returns the nearest chunks. Your retriever
should do better on questions that need structure: "who proposed KIP-X and who voted on
it", "what changed between these two KIPs", "which tickets shipped in release 3.9". Link
the question to graph entities, traverse to related entities, and turn what you find
into text the answer model can read.

The contract
------------
`retrieve(question, kg, budget_tokens) -> RetrievalResult`

  * `evidence`: `Evidence(doc_id, text, score)` rows, best first. Each `doc_id` must exist
    in the `documents` table, i.e. `email:{email_id}:{chunk}` or `jira:{event_id}`, and
    `text` is that row's text (use `kg.documents` / `kg.search_chunks`, or take the ids
    from `kg.source_docs(edge)`). Never invent evidence text; the answer model only sees it.
  * The evidence must fit `budget_tokens` in total (`kgrag.tokens.count_tokens`). Finish
    with `pack_to_budget(candidates, budget_tokens)`, the same packing the baseline uses.
  * Deterministic: the same question on the same graph gives the same evidence.
  * Never raise on a question with no recognizable entity ("What's a good pizza recipe?");
    return a valid result, e.g. fall back to `kg.search_chunks`.
  * `trace`: a dict that makes the result explainable. Record the entities you linked, the
    paths or edges you followed, and anything you dropped.

The toolkit: `kg` is a `KnowledgeGraph` (src/kgrag/graph/kg.py)
  * `kg.find_nodes("KIP-848")`                      exact name/id lookup (case-insensitive)
  * `kg.find_nodes("Chia Ping Tsai", fuzzy=True)`   fuzzy name lookup (score >= 80)
  * `kg.search_nodes("the Jakarta upgrade KIP")`    embedding search over node names
  * `kg.neighbors(id, edge_types=["VOTED"], direction="in", since=date(2024, 1, 1))`
                                                    (Edge, Node) pairs around a node
  * `kg.source_docs(edge)`                          document ids that justify an edge
  * `kg.documents(doc_ids)`                         the `documents` rows for those ids
  * `kg.search_chunks(question, k=20)`              plain vector retrieval, as `Evidence`
  * `kg.cypher("MATCH (k:KIP) RETURN count(k) AS c")`  read-only Cypher, rows as dicts
Node labels: Person, KIP, JiraTicket, Release, Component, Email. Edge types: SENT, REPLIED_TO,
MENTIONS, FILED, ASSIGNED, RESOLVED, FIX_VERSION, PROPOSED, VOTED, REVIEWED, BLOCKS, AFFECTS,
DEPENDS_ON, SUPERSEDES. Edge props: source_doc_ids, date, extractor, confidence.

Fairness rules (the comparison with the baseline depends on them)
  * Evidence must be chunk-level `documents` rows, never graph summaries or edge listings.
  * Same budget as the baseline, packed with `pack_to_budget`.
  * Tune only on the fixture and the dev split (tests/fixtures/dev_questions.json). The
    test split is for the final headline run; do not look at it while developing.

Check your work
---------------
    uv run pytest tests/retrieve/test_graph_contract.py tests/retrieve/test_graph_quality.py -v -s

`test_graph_contract.py` checks the rules above. `test_graph_quality.py` prints gold-source
recall for graph vs vector baseline, split by single-hop and multi-hop questions; it asserts
nothing, it is there to guide you. Both xfail until this stub stops raising
NotImplementedError, then run for real. `tests/graph/test_kg.py` covers the toolkit itself.
"""

from kgrag.graph.kg import KnowledgeGraph
from kgrag.retrieve.packing import pack_to_budget  # noqa: F401 - for your convenience
from kgrag.retrieve.types import Evidence, RetrievalResult  # noqa: F401


def retrieve(question: str, kg: KnowledgeGraph, budget_tokens: int) -> RetrievalResult:
    raise NotImplementedError("👤 implement graph retrieval")
