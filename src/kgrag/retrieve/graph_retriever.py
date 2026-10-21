"""Graph retrieval: find the evidence for a question by walking the knowledge graph.

The contract
------------
`retrieve(question, kg, budget_tokens) -> RetrievalResult`

  * `evidence`: `Evidence(doc_id, text, score)` rows, best first. Every `doc_id` is a
    `documents` row (`email:{email_id}:{chunk}` or `jira:{event_id}`) and `text` is that
    row's stored text. No graph summaries or edge listings: the answer model sees only chunks.
  * Packed with `pack_to_budget(candidates, budget_tokens)`, the same packing the vector
    baseline uses, so both systems get the same token budget.
  * Deterministic: same question + same graph -> same evidence (every sort has a doc_id or
    node id tie-break; scores are rounded).
  * Never raises on a question with no recognisable entity: falls back to plain vector
    search (`kg.search_chunks`) and says so in the trace.
  * No LLM calls: the only model calls are question embeddings (semantic rerank, the vector
    backfill/fallback, and the node-name search when no entity is linked), so cost stays
    comparable to the baseline.

How it works
------------
1. Link (graph_linking.py). The question's KIP / Jira / release ids and capitalised names
   become *seeds*, each with a trust weight (id/exact 1.0, fuzzy 0.8, node-name search 0.3).

2. Walk. From every seed we follow edges up to 2 hops (one batched neighbours query per hop;
   same rows as `kg.neighbors`). Every edge carries `source_doc_ids`, the chunks that justify
   it, and each of those docs gets a *support* from the seed it was reached from:

       path weight = seed weight x EDGE_WEIGHT[type] x confidence x DECAY^(hop-1)
                     x CUE_BOOST (asked-for type) x BRIDGE_BOOST x OFF_FOCUS (hop 1 only)
       doc support = path weight x doc_share(#docs on the edge)

   multiplied along the path at hop 2.
   * EDGE_WEIGHT favours facts (PROPOSED, VOTED, RESOLVED, ...) over plumbing (MENTIONS,
     SENT, REPLIED_TO); `confidence` is the extractor's.
   * CUE_BOOST: edge types the question asks about ("who *voted*" -> VOTED).
   * BRIDGE_BOOST: hop-1 edges between two seeds (Jun Rao -REVIEWED-> KIP-966) or into an
     entity several seeds reach (the person who reviewed *both* KIPs).
   * OFF_FOCUS: hop-1 edges from a seed into *another* node of a type the question names
     (Jun Rao's votes on other KIPs, when the question is about KIP-966).
   * doc_share: the lone vote email behind a VOTED edge is stronger evidence than any one
     of 16 thread emails that all restate "Andrew proposed KIP-1043".
   Hop 2 is deliberately narrow: only fact edges (a person's other emails say little about
   the seed), at most MAX_EDGES_PER_NODE per node (keeps hubs like release 4.0.0 in check),
   each hop-1 entity expanded once for the seed that reaches it best (else every doc behind
   a shared neighbour looks like a bridge), and never back onto the seed's own type (KIP ->
   voter -> *another* KIP is about that sibling). Email nodes are leaves: their chunks
   already come with the edge that reached them.

3. Score. A doc's graph score is the sum over seeds of its best support from each seed, so
   a chunk that connects several seeds (the multi-hop "bridge") beats one that touches a
   single seed. Text anchoring repairs missing edges: a candidate whose text names a strong
   seed ("... (KIP-1032) ...") counts as a MENTIONS of it. Then a hybrid rerank adds
   SEM_WEIGHT x the min-max-normalised embedding similarity between the question and each
   candidate (vector search restricted to the candidates).

4. Diversify and pack. With several strong seeds we round-robin: each seed gets its best
   DIVERSIFY_ROUNDS docs before the rest follow by score, so "compare KIP-A and KIP-B" gets
   evidence for both. Budget left after the graph candidates is backfilled with plain
   vector hits (BACKFILL), then `pack_to_budget` cuts to the budget.

5. Trace. Linked seeds (and how), dropped mentions, cues, bridge entities, walk stats
   (expansions, skipped siblings, truncations), phase timings, and for each returned doc
   its score parts, seeds and best paths.
"""

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass

from kgrag.graph.kg import KnowledgeGraph
from kgrag.graph.types import Edge, Node
from kgrag.observability import traced
from kgrag.retrieve.graph_linking import Seed, link_entities
from kgrag.retrieve.packing import pack_to_budget
from kgrag.retrieve.types import Evidence, RetrievalResult
from kgrag.retrieve.vector import K as VECTOR_K
from kgrag.tokens import count_tokens

# How much an edge of each type says about its endpoints. Facts extracted from the thread
# (who proposed/voted/reviewed, what a ticket fixed) beat structural plumbing.
EDGE_WEIGHT = {
    "PROPOSED": 1.0, "VOTED": 1.0, "REVIEWED": 0.9, "RESOLVED": 0.9, "FILED": 0.9,
    "FIX_VERSION": 0.9, "ASSIGNED": 0.8, "BLOCKS": 0.9, "AFFECTS": 0.8, "DEPENDS_ON": 0.9,
    "SUPERSEDES": 0.9, "MENTIONS": 0.6, "SENT": 0.5, "REPLIED_TO": 0.3,
}
FACT_TYPES = sorted(t for t in EDGE_WEIGHT if t not in ("MENTIONS", "SENT", "REPLIED_TO"))
DECAY = 0.5               # penalty per hop beyond the first
DOC_SHARE_EXP = 0.25      # edge with n docs gives each n^-0.25 of its weight (see doc_share)
CUE_BOOST = 1.5           # edge type the question asks about
BRIDGE_BOOST = 1.5        # hop-1 edge between two seeds, or into an entity several seeds reach
OFF_FOCUS = 0.5           # hop-1 edge into another node of a type the question names
MAX_EDGES_PER_NODE = 200  # per hop-2 expansion, strongest edge types first
MAX_EXPAND_PER_SEED = 25  # hop-1 entities expanded to hop 2, best-reached first
SEMANTIC_POOL = 2000      # best graph candidates that get an embedding similarity
MAX_CANDIDATES = 300      # best combined candidates whose text is fetched and ranked
SEM_WEIGHT = 0.5          # normalised question-chunk similarity (tie-breaker scale)
DIVERSIFY_ROUNDS = 2      # docs guaranteed per strong seed before ranking by score
BACKFILL = True           # fill leftover budget with plain vector hits
TRACE_PATHS = 3           # paths kept per returned doc in the trace

# Question words that signal an edge type ("who voted" -> VOTED). Prefix match on words.
CUES = {
    "PROPOSED": ("propos", "author", "start", "introduc", "drove", "led "),
    "VOTED": ("vote", "voting", "binding", "+1", "approv", "accept"),
    "REVIEWED": ("review", "feedback", "comment", "raise", "suggest", "concern", "object"),
    "FILED": ("creat", "filed", "file ", "open", "report"),
    "ASSIGNED": ("assign", "work on", "working on"),
    "RESOLVED": ("resolv", "resolution", "fixed", "closed"),
    "FIX_VERSION": ("fix version", "ship", "landed", "released in", "fixed in", "which release"),
    "AFFECTS": ("affect", "impact", "target", "component"),
    "BLOCKS": ("block",),
    "DEPENDS_ON": ("depend", "prerequisite", "require", "build on", "builds on"),
    "SUPERSEDES": ("supersed", "replac", "obsolete"),
}


@dataclass(frozen=True)
class Support:
    score: float
    path: str


@dataclass(frozen=True)
class Ranked:
    doc_id: str
    score: float
    graph: float
    semantic: float
    seeds: tuple[str, ...]
    via: tuple[str, ...] = ()


# --- pure scoring helpers -----------------------------------------------------------------

def relation_cues(question: str) -> frozenset[str]:
    """Edge types the question asks about, from CUES (word-prefix match, case-insensitive)."""
    q = " " + re.sub(r"\s+", " ", question.lower()) + " "
    return frozenset(t for t, words in CUES.items()
                     if any(re.search(r"(?<![a-z])" + re.escape(w), q) for w in words))


def edge_score(edge_type: str, confidence: float | None, hop: int, incoming: float,
               cues: frozenset[str], bridge: bool = False) -> float:
    """Path weight after crossing one edge; `incoming` is the seed weight or the hop-1 path weight."""
    score = incoming * EDGE_WEIGHT.get(edge_type, 0.3) * (1.0 if confidence is None else confidence)
    score *= DECAY ** (hop - 1)
    if edge_type in cues:
        score *= CUE_BOOST
    if bridge:
        score *= BRIDGE_BOOST
    return score


def doc_share(n_docs: int) -> float:
    """Specificity: one of an edge's n justifying docs gets n^-DOC_SHARE_EXP of its weight.
    The single vote email for a VOTED edge is the evidence; any one of 16 thread emails that
    all restate "Andrew proposed KIP-1043" is interchangeable."""
    return max(n_docs, 1) ** -DOC_SHARE_EXP


def add_support(supports: dict[str, dict[str, Support]], doc_id: str, seed_id: str,
                score: float, path: str) -> None:
    """Record that `seed_id` reaches `doc_id` with `score`; keep only the best path per seed."""
    cur = supports.setdefault(doc_id, {}).get(seed_id)
    if cur is None or score > cur.score:
        supports[doc_id][seed_id] = Support(score, path)


def off_focus(node_label: str, node_id: str, source: Seed, seeds: list[Seed]) -> bool:
    """True if `source` steps to a node of a *different* type that the question names, but not
    to the named node: for "Jun Rao on KIP-966", Jun Rao's votes on *other* KIPs are off focus.
    Same-type steps (KIP-1071 -DEPENDS_ON-> KIP-848) describe the seed itself and stay on focus."""
    return node_label != source.label and all(s.id != node_id for s in seeds) and any(
        s.strong and s.label == node_label for s in seeds)


def minmax(values: dict[str, float]) -> dict[str, float]:
    """Scale to [0, 1]; all-equal values map to 1.0 (a uniform signal, not no signal)."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def rank_docs(supports: dict[str, dict[str, Support]], semantic: dict[str, float],
              sem_weight: float | None = None) -> list[Ranked]:
    """Score = sum of the best support per seed + sem_weight x semantic; ties broken by doc_id."""
    sem_weight = SEM_WEIGHT if sem_weight is None else sem_weight
    ranked = []
    for doc_id, by_seed in supports.items():
        seeds = tuple(sorted(by_seed))
        graph = round(sum(by_seed[s].score for s in seeds), 6)
        sem = round(semantic.get(doc_id, 0.0), 6)
        best = sorted(by_seed.values(), key=lambda s: (-s.score, s.path))
        ranked.append(Ranked(doc_id, round(graph + sem_weight * sem, 6), graph, sem, seeds,
                             tuple(s.path for s in best[:TRACE_PATHS])))
    return sorted(ranked, key=lambda r: (-r.score, r.doc_id))


def diversify(ranked: list[Ranked], seed_ids: list[str],
              rounds: int | None = None) -> list[Ranked]:
    """Round-robin: each seed gets `rounds` of its best docs (a bridging doc counts for every
    seed it supports) before the remaining docs follow in score order."""
    rounds = DIVERSIFY_ROUNDS if rounds is None else rounds
    if len(seed_ids) < 2:
        return list(ranked)
    picked: list[Ranked] = []
    taken: set[str] = set()
    count = dict.fromkeys(seed_ids, 0)
    for r in range(1, rounds + 1):
        for seed in seed_ids:
            if count[seed] >= r:
                continue
            nxt = next((d for d in ranked if d.doc_id not in taken and seed in d.seeds), None)
            if nxt is None:
                continue
            picked.append(nxt)
            taken.add(nxt.doc_id)
            for s in nxt.seeds:
                if s in count:
                    count[s] += 1
    return picked + [d for d in ranked if d.doc_id not in taken]


# --- graph walk ---------------------------------------------------------------------------

def _label(node_id: str, names: dict[str, str]) -> str:
    return node_id if node_id.startswith("email:") else (names.get(node_id) or node_id)


def _step(edge: Edge, here: str, names: dict[str, str]) -> str:
    """Readable hop, e.g. `Andrew Schofield -PROPOSED-> KIP-1043` or `KIP-1043 <-VOTED- Lucas`."""
    if edge.source_id == here:
        return f"{_label(here, names)} -{edge.type}-> {_label(edge.target_id, names)}"
    return f"{_label(here, names)} <-{edge.type}- {_label(edge.source_id, names)}"


@dataclass
class Walk:
    supports: dict[str, dict[str, Support]]
    bridges: list[str]
    stats: dict


_NEIGHBORS = (
    "MATCH (a)-[r]-(b) WHERE a.id IN $ids AND ($types IS NULL OR type(r) IN $types) "
    "RETURN a.id AS here, type(r) AS type, startNode(r).id AS src, endNode(r).id AS dst, "
    "r.source_doc_ids AS docs, r.confidence AS confidence, "
    "b.id AS id, labels(b)[0] AS label, b.name AS name "
    "ORDER BY here, b.id, type(r), elementId(r)")


def neighbors_many(kg: KnowledgeGraph, ids: list[str],
                   edge_types: list[str] | None = None) -> dict[str, list[tuple[Edge, Node]]]:
    """`kg.neighbors` for many nodes in one round trip (only the props the walk uses).
    A 2-hop walk touches ~50 nodes; one query each made the walk ~5x slower on the real graph."""
    out: dict[str, list[tuple[Edge, Node]]] = {i: [] for i in ids}
    for r in kg.cypher(_NEIGHBORS, {"ids": ids, "types": edge_types}):
        edge = Edge(r["type"], r["src"], r["dst"],
                    {"source_doc_ids": r["docs"] or [], "confidence": r["confidence"]})
        out[r["here"]].append((edge, Node(r["id"], r["label"], r["name"] or "")))
    return out


def _strongest(pairs: list[tuple[Edge, Node]], limit: int) -> list[tuple[Edge, Node]]:
    """Keep the `limit` pairs with the heaviest edge types (stable: neighbors() is id-ordered)."""
    if len(pairs) <= limit:
        return pairs
    return sorted(pairs, key=lambda p: -EDGE_WEIGHT.get(p[0].type, 0.3))[:limit]


def _bridges(seeds: list[Seed], hop1: dict[str, list[tuple[Edge, Node]]]) -> set[str]:
    """Entities (not emails, not seeds) that two or more seeds reach in one hop."""
    seed_ids = {s.id for s in seeds}
    reached_by: dict[str, set[str]] = {}
    for s in seeds:
        for _, n in hop1[s.id]:
            if n.label != "Email" and n.id not in seed_ids:
                reached_by.setdefault(n.id, set()).add(s.id)
    return {n for n, by in reached_by.items() if len(by) >= 2}


def _frontier(per_seed: dict[str, list[tuple[float, str, str]]]) -> dict[str, tuple[float, str, str]]:
    """Hop-2 starting points: each seed's MAX_EXPAND_PER_SEED best-reached entities, merged so
    an entity is expanded once, credited to the seed that reaches it best -> {node: (w, seed, path)}.
    (Crediting every seed would make each doc behind a bridge look like it connects them all.)"""
    best: dict[str, tuple[float, str, str]] = {}
    for seed_id in sorted(per_seed):
        ranked = sorted(per_seed[seed_id], key=lambda x: (-x[0], x[1]))
        for w, node_id, path in ranked[:MAX_EXPAND_PER_SEED]:
            if node_id not in best or w > best[node_id][0]:
                best[node_id] = (w, seed_id, path)
    return dict(sorted(best.items()))


def walk(kg: KnowledgeGraph, seeds: list[Seed], cues: frozenset[str]) -> Walk:
    """Collect doc supports from every seed: hop 1 for all seeds, hop 2 for strong seeds."""
    names = {s.id: s.name for s in seeds}
    seed_ids = {s.id for s in seeds}
    label_of = {s.id: s.label for s in seeds}
    hop1 = neighbors_many(kg, [s.id for s in seeds])
    for pairs in hop1.values():
        names.update((n.id, n.name) for _, n in pairs)
    bridges = _bridges(seeds, hop1)

    supports: dict[str, dict[str, Support]] = {}
    reached: dict[str, list[tuple[float, str, str]]] = {}  # seed -> [(path weight, node, path)]
    for s in seeds:
        for e, n in hop1[s.id]:
            bridge = n.id in bridges or n.id in seed_ids
            w = edge_score(e.type, e.props.get("confidence"), 1, s.weight, cues, bridge)
            if off_focus(n.label, n.id, s, seeds):
                w *= OFF_FOCUS
            path = _step(e, s.id, names)
            docs = kg.source_docs(e)
            for d in docs:
                add_support(supports, d, s.id, w * doc_share(len(docs)), path)
            if s.strong and n.label != "Email" and n.id not in seed_ids:
                reached.setdefault(s.id, []).append((w, n.id, path))

    frontier = _frontier(reached)
    hop2 = neighbors_many(kg, list(frontier), FACT_TYPES)
    truncated, siblings = [], 0
    for node_id, (w1, seed_id, path1) in frontier.items():
        pairs = hop2[node_id]
        if len(pairs) > MAX_EDGES_PER_NODE:
            truncated.append(node_id)
        for e, n in _strongest(pairs, MAX_EDGES_PER_NODE):
            if n.id in seed_ids:
                continue  # seed-to-entity edges were scored at hop 1 (with BRIDGE_BOOST)
            if n.label == label_of[seed_id]:
                siblings += 1
                continue  # KIP -> voter -> *another* KIP: about the sibling, not the seed
            names.setdefault(n.id, n.name)
            w = edge_score(e.type, e.props.get("confidence"), 2, w1, cues)
            path = f"{path1} | {_step(e, node_id, names)}"
            docs = kg.source_docs(e)
            for d in docs:
                add_support(supports, d, seed_id, w * doc_share(len(docs)), path)

    stats = {"hop1_edges": sum(len(p) for p in hop1.values()), "expanded": len(frontier),
             "siblings_skipped": siblings, "truncated": truncated}
    return Walk(supports, sorted(_label(b, names) for b in bridges), stats)


# --- rerank inputs ------------------------------------------------------------------------

def semantic_scores(kg: KnowledgeGraph, question: str, doc_ids: list[str]) -> dict[str, float]:
    """Question-chunk cosine similarity for `doc_ids` only (vector search filtered to them)."""
    if not doc_ids:
        return {}
    hits = kg.chunk_store.similarity_search_with_score(
        question, k=len(doc_ids), filter={"doc_id": {"$in": doc_ids}})
    return {d.metadata["doc_id"]: 1.0 - dist for d, dist in hits}


def _anchor_pattern(seed: Seed) -> re.Pattern | None:
    """Regex for a seed's name in raw text; None when the name is too generic to trust."""
    name = seed.name or ""
    if seed.how != "id" and len(name.split()) < 2:
        return None  # "Andrew" or "Connect" alone would anchor half the corpus
    return re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-]|\.\d)", re.IGNORECASE)


def anchor(supports: dict[str, dict[str, Support]], texts: dict[str, str],
           seeds: Iterable[Seed]) -> int:
    """A candidate whose text names a strong seed gets at least a MENTIONS-strength support from
    it (repairs missing edges, e.g. Jira events never link to the KIP in their title).
    Returns how many supports were added or raised."""
    added = 0
    for s in seeds:
        pat = _anchor_pattern(s) if s.strong else None
        if pat is None:
            continue
        w = edge_score("MENTIONS", 1.0, 1, s.weight, frozenset())
        for doc_id in sorted(texts):
            cur = supports[doc_id].get(s.id)
            if (cur is None or cur.score < w) and pat.search(texts[doc_id]):
                add_support(supports, doc_id, s.id, w, f"text names {s.name}")
                added += 1
    return added


# --- entry point --------------------------------------------------------------------------

def _seed_trace(s: Seed) -> dict:
    return {"id": s.id, "label": s.label, "name": s.name, "how": s.how,
            "surface": s.surface if s.how != "node_search" else None, "weight": s.weight}


class _Clock:
    """Phase timings for the trace (`ms`): retrieval latency is part of the comparison."""

    def __init__(self):
        self.t0 = self.last = time.perf_counter()
        self.ms: dict[str, float] = {}

    def lap(self, phase: str) -> None:
        now = time.perf_counter()
        self.ms[phase] = round((now - self.last) * 1000, 1)
        self.last = now

    def done(self) -> dict[str, float]:
        return {**self.ms, "total": round((time.perf_counter() - self.t0) * 1000, 1)}


def _vector_fallback(question: str, kg: KnowledgeGraph, budget_tokens: int, trace: dict,
                     reason: str, clock: _Clock) -> RetrievalResult:
    evidence = pack_to_budget(kg.search_chunks(question, k=VECTOR_K), budget_tokens)
    clock.lap("vector")
    trace.update(strategy="vector_fallback", fallback_reason=reason,
                 selected=[{"doc_id": e.doc_id, "score": round(e.score, 6), "source": "vector"}
                           for e in evidence],
                 ms=clock.done())
    return RetrievalResult(evidence=evidence, trace=trace)


def _selected_trace(evidence: list[Evidence], ranked: list[Ranked]) -> list[dict]:
    info = {r.doc_id: r for r in ranked}
    return [{"doc_id": e.doc_id, "score": e.score, "source": "graph", "graph": info[e.doc_id].graph,
             "semantic": info[e.doc_id].semantic, "seeds": list(info[e.doc_id].seeds),
             "paths": list(info[e.doc_id].via)}
            if e.doc_id in info else
            {"doc_id": e.doc_id, "score": round(e.score, 6), "source": "vector_backfill"}
            for e in evidence]


@traced("retrieve.graph")
def retrieve(question: str, kg: KnowledgeGraph, budget_tokens: int) -> RetrievalResult:
    clock = _Clock()
    linking = link_entities(question, kg)
    clock.lap("link")
    trace: dict = {"budget_tokens": budget_tokens,
                   "linked": [_seed_trace(s) for s in linking.seeds], "dropped": linking.dropped}
    if not linking.seeds:
        return _vector_fallback(question, kg, budget_tokens, trace, "no entity linked", clock)

    cues = relation_cues(question)
    w = walk(kg, linking.seeds, cues)
    clock.lap("walk")
    trace.update(cues=sorted(cues), bridges=w.bridges, walk=w.stats)
    if not w.supports:
        return _vector_fallback(question, kg, budget_tokens, trace,
                                "linked entities have no evidence", clock)

    # Rerank: graph score -> top SEMANTIC_POOL get a similarity -> top MAX_CANDIDATES get text.
    pool = [r.doc_id for r in rank_docs(w.supports, {})[:SEMANTIC_POOL]]
    sem = minmax(semantic_scores(kg, question, pool))
    clock.lap("semantic")
    top = [r.doc_id for r in rank_docs({d: w.supports[d] for d in pool}, sem)[:MAX_CANDIDATES]]
    texts = {d.doc_id: d.text for d in kg.documents(top)}
    cand = {d: w.supports[d] for d in top if d in texts}
    anchored = anchor(cand, texts, linking.seeds)
    ranked = diversify(rank_docs(cand, sem), [s.id for s in linking.seeds if s.strong])
    clock.lap("rank")

    candidates = [Evidence(doc_id=r.doc_id, text=texts[r.doc_id], score=r.score) for r in ranked]
    backfill = []
    if BACKFILL and sum(count_tokens(e.text) for e in candidates) < budget_tokens:
        backfill = [e for e in kg.search_chunks(question, k=VECTOR_K) if e.doc_id not in cand]
    evidence = pack_to_budget(candidates + backfill, budget_tokens)
    clock.lap("pack")
    trace.update(strategy="graph", candidates=len(w.supports), reranked=len(cand),
                 anchored=anchored, backfilled=sum(e.doc_id not in cand for e in evidence),
                 selected=_selected_trace(evidence, ranked), ms=clock.done())
    return RetrievalResult(evidence=evidence, trace=trace)
