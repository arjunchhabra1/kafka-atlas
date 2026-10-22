"""Graph retrieval: find the evidence for a question by walking the knowledge graph.

The contract
------------
`retrieve(question, kg, budget_tokens, config=None) -> RetrievalResult`

  * `evidence`: `Evidence(doc_id, text, score)` rows, best first. Every `doc_id` is a
    `documents` row (`email:{email_id}:{chunk}` or `jira:{event_id}`), and `text` is that
    row's stored text. There are no graph summaries or edge listings: the answer model sees
    only chunks.
  * Evidence is packed with `pack_to_budget(candidates, budget_tokens)`, the same packing the
    vector baseline uses, so both systems get the same token budget.
  * Deterministic: the same question on the same graph gives the same evidence. Every sort
    has a doc_id, node id or path tie-break, and scores are rounded.
  * It never raises on a question with no recognisable entity. With `vector_fallback` on it
    falls back to plain vector search (`kg.search_chunks`); with it off (graph-only) it returns
    no evidence. Either way the trace says so (`fallback`, `unlinked`).
  * No LLM calls. The question is embedded at most once per call, and that one vector is
    reused for the rerank, the vector backfill or fallback, and the node-name fallback. So
    cost stays comparable to the baseline. With every embedding switch off
    (`GRAPH_ONLY_CONFIG`) it is never embedded.

All knobs live in `GraphRetrieverConfig` (graph_config.py). Booleans switch whole terms off
for ablations; graph_ablation.py runs those ablations.

Tuning provenance
-----------------
Tuned on: the dev fixture (tests/fixtures/dev_questions.json) plus the 5 hand-written
real-graph questions in tests/fixtures/tuning_log.json. That file also records which rule
each of those questions motivated. The rules and values must be re-validated by a dev-split
ablation before the headline run.

How it works
------------
1. Link (graph_linking.py). The question's KIP / Jira / release ids and its names become
   *seeds*, each with a trust weight. Ids and multi-word names are strong seeds; one-word
   names and the node-name search are weak.

2. Walk. From every seed we follow edges up to 2 hops, with one batched neighbours query per
   hop (`neighbors_many`). The query keeps only each node's strongest edges, and that cap is
   applied inside Cypher, so hubs with thousands of edges never cross the wire. Every edge
   carries `source_doc_ids`, the chunks that justify it. Each of those docs gets a
   *support* from the seed it was reached from:

       path weight = seed weight x weight(type) x confidence x decay^(hop-1)
                     x cue_factor (asked-for type) x bridge_factor x off_focus_factor (hop 1)
       doc support = path weight x n_docs^-doc_share_exp

   The weight is multiplied along the path at hop 2. The terms:
   * `weight(type)`: facts (PROPOSED, VOTED, RESOLVED, ...) count more than plumbing
     (MENTIONS, SENT, REPLIED_TO). `confidence` is the extractor's own.
   * cues: edge types the question asks about ("who *voted*" -> VOTED), from a
     word-bounded lexicon.
   * bridge: hop-1 edges between two seeds, or into an entity that several seeds reach
     (for example, the person who reviewed *both* KIPs).
   * off_focus: hop-1 edges from a seed into *another* node of a different type the
     question names. For "Jane Doe on KIP-9", Jane's votes on other KIPs are off focus.
   * doc_share: when an edge has one justifying doc, that doc is the evidence. When 16
     thread emails all restate the same fact, any one of them is interchangeable.

   Hop 2 is deliberately narrow:
   * It follows only fact edges, because a person's other emails say little about the seed.
   * Each hop-1 entity is expanded once, for the seed that reaches it best. Otherwise every
     doc behind a shared neighbour looks like a bridge.
   * It never lands back on the seed's own type: KIP -> voter -> *another* KIP is evidence
     about that sibling, not the seed.
   Email nodes are leaves: their chunks already come with the edge that reached them.

3. Score. A doc's graph score is the sum over seeds of its best support from each seed. So
   a chunk that connects several seeds (the multi-hop "bridge") beats one that touches a
   single seed. Text anchoring repairs missing edges: a candidate whose text names a strong
   seed counts as a MENTIONS of it. The hybrid rerank then adds sem_weight x the min-max
   normalised similarity between the question and each candidate, using vector search
   restricted to the candidates.

4. Diversify and pack. With several strong seeds we round-robin: each seed gets its best
   `diversify_rounds` docs before the rest follow by score, so "compare A and B" gets
   evidence for both. Budget left after the graph candidates is backfilled with plain
   vector hits. Then `pack_to_budget` cuts the list to the budget.

4b. Doc bridge and vector fusion (added after the first dev-split ablation, 2026-10-22).
   Inference questions name a KIP / ticket only through another discussion ("the KIP Jane
   said ..."), so linking finds just the person and the KIP is three hops away. The best
   first-pass docs and the question's best vector hits are pivots; the KIPs and tickets they
   mention become weak hop-1 seeds (`doc_bridge_*`). The best vector hits also join the pool
   as one pseudo seed (`vector_seeds`), so a doc no linked entity reaches still competes.
   Graph-only turns the vector parts off. Dev recall (44 multi-hop questions): 0.43 -> 0.65.

5. Trace. The trace records:
   * the linked seeds (and how each was linked), dropped mentions, cues and bridge entities
   * walk stats: rows per hop, capped nodes, skipped siblings
   * the config switches and phase timings
   * for each returned doc: its score parts, its seeds and its best paths
"""

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, fields

from kgrag.graph.kg import KnowledgeGraph
from kgrag.graph.types import Edge, Node
from kgrag.observability import traced
from kgrag.retrieve.graph_config import (
    DEFAULT_CONFIG,
    STRUCTURAL_TYPES,
    GraphRetrieverConfig,
)
from kgrag.retrieve.graph_linking import Seed, _valid, link_entities
from kgrag.retrieve.packing import pack_to_budget
from kgrag.retrieve.types import Evidence, RetrievalResult
from kgrag.retrieve.vector import K as VECTOR_K
from kgrag.tokens import count_tokens

# Question phrases that signal an edge type ("who voted" -> VOTED). Word-bounded regexes,
# so "open-source" is not "opened an issue" and "target" is not an AFFECTS cue.
_W = r"(?:\s+\w+){0,2}\s+"  # up to two words in between ("created the Jira ticket")
CUES: dict[str, tuple[str, ...]] = {
    "PROPOSED": (r"\bpropos(?:e|ed|es|al|als|ing|er)\b", r"\bauthor(?:s|ed)?\b",
                 r"\bstarted\b", r"\bkick(?:ed)? off\b"),
    "VOTED": (r"\bvot(?:e|es|ed|ing|er|ers)\b", r"\bbinding\b", r"(?<!\w)\+1\b",
              r"\bapprov(?:e|ed|al)\b", r"\baccepted\b"),
    "REVIEWED": (r"\breview(?:s|ed|er|ers|ing)?\b", r"\bfeedback\b", r"\bcomment(?:s|ed)?\b",
                 r"\brais(?:e|ed|es)\b", r"\bsuggest(?:s|ed|ion|ions)?\b", r"\bconcerns?\b",
                 r"\bobject(?:s|ed|ion|ions)\b"),
    "FILED": (r"\bfiled\b", r"\breporter\b",
              rf"\b(?:open(?:ed)?|creat(?:e|ed)|fil(?:e|ed)|report(?:ed)?|rais(?:e|ed)){_W}"
              r"(?:jira|ticket|issue|bug)s?\b"),
    "ASSIGNED": (r"\bassign(?:s|ed|ee|ment)?\b",),
    "RESOLVED": (r"\bresolv(?:e|ed|es|ing)\b", r"\bresolutions?\b", r"\bfixed\b", r"\bclosed\b"),
    "FIX_VERSION": (r"\bfix versions?\b", r"\bship(?:s|ped)?\b", r"\blanded\b",
                    r"\breleased in\b", r"\bfixed in\b", r"\bwhich (?:release|version)s?\b"),
    "AFFECTS": (r"\baffect(?:s|ed|ing)?\b", r"\bimpact(?:s|ed|ing)?\b", r"\bcomponents?\b"),
    "BLOCKS": (r"\bblock(?:s|ed|ing|er|ers)?\b",),
    "DEPENDS_ON": (r"\bdepend(?:s|ed|ing|ency|encies)?\b", r"\bprerequisites?\b",
                   r"\brequires?\b", r"\bbuilds? on\b"),
    "SUPERSEDES": (r"\bsupersed(?:e|es|ed|ing)\b", r"\breplac(?:e|es|ed|ing)\b",
                   r"\bobsolete\b"),
}
_CUE_RE = {t: re.compile("|".join(p), re.IGNORECASE) for t, p in CUES.items()}


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
    """Edge types the question asks about (word-bounded phrase match, case-insensitive)."""
    return frozenset(t for t, rx in _CUE_RE.items() if rx.search(question))


def fact_types(config: GraphRetrieverConfig = DEFAULT_CONFIG) -> list[str]:
    return sorted(t for t in config.edge_weight if t not in STRUCTURAL_TYPES)


def edge_score(edge_type: str, confidence: float | None, hop: int, incoming: float,
               cues: frozenset[str], bridge: bool = False,
               config: GraphRetrieverConfig = DEFAULT_CONFIG) -> float:
    """Path weight after crossing one edge; `incoming` is the seed weight or the hop-1 path weight."""
    score = incoming * config.weight(edge_type) * (1.0 if confidence is None else confidence)
    score *= config.decay ** (hop - 1)
    if config.cues and edge_type in cues:
        score *= config.cue_factor
    if config.bridge_boost and bridge:
        score *= config.bridge_factor
    return score


def doc_share(n_docs: int, config: GraphRetrieverConfig = DEFAULT_CONFIG) -> float:
    """Specificity: each of an edge's n justifying docs gets n^-doc_share_exp of its weight."""
    return max(n_docs, 1) ** -config.doc_share_exp if config.doc_share else 1.0


def add_support(supports: dict[str, dict[str, Support]], doc_id: str, seed_id: str,
                score: float, path: str) -> None:
    """Record that `seed_id` reaches `doc_id` with `score`; keep the best path per seed
    (equal scores: the lexicographically smaller path, so the trace is deterministic)."""
    cur = supports.setdefault(doc_id, {}).get(seed_id)
    if cur is None or score > cur.score or (score == cur.score and path < cur.path):
        supports[doc_id][seed_id] = Support(score, path)


def off_focus(node_label: str, node_id: str, source: Seed, seeds: list[Seed]) -> bool:
    """True if `source` steps to a node of a *different* type that the question names, but not
    to the named node ("Jane Doe on KIP-9": Jane's votes on other KIPs). Same-type steps
    (KIP-9 -DEPENDS_ON-> KIP-5) describe the seed itself and stay on focus."""
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
              config: GraphRetrieverConfig = DEFAULT_CONFIG) -> list[Ranked]:
    """Score = sum of the best support per seed + sem_weight x semantic; ties broken by doc_id."""
    ranked = []
    for doc_id, by_seed in supports.items():
        seeds = tuple(sorted(by_seed))
        graph = round(sum(by_seed[s].score for s in seeds), 6)
        sem = round(semantic.get(doc_id, 0.0), 6)
        best = sorted(by_seed.values(), key=lambda s: (-s.score, s.path))
        ranked.append(Ranked(doc_id, round(graph + config.sem_weight * sem, 6), graph, sem, seeds,
                             tuple(s.path for s in best[:config.trace_paths])))
    return sorted(ranked, key=lambda r: (-r.score, r.doc_id))


def diversify(ranked: list[Ranked], seed_ids: list[str], rounds: int) -> list[Ranked]:
    """Round-robin: each seed gets `rounds` of its best docs (a bridging doc counts for every
    seed it supports) before the remaining docs follow in score order."""
    if len(seed_ids) < 2 or rounds <= 0:
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


def _anchor_pattern(seed: Seed) -> re.Pattern | None:
    """Regex for a seed's name in raw text; None when the name is too generic to trust."""
    name = seed.name or ""
    if not seed.strong or (seed.how != "id" and len(name.split()) < 2):
        return None  # a first name or a bare component word would anchor half the corpus
    return re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-]|\.\d)", re.IGNORECASE)


def anchor(supports: dict[str, dict[str, Support]], texts: dict[str, str], seeds: Iterable[Seed],
           config: GraphRetrieverConfig = DEFAULT_CONFIG) -> tuple[dict[str, dict[str, Support]], int]:
    """A candidate whose text names a strong seed gets at least a MENTIONS-strength support
    from it. This repairs missing edges, for example Jira events never link to the KIP named
    in their title. Returns (a new supports mapping, number of supports added or raised).
    The input is not modified."""
    out = {d: dict(by_seed) for d, by_seed in supports.items()}
    added = 0
    for s in seeds:
        pat = _anchor_pattern(s)
        if pat is None:
            continue
        w = s.weight * config.weight("MENTIONS")
        for doc_id in sorted(out):
            cur, text = out[doc_id].get(s.id), texts.get(doc_id)
            if text is not None and (cur is None or cur.score < w) and pat.search(text):
                add_support(out, doc_id, s.id, w, f"text names {s.name}")
                added += 1
    return out, added


# --- graph walk ---------------------------------------------------------------------------

# The strongest `limit` edges of each node, chosen inside the database: per-node subquery
# ordered by (edge weight x cue) x confidence, ties by neighbour id. `total` is the node's
# full edge count (of the requested types), so the trace can say what the cap dropped.
_NEIGHBORS = """
MATCH (a) WHERE a.id IN $ids
CALL (a) {
  WITH a, COUNT { (a)-[x]-() WHERE $types IS NULL OR type(x) IN $types } AS total
  MATCH (a)-[r]-(b) WHERE $types IS NULL OR type(r) IN $types
  WITH total, r, b, coalesce($weights[type(r)], $unknown) * coalesce(r.confidence, 1.0) AS strength
  ORDER BY strength DESC, b.id, type(r), elementId(r)
  LIMIT $limit
  RETURN total, r, b
}
RETURN a.id AS here, total, type(r) AS type, startNode(r).id AS src, endNode(r).id AS dst,
       r.source_doc_ids AS docs, r.confidence AS confidence,
       b.id AS id, labels(b)[0] AS label, b.name AS name
ORDER BY here, id, type, elementId(r)
"""


def _strength_weights(cues: frozenset[str], config: GraphRetrieverConfig) -> dict[str, float]:
    return {t: w * (config.cue_factor if config.cues and t in cues else 1.0)
            for t, w in config.edge_weight.items()}


def neighbors_many(kg: KnowledgeGraph, ids: list[str], edge_types: list[str] | None = None,
                   limit: int = 1_000_000, cues: frozenset[str] = frozenset(),
                   config: GraphRetrieverConfig = DEFAULT_CONFIG,
                   ) -> tuple[dict[str, list[tuple[Edge, Node]]], dict[str, int]]:
    """`kg.neighbors` for many nodes in one round trip, keeping each node's `limit` strongest
    edges (cap applied in Cypher). Returns ({node: [(edge, neighbour)] in kg.neighbors
    order}, {node: total edges of the requested types})."""
    rows = kg.cypher(_NEIGHBORS, {"ids": ids, "types": edge_types, "limit": limit,
                                  "weights": _strength_weights(cues, config),
                                  "unknown": config.unknown_edge_weight},
                     timeout=config.query_timeout_s)
    pairs: dict[str, list[tuple[Edge, Node]]] = {i: [] for i in ids}
    totals: dict[str, int] = dict.fromkeys(ids, 0)
    for r in rows:
        edge = Edge(r["type"], r["src"], r["dst"],
                    {"source_doc_ids": r["docs"] or [], "confidence": r["confidence"]})
        pairs[r["here"]].append((edge, Node(r["id"], r["label"], r["name"] or "")))
        totals[r["here"]] = r["total"]
    return pairs, totals


def _label(node_id: str, names: dict[str, str]) -> str:
    return node_id if node_id.startswith("email:") else (names.get(node_id) or node_id)


def _step(edge: Edge, here: str, names: dict[str, str]) -> str:
    """Readable hop, e.g. `Jane Doe -PROPOSED-> KIP-9` or `KIP-9 <-VOTED- Jane Doe`."""
    if edge.source_id == here:
        return f"{_label(here, names)} -{edge.type}-> {_label(edge.target_id, names)}"
    return f"{_label(here, names)} <-{edge.type}- {_label(edge.source_id, names)}"


@dataclass
class Walk:
    supports: dict[str, dict[str, Support]]
    bridges: list[str]
    stats: dict


def _bridges(seeds: list[Seed], hop1: dict[str, list[tuple[Edge, Node]]]) -> set[str]:
    """Entities (not emails, not seeds) that two or more seeds reach in one hop."""
    seed_ids = {s.id for s in seeds}
    reached_by: dict[str, set[str]] = {}
    for s in seeds:
        for _, n in hop1[s.id]:
            if n.label != "Email" and n.id not in seed_ids:
                reached_by.setdefault(n.id, set()).add(s.id)
    return {n for n, by in reached_by.items() if len(by) >= 2}


def _frontier(per_seed: dict[str, list[tuple[float, str, str]]],
              max_per_seed: int) -> dict[str, tuple[float, str, str]]:
    """Hop-2 starting points: each seed's `max_per_seed` best-reached entities, merged so an
    entity is expanded once, credited to the seed that reaches it best -> {node: (w, seed, path)}.
    (Crediting every seed would make each doc behind a shared neighbour look like a bridge.)"""
    best: dict[str, tuple[float, str, str]] = {}
    for seed_id in sorted(per_seed):
        ranked = sorted(per_seed[seed_id], key=lambda x: (-x[0], x[1]))
        for w, node_id, path in ranked[:max_per_seed]:
            if node_id not in best or w > best[node_id][0]:
                best[node_id] = (w, seed_id, path)
    return dict(sorted(best.items()))


def _add_edge_docs(kg: KnowledgeGraph, supports: dict[str, dict[str, Support]], edge: Edge,
                   seed_id: str, w: float, path: str, config: GraphRetrieverConfig) -> None:
    docs = kg.source_docs(edge)
    for d in docs:
        add_support(supports, d, seed_id, w * doc_share(len(docs), config), path)


def walk(kg: KnowledgeGraph, seeds: list[Seed], cues: frozenset[str],
         config: GraphRetrieverConfig = DEFAULT_CONFIG) -> Walk:
    """Collect doc supports from every seed: hop 1 for all seeds, hop 2 for strong seeds."""
    names = {s.id: s.name for s in seeds}
    seed_ids = {s.id for s in seeds}
    label_of = {s.id: s.label for s in seeds}
    hop1, total1 = neighbors_many(kg, [s.id for s in seeds], None, config.hop1_max_edges, cues, config)
    for pairs in hop1.values():
        names.update((n.id, n.name) for _, n in pairs)
    bridges = _bridges(seeds, hop1)

    supports: dict[str, dict[str, Support]] = {}
    reached: dict[str, list[tuple[float, str, str]]] = {}  # seed -> [(path weight, node, path)]
    for s in seeds:
        for e, n in hop1[s.id]:
            bridge = n.id in bridges or n.id in seed_ids
            w = edge_score(e.type, e.props.get("confidence"), 1, s.weight, cues, bridge, config)
            if config.off_focus and off_focus(n.label, n.id, s, seeds):
                w *= config.off_focus_factor
            path = _step(e, s.id, names)
            _add_edge_docs(kg, supports, e, s.id, w, path, config)
            if s.strong and n.label != "Email" and n.id not in seed_ids:
                reached.setdefault(s.id, []).append((w, n.id, path))

    frontier = _frontier(reached, config.max_expand_per_seed)
    hop2, total2 = neighbors_many(kg, list(frontier), fact_types(config), config.hop2_max_edges,
                                  cues, config)
    siblings = 0
    for node_id, (w1, seed_id, path1) in frontier.items():
        for e, n in hop2[node_id]:
            if n.id in seed_ids:
                continue  # seed-to-entity edges were scored at hop 1 (with the bridge boost)
            if n.label == label_of[seed_id]:
                siblings += 1
                continue  # KIP -> voter -> *another* KIP: about the sibling, not the seed
            names.setdefault(n.id, n.name)
            w = edge_score(e.type, e.props.get("confidence"), 2, w1, cues, False, config)
            _add_edge_docs(kg, supports, e, seed_id, w, f"{path1} | {_step(e, node_id, names)}", config)

    stats = {
        "hop1_rows": sum(len(p) for p in hop1.values()),
        "hop1_capped": {n: t for n, t in total1.items() if t > config.hop1_max_edges},
        "expanded": len(frontier),
        "hop2_rows": sum(len(p) for p in hop2.values()),
        "hop2_capped": {n: t for n, t in total2.items() if t > config.hop2_max_edges},
        "siblings_skipped": siblings,
    }
    return Walk(supports, sorted(_label(b, names) for b in bridges), stats)


VECTOR_SEED = "vector:question"  # pseudo seed of the question's best vector hits

# --- doc bridge -----------------------------------------------------------------------------

# KIPs and tickets an edge justified by one of the given docs points at (an email's
# MENTIONS, a Jira event's FILED / RESOLVED ...).
_DOC_ENTITIES = (
    "MATCH ()-[r]->(t) WHERE (t:KIP OR t:JiraTicket) "
    "AND any(d IN r.source_doc_ids WHERE d IN $docs) "
    "RETURN DISTINCT t.id AS id, labels(t)[0] AS label, coalesce(t.name, t.id) AS name, "
    "[d IN r.source_doc_ids WHERE d IN $docs] AS via ORDER BY id"
)


def doc_bridge_seeds(kg: KnowledgeGraph, pivots: list[str], known: set[str],
                     config: GraphRetrieverConfig = DEFAULT_CONFIG) -> list[Seed]:
    """The KIPs / tickets the pivot docs are about or mention, as weak hop-1 seeds, best pivot
    first (at most doc_bridge_max_seeds; seeds already linked from the question are skipped)."""
    if not pivots:
        return []
    rank = {d: i for i, d in enumerate(pivots)}
    best: dict[str, tuple[int, dict]] = {}
    for r in kg.cypher(_DOC_ENTITIES, {"docs": pivots}, timeout=config.query_timeout_s):
        if r["id"] in known or not _valid(r["id"]):
            continue
        i = min(rank[d] for d in r["via"])
        if r["id"] not in best or i < best[r["id"]][0]:
            best[r["id"]] = (i, r)
    order = sorted(best.values(), key=lambda x: (x[0], x[1]["id"]))[:config.doc_bridge_max_seeds]
    return [Seed(r["id"], r["label"], r["name"], "doc_bridge", pivots[i], config.doc_bridge_weight)
            for i, r in order]


def _merge(into: dict[str, dict[str, Support]], more: dict[str, dict[str, Support]]) -> None:
    for doc_id, by_seed in more.items():
        into.setdefault(doc_id, {}).update(by_seed)


# --- entry point --------------------------------------------------------------------------

class _QuestionVector:
    """The question embedding, computed at most once per retrieve() and only if needed."""

    def __init__(self, kg: KnowledgeGraph, question: str):
        self.kg, self.question, self._vec = kg, question, None

    def get(self) -> list[float]:
        if self._vec is None:
            self._vec = self.kg.embed_query(self.question)
        return self._vec

    @property
    def calls(self) -> int:
        return 0 if self._vec is None else 1


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


def _seed_trace(s: Seed) -> dict:
    return {"id": s.id, "label": s.label, "name": s.name, "how": s.how,
            "surface": s.surface if s.how != "node_search" else None, "weight": s.weight}


def _switches(config: GraphRetrieverConfig) -> dict[str, bool]:
    return {f.name: getattr(config, f.name) for f in fields(config)
            if isinstance(getattr(config, f.name), bool)}


def _no_graph_evidence(kg: KnowledgeGraph, budget_tokens: int, trace: dict, reason: str,
                       qvec: _QuestionVector, clock: _Clock,
                       config: GraphRetrieverConfig) -> RetrievalResult:
    """Nothing linked, or the seeds reach no evidence: plain vector search when
    `vector_fallback` is on, otherwise an empty result (graph-only)."""
    if not config.vector_fallback:
        trace.update(strategy="empty", fallback_reason=reason, embeddings=qvec.calls,
                     selected=[], ms=clock.done())
        return RetrievalResult(evidence=[], trace=trace)
    hits = kg.search_chunks(qvec.question, k=VECTOR_K, embedding=qvec.get())
    evidence = pack_to_budget(hits, budget_tokens)
    clock.lap("vector")
    trace.update(strategy="vector_fallback", fallback=True, fallback_reason=reason,
                 embeddings=qvec.calls,
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


def _rank(kg: KnowledgeGraph, question: str, supports: dict[str, dict[str, Support]],
          seeds: list[Seed], qvec: _QuestionVector, config: GraphRetrieverConfig,
          ) -> tuple[list[Ranked], dict[str, str], dict[str, dict[str, Support]], int]:
    """Graph score -> top semantic_pool get a similarity (and their text comes back with it)
    -> top max_candidates are anchored, diversified and ranked. Returns (ranked, texts,
    anchored candidates, number anchored)."""
    pool = [r.doc_id for r in rank_docs(supports, {}, config)[:config.semantic_pool]]
    texts: dict[str, str] = {}
    sem: dict[str, float] = {}
    if config.semantic_rerank:
        hits = kg.search_chunks(question, k=len(pool), restrict_to=pool, embedding=qvec.get())
        texts = {e.doc_id: e.text for e in hits}
        sem = minmax({e.doc_id: e.score for e in hits})
    top = [r.doc_id for r in rank_docs({d: supports[d] for d in pool}, sem, config)
           [:config.max_candidates]]
    missing = [d for d in top if d not in texts]
    texts.update((d.doc_id, d.text) for d in (kg.documents(missing) if missing else []))
    cand, anchored = anchor({d: supports[d] for d in top if d in texts}, texts, seeds, config)
    rounds = config.diversify_rounds if config.diversify else 0
    strong = [s.id for s in seeds
              if s.strong or (config.doc_bridge_diversify and s.how == "doc_bridge")]
    return diversify(rank_docs(cand, sem, config), strong, rounds), texts, cand, anchored


@traced("retrieve.graph")
def retrieve(question: str, kg: KnowledgeGraph, budget_tokens: int,
             config: GraphRetrieverConfig | None = None) -> RetrievalResult:
    config = config or DEFAULT_CONFIG
    clock, qvec = _Clock(), _QuestionVector(kg, question)
    linking = link_entities(question, kg, config,
                            embed=qvec.get if config.node_search else None)
    clock.lap("link")
    trace: dict = {"budget_tokens": budget_tokens, "config": _switches(config),
                   "unlinked": not linking.seeds, "fallback": False,
                   "linked": [_seed_trace(s) for s in linking.seeds], "dropped": linking.dropped}
    if not linking.seeds and not (config.semantic_rerank and (config.doc_bridge
                                                             or config.vector_seeds)):
        return _no_graph_evidence(kg, budget_tokens, trace, "no entity linked", qvec, clock,
                                  config)

    cues = relation_cues(question)
    supports: dict[str, dict[str, Support]] = {}
    seeds = list(linking.seeds)
    if seeds:
        w = walk(kg, seeds, cues, config)
        supports = w.supports
        trace.update(cues=sorted(cues), bridges=w.bridges, walk=w.stats)
    clock.lap("walk")

    top_hits: list[str] = []
    if config.semantic_rerank:
        k = max(config.doc_bridge_pivots if config.doc_bridge else 0, config.vector_seeds)
        if k:
            top_hits = [e.doc_id for e in kg.search_chunks(question, k=k, embedding=qvec.get())]

    if config.doc_bridge:
        # Multi-hop through a doc: the question describes a KIP / ticket only through another
        # discussion ("the KIP Jane said ..."). Pivot on the best first-pass docs (and, with
        # embeddings, the best vector hits), then walk from what they mention.
        pivots = [r.doc_id for r in _rank(kg, question, supports, seeds, qvec, config)[0]
                  ][:config.doc_bridge_pivots] if supports else []
        pivots += [d for d in top_hits[:config.doc_bridge_pivots] if d not in pivots]
        bridged = doc_bridge_seeds(kg, pivots, {s.id for s in seeds}, config)
        if bridged:
            _merge(supports, walk(kg, bridged, cues, config).supports)
            seeds += bridged
        trace.update(doc_bridge=[_seed_trace(s) | {"pivot": s.surface} for s in bridged])
        clock.lap("doc_bridge")

    if config.vector_seeds and top_hits:
        n = min(config.vector_seeds, len(top_hits))
        for i, doc_id in enumerate(top_hits[:n]):
            add_support(supports, doc_id, VECTOR_SEED,
                        round(config.vector_seed_weight * (1 - i / n), 6), f"vector hit #{i + 1}")

    if not supports:
        reason = "linked entities have no evidence" if seeds else "no entity linked"
        return _no_graph_evidence(kg, budget_tokens, trace, reason, qvec, clock, config)

    ranked, texts, cand, anchored = _rank(kg, question, supports, seeds, qvec, config)
    clock.lap("rank")

    candidates = [Evidence(doc_id=r.doc_id, text=texts[r.doc_id], score=r.score) for r in ranked]
    backfill: list[Evidence] = []
    if config.vector_backfill and sum(count_tokens(e.text) for e in candidates) < budget_tokens:
        backfill = [e for e in kg.search_chunks(question, k=VECTOR_K, embedding=qvec.get())
                    if e.doc_id not in cand]
    evidence = pack_to_budget(candidates + backfill, budget_tokens)
    clock.lap("pack")
    trace.update(strategy="graph", candidates=len(supports), reranked=len(cand),
                 anchored=anchored, backfilled=sum(e.doc_id not in cand for e in evidence),
                 embeddings=qvec.calls, selected=_selected_trace(evidence, ranked),
                 ms=clock.done())
    return RetrievalResult(evidence=evidence, trace=trace)
