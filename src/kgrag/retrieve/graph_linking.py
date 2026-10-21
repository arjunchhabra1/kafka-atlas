"""Entity linking for the graph retriever: which graph nodes does a question talk about?

Tiers, strongest first. Each seed records `how` it was linked, for the trace, and its weight
comes from GraphRetrieverConfig:

  1. `id`           Canonical KIP / Jira / release ids, found by the same regexes that built
                    the graph (`extract_ids`), so "KIP 123" hits node `KIP:KIP-123`. A release
                    `x.y` also links its `x.y.0` twin, because both forms exist as nodes.
  2. `exact_name`   Multi-word capitalised runs ("Jane Doe", "DescribeGroups API") whose text
     `fuzzy_name`   equals a Person/Component name (case-insensitive). Multi-word runs that
                    don't match exactly get a strict fuzzy lookup ("Jane Do").
  3. `single_name`  One-word names ("Jane"). These are often ambiguous (many people share a
                    first name), so they get a *weak* seed.
  4. `node_search`  Embedding search over node names with the whole question. This is a
                    fallback for lowercase or paraphrased names, used only when nothing else
                    links, because it is noisy next to precise seeds.

Tiers 1-2 are *strong*: they walk 2 hops, get diversified, anchor text and set the focus.
Tiers 3-4 are *weak*: they walk 1 hop only. When several nodes share a name, they are ranked
by degree (most connected first), then by id, and the first `max_per_span` are kept.

The pure text helpers (`candidate_ngrams`, `select_spans`, `name_tier`, `rank_same_name`,
`id_seeds`, `release_aliases`) are kept apart from the graph lookups, so they can be
unit-tested without a database.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rapidfuzz import fuzz, utils

from kgrag.extract.regex_ids import extract_ids
from kgrag.graph.kg import KnowledgeGraph
from kgrag.retrieve.graph_config import DEFAULT_CONFIG, GraphRetrieverConfig

# Words that start a capitalised run without being part of a name: question words,
# sentence-initial imperatives, schema words and calendar words. They split runs, so
# "After Jane Doe" becomes "Jane Doe" and "Compare KIP-1 and KIP-2" yields no name.
STOPWORDS = frozenset("""
    a an the and or but if is are was were be been am do does did has have had can could will
    would should may might must shall i we you he she it they this that these those there
    who whom whose what which when where why how after before since during between both
    each either neither in on at of for from to by with about into over under than then
    name names list compare describe explain summarize summarise show give tell find
    identify outline please what's who's
    kip kips jira ticket tickets kafka apache release releases version versions vote votes
    thread discussion proposal list dev mail email emails
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
""".split())
NAME_LABELS = ("Person", "Component")
STRONG_TIERS = frozenset({"id", "exact_name", "fuzzy_name"})

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_INITIAL_GAP = re.compile(r"\.\s+")


@dataclass(frozen=True)
class NameSpan:
    text: str
    start: int
    end: int
    ntokens: int


@dataclass(frozen=True)
class Seed:
    id: str
    label: str
    name: str
    how: str        # id | exact_name | fuzzy_name | single_name | node_search
    surface: str    # the question text it was linked from
    weight: float

    @property
    def strong(self) -> bool:
        return self.how in STRONG_TIERS


@dataclass
class Linking:
    seeds: list[Seed] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # mentions we could not link, and why


# --- pure text helpers --------------------------------------------------------------------

def _name_like(token: str) -> bool:
    return token[0].isupper() or any(c.isupper() for c in token[1:])


def _runs(question: str) -> list[list[re.Match]]:
    """Maximal runs of capitalised tokens, split at stopwords, punctuation and id mentions."""
    blocked = [m.span for m in extract_ids(question)]
    runs: list[list[re.Match]] = []
    cur: list[re.Match] = []
    for tok in _TOKEN.finditer(question):
        text = tok.group(0)
        in_id = any(s < tok.end() and tok.start() < e for s, e in blocked)
        continues = (_name_like(text) or (text.isdigit() and bool(cur))) and not in_id \
            and text.lower() not in STOPWORDS
        if continues and cur:
            gap = question[cur[-1].end():tok.start()]
            prev = cur[-1].group(0)
            initial = len(prev) == 1 and prev.isupper() and _INITIAL_GAP.fullmatch(gap)
            if not (gap.isspace() or initial):
                runs.append(cur)
                cur = []
        if continues:
            cur.append(tok)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def candidate_ngrams(question: str, config: GraphRetrieverConfig = DEFAULT_CONFIG) -> list[NameSpan]:
    """Every n-gram (<= max_ngram tokens) of every capitalised run, longest first then leftmost."""
    spans = []
    for run in _runs(question):
        for n in range(min(len(run), config.max_ngram), 0, -1):
            for i in range(len(run) - n + 1):
                first, last = run[i], run[i + n - 1]
                text = question[first.start():last.end()]
                if n == 1 and (len(text) < config.min_single_len or text.isdigit()):
                    continue
                spans.append(NameSpan(text, first.start(), last.end(), n))
    return sorted(spans, key=lambda s: (-s.ntokens, s.start))


def select_spans(spans: list[NameSpan], matched: set[str]) -> list[NameSpan]:
    """Greedy longest-first choice of non-overlapping spans whose lowercased text is in `matched`."""
    chosen: list[NameSpan] = []
    for s in spans:
        if s.text.lower() in matched and all(s.end <= c.start or c.end <= s.start for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: s.start)


def name_tier(span: NameSpan) -> str:
    """Exact name matches: multi-word names are strong, one-word names are weak."""
    return "exact_name" if span.ntokens >= 2 else "single_name"


def rank_same_name(rows: list[dict]) -> list[dict]:
    """Nodes sharing a name, most connected first (then by id): the likeliest referent."""
    return sorted(rows, key=lambda r: (-r["degree"], r["id"]))


def release_aliases(release: str) -> list[str]:
    """`4.0` and `4.0.0` name the same release line; the graph holds both forms as nodes."""
    parts = release.split(".")
    if len(parts) == 2:
        return [release, f"{release}.0"]
    if len(parts) == 3 and parts[2] == "0":
        return [release, ".".join(parts[:2])]
    return [release]


def id_seeds(question: str) -> list[tuple[str, str, str]]:
    """(kind, canonical id, surface) for each distinct KIP / Jira / release id in the question."""
    out, seen = [], set()
    for m in extract_ids(question):
        if (m.kind, m.id) not in seen:
            seen.add((m.kind, m.id))
            out.append((m.kind, m.id, question[m.span[0]:m.span[1]]))
    return out


def _aliases(kind: str, cid: str) -> list[str]:
    return [f"{kind}:{a}" for a in (release_aliases(cid) if kind == "Release" else [cid])]


# --- graph lookups ------------------------------------------------------------------------

def _exact_lookup(kg: KnowledgeGraph, ids: list[str], names: list[str],
                  config: GraphRetrieverConfig) -> list[dict]:
    """One query for all candidate ids and names (instead of one find_nodes call per n-gram)."""
    return kg.cypher(
        "MATCH (n) WHERE NOT n:GraphMeta AND NOT n:Email "
        "AND (n.id IN $ids OR toLower(n.name) IN $names) "
        "RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, "
        "COUNT { (n)--() } AS degree ORDER BY n.id",
        {"ids": ids, "names": names}, timeout=config.query_timeout_s)


def _valid(node_id: str) -> bool:
    return ":?" not in node_id  # unparseable id mentions ("the KIP") become `KIP:?<mention>` nodes


def _tier_weight(how: str, config: GraphRetrieverConfig) -> float:
    return {"id": config.weight_id, "exact_name": config.weight_exact,
            "single_name": config.weight_single_name}[how]


def link_entities(question: str, kg: KnowledgeGraph,
                  config: GraphRetrieverConfig = DEFAULT_CONFIG,
                  embed: Callable[[], list[float]] | None = None) -> Linking:
    """Link the question to graph nodes (see the module docstring for the tiers).
    `embed` returns the question embedding; it is called only for the node-search fallback."""
    out = Linking()
    ids = id_seeds(question)
    wanted = sorted({a for kind, cid, _ in ids for a in _aliases(kind, cid)})
    spans = candidate_ngrams(question, config)
    rows = [r for r in _exact_lookup(kg, wanted, sorted({s.text.lower() for s in spans}), config)
            if _valid(r["id"])]

    by_id = {r["id"]: r for r in rows}
    for kind, cid, surface in ids:
        hits = [by_id[a] for a in _aliases(kind, cid) if a in by_id]
        if not hits:
            out.dropped.append({"surface": surface, "reason": "id not in graph"})
        out.seeds += [Seed(r["id"], r["label"], r["name"], "id", surface, config.weight_id)
                      for r in hits]

    by_name: dict[str, list[dict]] = {}
    for r in rows:
        if r["label"] in NAME_LABELS:
            by_name.setdefault((r["name"] or "").lower(), []).append(r)
    chosen = select_spans(spans, set(by_name))
    for s in chosen:
        how = name_tier(s)
        out.seeds += [Seed(r["id"], r["label"], r["name"], how, s.text, _tier_weight(how, config))
                      for r in rank_same_name(by_name[s.text.lower()])[:config.max_per_span]]

    for s in spans:  # fuzzy: multi-word spans not overlapping an exact match
        if s.ntokens < 2 or any(not (s.end <= c.start or c.end <= s.start) for c in chosen):
            continue
        hits = _fuzzy(kg, s.text, config)
        if hits:
            chosen.append(s)
            out.seeds += [Seed(n.id, n.label, n.name, "fuzzy_name", s.text,
                               round(config.weight_fuzzy * ratio / 100, 4)) for n, ratio in hits]

    if out.seeds or embed is None:
        return _dedupe(out)
    found = kg.search_nodes(question, k=config.node_search_k, embedding=embed())
    extra = [(n, sim) for n, sim in found if n.label in NAME_LABELS
             and sim >= config.node_search_min_sim and _valid(n.id)]
    out.seeds += [Seed(n.id, n.label, n.name, "node_search", question,
                       round(config.weight_node_search * sim, 4))
                  for n, sim in extra[:config.node_search_max]]
    return _dedupe(out)


def _fuzzy(kg: KnowledgeGraph, text: str, config: GraphRetrieverConfig) -> list:
    """Person/Component nodes whose name is a near-exact match of `text`: [(node, ratio)]."""
    scored = []
    for n in kg.find_nodes(text, fuzzy=True, limit=10):
        if n.label in NAME_LABELS and _valid(n.id):
            ratio = fuzz.ratio(text, n.name, processor=utils.default_process)
            if ratio >= config.fuzzy_min_ratio:
                scored.append((n, ratio))
    if not scored:
        return []
    best = max(r for _, r in scored)
    keep = sorted((p for p in scored if p[1] >= best - config.fuzzy_tie),
                  key=lambda p: (-p[1], p[0].id))
    return keep[:config.max_per_span]


def _dedupe(linking: Linking) -> Linking:
    """One seed per node: the first (strongest-tier) link wins."""
    seen, seeds = set(), []
    for s in linking.seeds:
        if s.id not in seen:
            seen.add(s.id)
            seeds.append(s)
    linking.seeds = seeds
    return linking
