"""Every knob of the graph retriever, in one frozen dataclass.

`retrieve(question, kg, budget_tokens, config=None)` uses `DEFAULT_CONFIG` unless given
another. Booleans switch whole scoring terms off for ablations (graph_ablation.py); numbers
are the values in use. They were set on the dev fixture plus the 5 real-graph questions in
tests/fixtures/tuning_log.json, and must be re-validated by a dev-split ablation
(`kgrag ablate --split dev`) before the headline run. Do not tune them on the test split.

`GRAPH_CONFIGS` names the configurations a run can select (`kgrag eval --graph-config`);
each results file records the name and `fingerprint()` of the one it used.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType

# How much an edge of each type says about its endpoints. Facts extracted from the thread
# (who proposed/voted/reviewed, what a ticket fixed) beat structural plumbing.
EDGE_WEIGHT: Mapping[str, float] = MappingProxyType({
    "PROPOSED": 1.0, "VOTED": 1.0, "REVIEWED": 0.9, "RESOLVED": 0.9, "FILED": 0.9,
    "FIX_VERSION": 0.9, "ASSIGNED": 0.8, "BLOCKS": 0.9, "AFFECTS": 0.8, "DEPENDS_ON": 0.9,
    "SUPERSEDES": 0.9, "MENTIONS": 0.6, "SENT": 0.5, "REPLIED_TO": 0.3,
})
STRUCTURAL_TYPES = frozenset({"MENTIONS", "SENT", "REPLIED_TO"})


@dataclass(frozen=True)
class GraphRetrieverConfig:
    # --- switches (ablations turn whole terms off) ---------------------------------------
    cues: bool = True             # boost edge types the question asks about
    bridge_boost: bool = True     # boost seed-seed edges and edges into shared neighbours
    doc_share: bool = True        # an edge's weight is shared among its n justifying docs
    off_focus: bool = True        # damp seed edges into *other* nodes of a named type
    semantic_rerank: bool = True  # add question-chunk embedding similarity
    vector_backfill: bool = True  # fill leftover budget with plain vector hits
    vector_fallback: bool = True  # no seed / no graph evidence: answer with plain vector search
    node_search: bool = True      # link by embedding search over node names (last-resort tier)
    diversify: bool = True        # round-robin the best docs of each strong seed first
    doc_bridge: bool = True       # follow KIPs / tickets mentioned by the best first-pass docs

    # --- walk scoring --------------------------------------------------------------------
    edge_weight: Mapping[str, float] = field(default_factory=lambda: EDGE_WEIGHT)
    unknown_edge_weight: float = 0.3   # edge types missing from edge_weight
    decay: float = 0.5                 # penalty per hop beyond the first
    cue_factor: float = 1.5
    bridge_factor: float = 1.5
    off_focus_factor: float = 0.5
    doc_share_exp: float = 0.25        # each of an edge's n docs gets n^-exp of its weight

    # --- walk size (caps are enforced inside the Cypher query) ---------------------------
    hop1_max_edges: int = 500          # strongest edges per seed (hubs have thousands)
    hop2_max_edges: int = 200          # strongest fact edges per expanded hop-1 entity
    max_expand_per_seed: int = 25      # hop-1 entities expanded to hop 2, best-reached first
    query_timeout_s: float = 10.0      # Neo4j transaction timeout for lookups and walks

    # --- rerank, diversify, trace --------------------------------------------------------
    semantic_pool: int = 2000          # best graph candidates that get a similarity
    max_candidates: int = 300          # best combined candidates whose text is ranked
    sem_weight: float = 0.5            # weight of the min-max normalised similarity
    diversify_rounds: int = 2          # docs guaranteed per strong seed
    trace_paths: int = 3               # paths kept per returned doc in the trace

    # --- doc bridge: question -> best docs -> KIPs / tickets they mention -> their docs -----
    doc_bridge_pivots: int = 5         # best first-pass docs (and best vector hits) to pivot on
    doc_bridge_max_seeds: int = 6      # entities taken from the pivots, best pivot first
    doc_bridge_weight: float = 0.6     # seed weight of a doc-bridged entity (hop 1 only)
    doc_bridge_diversify: bool = True  # doc-bridged entities also get diversify rounds
    # Hybrid fusion: the question's best vector hits join the pool as one pseudo seed, so a
    # doc no linked entity reaches still competes (needs embeddings; off in graph-only).
    vector_seeds: int = 5
    vector_seed_weight: float = 1.0

    # --- entity linking ------------------------------------------------------------------
    weight_id: float = 1.0             # KIP / Jira / release id found by regex
    weight_exact: float = 1.0          # multi-word name, exact match
    weight_fuzzy: float = 0.8          # multi-word name, fuzzy match (x ratio/100)
    weight_single_name: float = 0.5    # one-word name ("Kirk"): weak, often ambiguous
    weight_node_search: float = 0.3    # embedding search over node names (fallback only)
    max_ngram: int = 4
    min_single_len: int = 3
    max_per_span: int = 3              # same-name nodes kept, highest degree first
    fuzzy_min_ratio: int = 85
    fuzzy_tie: int = 5
    node_search_k: int = 5
    node_search_min_sim: float = 0.5   # embedding-model dependent
    node_search_max: int = 2

    def weight(self, edge_type: str) -> float:
        return self.edge_weight.get(edge_type, self.unknown_edge_weight)

    def with_(self, **changes) -> "GraphRetrieverConfig":
        return replace(self, **changes)

    @property
    def uses_embeddings(self) -> bool:
        return self.semantic_rerank or self.vector_backfill or self.vector_fallback \
            or self.node_search or self.vector_seeds > 0

    def fingerprint(self) -> str:
        """Short stable hash of every field, recorded with each run that uses this config."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        payload = json.dumps({k: dict(v) if isinstance(v, Mapping) else v
                              for k, v in values.items()}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


DEFAULT_CONFIG = GraphRetrieverConfig()
# The graph alone: every embedding use is off (no rerank, no backfill, no vector fallback, no
# node-name search). A question that links to no node, or whose seeds reach no evidence,
# gets an empty result (recall 0) and the trace says so.
GRAPH_ONLY_CONFIG = DEFAULT_CONFIG.with_(semantic_rerank=False, vector_backfill=False,
                                         vector_fallback=False, node_search=False,
                                         vector_seeds=0)

# Named configurations for `kgrag eval --graph-config` and `kgrag ablate --configs`.
GRAPH_CONFIGS: Mapping[str, GraphRetrieverConfig] = MappingProxyType({
    "hybrid": DEFAULT_CONFIG,
    "graph-only": GRAPH_ONLY_CONFIG,
})
DEFAULT_CONFIG_NAME = "hybrid"


class UnknownGraphConfigError(ValueError):
    pass


def get_graph_config(name: str) -> GraphRetrieverConfig:
    try:
        return GRAPH_CONFIGS[name]
    except KeyError:
        raise UnknownGraphConfigError(
            f"unknown graph config {name!r}; choose one of {', '.join(GRAPH_CONFIGS)}") from None
