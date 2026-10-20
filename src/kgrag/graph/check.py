"""Graph reachability diagnostics. Not a retriever: no ranking, just "can traversal reach it?"."""

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from neo4j import Driver

QUESTIONS_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "dev_questions.json"
MULTI_HOP_TYPES = ("inference", "comparison", "temporal")
MIN_SEED_LEN = 3  # shorter names ("a", "3") would match almost any question


@dataclass
class Connectivity:
    nodes_by_label: dict[str, int]
    rels_by_type: dict[str, int]
    largest_component_share: float
    degree: dict[str, tuple[float, float]]  # label -> (mean, median)
    person_degree_one: int


@dataclass
class QuestionReach:
    id: str
    type: str
    seeds: list[str]
    gold: list[str]
    reached: list[str]


@dataclass
class Reachability:
    hops: int
    questions: list[QuestionReach] = field(default_factory=list)

    @property
    def share(self) -> float:
        total = sum(len(q.gold) for q in self.questions)
        return sum(len(q.reached) for q in self.questions) / total if total else 0.0


def connectivity(driver: Driver) -> Connectivity:
    with driver.session() as s:
        nodes = {r["l"]: r["c"] for r in s.run(
            "MATCH (n) WHERE NOT n:GraphMeta RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l")}
        rels = {r["t"]: r["c"] for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")}
        ids = [r["id"] for r in s.run("MATCH (n) WHERE NOT n:GraphMeta RETURN n.id AS id")]
        edges = [(r["a"], r["b"]) for r in s.run(
            "MATCH (a)-[]->(b) RETURN a.id AS a, b.id AS b")]
        degree: dict[str, tuple[float, float]] = {}
        person_one = 0
        for label in ("Person", "KIP"):
            degs = [r["d"] for r in s.run(
                f"MATCH (n:{label}) RETURN COUNT {{ (n)--() }} AS d")]
            if degs:
                degree[label] = (statistics.fmean(degs), statistics.median(degs))
            if label == "Person":
                person_one = sum(1 for d in degs if d == 1)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    sizes: dict[str, int] = defaultdict(int)
    for i in ids:
        sizes[find(i)] += 1
    share = max(sizes.values()) / len(ids) if ids else 0.0
    return Connectivity(nodes, rels, share, degree, person_one)


def find_seeds(driver: Driver, question: str) -> list[str]:
    """Entity ids whose canonical name or id appears verbatim (case-insensitive) in the question."""
    with driver.session() as s:
        entities = s.run(
            "MATCH (n) WHERE NOT n:Email AND NOT n:GraphMeta RETURN n.id AS id, n.name AS name"
        ).data()
    seeds = []
    for e in entities:
        forms = {e["name"] or "", e["id"].split(":", 1)[-1], e["id"]}
        for form in forms:
            if len(form) >= MIN_SEED_LEN and re.search(
                rf"(?<![\w-]){re.escape(form)}(?![\w-])", question, re.IGNORECASE
            ):
                seeds.append(e["id"])
                break
    return sorted(seeds)


def reachable(driver: Driver, seeds: list[str], hops: int) -> tuple[set[str], set[str]]:
    """BFS up to `hops` over all relationships (undirected).

    Returns (node ids reached incl. seeds, source_doc_ids of every relationship traversed).
    """
    seen, frontier, docs = set(seeds), set(seeds), set()
    with driver.session() as s:
        for _ in range(hops):
            if not frontier:
                break
            rows = s.run(
                "MATCH (n)-[r]-(m) WHERE n.id IN $ids "
                "RETURN m.id AS m, r.source_doc_ids AS docs", ids=sorted(frontier)).data()
            frontier = set()
            for r in rows:
                docs.update(r["docs"] or [])
                if r["m"] not in seen:
                    seen.add(r["m"])
                    frontier.add(r["m"])
    return seen, docs


def doc_reached(doc_id: str, nodes: set[str], docs: set[str]) -> bool:
    """An Email node reached, or a traversed relationship cites the doc."""
    if doc_id in docs:
        return True
    parts = doc_id.split(":")
    return parts[0] == "email" and f"email:{parts[1]}" in nodes


def reachability(driver: Driver, questions: list[dict], hops: int,
                 types: tuple[str, ...] = MULTI_HOP_TYPES) -> Reachability:
    out = Reachability(hops)
    for q in questions:
        if q["type"] not in types:
            continue
        seeds = find_seeds(driver, q["question"])
        nodes, docs = reachable(driver, seeds, hops)
        gold = list(q["gold_doc_ids"])
        out.questions.append(QuestionReach(
            q["id"], q["type"], seeds, gold, [d for d in gold if doc_reached(d, nodes, docs)]))
    return out


def load_dev_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text())


def format_connectivity(c: Connectivity) -> str:
    lines = ["nodes: " + ", ".join(f"{k}={v}" for k, v in c.nodes_by_label.items()),
             "rels:  " + ", ".join(f"{k}={v}" for k, v in c.rels_by_type.items()),
             f"largest weakly connected component: {c.largest_component_share:.1%} of nodes"]
    for label, (mean, median) in c.degree.items():
        lines.append(f"{label} degree: mean={mean:.2f} median={median:g}")
    lines.append(f"Person nodes with degree 1: {c.person_degree_one}")
    return "\n".join(lines)


def format_reachability(r: Reachability) -> str:
    lines = [f"reachability within {r.hops} hops (multi-hop questions)"]
    for q in r.questions:
        lines.append(f"  {q.id} [{q.type}] seeds={len(q.seeds)} "
                     f"gold reached {len(q.reached)}/{len(q.gold)}  seeds: {', '.join(q.seeds) or '-'}")
    total = sum(len(q.gold) for q in r.questions)
    lines.append(f"overall: {sum(len(q.reached) for q in r.questions)}/{total} = {r.share:.1%}")
    return "\n".join(lines)
