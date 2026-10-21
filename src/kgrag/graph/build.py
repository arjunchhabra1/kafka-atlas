"""Build the Neo4j knowledge graph from Postgres facts plus one stored entity resolution."""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice

import psycopg
from neo4j import Driver

from kgrag.config import build_hash, get_settings
from kgrag.extract.schema import Mention
from kgrag.graph.types import EDGE_RULES, NODE_LABELS, BuildStats, GraphMeta
from kgrag.observability import traced
from kgrag.resolve.interface import ID_TYPES, Resolution, normalize_id
from kgrag.store.repo import load_mentions, load_resolution, resolution_hash

BATCH = 1000
_EXTRACTOR_RANK = {"header": 0, "regex": 1, "jira": 2, "llm": 3}
_JIRA_EDGE = {"created": "FILED", "assigned": "ASSIGNED", "resolved": "RESOLVED"}


@dataclass
class _Edge:
    type: str
    src: str
    dst: str
    src_label: str
    dst_label: str
    docs: list[str]
    date: datetime
    extractor: str
    confidence: float
    vote_value: int | None = None
    binding: bool | None = None
    vote_date: datetime | None = None


@dataclass
class _Graph:
    nodes: dict[str, dict[str, dict]] = field(default_factory=lambda: defaultdict(dict))
    edges: dict[tuple[str, str, str], _Edge] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def node(self, label: str, node_id: str, **props) -> None:
        self.nodes[label].setdefault(node_id, {"id": node_id, **props})

    def edge(self, e: _Edge) -> None:
        """Add a fact; parallel facts merge. Facts violating the §4 endpoint rules are counted."""
        sources, targets = EDGE_RULES[e.type]
        if e.src_label not in sources or e.dst_label not in targets:
            self.skipped[f"type_violation:{e.type}"] += 1
            return
        cur = self.edges.get((e.type, e.src, e.dst))
        if cur is None:
            e.docs = list(dict.fromkeys(e.docs))  # own the list: never alias the caller's
            self.edges[(e.type, e.src, e.dst)] = e
            return
        cur.docs.extend(d for d in e.docs if d not in cur.docs)
        cur.date = min(cur.date, e.date)
        cur.confidence = max(cur.confidence, e.confidence)
        if _EXTRACTOR_RANK[e.extractor] < _EXTRACTOR_RANK[cur.extractor]:
            cur.extractor = e.extractor
        if e.type == "VOTED" and e.vote_date and (
            cur.vote_date is None or e.vote_date >= cur.vote_date
        ):
            cur.vote_value, cur.binding, cur.vote_date = e.vote_value, e.binding, e.vote_date


def _id_mention(surface: str, mtype: str) -> str | None:
    """Canonical entity id for an ID-type surface, mirroring resolve_all's `{type}:{id}`."""
    canonical = normalize_id(
        Mention(mention_id="x", surface=surface, type=mtype, email_id="", context="")  # type: ignore[arg-type]
    )
    return f"{mtype}:{canonical}" if canonical else None


def _contains(text: str, surface: str) -> bool:
    return bool(surface) and re.search(
        rf"(?<![\w-]){re.escape(surface)}(?![\w-])", text, re.IGNORECASE) is not None


def _chunks_containing(chunks: list[tuple[str, str]], *surfaces: str) -> list[str]:
    """Doc ids of the chunks citing the surfaces: those with ALL of them, else ANY, else chunk 0.

    Matching is case-insensitive and token-bounded (`KIP-10` does not match `KIP-1043`).
    """
    both = [d for d, t in chunks if all(_contains(t, x) for x in surfaces)]
    if both:
        return both
    either = [d for d, t in chunks if any(_contains(t, x) for x in surfaces)]
    return either or [chunks[0][0]]


def _assemble(conn: psycopg.Connection, resolution: Resolution) -> _Graph:
    g = _Graph()
    emails = {
        r[0]: r
        for r in conn.execute(
            "select id, message_id, date, subject, sender_name, sender_address, parent_message_id "
            "from emails where kind = 'human' order by id"
        )
    }
    chunks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc_id, source_id, text in conn.execute(
        "select doc_id, source_id, text from documents where source_kind = 'email' "
        "order by source_id, chunk_index"
    ):
        chunks[source_id].append((doc_id, text or ""))

    for eid, (_, _, date, subject, name, address, _) in emails.items():
        if eid not in chunks:  # an empty body yields no chunk; fall back to the nominal chunk 0
            chunks[eid] = [(f"email:{eid}:0", "")]
        g.node("Email", f"email:{eid}", name=subject or "", date=date, subject=subject or "",
               sender=name or address or "")

    types = resolution.entity_types
    for entity, etype in types.items():
        g.node(etype, entity, name=resolution.canonical_names.get(entity, entity.split(":", 1)[-1]))

    def entity_of(mention_id: str) -> str | None:
        return resolution.assignments.get(mention_id)

    # Headers: SENT; replies.
    mentions = load_mentions(conn)
    source = dict(conn.execute("select mention_id, source from mentions"))
    for m in mentions:
        eid = m.email_id
        if eid not in emails:
            continue
        date, docs = emails[eid][2], [d for d, _ in chunks[eid]]
        entity = entity_of(m.mention_id)
        if entity is None:
            g.skipped["unresolved_mention"] += 1
            continue
        src = source[m.mention_id]
        if src == "header":
            g.edge(_Edge("SENT", entity, f"email:{eid}", types[entity], "Email", list(docs), date,
                         "header", 1.0))
        elif src in ("regex", "llm") and m.type in ID_TYPES:
            g.edge(_Edge("MENTIONS", f"email:{eid}", entity, "Email", types[entity],
                         _chunks_containing(chunks[eid], m.surface), date, src, 1.0))

    by_message = {r[1]: r[0] for r in emails.values()}
    for eid, row in emails.items():
        parent = by_message.get(row[6]) if row[6] else None
        if parent and parent != eid:
            g.edge(_Edge("REPLIED_TO", f"email:{eid}", f"email:{parent}", "Email", "Email",
                         [d for d, _ in chunks[eid]], row[2], "header", 1.0))

    # Jira events.
    actor_entity = {
        int(mid.split(":")[1]): entity_of(mid) for mid in resolution.assignments
        if mid.startswith("jira:") and mid.endswith(":a0")
    }
    for event_id, ticket, action, date, fix_version in conn.execute(
        "select id, ticket, action, date, fix_version from jira_events order by id"
    ):
        ticket_id = _id_mention(ticket, "JiraTicket") or f"JiraTicket:{ticket}"
        g.node("JiraTicket", ticket_id, name=ticket_id.split(":", 1)[1])
        etype = _JIRA_EDGE.get(action)
        actor = actor_entity.get(event_id)
        if etype and actor:
            g.edge(_Edge(etype, actor, ticket_id, types.get(actor, "?"), "JiraTicket",
                         [f"jira:{event_id}"], date, "jira", 1.0))
        elif etype:
            g.skipped["jira_event_no_actor"] += 1
        else:
            g.skipped[f"jira_action_{action}"] += 1
        if fix_version:
            release = _id_mention(fix_version, "Release")
            if release is None:
                g.skipped["jira_unparseable_fix_version"] += 1
            else:
                g.node("Release", release, name=release.split(":", 1)[1])
                g.edge(_Edge("FIX_VERSION", ticket_id, release, "JiraTicket", "Release",
                             [f"jira:{event_id}"], date, "jira", 1.0))

    # LLM relations.
    by_mention = {m.mention_id: m for m in mentions}
    cur = conn.execute(
        "select type, source_mention_id, target_mention_id, vote_value, binding, confidence, "
        "email_id from relations order by id"
    )
    for rtype, smid, tmid, vote, binding, conf, eid in cur.fetchall():
        s_ent, t_ent = entity_of(smid), entity_of(tmid)
        if eid not in emails or s_ent is None or t_ent is None:
            g.skipped["relation_unresolved_endpoint"] += 1
            continue
        s_label, t_label = types[s_ent], types[t_ent]
        if s_ent == t_ent:
            g.skipped["relation_self_loop"] += 1
            continue
        date = emails[eid][2]
        docs = _chunks_containing(chunks[eid], by_mention[smid].surface, by_mention[tmid].surface)
        g.edge(_Edge(rtype, s_ent, t_ent, s_label, t_label, docs, date, "llm", float(conf or 0.0),
                     vote, binding, date if rtype == "VOTED" else None))
    return g


def _batches(rows: list[dict]):
    it = iter(rows)
    while batch := list(islice(it, BATCH)):
        yield batch


def _clear(driver: Driver) -> None:
    with driver.session() as s:
        while s.run(
            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c"
        ).single()["c"]:
            pass
        for label in (*NODE_LABELS, "GraphMeta"):
            s.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                  f"FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()


def _load(driver: Driver, g: _Graph) -> None:
    with driver.session() as s:
        for label, nodes in g.nodes.items():
            for batch in _batches(sorted(nodes.values(), key=lambda r: r["id"])):
                s.run(f"UNWIND $rows AS r MERGE (n:{label} {{id: r.id}}) SET n += r", rows=batch)
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for e in g.edges.values():
            groups[(e.type, e.src_label, e.dst_label)].append({
                "src": e.src, "dst": e.dst, "docs": e.docs, "date": e.date,
                "extractor": e.extractor, "confidence": e.confidence,
                "vote_value": e.vote_value, "binding": e.binding,
            })
        for (rtype, a, b), rows in sorted(groups.items()):
            extra = ", x.vote_value = r.vote_value, x.binding = r.binding" if rtype == "VOTED" else ""
            query = (
                f"UNWIND $rows AS r MATCH (a:{a} {{id: r.src}}) MATCH (b:{b} {{id: r.dst}}) "
                f"MERGE (a)-[x:{rtype}]->(b) SET x.source_doc_ids = r.docs, x.date = r.date, "
                f"x.extractor = r.extractor, x.confidence = r.confidence{extra}"
            )
            for batch in _batches(sorted(rows, key=lambda r: (r["src"], r["dst"]))):
                s.run(query, rows=batch)


@traced("graph.build")
def build_graph(conn: psycopg.Connection, driver: Driver, resolver_name: str) -> BuildStats:
    """Clear the database and rebuild the graph for one stored resolution."""
    resolution = load_resolution(conn, resolver_name)
    res_hash = resolution_hash(conn, resolver_name)  # same snapshot as the rows just loaded
    if not resolution.assignments:
        raise ValueError(f"no stored resolution for resolver {resolver_name!r}; run `kgrag resolve`")
    g = _assemble(conn, resolution)
    _clear(driver)
    _load(driver, g)
    with driver.session() as sess:  # counts come from the database, not from what we meant to load
        nodes = {r["l"]: r["c"] for r in sess.run(
            "MATCH (n) WHERE NOT n:GraphMeta RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l")}
        rels = {r["t"]: r["c"] for r in sess.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")}
    expected = sum(len(v) for v in g.nodes.values()), len(g.edges)
    if (sum(nodes.values()), sum(rels.values())) != expected:
        raise RuntimeError(f"graph load mismatch: expected {expected} nodes/rels, "
                           f"database has {sum(nodes.values())}/{sum(rels.values())}")
    warnings = []
    humans, with_parent = conn.execute(
        "select count(*), count(parent_message_id) from emails where kind = 'human'").fetchone()
    if humans and not with_parent:
        warnings.append("no emails have parent_message_id: REPLIED_TO is empty; "
                        "run scripts/backfill_parents.py")
    meta = GraphMeta(resolver_name, datetime.now(UTC), build_hash(get_settings()),
                     sum(nodes.values()), sum(rels.values()), res_hash)
    with driver.session() as s:
        s.run(
            "CREATE (m:GraphMeta {id: 'meta', resolver: $resolver, built_at: $built_at, "
            "build_hash: $build_hash, resolution_hash: $resolution_hash, "
            "node_count: $node_count, rel_count: $rel_count})",
            resolver=meta.resolver, built_at=meta.built_at, build_hash=meta.build_hash,
            resolution_hash=meta.resolution_hash, node_count=meta.node_count,
            rel_count=meta.rel_count,
        ).consume()
    return BuildStats(nodes=nodes, rels=rels, skipped=dict(sorted(g.skipped.items())),
                      warnings=warnings)


def read_graph_meta(driver: Driver) -> GraphMeta | None:
    with driver.session() as s:
        rec = s.run("MATCH (m:GraphMeta) RETURN m").single()
    if rec is None:
        return None
    m = rec["m"]
    return GraphMeta(m["resolver"], m["built_at"].to_native(), m.get("build_hash"),
                     m["node_count"], m["rel_count"], m.get("resolution_hash"))
