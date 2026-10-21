"""Graph schema (spec §4) and the small value types shared by build and check."""

from dataclasses import dataclass, field
from datetime import datetime

NODE_LABELS = ("Person", "KIP", "JiraTicket", "Release", "Component", "Email")

# edge type -> (allowed source labels, allowed target labels)
EDGE_RULES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "SENT": (frozenset({"Person"}), frozenset({"Email"})),
    "REPLIED_TO": (frozenset({"Email"}), frozenset({"Email"})),
    "MENTIONS": (frozenset({"Email"}), frozenset({"KIP", "JiraTicket", "Release"})),
    "FILED": (frozenset({"Person"}), frozenset({"JiraTicket"})),
    "ASSIGNED": (frozenset({"Person"}), frozenset({"JiraTicket"})),
    "RESOLVED": (frozenset({"Person"}), frozenset({"JiraTicket"})),
    "FIX_VERSION": (frozenset({"JiraTicket"}), frozenset({"Release"})),
    "PROPOSED": (frozenset({"Person"}), frozenset({"KIP"})),
    "VOTED": (frozenset({"Person"}), frozenset({"KIP", "Release"})),
    "REVIEWED": (frozenset({"Person"}), frozenset({"KIP", "JiraTicket"})),
    "BLOCKS": (frozenset({"KIP", "JiraTicket"}), frozenset({"Release"})),
    "AFFECTS": (frozenset({"KIP", "JiraTicket"}), frozenset({"Component", "Release"})),
    "DEPENDS_ON": (frozenset({"KIP"}), frozenset({"KIP"})),
    "SUPERSEDES": (frozenset({"KIP"}), frozenset({"KIP"})),
}


@dataclass(frozen=True)
class GraphMeta:
    resolver: str
    built_at: datetime
    build_hash: str | None       # config.build_hash at build time (None: built before it existed)
    node_count: int
    rel_count: int
    resolution_hash: str | None = None  # repo.resolution_hash of the stored resolution used


@dataclass
class BuildStats:
    nodes: dict[str, int]
    rels: dict[str, int]
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count of dropped facts
    warnings: list[str] = field(default_factory=list)


@dataclass
class Node:
    id: str
    label: str
    name: str
    props: dict = field(default_factory=dict)


@dataclass
class Edge:
    type: str
    source_id: str
    target_id: str
    props: dict = field(default_factory=dict)
