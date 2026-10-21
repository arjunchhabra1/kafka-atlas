"""Person resolution: anchor identities from headers/Jira, then attach every mention.

See the "How it works" section in resolver.py for the overview.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from kgrag.extract.schema import AnchorHints, Mention
from kgrag.resolve.names import (
    EMAIL_RE,
    PersonName,
    address_local_part,
    clean_display_name,
    first_names_compatible,
    fold,
    formal,
    full_key,
    is_relay_name,
    is_surname_typo,
    name_role,
    parse_name,
)

HEADER, JIRA, TEXT = 0, 1, 2  # name sources, best first (used to pick canonical names)


class _UnionFind:
    """Minimal union-find over string node keys."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> str:
        self.parent.setdefault(node, node)
        return node

    def find(self, node: str) -> str:
        self.add(node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the smaller key as root so the structure does not depend on call order.
            ra, rb = min(ra, rb), max(ra, rb)
            self.parent[rb] = ra


@dataclass
class _Identity:
    """Everything known about one person: name spellings, first/last names, address local parts."""

    root: str  # union-find root; also the fallback display name
    names: Counter = field(default_factory=Counter)  # (is_abbreviated, source, display) -> count
    first_names: set[str] = field(default_factory=set)
    last_names: set[str] = field(default_factory=set)
    local_parts: set[str] = field(default_factory=set)
    addresses: set[str] = field(default_factory=set)

    def add_name(self, display: str, source: int, count: int = 1) -> None:
        """Record a spelling of this person's name."""
        name = parse_name(display)
        if not name.tokens:
            return
        self.names[(not name.is_full, source, display)] += count
        self.first_names.add(name.first)
        if name.last:
            self.last_names.add(name.last)

    def matches(self, name: PersonName) -> bool:
        """Could a partial mention ("Jane", "Jane D") refer to this person?"""
        if not any(first_names_compatible(name.first, f) for f in self.first_names):
            return False
        if name.last_is_initial:
            return any(len(last) > 1 and last[0] == name.last for last in self.last_names)
        return True

    def canonical(self) -> str:
        """Most informative spelling: full over abbreviated, header > Jira > text, then frequency."""
        if not self.names:
            return min(self.addresses, default=self.root)
        def rank(k):
            abbreviated, source, display = k
            return abbreviated, source, -self.names[k], -len(display), display

        return min(self.names, key=rank)[2]


def _header_parts(m: Mention) -> tuple[str, str]:
    """(sender name, address) of a header mention, whose context is "<name> <address>"."""
    address = m.context.rsplit(None, 1)[-1] if m.context.strip() else ""
    return m.surface, address


def _anchor_node(name: str, address: str, email_id: str) -> str | None:
    """Union-find node for a header sender: the address, unless a mailing-list relay rewrote it.

    A relay sender with only a first name ("Ann via dev") carries no identity evidence, so
    each such email gets its own node; from hints alone (no email_id) there is no node.
    """
    if address and not is_relay_name(name):
        return f"addr:{fold(address)}"
    key = full_key(parse_name(name))
    if key:
        return f"name:{key}"
    return f"lone:{email_id}" if email_id else None


def _jira_node(name: str) -> str:
    """Union-find node for a Jira display name."""
    key = full_key(parse_name(name))
    return f"name:{key}" if key else f"lone:jira:{fold(name)}"


def _is_header(m: Mention) -> bool:
    return m.mention_id.endswith(":h0")


def _is_jira(m: Mention) -> bool:
    return m.mention_id.startswith("jira:")


class PeopleResolver:
    """Builds identities from strong evidence, then assigns each mention (see resolve_people).

    `assign()` runs the steps in order and the order matters: the three `_link_*` steps
    mutate the union-find, and `_build_identities` snapshots it into `identities` and the
    lookup indexes, so it must run after the last link and before any mention is attached.
    """

    def __init__(self, mentions: list[Mention], hints: AnchorHints) -> None:
        self.mentions = sorted(mentions, key=lambda m: m.mention_id)
        self.uf = _UnionFind()
        # (name, address, email_id); hint pairs have no email ("") -- see _anchor_node.
        self.senders = sorted({(n, a, "") for n, a in hints.sender_pairs}
                              | {(*_header_parts(m), m.email_id) for m in self.mentions if _is_header(m)})
        self.jira_names = sorted(set(hints.jira_actors)
                                 | {m.surface for m in self.mentions if _is_jira(m)})
        self.text_full = [m for m in self.mentions if not _is_header(m) and not _is_jira(m)
                          and parse_name(m.surface).is_full]
        self.identities: dict[str, _Identity] = {}

    def _sender_nodes(self):
        """(name, address, node) for every sender that has a union-find node."""
        for name, address, email_id in self.senders:
            node = _anchor_node(name, address, email_id)
            if node is not None:
                yield name, address, node

    # --- step 1: identities from headers, Jira and full names --------------------------------

    def _link_anchors(self) -> None:
        """Same address => same person; same full name => same person (across addresses/Jira)."""
        for name, _, node in self._sender_nodes():
            self.uf.add(node)
            key = full_key(parse_name(name))
            if key:
                self.uf.union(node, f"name:{key}")
        for name in self.jira_names:
            self.uf.add(_jira_node(name))
        for m in self.text_full:
            self.uf.add(f"name:{full_key(parse_name(m.surface))}")

    def _link_abbreviated(self) -> None:
        """Join an identity known only as "Jane D." to the single full name "Jane Doe" that fits."""
        abbreviated: dict[str, list[PersonName]] = defaultdict(list)
        has_full: set[str] = set()
        for name, _, node in self._sender_nodes():
            root, parsed = self.uf.find(node), parse_name(name)
            if parsed.is_full:
                has_full.add(root)
            elif parsed.last_is_initial:
                abbreviated[root].append(parsed)
        full_keys = sorted(n for n in self.uf.parent if n.startswith("name:"))
        fits: dict[str, list[str]] = defaultdict(list)  # abbreviated root -> fitting full names
        for root, names in sorted(abbreviated.items()):
            if root in has_full:
                continue
            for node in full_keys:
                first, last = node[len("name:"):].split(" ", 1)
                if any(formal(n.first) == first and last.split()[-1][0] == n.last for n in names):
                    fits[root].append(node)
        claimed = Counter(node for nodes in fits.values() for node in nodes)
        for root, nodes in sorted(fits.items()):
            if len(nodes) == 1 and claimed[nodes[0]] == 1:
                self.uf.union(root, nodes[0])

    def _anchored_roots(self) -> set[str]:
        """Roots of identities backed by a mail header or a Jira account (list participants)."""
        return ({self.uf.find(node) for _, _, node in self._sender_nodes()}
                | {self.uf.find(_jira_node(n)) for n in self.jira_names})

    def _link_typos(self) -> None:
        """Join a name seen only in text ("Jane Doee") to the one participant it is a typo of."""
        anchored = self._anchored_roots()
        participants: dict[str, list[tuple[str, str]]] = defaultdict(list)  # first -> (last, root)
        text_only: list[tuple[str, str, str]] = []
        for node in sorted(n for n in self.uf.parent if n.startswith("name:")):
            first, last = node[len("name:"):].split(" ", 1)
            root = self.uf.find(node)
            if root in anchored:
                participants[first].append((last, root))
            else:
                text_only.append((node, first, last))
        for node, first, last in text_only:
            roots = {r for known, r in participants[first] if is_surname_typo(last, known)}
            if len(roots) == 1:
                self.uf.union(node, roots.pop())

    def _identity(self, node: str) -> _Identity:
        """The identity of `node`'s union-find root, created on first use."""
        root = self.uf.find(node)
        return self.identities.setdefault(root, _Identity(root))

    def _build_identities(self) -> None:
        """Collect names, first names and address local parts per union-find root."""
        # Names are registered with zero votes here: they give each identity its first names,
        # surnames and spellings; resolve_people later adds one vote per mention actually seen.
        for name, address, node in self._sender_nodes():
            ident = self._identity(node)
            ident.add_name(clean_display_name(name), HEADER, count=0)
            if address and not is_relay_name(name):  # a relay address is everyone's
                ident.addresses.add(fold(address))
                local = address_local_part(address)
                if local:
                    ident.local_parts.add(local)
        for name in self.jira_names:
            self._identity(_jira_node(name)).add_name(clean_display_name(name), JIRA, count=0)
        for m in self.text_full:
            node = f"name:{full_key(parse_name(m.surface))}"
            self._identity(node).add_name(" ".join(m.surface.split()), TEXT, count=0)
        self.anchored = self._anchored_roots()
        self.by_first: dict[str, list[str]] = defaultdict(list)  # formal first name -> roots
        self.by_local: dict[str, set[str]] = defaultdict(set)
        self.by_address: dict[str, str] = {}
        for root in sorted(self.identities):
            ident = self.identities[root]
            for first in sorted({formal(f) for f in ident.first_names}):
                self.by_first[first].append(root)
            for local in ident.local_parts:
                self.by_local[local].add(root)
            for address in ident.addresses:
                self.by_address.setdefault(address, root)

    # --- step 2: attach mentions -------------------------------------------------------------

    def _strong_root(self, m: Mention) -> str | None:
        """Identity for header, Jira and full-name mentions (no guessing involved)."""
        if _is_header(m):
            return self.uf.find(_anchor_node(*_header_parts(m), m.email_id))
        if _is_jira(m):
            return self.uf.find(_jira_node(m.surface))
        key = full_key(parse_name(m.surface))
        return self.uf.find(f"name:{key}") if key else None

    def _by_address(self, m: Mention, name: PersonName) -> str | None:
        """"Jane <jd@x.org> wrote:" -- a known address in the sentence pins the mention."""
        found = {fold(a.group(0)) for a in EMAIL_RE.finditer(f"{m.surface} {m.context}")}
        roots = {self.by_address[a] for a in found if a in self.by_address}
        if len(roots) == 1:
            root = roots.pop()
            if "@" in m.surface or self.identities[root].matches(name):
                return root
        return None

    def _by_username(self, name: PersonName) -> str | None:
        """A handle like "jdoe42" that is exactly one person's address local part.

        Skipped for words that are someone's first name: "Jane" is a name, not a handle.
        """
        if len(name.tokens) != 1 or formal(name.first) in self.by_first:
            return None
        roots = self.by_local.get(name.first, set())
        return next(iter(roots)) if len(roots) == 1 else None

    def _candidates(self, name: PersonName, roots) -> list[str]:
        """The roots (sorted, distinct) whose identity fits the partial name."""
        return sorted({r for r in roots if self.identities[r].matches(name)})

    def _by_context(self, m: Mention, name: PersonName, sender: str | None,
                    local: set[str]) -> str | None:
        """Resolve "Jane"/"Jane D" from the email it appears in, falling back to a unique namesake.

        1. The writer naming themselves (a sign-off like "Thanks,\nJane"; see `name_role`)
           and the sender fits -> the sender. So does a name absent from its sentence (the
           extractor quoted a neighbouring line).
        2. Exactly one fitting person named in full in the same email -> them.
        3. Exactly one fitting person in the whole corpus -> them. List participants (header
           senders, Jira users) come first: someone only ever named in passing is counted
           only when no participant fits.
        The sender is never chosen in steps 2-3 ("Hi Jane", "Jane's point", "@Jane" address
        someone else). When the sender fits but the mention is neither a sign-off nor a
        greeting (a vote tally, "Jane's point"), step 3 is skipped: a Jane writing "Jane" is
        ambiguous by construction, so only local evidence may decide.
        """
        sender_fits = bool(sender) and self.identities[sender].matches(name)
        role = name_role(m.context, m.surface,
                         surnames=self.identities[sender].last_names if sender else ())
        if sender_fits and role in ("self", "absent"):
            return sender
        excluded = {sender} if sender else set()
        local_fits = self._candidates(name, local - excluded)
        if local_fits or (sender_fits and role == "other"):
            return local_fits[0] if len(local_fits) == 1 else None
        namesakes = set(self.by_first.get(formal(name.first), [])) - excluded
        corpus_fits = self._candidates(name, namesakes)
        participants = [r for r in corpus_fits if r in self.anchored]
        fits = participants or corpus_fits
        return fits[0] if len(fits) == 1 else None

    def assign(self) -> dict[str, str | None]:
        """mention_id -> identity root, or None when the mention should stay a singleton."""
        self._link_anchors()
        self._link_abbreviated()
        self._link_typos()
        self._build_identities()
        out: dict[str, str | None] = {m.mention_id: self._strong_root(m) for m in self.mentions}
        sender_of = {m.email_id: out[m.mention_id] for m in self.mentions if _is_header(m)}
        local_of: dict[str, set[str]] = defaultdict(set)
        for m in self.mentions:
            if out[m.mention_id] and not _is_header(m):
                local_of[m.email_id].add(out[m.mention_id])
        for m in self.mentions:
            if out[m.mention_id] is not None:
                continue
            name = parse_name(m.surface)
            if not name.tokens:
                continue
            out[m.mention_id] = (self._by_address(m, name) or self._by_username(name)
                                 or self._by_context(m, name, sender_of.get(m.email_id),
                                                     local_of[m.email_id]))
        return out


def _spelling(m: Mention) -> tuple[str, int] | None:
    """(display name, source) a mention contributes to its identity's name votes, if any."""
    if _is_header(m):
        return clean_display_name(m.surface), HEADER
    if _is_jira(m):
        return clean_display_name(m.surface), JIRA
    if parse_name(m.surface).is_full:
        return " ".join(m.surface.split()), TEXT
    return None  # "Hi Jane" says nothing about how to spell the full name


def resolve_people(mentions: list[Mention], hints: AnchorHints) -> list[tuple[str, list[Mention]]]:
    """Group Person mentions into entities: [(canonical name, members)], sorted.

    Unresolvable mentions (an ambiguous "Chris") come back as singletons named by their surface.
    """
    resolver = PeopleResolver(mentions, hints)
    assigned = resolver.assign()
    groups: dict[str, list[Mention]] = defaultdict(list)
    for m in resolver.mentions:
        groups[assigned[m.mention_id] or f"single:{m.mention_id}"].append(m)
    out = []
    for root, ms in groups.items():
        if root.startswith("single:"):
            out.append((" ".join(ms[0].surface.split()) or ms[0].mention_id, ms))
            continue
        ident = resolver.identities[root]
        for spelling in filter(None, map(_spelling, ms)):  # weight spellings by actual use
            ident.add_name(*spelling)
        out.append((ident.canonical(), ms))
    return sorted(out, key=lambda g: (g[0], g[1][0].mention_id))
