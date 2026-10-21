"""Entity resolution for Person and Component mentions (`UserResolver`).

Contract
--------
`resolve(mentions, hints) -> Resolution`

  * `mentions`: only Person and Component mentions (KIP/JiraTicket/Release are normalized
    by `resolve_all`). Header mentions (`{email_id}:h0`) have context
    "<sender name> <sender address>"; Jira mentions (`jira:{event}:a0`) are Jira display
    names; LLM mentions (`{email_id}:m{n}`) carry the sentence they appear in.
  * `hints`: distinct (sender_name, sender_address) pairs and distinct Jira display names.
  * Output: every mention assigned to an entity; every entity has a canonical name and a
    type. Never merge across types.
  * Pure (no database, network or LLM), and deterministic even if the input is shuffled.
  * Two different people who share a first name must never be merged.

How it works
------------
Components (components.py): each surface is normalized to a key -- case, spaces, hyphens,
underscores, a leading "the/apache/kafka" word, a ".java" suffix, a package prefix and a
plural "s" on ordinary words are ignored -- and mentions with the same key are one entity.
A versioned spelling joins an existing product: "Widget 2.0" -> "Widget2" if present, else
"Widget 12" -> "Widget" if the bare name occurs. CamelCase class names are
never stemmed, so `FooExtension` and `FooExtensionContext` stay apart.

People (people.py, names.py), in two steps:

1. Identities from strong evidence. A union-find links (a) header names that share an
   address (except mailing-list relays: "Jane via dev"), (b) any two names with the same
   normalized full name (accents, case, punctuation, middle initials and common nicknames
   like Bob/Robert ignored) -- this joins a person's several addresses and their Jira
   display name -- (c) a header known only as "Jane D." to the single full name
   "Jane Doe" that fits it, and (d) a name seen only in text to the one participant whose
   surname it misspells by a single letter ("Jane Doee").
2. Attach every mention. Header, Jira and full-name mentions go straight to their
   identity. A first-name-only mention ("Chris", "David A") is resolved from its own
   email: a sign-off matching the sender -> the sender; a greeting ("Hi Chris") never
   means the sender; otherwise the one fitting person named in full in the same email,
   else the one fitting list participant (header sender or Jira user) in the whole corpus
   (or, if no participant fits, the one fitting person ever named in full). Anything still ambiguous stays a
   singleton: a missed link costs recall, a wrong merge corrupts the graph.

Canonical names prefer full over abbreviated spellings, then header > Jira > text, then
the most used spelling. Every loop runs over sorted data, so the output never depends on
input order or hash order.
"""

from kgrag.extract.schema import AnchorHints, Mention
from kgrag.resolve.components import group_components
from kgrag.resolve.interface import Resolution
from kgrag.resolve.people import resolve_people


class UserResolver:
    name = "user"

    def resolve(self, mentions: list[Mention], hints: AnchorHints) -> Resolution:
        """Resolve Person and Component mentions into entities (see module docstring)."""
        out = Resolution(assignments={}, canonical_names={}, entity_types={})
        people = [m for m in mentions if m.type == "Person"]
        components = [m for m in mentions if m.type == "Component"]
        for etype, groups in (("Person", resolve_people(people, hints)),
                              ("Component", group_components(components))):
            for canonical, members in groups:
                entity_id = f"{etype}:{members[0].mention_id}"  # first member id: unique, stable
                out.canonical_names[entity_id] = canonical
                out.entity_types[entity_id] = etype
                for m in members:
                    out.assignments[m.mention_id] = entity_id
        return out
