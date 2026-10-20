"""YOUR TASK: entity resolution for Person and Component mentions.

The problem
-----------
The extractor emits one mention per occurrence ("Chia-Ping", "Chia-Ping Tsai",
"chia7712", "MM2", "MirrorMaker 2", ...). Group the mentions that refer to the same
real-world person or component, and name each group. Everything downstream (the graph,
retrieval, answers) is built on the entities you produce here.

The contract
------------
`resolve(mentions, hints) -> Resolution`

Inputs
  * `mentions`: only `Person` and `Component` mentions (KIP/JiraTicket/Release are
    normalized for you by `resolve_all`). Sorted by mention_id. Each has `surface`,
    `type`, `email_id` and `context`. For header-sourced Person mentions the context is
    "<sender name> <sender address>"; Jira-sourced ones are Jira display names.
  * `hints`: `AnchorHints(sender_pairs, jira_actors)` -- distinct (name, address) pairs
    from email headers and distinct Jira display names. Use them as identity evidence
    (e.g. the same address behind two name spellings; "Chia-Ping Tsai" in Jira vs a
    personal address in the mail).

Output: `Resolution`
  * `assignments`: mention_id -> entity_id, for EVERY mention you were given.
  * `canonical_names`: entity_id -> display name, for EVERY entity.
  * `entity_types`: entity_id -> "Person" or "Component". Never merge across types.
  entity_id is any opaque string you like; `resolve_all` prefixes it with the type.

Rules
  * Pure function: no database, network or LLM-cache access, just these two arguments.
  * Deterministic: same input -> same output (the contract test shuffles input order).
  * Be careful with shared first names (two different Davids/Chrises must NOT merge).

Check your work
---------------
    uv run pytest tests/resolve tests/eval/test_bcubed.py -v -s

`test_contract.py` checks the rules above; `test_quality.py` needs B-cubed F1 >= 0.85
against tests/fixtures/gold_entities.json plus the tagged edge cases (shared first
names, multi-identity people, variant spellings). Both xfail until this stub stops
raising NotImplementedError, then run for real. The exact-match baseline is in
`exact_match.py` for comparison. Try it on the real data with
`kgrag resolve --resolver user`.
"""

from kgrag.extract.schema import AnchorHints, Mention
from kgrag.resolve.interface import Resolution


class UserResolver:
    name = "user"

    def resolve(self, mentions: list[Mention], hints: AnchorHints) -> Resolution:
        raise NotImplementedError("👤 implement entity resolution")
