"""Deterministic ID extraction (KIPs, Jira tickets, releases) from free text.

These IDs become canonical graph nodes, so precision beats recall: ambiguous numbers
(``3.5 seconds``, IP addresses, 4-part versions) must not match.
"""

import re
from typing import Literal

from pydantic import BaseModel

Kind = Literal["KIP", "JiraTicket", "Release"]


class IdMention(BaseModel):
    kind: Kind
    id: str
    span: tuple[int, int]


_KIP = re.compile(r"\bKIP(?:-| )?(\d+)\b", re.IGNORECASE)
_TICKET = re.compile(r"\bKAFKA-(\d+)\b", re.IGNORECASE)

# Single-digit major; not part of a longer dotted/digit run on either side.
_RELEASE = re.compile(r"(?<![\d.])(?<!\d\.)\b[0-9]\.[0-9]{1,2}(\.[0-9]{1,2})?(?!\d|\.\d)\b")
# Suffixes that belong to the version token (span still covers only the version).
_SUFFIX = re.compile(r"-(?:rc\d*|snapshot)\b", re.IGNORECASE)
_KEYWORD = re.compile(r"(?:releases?|versions?|branch|rc\d*)", re.IGNORECASE)

_MAX_WORDS_BETWEEN = 3
_WINDOW = _MAX_WORDS_BETWEEN + 1  # keyword may be this many tokens away (3 between)


def _is_keyword(token: str) -> bool:
    return _KEYWORD.fullmatch(token.strip(".,;:!?()[]{}<>\"'`*_")) is not None


def _has_release_context(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 200) : start].split()
    if any(_is_keyword(t) for t in before[-_WINDOW:]):
        return True
    if _SUFFIX.match(text, end):
        return True
    after = text[end : end + 200].split()
    return any(_is_keyword(t) for t in after[:_WINDOW])


def extract_ids(text: str) -> list[IdMention]:
    """Return KIP, Jira ticket and release mentions ordered by span start."""
    found: set[tuple[int, int, Kind, str]] = set()
    for m in _KIP.finditer(text):
        found.add((m.start(), m.end(), "KIP", f"KIP-{int(m.group(1))}"))
    for m in _TICKET.finditer(text):
        found.add((m.start(), m.end(), "JiraTicket", f"KAFKA-{int(m.group(1))}"))
    for m in _RELEASE.finditer(text):
        if m.group(1) or _has_release_context(text, m.start(), m.end()):
            found.add((m.start(), m.end(), "Release", m.group(0)))
    return [IdMention(kind=k, id=i, span=(s, e)) for s, e, k, i in sorted(found)]
