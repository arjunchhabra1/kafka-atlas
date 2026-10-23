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


_KIP = re.compile(r"\bKIP(?:-| )?(\d{1,7})\b", re.IGNORECASE)
_TICKET = re.compile(r"\bKAFKA-(\d{1,7})\b(?!\.\d)", re.IGNORECASE)

# Single-digit major; not part of a longer dotted/digit run on either side.
_RELEASE = re.compile(r"(?<![\d.])(?<!\d\.)\b[0-9]\.[0-9]{1,2}(\.[0-9]{1,2})?(?!\d|\.\d)\b")
# Suffixes that belong to the version token (span still covers only the version).
_SUFFIX = re.compile(r"-(?:rc\d*|snapshot)\b", re.IGNORECASE)
_KEYWORD = re.compile(r"(?:releases?|versions?|branch|rc\d*)", re.IGNORECASE)

# Real Kafka release lines (major, minor). A bare x.y.z with no keyword context only counts
# as a Release when its (major, minor) is listed here; this rejects outline numbering like
# "4.6.4". MUST be extended when a new release line ships.
KAFKA_RELEASE_LINES = frozenset(
    {(0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1)}
    | {(2, n) for n in range(0, 9)}
    | {(3, n) for n in range(0, 10)}
    | {(4, 0), (4, 1), (4, 2)}
)

# Names of third-party software whose version numbers collide with Kafka lines (ZooKeeper
# 3.4.x, Scala 2.x, ...). A version preceded by one of these (within 2 words) is not a Release.
THIRD_PARTY_NAMES = frozenset(
    "zookeeper zk scala java jdk jre jvm gradle maven jackson netty jetty log4j log4j2 slf4j "
    "rocksdb junit mockito plugin python docker ubuntu debian centos spring guava "
    "protobuf grpc zstd lz4 snappy".split()
)

_MAX_WORDS_BETWEEN = 3
_WINDOW = _MAX_WORDS_BETWEEN + 1  # keyword may be this many tokens away (3 between)


def _is_keyword(token: str) -> bool:
    return _KEYWORD.fullmatch(token.strip(".,;:!?()[]{}<>\"'`*_")) is not None


def _after_third_party_name(text: str, start: int) -> bool:
    """True if one of the 2 words before `start` is (or is hyphen-joined to) a third-party name."""
    for token in text[max(0, start - 200) : start].split()[-2:]:
        for part in re.split(r"[-_/]", token.strip(".,;:!?()[]{}<>\"'`*")):
            if part.lower() in THIRD_PARTY_NAMES:
                return True
    return False


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
        major, minor = m.group(0).split(".")[:2]
        known_line = bool(m.group(1)) and (int(major), int(minor)) in KAFKA_RELEASE_LINES
        if _after_third_party_name(text, m.start()):
            continue
        if known_line or _has_release_context(text, m.start(), m.end()):
            found.add((m.start(), m.end(), "Release", m.group(0)))
    return [IdMention(kind=k, id=i, span=(s, e)) for s, e, k, i in sorted(found)]
