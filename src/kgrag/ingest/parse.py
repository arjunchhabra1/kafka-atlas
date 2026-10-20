"""Parse monthly mbox files into RawEmail records; dedupe, classify, thread."""

import email
import mailbox
import re
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from email import policy
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

_ANGLE_ID = re.compile(r"<([^<>]*)>")
_AUTOMATED = re.compile(r"jenkins|github|buildbot|notifications@", re.IGNORECASE)
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}


class RawEmail(BaseModel):
    message_id: str
    in_reply_to: str | None
    references: list[str]
    date: datetime
    sender_name: str
    sender_address: str
    subject: str
    body: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    p.close()
    text = "".join(p.parts)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _part_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_body(msg: Message) -> str:
    plain = html = None
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain" and plain is None:
            plain = _part_text(part)
        elif ctype == "text/html" and html is None:
            html = _part_text(part)
    if plain and plain.strip():
        return plain
    if html and html.strip():
        return _html_to_text(html)
    return plain or ""


def _ids(value: str) -> list[str]:
    found = [i.strip() for i in _ANGLE_ID.findall(value) if i.strip()]
    return found or [t for t in value.split() if t]


def _header(msg: Message, name: str) -> str:
    try:
        return str(msg.get(name, "") or "")
    except Exception:  # malformed header that the policy cannot parse
        return str(msg.get_all(name, [""])[0]) if msg.get_all(name) else ""


def parse_mbox(
    path: Path, on_skip: Callable[[str, str], None] | None = None
) -> Iterator[RawEmail]:
    """Yield one RawEmail per usable message in the mbox.

    Messages without a Message-ID or with an unparseable Date are skipped and
    reported through on_skip(item_id, reason), item_id being "<file>#<index>".
    """
    path = Path(path)
    mbx = mailbox.mbox(
        str(path),
        factory=lambda f: email.message_from_binary_file(f, policy=policy.default),
        create=False,
    )
    try:
        for index, msg in enumerate(mbx):
            item_id = f"{path.name}#{index}"

            def skip(reason: str) -> None:
                if on_skip:
                    on_skip(item_id, reason)

            ids = _ids(_header(msg, "Message-ID"))
            if not ids:
                skip("missing Message-ID")
                continue
            try:
                date = parsedate_to_datetime(_header(msg, "Date"))
            except (TypeError, ValueError, IndexError):
                date = None
            if date is None:
                skip("missing or unparseable Date")
                continue
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)

            name, address = parseaddr(_header(msg, "From"))
            irt = _ids(_header(msg, "In-Reply-To"))
            yield RawEmail(
                message_id=ids[0],
                in_reply_to=irt[-1] if irt else None,
                references=_ids(_header(msg, "References")),
                date=date,
                sender_name=name.strip(),
                sender_address=address.strip().lower(),
                subject=" ".join(_header(msg, "Subject").split()),
                body=_extract_body(msg),
            )
    finally:
        mbx.close()


def dedupe(emails: Iterable[RawEmail]) -> list[RawEmail]:
    """Keep the first occurrence of each message_id."""
    seen: set[str] = set()
    out = []
    for e in emails:
        if e.message_id not in seen:
            seen.add(e.message_id)
            out.append(e)
    return out


def classify_kind(sender_name: str, sender_address: str) -> Literal["human", "jira"]:
    """'jira' for Jira and other automated senders, else 'human'."""
    if (
        sender_name.rstrip().endswith("(Jira)")
        or sender_address.lower().startswith("jira@")
        or _AUTOMATED.search(sender_name)
        or _AUTOMATED.search(sender_address)
    ):
        return "jira"
    return "human"


def assign_parents(emails: list[RawEmail]) -> dict[str, str]:
    """Map message_id -> parent message_id (In-Reply-To, else the last References entry in the corpus)."""
    known = {e.message_id for e in emails}
    parent: dict[str, str] = {}
    for e in emails:
        candidates = [e.in_reply_to] if e.in_reply_to else []
        candidates += reversed(e.references)
        for c in candidates:
            if c in known and c != e.message_id:
                parent[e.message_id] = c
                break
    return parent


def assign_threads(
    emails: list[RawEmail], parent: dict[str, str] | None = None
) -> dict[str, str]:
    """Map message_id -> thread_id (the root message_id).

    The parent is In-Reply-To if present in `emails`, else the last
    References entry present. Messages whose parents are unknown are roots.
    """
    known = {e.message_id for e in emails}
    if parent is None:
        parent = assign_parents(emails)
    threads = {}
    for mid in known:
        path = [mid]
        cur = mid
        while cur in parent and parent[cur] not in path:
            cur = parent[cur]
            path.append(cur)
        if cur in parent:  # parent[cur] is on the path: a reply cycle
            cur = min(path[path.index(parent[cur]) :])
        threads[mid] = cur
    return threads
