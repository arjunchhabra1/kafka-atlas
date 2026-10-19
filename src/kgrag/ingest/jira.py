"""Parse automated Jira notification emails into structured events.

Observed subject form on dev@: ``[jira] [Action] (KAFKA-N) Title``. Only Created, Resolved
and Reopened actually reach the list; the other actions are handled for completeness.
"""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from kgrag.ingest.parse import RawEmail

Action = Literal["created", "assigned", "resolved", "commented", "updated", "reopened", "closed"]

_ACTIONS: dict[str, Action] = {
    a: a for a in ("created", "assigned", "resolved", "commented", "updated", "reopened", "closed")
}
_SUBJECT = re.compile(r"^\s*\[jira\]\s*\[([^\]]+)\]\s*\((KAFKA-\d+)\)\s*(.*)$", re.IGNORECASE)
_SENDER_SUFFIX = re.compile(r"\s*\(jira\)\s*$", re.IGNORECASE)
_RESOLUTION = re.compile(r"^[ \t]*Resolution:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_FIX_VERSION = re.compile(r"^[ \t]*Fix Version/s:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


class JiraEvent(BaseModel):
    ticket: str
    action: Action
    actor: str
    date: datetime
    status: str | None
    fix_version: str | None
    source_email_id: str
    title: str


def _header_block(body: str) -> str:
    """Lines before the quoted issue summary (first line starting with '>')."""
    lines = []
    for line in body.splitlines():
        if line.startswith(">"):
            break
        lines.append(line)
    return "\n".join(lines)


def parse_jira_notification(email: RawEmail, email_id: str) -> JiraEvent | None:
    m = _SUBJECT.match(email.subject)
    if not m:
        return None
    action = _ACTIONS.get(m.group(1).strip().lower())
    if action is None:
        return None
    head = _header_block(email.body)
    status = _RESOLUTION.search(head)
    fix = _FIX_VERSION.search(head)
    return JiraEvent(
        ticket=m.group(2),
        action=action,
        actor=_SENDER_SUFFIX.sub("", email.sender_name).strip(),
        date=email.date,
        status=status.group(1) if status else None,
        fix_version=fix.group(1).split(",")[0].strip() if fix else None,
        source_email_id=email_id,
        title=m.group(3).strip(),
    )


def render_event(e: JiraEvent) -> str:
    text = f"{e.actor} {e.action} {e.ticket} ({e.title})"
    if e.status:
        text += f" as {e.status}"
    if e.fix_version:
        text += f" in {e.fix_version}"
    return f"{text} on {e.date.date().isoformat()}."
