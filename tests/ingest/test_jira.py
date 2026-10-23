from datetime import UTC, datetime
from pathlib import Path

import pytest

from kgrag.ingest.jira import JiraEvent, parse_jira_notification, render_event
from kgrag.ingest.parse import RawEmail, parse_mbox

SAMPLES = Path(__file__).parent.parent / "fixtures" / "jira_samples"


def load(name: str, tmp_path: Path) -> RawEmail:
    """Parse a sample .eml through the real mbox parser."""
    mbox = tmp_path / f"{name}.mbox"
    mbox.write_bytes(b"From jira@apache.org Tue Mar  5 10:00:00 2024\n" + (SAMPLES / f"{name}.eml").read_bytes() + b"\n")
    (email,) = list(parse_mbox(mbox))
    return email


# (sample, ticket, action, actor)
CASES = [
    ("created", "KAFKA-16318", "created", "Mickael Maison"),
    ("resolved", "KAFKA-8090", "resolved", "Manikumar"),
    ("resolved_no_fix_version", "KAFKA-16071", "resolved", "Luke Chen"),
    ("reopened", "KAFKA-4600", "reopened", "Braedon Vickers"),
    ("assigned", "KAFKA-15001", "assigned", "Bob Jones"),
    ("commented", "KAFKA-15002", "commented", "Carol Lee"),
    ("updated", "KAFKA-15003", "updated", "Dan Wu"),
    ("closed", "KAFKA-15004", "closed", "Erin Park"),
]


@pytest.mark.parametrize("name,ticket,action,actor", CASES)
def test_parse_sample(name, ticket, action, actor, tmp_path):
    raw = load(name, tmp_path)
    ev = parse_jira_notification(raw, "email-1")
    assert ev is not None
    assert ev.ticket == ticket
    assert ev.action == action
    assert ev.actor == actor
    assert "(Jira)" not in ev.actor and "(JIRA)" not in ev.actor
    assert ev.source_email_id == "email-1"
    assert ev.date == raw.date


def test_title_collapses_whitespace(tmp_path):
    ev = parse_jira_notification(load("resolved", tmp_path), "e")
    assert ev.title == "Replace ControlledShutdown request/response with automated protocol"


def test_resolved_status_and_fix_version(tmp_path):
    ev = parse_jira_notification(load("resolved", tmp_path), "e")
    assert ev.status == "Fixed"
    assert ev.fix_version == "2.3.0"


def test_resolved_without_fix_version(tmp_path):
    ev = parse_jira_notification(load("resolved_no_fix_version", tmp_path), "e")
    assert ev.status == "Fixed"
    assert ev.fix_version is None


def test_created_has_no_status(tmp_path):
    ev = parse_jira_notification(load("created", tmp_path), "e")
    assert ev.status is None and ev.fix_version is None


def test_closed_status_from_body(tmp_path):
    ev = parse_jira_notification(load("closed", tmp_path), "e")
    assert ev.status == "Won't Fix"


def test_first_fix_version_taken(tmp_path):
    raw = load("resolved", tmp_path)
    raw = raw.model_copy(update={"body": raw.body.replace("Fix Version/s: 2.3.0", "Fix Version/s: 2.3.0, 2.2.1")})
    assert parse_jira_notification(raw, "e").fix_version == "2.3.0"


def test_created_description_fields_ignored(tmp_path):
    raw = load("created", tmp_path)
    body = raw.body.replace("KafkaMetric is part", "Resolution: Fixed\nFix Version/s: 9.9.9\nKafkaMetric is part")
    assert body != raw.body
    ev = parse_jira_notification(raw.model_copy(update={"body": body}), "e")
    assert ev.status is None and ev.fix_version is None


def test_comment_text_fields_ignored(tmp_path):
    raw = load("resolved", tmp_path)
    body = raw.body.replace("Issue resolved by pull request", "Fix Version/s: 9.9.9\nResolution: Duplicate\n\nIssue resolved by pull request")
    assert body != raw.body
    ev = parse_jira_notification(raw.model_copy(update={"body": body}), "e")
    assert ev.status == "Fixed" and ev.fix_version == "2.3.0"


def test_unknown_subject_returns_none(tmp_path):
    assert parse_jira_notification(load("unknown_comment_edited", tmp_path), "e") is None


def test_non_jira_subject_returns_none(tmp_path):
    raw = load("created", tmp_path).model_copy(update={"subject": "[DISCUSS] KIP-1000"})
    assert parse_jira_notification(raw, "e") is None


def test_action_is_case_insensitive(tmp_path):
    raw = load("created", tmp_path)
    raw = raw.model_copy(update={"subject": raw.subject.replace("[Created]", "[CREATED]")})
    assert parse_jira_notification(raw, "e").action == "created"


def test_render_event():
    e = JiraEvent(
        ticket="KAFKA-1234", action="resolved", actor="Alice Smith",
        date=datetime(2024, 3, 5, 10, 0, tzinfo=UTC), status="Fixed", fix_version="3.7.0",
        source_email_id="x", title="Fix consumer rebalance",
    )
    assert render_event(e) == "Alice Smith resolved KAFKA-1234 (Fix consumer rebalance) as Fixed in 3.7.0 on 2024-03-05."


def test_render_event_optional_clauses():
    base = dict(
        ticket="KAFKA-1234", action="created", actor="Alice Smith",
        date=datetime(2024, 3, 5, 10, 0, tzinfo=UTC), source_email_id="x", title="T",
    )
    assert render_event(JiraEvent(**base, status=None, fix_version=None)) == "Alice Smith created KAFKA-1234 (T) on 2024-03-05."
    assert render_event(JiraEvent(**base, status=None, fix_version="3.7.0")) == "Alice Smith created KAFKA-1234 (T) in 3.7.0 on 2024-03-05."
    assert render_event(JiraEvent(**base, status="Fixed", fix_version=None)) == "Alice Smith created KAFKA-1234 (T) as Fixed on 2024-03-05."


def test_fix_version_from_quoted_summary_fix_for(tmp_path):
    ev = parse_jira_notification(load("resolved_fix_for_only", tmp_path), "e")
    assert ev.status == "Fixed"
    assert ev.fix_version == "2.0.2"


def test_fix_for_in_description_text_ignored(tmp_path):
    raw = load("resolved_fix_for_only", tmp_path)
    body = raw.body.replace("Fix For: 2.0.2, 2.3.0, 2.1.2, 2.2.1", "Priority: Major")
    body += "\n> Some description\n>   Fix For: 9.9.9\n"
    assert parse_jira_notification(raw.model_copy(update={"body": body}), "e").fix_version is None
