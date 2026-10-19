"""LLM extraction: request building, response parsing, batch runner, cache."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from kgrag.config import get_settings
from kgrag.extract.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from kgrag.extract.schema import (
    EXTRACTION_SCHEMA,
    ExtractionResult,
    Mention,
    RawExtraction,
    Relation,
)
from kgrag.observability import traced
from kgrag.store.repo import EmailRow, log_failure
from kgrag.tokens import count_tokens

MAX_TOKENS = 8000
# Assumption: a typical response (a handful of mentions + relations as JSON) is ~600 tokens.
EST_OUTPUT_TOKENS = 600
# Fixed per-request input: system prompt plus the schema the API injects. Computed once.
PROMPT_OVERHEAD_TOKENS = count_tokens(SYSTEM_PROMPT + json.dumps(EXTRACTION_SCHEMA))

__all__ = [
    "PROMPT_VERSION",
    "ExtractionError",
    "ExtractionStats",
    "build_request",
    "estimate_cost_usd",
    "parse_response",
    "run_extraction",
]


class ExtractionError(ValueError):
    """The model response could not be turned into a valid ExtractionResult."""


@dataclass
class ExtractionStats:
    done: int = 0
    cached: int = 0
    failed: int = 0


def build_request(email: EmailRow, thread_subject: str) -> dict[str, Any]:
    user = (
        f"From: {email.sender_name} <{email.sender_address}>\n"
        f"Date: {email.date.isoformat()}\n"
        f"Thread subject: {thread_subject}\n"
        f"Subject: {email.subject}\n\n"
        f"{email.body_clean}"
    )
    return {
        "model": get_settings().models.extract,
        "max_tokens": MAX_TOKENS,
        "system": [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
            "effort": "low",
        },
    }


def _as_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message


def parse_response(email_id: str, message: Any) -> ExtractionResult:
    msg = _as_dict(message)
    stop = msg.get("stop_reason")
    if stop in ("refusal", "max_tokens"):
        raise ExtractionError(f"stop_reason={stop}")
    text = next(
        (b.get("text") for b in msg.get("content") or [] if b.get("type") == "text"), None
    )
    if text is None:
        raise ExtractionError("no text block in response")
    try:
        raw = RawExtraction.model_validate_json(text)
    except ValidationError as e:
        raise ExtractionError(f"schema violation: {e}") from e

    ids: dict[str, str] = {}
    for m in raw.mentions:
        if m.local_id in ids:
            raise ExtractionError(f"duplicate mention id {m.local_id}")
        ids[m.local_id] = f"{email_id}:m{len(ids) + 1}"
    for r in raw.relations:
        for end in (r.source, r.target):
            if end not in ids:
                raise ExtractionError(f"relation references unknown mention {end}")
    return ExtractionResult(
        email_id=email_id,
        mentions=[
            Mention(
                mention_id=ids[m.local_id],
                surface=m.surface,
                type=m.type,
                email_id=email_id,
                context=m.context,
            )
            for m in raw.mentions
        ],
        relations=[
            Relation(
                type=r.type,
                source_mention_id=ids[r.source],
                target_mention_id=ids[r.target],
                vote_value=r.vote_value,
                binding=r.binding,
                confidence=r.confidence,
            )
            for r in raw.relations
        ],
    )


def estimate_cost_usd(n_emails: int, avg_tokens: int, model: str, *, batch: bool = True) -> float:
    s = get_settings()
    price = s.pricing[model]
    cost = (
        n_emails * (avg_tokens + PROMPT_OVERHEAD_TOKENS) * price.input
        + n_emails * EST_OUTPUT_TOKENS * price.output
    ) / 1_000_000
    return cost * s.batch_discount if batch else cost


@traced("extract")
def _create(client: Any, params: dict[str, Any]) -> Any:
    return client.messages.create(**params)


@traced("extract")
def _run_batch(
    client: Any, requests: list[dict[str, Any]], poll_interval: float, sleep: Callable[[float], None]
) -> dict[str, Any]:
    """Submit, poll until ended, return {custom_id: message} for succeeded items."""
    batch = client.messages.batches.create(requests=requests)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        sleep(poll_interval)
    out: dict[str, Any] = {}
    for item in client.messages.batches.results(batch.id):
        if item.result.type == "succeeded":
            out[item.custom_id] = item.result.message
    return out


def _store(conn: psycopg.Connection, email_id: str, model: str, message: dict[str, Any]) -> None:
    conn.execute(
        "insert into llm_cache (email_id, prompt_version, model, response) "
        "values (%s, %s, %s, %s) on conflict (email_id, prompt_version, model) "
        "do update set response = excluded.response",
        (email_id, PROMPT_VERSION, model, Jsonb(message)),
    )


def _thread_subjects(emails: list[EmailRow]) -> dict[str, str]:
    """Thread subject = subject of the earliest email of the thread among those given."""
    first: dict[str, EmailRow] = {}
    for e in emails:
        if e.thread_id not in first or e.date < first[e.thread_id].date:
            first[e.thread_id] = e
    return {t: e.subject for t, e in first.items()}


def run_extraction(
    conn: psycopg.Connection,
    emails: list[EmailRow],
    client: Any,
    *,
    batch: bool = True,
    poll_interval: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ExtractionStats:
    model = get_settings().models.extract
    stats = ExtractionStats()
    have = {
        row[0]
        for row in conn.execute(
            "select email_id from llm_cache where prompt_version = %s and model = %s",
            (PROMPT_VERSION, model),
        )
    }
    todo = [e for e in emails if e.id not in have]
    stats.cached = len(emails) - len(todo)
    if not todo:
        return stats

    subjects = _thread_subjects(emails)
    params = {e.id: build_request(e, subjects[e.thread_id]) for e in todo}

    # email id -> validated raw message dict
    good: dict[str, dict[str, Any]] = {}
    retry: list[EmailRow] = []
    if batch:
        # custom_id must match [A-Za-z0-9_-]{1,64}; email ids may not, so use the index.
        requests = [{"custom_id": f"e{i}", "params": params[e.id]} for i, e in enumerate(todo)]
        results = _run_batch(client, requests, poll_interval, sleep)
        for i, e in enumerate(todo):
            msg = results.get(f"e{i}")
            try:
                if msg is None:
                    raise ExtractionError("batch item did not succeed")
                dumped = _as_dict(msg)
                parse_response(e.id, dumped)
                good[e.id] = dumped
            except ExtractionError:
                retry.append(e)
    else:
        retry = todo

    for e in retry:
        try:
            dumped = _as_dict(_create(client, params[e.id]))
            parse_response(e.id, dumped)
            good[e.id] = dumped
        except Exception as exc:  # noqa: BLE001 - any API/parse error is a per-item failure
            log_failure(conn, "extract", e.id, f"{type(exc).__name__}: {exc}")
            stats.failed += 1

    for eid, dumped in good.items():
        _store(conn, eid, model, dumped)
    stats.done = len(good)
    conn.commit()
    return stats
