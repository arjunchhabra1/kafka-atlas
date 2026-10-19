"""LLM extraction: request building, response parsing, batch runner, cache."""

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
# Multiplier on the estimate: covers thinking tokens, API header tokens and tokenizer
# differences (we count with cl100k_base). Task 8's pilot calibrates it from stored usage.
COST_SAFETY_MARGIN = 1.3
RETRYABLE_BATCH_ERRORS = frozenset({"api_error", "overloaded_error", "rate_limit_error"})
log = logging.getLogger(__name__)
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
    if hasattr(message, "to_dict"):
        return message.to_dict()
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
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
                confidence=min(1.0, max(0.0, r.confidence)),
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
    cost *= COST_SAFETY_MARGIN
    return cost * s.batch_discount if batch else cost


@traced("extract")
def _create(client: Any, params: dict[str, Any]) -> Any:
    return client.messages.create(**params)


@traced("extract")
def _trace_batch(batch_id: str, n_requests: int) -> None:
    """Langfuse observation carrying only batch metadata, not request bodies."""


def _batch_dir() -> Path:
    return get_settings().data_dir / "batches"


def _batch_path(batch_id: str) -> Path:
    return _batch_dir() / f"{batch_id}.json"


def _error_type(result: Any) -> str | None:
    err = getattr(result, "error", None)
    inner = getattr(err, "error", None)
    return getattr(inner, "type", None) or getattr(err, "type", None)


def _store(conn: psycopg.Connection, email_id: str, model: str, message: dict[str, Any]) -> None:
    conn.execute(
        "insert into llm_cache (email_id, prompt_version, model, response) "
        "values (%s, %s, %s, %s) on conflict (email_id, prompt_version, model) "
        "do update set response = excluded.response",
        (email_id, PROMPT_VERSION, model, Jsonb(message)),
    )


def _thread_subjects(conn: psycopg.Connection, emails: list[EmailRow]) -> dict[str, str]:
    """threads.subject from the DB; fall back to the earliest given email's subject."""
    ids = sorted({e.thread_id for e in emails})
    subjects = {
        tid: subj
        for tid, subj in conn.execute(
            "select thread_id, subject from threads where thread_id = any(%s)", (ids,)
        )
        if subj is not None
    }
    first: dict[str, EmailRow] = {}
    for e in emails:
        if e.thread_id not in first or e.date < first[e.thread_id].date:
            first[e.thread_id] = e
    return {t: subjects.get(t, first[t].subject) for t in ids}


def run_extraction(
    conn: psycopg.Connection,
    emails: list[EmailRow],
    client: Any,
    *,
    batch: bool = True,
    poll_interval: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    resume_batch_id: str | None = None,
    max_sync_retries: int | None = None,
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

    subjects = _thread_subjects(conn, todo)
    params = {e.id: build_request(e, subjects[e.thread_id]) for e in todo}
    by_id = {e.id: e for e in todo}

    def accept(email_id: str, msg: Any) -> bool:
        """Validate and cache one response; False when it is unparseable."""
        dumped = _as_dict(msg)
        try:
            parse_response(email_id, dumped)
        except ExtractionError:
            return False
        _store(conn, email_id, model, dumped)
        stats.done += 1
        return True

    def fail(email_id: str, error: str) -> None:
        log_failure(conn, "extract", email_id, error)
        stats.failed += 1

    retry: list[EmailRow] = []
    if batch:
        if resume_batch_id:
            batch_id = resume_batch_id
            path = _batch_path(batch_id)
            if not path.exists():
                raise FileNotFoundError(f"no custom_id mapping for batch {batch_id}: {path}")
            mapping: dict[str, str] = json.loads(path.read_text())
        else:
            # custom_id must match [A-Za-z0-9_-]{1,64}; email ids may not, so use the index.
            mapping = {f"e{i}": e.id for i, e in enumerate(todo)}
            requests = [{"custom_id": cid, "params": params[eid]} for cid, eid in mapping.items()]
            batch_id = client.messages.batches.create(requests=requests).id
            _batch_dir().mkdir(parents=True, exist_ok=True)
            _batch_path(batch_id).write_text(json.dumps(mapping))
            log.info("submitted extraction batch %s (%d requests)", batch_id, len(requests))
        _trace_batch(batch_id, len(mapping))
        while client.messages.batches.retrieve(batch_id).processing_status != "ended":
            sleep(poll_interval)

        seen: set[str] = set()
        for item in client.messages.batches.results(batch_id):
            eid = mapping.get(item.custom_id)
            if eid not in by_id:
                continue
            seen.add(eid)
            res = item.result
            if res.type == "succeeded":
                if not accept(eid, res.message):
                    retry.append(by_id[eid])
            elif res.type == "errored" and _error_type(res) not in RETRYABLE_BATCH_ERRORS:
                fail(eid, f"batch item errored: {_error_type(res)}")
            else:  # expired, canceled, or retryable error
                retry.append(by_id[eid])
        retry += [e for e in todo if e.id not in seen and e.id in mapping.values()]
        conn.commit()  # paid results are durable before any retry can raise
        cap = max_sync_retries
        if cap is None:
            cap = max(20, math.ceil(0.02 * len(todo)))
        for e in retry[cap:]:
            fail(e.id, "retry cap reached")
        retry = retry[:cap]
    else:
        retry = todo

    for e in retry:
        try:
            msg = _create(client, params[e.id])
            if not accept(e.id, msg):
                fail(e.id, "ExtractionError: unparseable response after retry")
        except Exception as exc:  # noqa: BLE001 - any API error is a per-item failure
            fail(e.id, f"{type(exc).__name__}: {exc}")
        conn.commit()
    conn.commit()
    return stats
