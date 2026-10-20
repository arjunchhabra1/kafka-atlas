import random
import sys
import time
from collections import Counter
from typing import Any

import psycopg
import typer

from kgrag.config import get_settings
from kgrag.extract.llm import (
    PROMPT_VERSION,
    batch_email_ids,
    estimate_cost_usd,
    parse_response,
    run_extraction,
)
from kgrag.extract.pipeline import persist_extractions
from kgrag.ingest.fetch import fetch_month, month_range
from kgrag.ingest.pipeline import ingest as run_ingest
from kgrag.questions.draft import avg_doc_tokens, draft_questions, estimate_draft_cost_usd
from kgrag.questions.review import review_session
from kgrag.questions.split import AlreadySplitError, persist_splits
from kgrag.store.db import apply_migrations, connect
from kgrag.resolve.exact_match import ExactMatchResolver
from kgrag.resolve.interface import resolve_all
from kgrag.resolve.resolver import UserResolver
from kgrag.store.repo import (
    load_anchor_hints,
    load_mentions,
    save_resolution,
    select_unextracted,
)
from kgrag.tokens import count_tokens

app = typer.Typer(help="Kafka Atlas: knowledge-graph RAG over the Kafka mailing lists.", no_args_is_help=True)
questions_app = typer.Typer(help="Build the evaluation question set.", no_args_is_help=True)
app.add_typer(questions_app, name="questions")


@app.callback()
def main() -> None:
    """Kafka Atlas command line."""


@app.command()
def ingest(
    start: str = typer.Option(..., "--from", help="First month, YYYY-MM."),
    end: str = typer.Option(..., "--to", help="Last month, YYYY-MM."),
) -> None:
    """Download the monthly archives in the range and ingest exactly those files."""
    settings = get_settings()
    paths = [fetch_month(m, settings.data_dir / "raw") for m in month_range(start, end)]
    conn = connect()
    try:
        apply_migrations(conn)
        stats = run_ingest(conn, paths)
    finally:
        conn.close()
    typer.echo(stats)


def _make_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def _show(conn: psycopg.Connection, n: int) -> None:
    model = get_settings().models.extract
    rows = conn.execute(
        "select c.email_id, c.response, e.subject from llm_cache c "
        "left join emails e on e.id = c.email_id "
        "where c.prompt_version = %s and c.model = %s order by c.email_id",
        (PROMPT_VERSION, model),
    ).fetchall()
    for email_id, response, subject in random.Random(0).sample(rows, min(n, len(rows))):
        result = parse_response(email_id, response)
        surface = {m.mention_id: m.surface for m in result.mentions}
        typer.echo(f"=== {email_id}: {subject}")
        for m in result.mentions:
            typer.echo(f"  [{m.type}] {m.surface!r}  <- {m.context}")
        for r in result.relations:
            extra = f" vote={r.vote_value} binding={r.binding}" if r.type == "VOTED" else ""
            typer.echo(
                f"  {surface[r.source_mention_id]} -{r.type}-> "
                f"{surface[r.target_mention_id]} (conf {r.confidence:.2f}){extra}"
            )


@app.command()
def extract(
    limit: int | None = typer.Option(None, help="Extract at most this many emails."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-guard confirmation."),
    resume_batch: str | None = typer.Option(None, "--resume-batch", help="Resume a batch id."),
    max_sync_retries: int | None = typer.Option(None, help="Cap on synchronous retries."),
    show: int | None = typer.Option(None, help="Print N cached results and exit (no API)."),
    sync: bool = typer.Option(False, "--sync", help="Use synchronous calls, not the Batch API."),
) -> None:
    """Run LLM extraction on emails not yet cached, then rebuild mentions and relations."""
    settings = get_settings()
    conn = connect()
    try:
        apply_migrations(conn)
        if show is not None:
            _show(conn, show)
            return
        model = settings.models.extract
        emails = select_unextracted(conn, PROMPT_VERSION, model, limit)
        if emails:
            avg = round(sum(count_tokens(e.body_clean or "") for e in emails) / len(emails))
            cost = estimate_cost_usd(len(emails), avg, model, batch=not sync)
            typer.echo(
                f"{len(emails)} emails, avg {avg} tokens; estimated cost ${cost:.2f} "
                f"(guard ${settings.cost_guard_usd:.2f})"
            )
            if cost > settings.cost_guard_usd and not yes:
                if not sys.stdin.isatty():
                    typer.echo("Estimated cost exceeds the guard; pass --yes to proceed.", err=True)
                    raise typer.Exit(2)
                if not typer.confirm("Proceed?", default=False):
                    raise typer.Exit(1)
            if resume_batch:
                outside = len({e.id for e in emails} - batch_email_ids(resume_batch))
                typer.echo(
                    f"{outside} selected emails were not part of batch {resume_batch} "
                    "and will not be extracted by this run"
                )
            stats = run_extraction(
                conn, emails, _make_client(), batch=not sync,
                resume_batch_id=resume_batch, max_sync_retries=max_sync_retries,
            )
            typer.echo(stats)
        else:
            typer.echo("No emails need extraction.")
        typer.echo(persist_extractions(conn))
    finally:
        conn.close()


@app.command()
def resolve(
    resolver: str = typer.Option("exact", help="Resolver to run: exact or user."),
) -> None:
    """Resolve all mentions into entities and store the result (replacing that resolver's rows)."""
    resolvers = {"exact": ExactMatchResolver, "user": UserResolver}
    if resolver not in resolvers:
        raise typer.BadParameter(f"resolver must be one of {sorted(resolvers)}")
    conn = connect()
    try:
        apply_migrations(conn)
        mentions, hints = load_mentions(conn), load_anchor_hints(conn)
        start = time.perf_counter()
        try:
            result = resolve_all(resolvers[resolver](), mentions, hints)
        except NotImplementedError:
            typer.echo("UserResolver not implemented yet — see src/kgrag/resolve/resolver.py")
            raise typer.Exit(1) from None
        elapsed = time.perf_counter() - start
        save_resolution(conn, resolver, result)
    finally:
        conn.close()
    typer.echo(f"{resolver}: {len(mentions)} mentions -> {len(result.canonical_names)} entities "
               f"in {elapsed:.1f}s")
    for etype, n in sorted(Counter(result.entity_types.values()).items()):
        typer.echo(f"  {etype}: {n}")


QUESTION_TYPES = ("single_hop", "inference", "comparison", "temporal", "null")
DEFAULT_OVERSAMPLE = 1.3  # draft extra to absorb the human-review drop rate


def _parse_counts(spec: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        for part in spec.split(","):
            name, value = part.split("=")
            name = name.strip()
            if name in counts:
                raise typer.BadParameter(f"duplicate question type: {name}")
            counts[name] = int(value)
            if counts[name] < 0:
                raise typer.BadParameter(f"negative count for {name}")
    except ValueError:
        raise typer.BadParameter("expected type=N,type=N,...") from None
    unknown = set(counts) - set(QUESTION_TYPES)
    if unknown:
        raise typer.BadParameter(f"unknown question types: {sorted(unknown)}")
    return counts


@questions_app.command("draft")
def questions_draft(
    pilot: bool = typer.Option(False, "--pilot", help="Draft 6 questions per type."),
    counts: str | None = typer.Option(None, help="e.g. single_hop=100,inference=55,null=40"),
    seed: int = typer.Option(13, help="Sampling seed."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-guard confirmation."),
) -> None:
    """Sample evidence from Postgres and draft questions with an LLM (status 'draft')."""
    if pilot == (counts is not None):
        raise typer.BadParameter("pass exactly one of --pilot or --counts")
    wanted = dict.fromkeys(QUESTION_TYPES, 6) if pilot else _parse_counts(counts or "")
    oversample = 1.0 if pilot else DEFAULT_OVERSAMPLE
    settings = get_settings()
    conn = connect()
    try:
        apply_migrations(conn)
        model = settings.models.draft
        avg = avg_doc_tokens(conn)
        expected = estimate_draft_cost_usd(wanted, avg, model, oversample=oversample)
        cost = estimate_draft_cost_usd(wanted, avg, model, oversample=oversample, worst_case=True)
        typer.echo(
            f"{sum(wanted.values())} questions (x{oversample} oversample), avg doc {avg} tokens; "
            f"expected estimated cost ${expected:.2f}, worst-case ${cost:.2f} "
            f"(guard ${settings.cost_guard_usd:.2f} applies to the worst case)"
        )
        if cost > settings.cost_guard_usd and not yes:
            if not sys.stdin.isatty():
                typer.echo("Estimated cost exceeds the guard; pass --yes to proceed.", err=True)
                raise typer.Exit(2)
            if not typer.confirm("Proceed?", default=False):
                raise typer.Exit(1)
        stats = draft_questions(conn, wanted, seed=seed, oversample=oversample)
        typer.echo(stats)
        if stats.failed and not stats.drafted:
            typer.echo("Every draft attempt failed; see the failures table.", err=True)
            raise typer.Exit(1)
    finally:
        conn.close()


@questions_app.command("review")
def questions_review(
    qtype: str | None = typer.Option(None, "--type", help="Only review this question type."),
) -> None:
    """Review drafts one at a time: keep, edit, drop or skip. Every decision is saved at once."""
    if qtype is not None and qtype not in QUESTION_TYPES:
        raise typer.BadParameter(f"unknown question type: {qtype}")
    conn = connect()
    try:
        apply_migrations(conn)
        review_session(conn, qtype=qtype)
    finally:
        conn.close()


@questions_app.command("split")
def questions_split(
    force: bool = typer.Option(False, "--force", help="Reassign even if splits already exist."),
) -> None:
    """Assign kept/edited questions to dev (20%) and test (80%), stratified by type."""
    conn = connect()
    try:
        apply_migrations(conn)
        try:
            counts = persist_splits(conn, force=force)
        except AlreadySplitError:
            typer.echo("Kept questions already have a split; pass --force to reassign.", err=True)
            raise typer.Exit(1) from None
        for qtype, c in counts.items():
            typer.echo(f"{qtype}: dev {c['dev']}, test {c['test']}")
        remaining = conn.execute("select count(*) from questions where status = 'draft'").fetchone()[0]
        if remaining:
            typer.echo(f"warning: {remaining} draft(s) not yet reviewed were ignored", err=True)
    finally:
        conn.close()
