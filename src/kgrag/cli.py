import random
import sys
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
from kgrag.store.db import apply_migrations, connect
from kgrag.store.repo import select_unextracted
from kgrag.tokens import count_tokens

app = typer.Typer(help="Kafka Atlas: knowledge-graph RAG over the Kafka mailing lists.", no_args_is_help=True)


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
