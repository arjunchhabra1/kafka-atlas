import typer

from kgrag.config import get_settings
from kgrag.ingest.fetch import fetch_month, month_range
from kgrag.ingest.pipeline import ingest as run_ingest
from kgrag.store.db import apply_migrations, connect

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
