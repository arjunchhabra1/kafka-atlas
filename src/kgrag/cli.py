import typer

app = typer.Typer(help="Kafka Atlas: knowledge-graph RAG over the Kafka mailing lists.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Kafka Atlas command line."""
