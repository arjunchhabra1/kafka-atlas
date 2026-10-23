import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
import typer

from kgrag.config import get_settings, load_env
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
from kgrag.questions.draft import (
    attempts_per_kept,
    avg_doc_tokens,
    draft_questions,
    estimate_draft_costs,
    format_cost_line,
)
from kgrag.questions.review import review_session
from kgrag.questions.split import AlreadySplitError, persist_splits
from kgrag.graph.build import build_graph
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
    load_env()


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
        result = resolve_all(resolvers[resolver](), mentions, hints)
        elapsed = time.perf_counter() - start
        save_resolution(conn, resolver, result)
        conn.commit()  # save_resolution joins the open transaction; close() would roll it back
    finally:
        conn.close()
    typer.echo(f"{resolver}: {len(mentions)} mentions -> {len(result.canonical_names)} entities "
               f"in {elapsed:.1f}s")
    for etype, n in sorted(Counter(result.entity_types.values()).items()):
        typer.echo(f"  {etype}: {n}")


def _neo4j_driver() -> Any:
    from neo4j import GraphDatabase

    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _guard_wipe(driver: Any, uri: str, force: bool, yes: bool) -> None:
    """`build` clears the whole database: show the target and refuse unsafe wipes."""
    from urllib.parse import urlparse

    from kgrag.graph.build import read_graph_meta

    meta = read_graph_meta(driver)
    with driver.session() as s:
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    typer.echo(f"target: {uri}; current graph: "
               + (f"built from '{meta.resolver}' at {meta.built_at:%Y-%m-%d %H:%M}, "
                  f"{meta.node_count} nodes" if meta else f"no GraphMeta, {nodes} nodes"))
    if nodes and meta is None and not force:
        typer.echo("refusing to wipe a non-empty database that was not built by kgrag; "
                   "pass --force to override")
        raise typer.Exit(2)
    if urlparse(uri).hostname not in {h.strip("[]") for h in LOCAL_HOSTS} and not yes:
        if not sys.stdin.isatty():
            typer.echo("remote Neo4j target: pass --yes to confirm the wipe")
            raise typer.Exit(2)
        typer.confirm(f"Wipe the database at {uri}?", abort=True)


@app.command()
def build(
    resolver: str = typer.Option("exact", help="Stored resolution to build from: exact or user."),
    force: bool = typer.Option(False, "--force", help="Wipe a non-empty database lacking GraphMeta."),
    yes: bool = typer.Option(False, "--yes", help="Confirm wiping a non-local Neo4j."),
) -> None:
    """Clear Neo4j and rebuild the knowledge graph from Postgres using a stored resolution."""
    if resolver not in ("exact", "user"):
        raise typer.BadParameter("resolver must be one of ['exact', 'user']")
    conn, driver = connect(), _neo4j_driver()
    try:
        apply_migrations(conn)
        _guard_wipe(driver, get_settings().neo4j_uri, force, yes)
        start = time.perf_counter()
        try:
            stats = build_graph(conn, driver, resolver)
        except ValueError as e:
            typer.echo(str(e))
            raise typer.Exit(1) from None
        elapsed = time.perf_counter() - start
    finally:
        conn.close()
        driver.close()
    typer.echo(f"graph built from '{resolver}' in {elapsed:.1f}s: "
               f"{sum(stats.nodes.values())} nodes, {sum(stats.rels.values())} relationships")
    for label, n in stats.nodes.items():
        typer.echo(f"  node {label}: {n}")
    for rtype, n in stats.rels.items():
        typer.echo(f"  rel {rtype}: {n}")
    for reason, n in stats.skipped.items():
        typer.echo(f"  skipped {reason}: {n}")
    for w in stats.warnings:
        typer.echo(f"WARNING: {w}")


@app.command()
def index() -> None:
    """Embed documents and graph node names into pgvector (only items not yet indexed)."""
    from kgrag.retrieve.embeddings import EmbeddingsNotConfiguredError, get_embeddings
    from kgrag.retrieve.index import EmbeddingsModelMismatchError, build_index

    try:
        get_embeddings()
    except EmbeddingsNotConfiguredError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    conn, driver = connect(), _neo4j_driver()
    try:
        apply_migrations(conn)
        try:
            stats = build_index(conn, driver)
        except EmbeddingsModelMismatchError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from None
        for name in ("chunks", "node_names"):
            c = getattr(stats, name)
            typer.echo(f"{name}: {c.added} added, {c.updated} updated, {c.deleted} deleted, "
                       f"{c.unchanged} unchanged ({c.total} indexed)")
    finally:
        conn.close()
        driver.close()


@app.command("graph-check")
def graph_check(
    resolver: str = typer.Option("exact", help="Resolver used for the fixture graph."),
    hops: int = typer.Option(3, help="BFS depth for the reachability check."),
    fixture: bool = typer.Option(False, "--fixture", help="Also build the dev-fixture graph in a "
                                 "throwaway Neo4j container and report reachability."),
) -> None:
    """Diagnostic: connectivity of the current graph, and optionally whether traversal reaches gold docs."""
    from kgrag.graph.check import (
        connectivity,
        format_connectivity,
        format_reachability,
        load_dev_questions,
        reachability,
    )

    driver = _neo4j_driver()
    try:
        typer.echo("current graph\n" + format_connectivity(connectivity(driver)))
    finally:
        driver.close()
    if fixture:
        from kgrag.graph.fixture_graph import fixture_graph

        with fixture_graph(resolver) as fx_driver:
            typer.echo("\nfixture graph\n" + format_connectivity(connectivity(fx_driver)))
            typer.echo("\n" + format_reachability(
                reachability(fx_driver, load_dev_questions(), hops)))


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


def _parse_types(spec: str, per_type: int) -> dict[str, int]:
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        raise typer.BadParameter("expected type,type,...")
    unknown = set(names) - set(QUESTION_TYPES)
    if unknown:
        raise typer.BadParameter(f"unknown question types: {sorted(unknown)}")
    if len(set(names)) != len(names):
        raise typer.BadParameter("duplicate question type")
    if per_type < 1:
        raise typer.BadParameter("--per-type must be at least 1")
    return dict.fromkeys(names, per_type)


@questions_app.command("draft")
def questions_draft(
    pilot: bool = typer.Option(False, "--pilot", help="Draft 6 questions per type."),
    counts: str | None = typer.Option(None, help="e.g. single_hop=100,inference=80,null=40"),
    types: str | None = typer.Option(
        None, "--types", help="Targeted pilot of only these types, e.g. inference,temporal "
        "(no oversample)."),
    per_type: int | None = typer.Option(
        None, "--per-type", help="Questions per type with --types (default 6)."),
    seed: int = typer.Option(13, help="Sampling seed."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-guard confirmation."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the cost estimate and exit."),
) -> None:
    """Sample evidence from Postgres and draft questions with an LLM (status 'draft')."""
    if sum((pilot, counts is not None, types is not None)) != 1:
        raise typer.BadParameter("pass exactly one of --pilot, --counts or --types")
    if per_type is not None and types is None:
        raise typer.BadParameter("--per-type needs --types")
    if types is not None:
        wanted = _parse_types(types, 6 if per_type is None else per_type)
    else:
        wanted = dict.fromkeys(QUESTION_TYPES, 6) if pilot else _parse_counts(counts or "")
    oversample = DEFAULT_OVERSAMPLE if counts is not None else 1.0
    settings = get_settings()
    conn = connect()
    try:
        apply_migrations(conn)
        model = settings.models.draft
        avg = avg_doc_tokens(conn)
        factors = attempts_per_kept(conn)
        est = estimate_draft_costs(wanted, avg, model, oversample=oversample, factors=factors)
        typer.echo(
            f"{sum(wanted.values())} questions (x{oversample} oversample), avg doc {avg} tokens; "
            f"attempts per kept: "
            + ", ".join(f"{t} {factors[t]:.2f}" for t in wanted)
            + f"\nestimated cost: {format_cost_line(est)} "
            f"(guard ${settings.cost_guard_usd:.2f} applies to the worst case)"
        )
        if dry_run:
            return
        cost = est.worst
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
    force: bool = typer.Option(False, "--force", help="Keep existing splits; assign only the "
                               "kept/edited questions that have none."),
    reshuffle: bool = typer.Option(False, "--reshuffle", help="Reassign ALL kept/edited "
                                   "questions (may move dev questions into test: leakage)."),
) -> None:
    """Assign kept/edited questions to dev (20%) and test (80%), stratified by type."""
    if force and reshuffle:
        raise typer.BadParameter("pass at most one of --force or --reshuffle")
    conn = connect()
    try:
        apply_migrations(conn)
        if reshuffle:
            typer.echo("warning: --reshuffle reassigns every kept question; questions already "
                       "used for dev tuning may land in test (dev/test leakage), and test runs "
                       "made before are no longer comparable", err=True)
        try:
            counts = persist_splits(conn, force=force, reshuffle=reshuffle)
        except AlreadySplitError:
            typer.echo("Kept questions already have a split; pass --force to assign only the "
                       "unsplit ones (or --reshuffle to reassign all).", err=True)
            raise typer.Exit(1) from None
        for qtype, c in counts.items():
            typer.echo(f"{qtype}: dev {c['dev']}, test {c['test']} ({c['new']} newly assigned)")
        remaining = conn.execute("select count(*) from questions where status = 'draft'").fetchone()[0]
        if remaining:
            typer.echo(f"warning: {remaining} draft(s) not yet reviewed were ignored", err=True)
    finally:
        conn.close()


@app.command("eval")
def eval_cmd(
    system: str = typer.Option(..., "--system", help="vector or graph."),
    split: str = typer.Option(..., "--split", help="dev or test."),
    resolver: str | None = typer.Option(None, "--resolver", help="exact or user (graph only)."),
    graph_config: str | None = typer.Option(
        None, "--graph-config", help="Named graph-retriever config (graph only): "
        "hybrid (default) or graph-only."),
    force: bool = typer.Option(False, "--force", help="Re-run a test split already spent."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-guard confirmation."),
) -> None:
    """Run the evaluation for one system on one split and write results/<run_id>.json."""
    from kgrag.eval.run import (
        EvalPreconditionError,
        estimate_eval_cost_usd,
        load_questions,
        run_eval,
    )
    from kgrag.retrieve.graph_config import UnknownGraphConfigError, get_graph_config

    if system not in ("vector", "graph"):
        raise typer.BadParameter("system must be vector or graph")
    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be dev or test")
    if system == "graph" and resolver not in ("exact", "user"):
        raise typer.BadParameter("--resolver exact|user is required for the graph system")
    if system == "vector" and graph_config is not None:
        raise typer.BadParameter("--graph-config applies to the graph system only")
    if graph_config is not None:
        try:
            get_graph_config(graph_config)
        except UnknownGraphConfigError as e:
            raise typer.BadParameter(str(e)) from None
    settings = get_settings()
    conn = connect()
    driver = _neo4j_driver() if system == "graph" else None
    try:
        apply_migrations(conn)
        try:
            questions = load_questions(conn, split)
            answer_usd, judge_usd = estimate_eval_cost_usd(questions)
        except EvalPreconditionError as e:
            typer.echo(f"cannot run: {e}", err=True)
            raise typer.Exit(1) from None
        cost = answer_usd + judge_usd
        typer.echo(f"{len(questions)} questions; estimated cost ${cost:.2f} (answer "
                   f"{settings.models.answer} ${answer_usd:.2f}, judge {settings.models.judge} "
                   f"${judge_usd:.2f}; guard ${settings.cost_guard_usd:.2f})")
        if cost > settings.cost_guard_usd and not yes:
            if not sys.stdin.isatty():
                typer.echo("Estimated cost exceeds the guard; pass --yes to proceed.", err=True)
                raise typer.Exit(2)
            if not typer.confirm("Proceed?", default=False):
                raise typer.Exit(1)
        try:
            path = run_eval(conn, driver, system=system, split=split,  # type: ignore[arg-type]
                            resolver=resolver, graph_config=graph_config, force=force)
        except EvalPreconditionError as e:
            typer.echo(f"cannot run: {e}", err=True)
            raise typer.Exit(1) from None
    finally:
        conn.close()
        if driver is not None:
            driver.close()
    import json

    header = json.loads(path.read_text())["header"]
    typer.echo(f"wrote {path}: {header['n_questions']} questions, {header['n_errors']} errors")
    if header["degraded"]:
        typer.echo("WARNING: run is degraded (more than 10% of questions errored); it does not "
                   "consume the test-split one-shot and the report ignores it when a "
                   "non-degraded run exists", err=True)


@app.command()
def ablate(
    split: str = typer.Option("dev", "--split", help="Question split: dev (test is refused)."),
    resolver: str = typer.Option(..., "--resolver", help="exact or user (must match the graph)."),
    configs: str = typer.Option("hybrid,graph-only", "--configs",
                                help="Comma-separated graph-config names."),
    budget: int | None = typer.Option(None, "--budget", help="Token budget (default: budget_tokens)."),
    vector: bool = typer.Option(True, "--vector/--no-vector", help="Add the vector baseline column."),
) -> None:
    """Gold-source recall per question type for each graph config (retrieval only, no LLM calls).

    Prints the table and writes results/ablation-<timestamp>.json.
    """
    from kgrag.eval.ablation import run_ablation
    from kgrag.eval.run import EvalPreconditionError
    from kgrag.retrieve.graph_ablation import format_table
    from kgrag.retrieve.graph_config import UnknownGraphConfigError, get_graph_config

    if split == "test":
        typer.echo("refusing --split test: the ablation tunes the retriever, and tuning on the "
                   "test split would leak it. Use --split dev.", err=True)
        raise typer.Exit(2)
    if split != "dev":
        raise typer.BadParameter("split must be dev")
    if resolver not in ("exact", "user"):
        raise typer.BadParameter("resolver must be one of ['exact', 'user']")
    names = list(dict.fromkeys(c.strip() for c in configs.split(",") if c.strip()))
    try:
        chosen = {n: get_graph_config(n) for n in names}
    except UnknownGraphConfigError as e:
        raise typer.BadParameter(str(e)) from None
    if not chosen:
        raise typer.BadParameter("--configs names no graph config")
    budget = budget if budget is not None else get_settings().budget_tokens
    conn, driver = connect(), _neo4j_driver()
    try:
        apply_migrations(conn)
        try:
            path, results = run_ablation(conn, driver, split=split, resolver=resolver,
                                         configs=chosen, budget=budget, vector_baseline=vector)
        except EvalPreconditionError as e:
            typer.echo(f"cannot run: {e}", err=True)
            raise typer.Exit(1) from None
    finally:
        conn.close()
        driver.close()
    typer.echo(f"gold-source recall within {budget} tokens ({split} split, resolver {resolver})")
    typer.echo(format_table(results))
    typer.echo(f"wrote {path}")


@app.command()
def report(
    runs: list[Path] = typer.Argument(..., help="Result files, e.g. results/*.json."),
    trace_examples: int = typer.Option(3, help="Multi-hop disagreement examples to show."),
    split: str = typer.Option("test", "--split", help="Report runs of this split: test or dev."),
) -> None:
    """Print the markdown report for the given results files (judge_check.json and
    ablation-*.json are skipped)."""
    import json

    from kgrag.eval.report import RESULTS_DIR, render_report

    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be dev or test")
    paths = [p for p in runs
             if p.name != "judge_check.json" and not p.name.endswith(".partial.jsonl")
             and not p.name.startswith("ablation-")]
    try:  # render fully first so a failure never leaves a truncated redirect target half-written
        text = render_report(paths, trace_examples=trace_examples, split=split,
                             judge_check_path=RESULTS_DIR / "judge_check.json")
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        typer.echo(f"cannot build report: {e}", err=True)
        raise typer.Exit(1) from None
    typer.echo(text)


@app.command("judge-check")
def judge_check(
    run: str = typer.Option(..., "--run", help="Run id (a file in results/)."),
    n: int = typer.Option(50, help="Number of judged answers to label."),
) -> None:
    """Label a sample of judged answers yourself; prints agreement and Cohen's kappa (resumable)."""
    from kgrag.eval.judge import LABELS, agreement_stats, run_judge_check
    from kgrag.eval.run import RESULTS_DIR

    run_path = RESULTS_DIR / f"{run}.json"
    if not run_path.exists():
        raise typer.BadParameter(f"no such run: {run_path}")
    keys = {"c": "correct", "p": "partial", "i": "incorrect"}

    def ask(item: dict) -> str | None:
        typer.echo(f"\nQuestion: {item['question']}\nGold answer: {item['gold_answer']}\n"
                   f"System answer: {item['answer']}")
        while True:
            key = typer.prompt("[c]orrect / [p]artial / [i]ncorrect / [q]uit").strip().lower()[:1]
            if key == "q":
                return None
            if key in keys:
                return keys[key]

    try:
        state = run_judge_check(run_path, RESULTS_DIR / "judge_check.json", n, ask)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    s = agreement_stats(state["items"])
    typer.echo(f"\nlabelled {s['labelled']} of {s['total']} (labels: {', '.join(LABELS)})")
    if s["labelled"]:
        typer.echo(f"agreement {s['agreement']:.0%}; Cohen's kappa "
                   + ("undefined" if s["kappa"] is None else f"{s['kappa']:.2f}"))


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", envvar="PORT", help="Port to listen on (PORT env honoured)."),
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind (0.0.0.0 to "
                             "listen on every interface, as the container does)."),
) -> None:
    """Run the HTTP API (POST /ask, GET /health)."""
    import uvicorn

    uvicorn.run("kgrag.api.main:app", host=host, port=port)
