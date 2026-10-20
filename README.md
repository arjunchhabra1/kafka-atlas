# Kafka Atlas

An engineering knowledge assistant over the Apache Kafka developer mailing list, built to answer one question: **does a knowledge graph retrieve better than plain vector search when an answer spans several documents?**

The corpus is three years (2023–2025) of `dev@kafka.apache.org` plus its Jira activity. It is a public stand-in for the Slack, email and ticket history that companies index internally. Two systems answer the same questions with the same model, prompt and context budget, so retrieval is the only thing that differs:

- **Vector RAG baseline:** LangChain + pgvector, nearest-neighbour search over email chunks.
- **Graph RAG:** an LLM-extracted knowledge graph in Neo4j, queried through entity resolution and multi-hop graph traversal.

**Stack:** Python 3.12, Claude (Haiku / Sonnet / Opus 5.5), Voyage embeddings, Postgres + pgvector, Neo4j, LangChain, FastAPI, Docker, GCP (Cloud Run, Cloud SQL), Langfuse, GitHub Actions.

## Status

| Area | State |
| --- | --- |
| Ingestion: download, parse, clean, thread, chunk | Done, run on the full corpus |
| Jira notification parsing into structured events | Done |
| Regex ID extraction (KIPs, tickets, releases) | Done |
| LLM extraction: batch runner, cache, cost guard | Done; pilot comparing Haiku vs Sonnet in progress |
| Answer chain shared by both systems | Done |
| Question sampling, drafting and review CLI | Done |
| Metrics with bootstrap confidence intervals | Done |
| Knowledge graph build (Neo4j) | Next |
| Entity resolution and graph retrieval | Next |
| Vector baseline index | Next (needs embedding key) |
| Evaluation runs and results | Planned |
| FastAPI service on Cloud Run with Langfuse | Planned |

259 automated tests, including Postgres and Neo4j integration tests in Docker, run in CI on every push.

## Data so far

| Measure | Count |
| --- | --- |
| Months ingested | 36 (Jan 2023 – Dec 2025) |
| Human-written emails | 11,847 |
| Automated mail (Jira + build bots) | 13,435 |
| Structured Jira events | 9,380 |
| Retrievable documents (email chunks ≤ 500 tokens + Jira events) | 22,138 |
| Regex ID mentions (KIPs, tickets, releases) | 17,553 |

## How it works

1. **Ingest.** Monthly mbox archives are parsed; quoted replies, signatures, stack traces, patches and log dumps are stripped or collapsed; emails are threaded across months and chunked to at most 500 tokens.
2. **Extract.** Headers, Jira events and IDs are extracted by rule. An LLM extracts only what rules can't: mentions of people and components, and relations such as *proposed*, *voted*, *reviewed*, *blocks* and *affects*. Extraction runs through the Message Batches API with a per-email cache, persisted batch IDs (a crash never loses paid results) and a cost guard that shows an estimate before spending.
3. **Resolve and build.** Entity resolution merges mentions into people, KIPs, tickets, releases and components, and the result is loaded into Neo4j with provenance on every edge.
4. **Retrieve and answer.** Each system fills the same 4,000-token context budget; a shared answer chain cites its sources.
5. **Evaluate.** About 300 human-reviewed single-hop and multi-hop questions, built with the MultiHop-RAG recipe from thread- and ID-linked evidence (never from the graph, to avoid bias), split 20% dev / 80% held-out test. Metrics: source recall, all-sources-found rate, judged answer accuracy, cost and latency, with paired bootstrap 95% confidence intervals.

Design spec: [docs/superpowers/specs/2026-09-28-kafka-kg-rag-design.md](docs/superpowers/specs/2026-09-28-kafka-kg-rag-design.md)

## Quickstart

Requires Docker, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

```bash
uv sync
cp .env.example .env          # add ANTHROPIC_API_KEY
docker compose up -d postgres
uv run kgrag ingest --from 2023-01 --to 2025-12
uv run kgrag extract --limit 200          # prints a cost estimate first
uv run kgrag extract --show 5             # inspect cached extractions, no API calls
uv run pytest                             # paid-API tests are excluded by default
```

## Project layout

```
src/kgrag/
  ingest/     fetch, parse, clean, Jira parser, ingest pipeline
  extract/    regex IDs, LLM extraction (batch + cache), persistence
  answer/     shared answer chain
  questions/  evidence sampling, drafting, review CLI, splits
  eval/       metrics and bootstrap CIs
  store/      Postgres access
migrations/   SQL schema
tests/        unit + Docker integration tests
```
