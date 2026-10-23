# Kafka Atlas

An engineering knowledge assistant over the Apache Kafka developer mailing list. It answers questions such as *"Which release fixed the ticket Colin McCabe cited when voting −1 on 3.6.2 RC1?"* with cited sources, and it is built to answer one question with a controlled experiment: **does a knowledge graph retrieve better than plain vector search when the answer spans several documents?**

The corpus is three years (2023–2025) of `dev@kafka.apache.org` plus its Jira activity: a public stand-in for the Slack, email and ticket history that companies index internally. Two systems answer the same questions with the same model, prompt and context budget, so retrieval is the only thing that differs:

- **Vector RAG baseline:** LangChain + pgvector, nearest-neighbour search over email and Jira chunks.
- **Graph RAG:** an LLM-extracted knowledge graph in Neo4j, queried through entity linking and multi-hop traversal, with every edge pointing back to the chunks that justify it.

**Stack:** Python 3.12 · Claude Sonnet 5.5 (extraction, question drafting, answering) and Claude Opus 5.5 (judge) · Voyage embeddings · Postgres + pgvector · Neo4j · LangChain · FastAPI · Docker · GCP (Cloud Run, Cloud SQL) · Langfuse · GitHub Actions

## Highlights

- **Corpus and knowledge graph:** 11,847 human emails and 9,380 Jira events become 22,138 retrievable documents and a graph of 24,376 nodes and 54,127 relationships, every edge traceable to its source chunks.
- **Entity resolution:** a deterministic resolver for people and components scores **B³ F1 0.965** (precision 1.000) against **0.750** for exact-match merging.
- **Evaluation set:** 379 questions across five types (single-hop, inference, comparison, temporal, and unanswerable "null" questions), reviewed against their source emails and split into a tuning set (76) and a held-out test set (303).
- **Multi-hop retrieval:** on the tuning set, hybrid graph retrieval finds **65%** of the gold source documents for multi-hop questions, against **47%** for vector search.

## Architecture

```mermaid
flowchart LR
    A[mbox archives<br/>dev@ 2023-2025] --> B[Ingest<br/>parse, clean, thread, chunk]
    B --> P[(Postgres + pgvector<br/>emails, documents, mentions)]
    B --> C[Extract<br/>rules + LLM batch]
    C --> P
    P --> D[Resolve<br/>exact or user resolver]
    D --> E[Build graph]
    E --> N[(Neo4j<br/>provenance on every edge)]
    P --> F[Index<br/>embed chunks and node names]
    F --> P
    P --> V[Vector retriever]
    N --> G[Graph retriever]
    P --> G
    V --> H[Shared answer chain<br/>same model, prompt, 4k budget]
    G --> H
    H --> I[Eval<br/>LLM judge, bootstrap CIs, report]
    H --> J[FastAPI /ask]
```

## How it works

**Ingestion and cleaning.** Monthly mbox archives are parsed. Quoted replies, signatures, stack traces, patches and log dumps are stripped or collapsed; emails are threaded across months and chunked to at most 500 tokens. Jira notification mails are parsed into structured events, and automated mail is kept out of the human corpus.

**Extraction.** Headers, Jira events and IDs (KIPs, tickets, releases) are extracted by rule. An LLM extracts only what rules cannot: mentions of people and components, and relations such as *proposed*, *voted*, *reviewed*, *blocks*, *affects*. Extraction runs through the Message Batches API with a per-email cache, persisted batch IDs (a crash never loses paid results), and a cost guard that prints an estimate before spending.

**Entity resolution.** `src/kgrag/resolve/` merges Person and Component mentions into entities, deterministically and without LLM calls. Components are normalised (case, hyphens, product prefixes, package prefixes, plurals, versioned spellings). People are resolved in two steps: strong identities first (shared address, same normalised full name with nicknames and middle initials ignored, abbreviated headers matched to the one full name that fits, single-letter misspellings), then every mention is attached, with first-name-only mentions resolved from the email they appear in. Sign-offs mean the sender; greetings never do. Anything still ambiguous stays unresolved, because a missed link costs recall while a wrong merge corrupts the graph. Scored on 188 hand-labelled mentions of 22 entities (13 people, 9 components):

| Resolver | B³ precision | B³ recall | B³ F1 |
| --- | --- | --- | --- |
| Exact match (baseline) | 0.991 | 0.603 | 0.750 |
| **Kafka Atlas resolver** | **1.000** | **0.933** | **0.965** |

**Graph build.** Resolved entities become Person, Component, KIP, JiraTicket, Release and Email nodes; relations become typed edges (PROPOSED, VOTED, REVIEWED, FILED, RESOLVED, BLOCKS, AFFECTS, …). Every edge carries `source_doc_ids`, the chunks that justify it, so graph retrieval returns evidence rather than graph summaries. A metadata node records the resolver and a config hash, so evaluation refuses a mismatched graph.

**Graph retrieval.** `src/kgrag/retrieve/graph_retriever.py`, with no LLM calls and deterministic output:

1. **Link.** The question's KIP, ticket and release IDs and its names become weighted seeds.
2. **Walk.** From each seed it walks up to two hops, one batched Cypher query per hop, with edge caps enforced inside Cypher so hub nodes never cross the wire. A path's weight combines seed trust, edge type, extractor confidence and per-hop decay, with boosts for edge types the question asks about ("who voted" favours VOTED) and for bridges between seeds.
3. **Doc bridge.** Multi-hop questions often identify a KIP only through another discussion ("the KIP Jane said was blocked by…"), which puts the answer three hops from anything named in the question. The retriever pivots on its best first-pass documents and the question's best vector hits, and walks from the KIPs and tickets *those* documents mention.
4. **Rank and pack.** A chunk's score sums its best support from each seed, so chunks connecting several seeds win. A hybrid rerank adds embedding similarity, results are diversified across seeds, and evidence is packed into the same 4,000-token budget the baseline uses.

All knobs live in `GraphRetrieverConfig`. A `graph-only` configuration turns off every use of embeddings, isolating what the graph contributes on its own.

**Vector baseline.** Email and Jira chunks embedded with Voyage into pgvector; top-k nearest neighbours packed into the same budget.

**Shared answer chain.** Both systems feed the same chain: one model, one prompt, the same evidence budget, structured output with cited document IDs and an explicit "insufficient evidence" option.

## Evaluation design

**Question set.** Built with the MultiHop-RAG recipe. Evidence is sampled from thread and ID links using headers and regexes only, never from the graph, so the question set does not favour either system. Questions are drafted by an LLM, then filtered by deterministic checks (single ask, length, date share, giveaways such as a question that reuses the linked ticket's title) and LLM checks that every gold document is actually needed. Every surviving question was then **LLM-reviewed against its source emails** (kept, edited or dropped, with a logged reason), and **a random sample was human-verified** (50 of 50 decisions confirmed).

| Type | What it tests | Questions |
| --- | --- | --- |
| Single-hop | One fact from one document | 121 |
| Inference | A bridge: one email identifies a KIP or ticket, another states the answer | 166 |
| Comparison | Two related KIPs or tickets compared on substance | 48 |
| Temporal | How a proposal changed between its first version and a revision | 6 |
| Null | Plausible but unanswerable from the list; the system should decline | 38 |
| **Total** | 76 tuning (dev) · 303 held-out (test) | **379** |

**Protocol.** Retriever tuning uses only the dev split. Each system runs on the test split once; a run where more than 10% of questions fail (for example an API outage) is flagged degraded and does not count. Answers are graded by an LLM judge (Claude Opus 5.5) against the reviewed gold answer, and the judge is validated against human labels. Metrics are source recall, full-evidence rate, judged answer accuracy, cost and latency, with 95% bootstrap confidence intervals and paired differences against the vector baseline.

## Retrieval on the dev split

Gold-source recall within the 4,000-token budget (68 dev questions with gold documents; null questions have none):

| Question type | n | Graph (hybrid) | Graph-only | Vector |
| --- | --- | --- | --- | --- |
| Single-hop | 24 | 0.88 | 0.50 | **0.96** |
| Inference | 33 | **0.58** | 0.30 | 0.41 |
| Comparison | 10 | **0.85** | **0.85** | 0.60 |
| **Multi-hop overall** | 44 | **0.65** | 0.44 | 0.47 |

The first ablation put the graph retriever *behind* vector search on inference questions (0.27 vs 0.41). Tracing each miss showed why: those questions name only a person, and 42% of the gold documents sat three hops away, beyond the walk. The doc-bridge step above was the fix, and it raised dev multi-hop recall from 0.43 to 0.65. The `graph-only` column shows the graph's own contribution: it wins on comparison questions, while the strongest system on bridge questions combines the graph with vector hits. Vector search remains best on single-hop questions, where the answer sits in one well-matched chunk.

## Data

| Measure | Count |
| --- | --- |
| Months ingested | 36 (Jan 2023 – Dec 2025) |
| Human-written emails | 11,847 |
| Structured Jira events | 9,380 |
| Retrievable documents (email chunks ≤ 500 tokens + Jira events) | 22,138 |
| LLM-extracted relations | 14,646 |
| Graph | 24,376 nodes, 54,127 relationships |
| Node counts | 4,694 Person · 751 Component · 700 KIP · 6,088 JiraTicket · 296 Release · 11,847 Email |

## Quickstart

Requires Docker, [uv](https://docs.astral.sh/uv/), an Anthropic API key and a Voyage API key. Commands that call paid APIs print a cost estimate and ask first.

```bash
uv sync
cp .env.example .env                      # add ANTHROPIC_API_KEY and VOYAGE_API_KEY
docker compose up -d                      # Postgres (pgvector) and Neo4j

uv run kgrag ingest --from 2023-01 --to 2025-12
uv run kgrag extract
uv run kgrag resolve --resolver user
uv run kgrag build --resolver user
uv run kgrag index

uv run kgrag questions draft --pilot      # draft a few questions per type
uv run kgrag questions review             # keep / edit / drop each draft
uv run kgrag questions split              # 20% dev / 80% test

uv run kgrag ablate --split dev --resolver user
uv run kgrag eval --system vector --split dev
uv run kgrag eval --system graph --resolver user --split dev
uv run kgrag report results/*.json --split dev

uv run kgrag serve                        # FastAPI: POST /ask {"question", "system"}
```

Evaluation runs a freshness preflight before any question and refuses, naming the command to run, when a gold document is missing, the index was embedded with another model, or the graph was built from a different resolution.

## Deployment

The API is containerised and deployable to Cloud Run with Cloud SQL Postgres (pgvector), AuraDB for the graph, Secret Manager for keys and Langfuse for traces. `/ask` requires an `X-API-Key` header and the service refuses to start in cloud mode without one; instance and concurrency caps bound spend. Steps and cost notes: [docs/DEPLOY.md](docs/DEPLOY.md).

## Limitations

- One corpus and one time window; results may not generalise to other projects or periods.
- Questions are drafted from header- and regex-linked evidence, so they over-represent relations visible through explicit IDs and threads.
- Temporal questions are scarce (6): Kafka design revisions mostly live on the wiki, and the list only links to them.
- The dev split is small (44 multi-hop questions), so dev numbers carry wide uncertainty; the held-out test split is the measure.
- Ambiguous first names ("Chris", "David") are left unresolved rather than guessed, so some people's mentions stay split across nodes.
- dev@ carries only Jira created / resolved / reopened notifications, so ticket assignments are not in the graph.

## Testing

```bash
uv run pytest                       # everything not marked live
uv run pytest -m "not integration"  # unit tests only, no Docker
```

Around 750 tests. Markers: `integration` (Postgres and Neo4j via testcontainers), `quality` (non-blocking quality checks), `live` (calls paid APIs; never run by default). An offline end-to-end smoke test ingests a fixture slice, extracts, resolves, builds, indexes, evaluates both systems with fake models and renders the report. GitHub Actions runs the suite on every push and pull request.

## Project layout

```
src/kgrag/
  ingest/     fetch, parse, clean, Jira parser, ingest pipeline
  extract/    regex IDs, LLM extraction (batch + cache), persistence
  resolve/    exact-match baseline and the entity resolver (names, people, components)
  graph/      Neo4j build, knowledge-graph access, graph checks
  retrieve/   embeddings, index, vector and graph retrievers, ablations
  answer/     shared answer chain
  questions/  evidence sampling, drafting, review, splits
  eval/       runner, judge, metrics, bootstrap CIs, report
  api/        FastAPI service
  store/      Postgres access
migrations/   SQL schema
scripts/      fixtures, embedding recording, cloud load and deploy
docs/         design spec, deployment guide
tests/        unit + Docker integration tests, fixtures
```
