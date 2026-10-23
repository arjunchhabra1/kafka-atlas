# Kafka Atlas evaluation report

Split: **test**

Ignored runs (superseded or degraded): graph-user-test-20261023T002824Z

## Runs

| System | Run | Questions | Errors | Models (answer / judge) | Prompts | Run hash |
|---|---|---|---|---|---|---|
| vector | vector-na-test-20261022T235747Z | 303 | 0 | claude-sonnet-5-5 / claude-opus-5-5 | answer=answer-v1, judge=judge-v1 | f470f56439fd |
| graph-user | graph-user-test-20261023T004554Z | 303 | 0 | claude-sonnet-5-5 / claude-opus-5-5 | answer=answer-v1, judge=judge-v1 | 569b5a5c5c34 |

Errored questions are excluded from answer accuracy (the n in each cell). Cells are `mean [95% bootstrap CI] (n)`. *Indicative*: the per-type groups have few questions each, so their intervals are wide.

## Results

### Source recall

| Question group | vector | graph-user |
|---|---|---|
| Single-hop | 0.90 [0.84, 0.95] (n=97) | 0.90 [0.84, 0.95] (n=97) |
| Multi-hop | 0.56 [0.51, 0.61] (n=176) | 0.61 [0.55, 0.67] (n=176) |
| Multi-hop: inference *(indicative)* | 0.52 [0.46, 0.58] (n=133) | 0.55 [0.48, 0.61] (n=133) |
| Multi-hop: comparison *(indicative)* | 0.66 [0.54, 0.78] (n=38) | 0.79 [0.68, 0.88] (n=38) |
| Multi-hop: temporal *(indicative)* | 0.90 [0.70, 1.00] (n=5) | 1.00 [1.00, 1.00] (n=5) |
| Null (abstention) | n/a | n/a |

### Full-evidence rate

| Question group | vector | graph-user |
|---|---|---|
| Single-hop | 0.90 [0.84, 0.95] (n=97) | 0.90 [0.84, 0.95] (n=97) |
| Multi-hop | 0.34 [0.27, 0.41] (n=176) | 0.45 [0.38, 0.52] (n=176) |
| Multi-hop: inference *(indicative)* | 0.27 [0.20, 0.35] (n=133) | 0.37 [0.29, 0.45] (n=133) |
| Multi-hop: comparison *(indicative)* | 0.50 [0.34, 0.66] (n=38) | 0.66 [0.50, 0.82] (n=38) |
| Multi-hop: temporal *(indicative)* | 0.80 [0.40, 1.00] (n=5) | 1.00 [1.00, 1.00] (n=5) |
| Null (abstention) | n/a | n/a |

### Answer accuracy

| Question group | vector | graph-user |
|---|---|---|
| Single-hop | 0.93 [0.88, 0.98] (n=97) | 0.93 [0.88, 0.98] (n=97) |
| Multi-hop | 0.55 [0.47, 0.62] (n=176) | 0.60 [0.53, 0.67] (n=176) |
| Multi-hop: inference *(indicative)* | 0.44 [0.36, 0.53] (n=133) | 0.50 [0.42, 0.59] (n=133) |
| Multi-hop: comparison *(indicative)* | 0.86 [0.75, 0.95] (n=38) | 0.89 [0.79, 0.97] (n=38) |
| Multi-hop: temporal *(indicative)* | 0.90 [0.70, 1.00] (n=5) | 1.00 [1.00, 1.00] (n=5) |
| Null (abstention) | 1.00 [1.00, 1.00] (n=30) | 1.00 [1.00, 1.00] (n=30) |

## Paired differences vs the vector baseline

Computed only over questions present in both runs with a value in both.

### graph-user minus vector (paired, 303 shared questions)

| Question group | Source recall | Full-evidence rate | Answer accuracy |
|---|---|---|---|
| Single-hop | +0.00 [-0.05, +0.06] (n=97) | +0.00 [-0.05, +0.06] (n=97) | +0.01 [-0.05, +0.06] (n=97) |
| Multi-hop | +0.05 [-0.01, +0.11] (n=176) | +0.11 [+0.04, +0.19] (n=176) | +0.06 [-0.01, +0.12] (n=176) |
| Multi-hop: inference *(indicative)* | +0.03 [-0.04, +0.10] (n=133) | +0.10 [+0.01, +0.19] (n=133) | +0.06 [-0.03, +0.15] (n=133) |
| Multi-hop: comparison *(indicative)* | +0.13 [+0.04, +0.24] (n=38) | +0.16 [+0.00, +0.32] (n=38) | +0.04 [-0.03, +0.11] (n=38) |
| Multi-hop: temporal *(indicative)* | +0.10 [+0.00, +0.30] (n=5) | +0.20 [+0.00, +0.60] (n=5) | +0.10 [+0.00, +0.30] (n=5) |
| Null (abstention) | n/a | n/a | +0.00 [+0.00, +0.00] (n=30) |

## Cost and latency per question

Token counts are tiktoken estimates of the answer call (sources + question in, answer out). Cost is answer-call cost only (excludes judge, embeddings, any LLM calls inside the graph retriever). Latency covers retrieval and answering.

| System | Latency ms (mean) | Tokens in | Tokens out | Est. cost USD |
|---|---|---|---|---|
| vector | 4143 | 3820 | 116 | 0.0088 |
| graph-user | 4608 | 4036 | 116 | 0.0092 |

