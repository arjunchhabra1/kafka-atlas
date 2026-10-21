"""Markdown report over one or more results files."""

import json
from pathlib import Path

from kgrag.config import get_settings
from kgrag.eval.judge import VERDICT_SCORE, agreement_stats
from kgrag.eval.metrics import bootstrap_ci, paired_diff_ci

MULTI_HOP = ("inference", "comparison", "temporal")
SYSTEM_ORDER = ("vector", "graph-exact", "graph-user")
# (row label, question types)
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Single-hop", ("single_hop",)),
    ("Multi-hop", MULTI_HOP),
    *[(f"Multi-hop: {t} *(indicative)*", (t,)) for t in MULTI_HOP],
    ("Null (abstention)", ("null",)),
]
METRICS = [("recall", "Source recall"), ("full_evidence", "Full-evidence rate"),
           ("accuracy", "Answer accuracy")]


def _label(header: dict) -> str:
    return "vector" if header["system"] == "vector" else f"graph-{header['resolver']}"


def _value(q: dict, metric: str) -> float | None:
    """One question's metric value, or None when it does not apply (null/errored questions)."""
    if metric == "accuracy":
        return VERDICT_SCORE.get(q["verdict"])  # error -> None, excluded
    v = q.get(metric)
    return None if v is None else float(v)


def _cell(values: list[float]) -> str:
    if not values:
        return "n/a (n=0)"
    m, lo, hi = bootstrap_ci(values)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}] (n={len(values)})"


def _diff_cell(a: list[float], b: list[float]) -> str:
    if not a:
        return "n/a (n=0)"
    m, lo, hi = paired_diff_ci(a, b)
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={len(a)})"


def _load(run_paths: list[Path]) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for p in run_paths:
        run = json.loads(Path(p).read_text())
        label = _label(run["header"])
        if label in runs:
            raise ValueError(f"two runs for system {label}: {runs[label]['header']['run_id']} "
                             f"and {run['header']['run_id']}")
        runs[label] = run
    splits = {r["header"]["split"] for r in runs.values()}
    if len(splits) > 1:
        raise ValueError(f"runs come from different splits: {sorted(splits)}")
    return dict(sorted(runs.items(), key=lambda kv: SYSTEM_ORDER.index(kv[0])
                       if kv[0] in SYSTEM_ORDER else len(SYSTEM_ORDER)))


def _table(title: str, runs: dict[str, dict], metric: str) -> list[str]:
    out = [f"### {title}", "", "| Question group | " + " | ".join(runs) + " |",
           "|---|" + "---|" * len(runs)]
    for name, types in GROUPS:
        cells = []
        for run in runs.values():
            qs = [q for q in run["questions"] if q["type"] in types]
            vals = [v for q in qs if (v := _value(q, metric)) is not None]
            cells.append("n/a" if types == ("null",) and metric != "accuracy" else _cell(vals))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out + [""]


def _paired(runs: dict[str, dict]) -> list[str]:
    base = runs.get("vector")
    if base is None or len(runs) < 2:
        return ["_No vector baseline run among the inputs; paired differences omitted._", ""]
    base_q = {q["question_id"]: q for q in base["questions"]}
    out: list[str] = []
    for label, run in runs.items():
        if label == "vector":
            continue
        other = {q["question_id"]: q for q in run["questions"]}
        shared = sorted(set(base_q) & set(other))
        out += [f"### {label} minus vector (paired, {len(shared)} shared questions)", "",
                "| Question group | " + " | ".join(t for _, t in METRICS) + " |",
                "|---|" + "---|" * len(METRICS)]
        for name, types in GROUPS:
            cells = []
            for metric, _ in METRICS:
                if types == ("null",) and metric != "accuracy":
                    cells.append("n/a")
                    continue
                a, b = [], []
                for qid in shared:
                    if other[qid]["type"] not in types:
                        continue
                    va, vb = _value(other[qid], metric), _value(base_q[qid], metric)
                    if va is not None and vb is not None:  # both systems must have a value
                        a.append(va)
                        b.append(vb)
                cells.append(_diff_cell(a, b))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _cost_latency(runs: dict[str, dict]) -> list[str]:
    pricing = get_settings().pricing
    out = ["## Cost and latency per question", "",
           "Token counts are tiktoken estimates of the answer call (sources + question in, answer "
           "out); judge calls are excluded. Latency covers retrieval and answering.", "",
           "| System | Latency ms (mean) | Tokens in | Tokens out | Est. cost USD |",
           "|---|---|---|---|---|"]
    for label, run in runs.items():
        qs = [q for q in run["questions"] if q["tokens_in"] is not None]
        lat, tin, tout = (_mean([q[k] for q in qs]) for k in ("latency_ms", "tokens_in", "tokens_out"))
        price = pricing.get(run["header"]["models"]["answer"])
        cost = (f"{(tin * price.input + tout * price.output) / 1e6:.4f}"
                if price and tin is not None else "n/a")
        f = lambda v, d=0: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
        out.append(f"| {label} | {f(lat)} | {f(tin)} | {f(tout)} | {cost} |")
    return out + [""]


def _traces(runs: dict[str, dict], n: int) -> list[str]:
    if n <= 0 or len(runs) < 2:
        return []
    by_sys = {label: {q["question_id"]: q for q in r["questions"]} for label, r in runs.items()}
    common = set.intersection(*(set(d) for d in by_sys.values()))
    picks = []
    for qid in sorted(common):
        qs = [d[qid] for d in by_sys.values()]
        if qs[0]["type"] in MULTI_HOP and all(q["verdict"] != "error" for q in qs) \
                and len({q["verdict"] for q in qs}) > 1:
            picks.append(qid)
    out = ["## Side-by-side traces (multi-hop questions where systems disagree)", ""]
    if not picks:
        return out + ["_No multi-hop question has differing verdicts across systems._", ""]
    for qid in picks[:n]:
        first = next(iter(by_sys.values()))[qid]
        out += [f"### {qid} ({first['type']})", "", f"**Question:** {first['question']}", "",
                f"**Gold answer:** {first['gold_answer']}", "",
                f"**Gold sources:** {', '.join(first['gold_doc_ids'])}", "",
                "| System | Verdict | Recall | Retrieved | Answer |", "|---|---|---|---|---|"]
        for label, d in by_sys.items():
            q = d[qid]
            text = q["answer"]["text"].replace("\n", " ").replace("|", "\\|")
            recall = "n/a" if q["recall"] is None else f"{q['recall']:.2f}"
            out.append(f"| {label} | {q['verdict']} | {recall} | "
                       f"{', '.join(q['retrieved_doc_ids'][:8])} | {text} |")
        out.append("")
    return out


def _judge_section(path: Path) -> list[str]:
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    s = agreement_stats(state["items"])
    out = ["## Judge validation", "",
           f"Human vs judge labels on run `{state['run_id']}`: {s['labelled']} of {s['total']} "
           "sampled answers labelled."]
    if s["labelled"]:
        kappa = "undefined" if s["kappa"] is None else f"{s['kappa']:.2f}"
        out.append(f"Agreement {s['agreement']:.0%}; Cohen's kappa {kappa}.")
    if s["labelled"] < s["total"]:
        out.append("**The check is incomplete.**")
    return out + [""]


def render_report(
    run_paths: list[Path], trace_examples: int = 3, judge_check_path: Path | None = None
) -> str:
    """Markdown report. Paired differences use only questions present in both runs."""
    runs = _load([Path(p) for p in run_paths])
    if not runs:
        raise ValueError("no result files given")
    first = next(iter(runs.values()))["header"]
    out = ["# Kafka Atlas evaluation report", "",
           f"Split: **{first['split']}**"
           + ("" if first["split"] == "test" else " (not the headline split; headline numbers "
              "come only from the test split)"), ""]
    degraded = [(label, r["header"]) for label, r in runs.items() if r["header"].get("degraded")]
    for label, h in degraded:
        out += [f"> **DEGRADED RUN: {label}** - {h['n_errors']} of {h['n_questions']} questions "
                "errored (more than 10%). Its numbers are not trustworthy.", ""]
    out += ["## Runs", "", "| System | Run | Questions | Errors | Models (answer / judge) | "
            "Prompts | Config hash |", "|---|---|---|---|---|---|---|"]
    for label, r in runs.items():
        h = r["header"]
        n_err = h.get("n_errors", sum(q["verdict"] == "error" for q in r["questions"]))
        out.append(f"| {label} | {h['run_id']} | {len(r['questions'])} | {n_err} | "
                   f"{h['models']['answer']} / {h['models']['judge']} | "
                   f"{', '.join(f'{k}={v}' for k, v in h['prompt_versions'].items())} | "
                   f"{h['config_hash']} |")
    out += ["", "Errored questions are excluded from answer accuracy (the n in each cell). "
            "Cells are `mean [95% bootstrap CI] (n)`.", "", "## Results", ""]
    for metric, title in METRICS:
        out += _table(title, runs, metric)
    out += ["## Paired differences vs the vector baseline", "",
            "Computed only over questions present in both runs with a value in both.", ""]
    out += _paired(runs)
    out += _cost_latency(runs)
    out += _traces(runs, trace_examples)
    check = judge_check_path or Path(next(iter(run_paths))).parent / "judge_check.json"
    out += _judge_section(Path(check))
    return "\n".join(out).rstrip() + "\n"
