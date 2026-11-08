"""Run the eval dataset through the pipeline, score it, and gate on the result.

The gate fails when
  * a gated metric is below its floor (or above its ceiling), in any gated scope, or
  * a gated metric dropped by more than `max_regression` against evals/baseline.json, when
    the baseline was produced with the same model, embedder and store.
Updating the baseline is an explicit command (`make eval-baseline`), so a lower score can
only be accepted through a reviewed change to a committed file.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cited_answers.evals.metrics import EvalItem, ItemOutcome, aggregate, score_item
from cited_answers.pipeline.graph import Pipeline

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "evals"
DATASET = EVAL_DIR / "dataset.jsonl"
GATE = EVAL_DIR / "gate.json"
BASELINE = EVAL_DIR / "baseline.json"
REPORT_DIR = EVAL_DIR / "reports"


def load_dataset(path: Path = DATASET) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                items.append(EvalItem(**row))
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in eval dataset")
    for i in items:
        if i.answerable != bool(i.gold):
            raise ValueError(f"{i.id}: answerable items need gold articles, others none")
    return items


def run_items(
    pipeline: Pipeline, items: Sequence[EvalItem], *, workers: int = 1
) -> list[ItemOutcome]:
    """Answer and score every item. Workers > 1 overlap live model calls; order is kept."""

    def one(it: EvalItem) -> ItemOutcome:
        return score_item(pipeline.corpus, it, pipeline.answer(it.question, debug=True))

    if workers <= 1:
        return [one(it) for it in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def build_report(pipeline: Pipeline, outcomes: Sequence[ItemOutcome]) -> dict[str, Any]:
    scopes: dict[str, Any] = {"all": aggregate(outcomes)}
    for split in sorted({o.item.split for o in outcomes}):
        scopes[split] = aggregate([o for o in outcomes if o.item.split == split])
    categories = {
        cat: aggregate([o for o in outcomes if o.item.category == cat])
        for cat in sorted({o.item.category for o in outcomes})
    }
    return {
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config_key(pipeline),
        "metrics": scopes,
        "by_category": categories,
        "items": [o.to_json() for o in outcomes],
    }


def config_key(pipeline: Pipeline) -> dict[str, str]:
    return {
        "model": pipeline.model.name,
        "embedder": pipeline.embedder_name,
        "store": pipeline.retriever.name,
        "corpus_version": pipeline.corpus.version,
    }


def check_gate(
    report: dict[str, Any], gate: dict[str, Any], baseline: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    """Return (failures, notes)."""
    failures: list[str] = []
    notes: list[str] = []
    metrics = report["metrics"]
    for scope in gate["scopes"]:
        if scope not in metrics:
            continue
        for name, bound in gate["metrics"].items():
            value = metrics[scope][name]
            if "min" in bound and value < bound["min"]:
                failures.append(f"{scope}.{name} = {value:.4f} is below the floor {bound['min']}")
            if "max" in bound and value > bound["max"]:
                failures.append(f"{scope}.{name} = {value:.4f} is above the ceiling {bound['max']}")
    if baseline is None:
        notes.append("no baseline file: regression check skipped")
        return failures, notes
    same = all(
        baseline["config"].get(k) == report["config"].get(k) for k in ("model", "embedder", "store")
    )
    if not same:
        notes.append(
            "baseline was produced with another model/embedder/store: only the floors apply"
        )
        return failures, notes
    if baseline["config"].get("corpus_version") != report["config"]["corpus_version"]:
        notes.append("corpus changed since the baseline; re-baseline after reviewing the diff")
    tolerance = gate["max_regression"]
    for scope in gate["scopes"]:
        for name, bound in gate["metrics"].items():
            before = baseline["metrics"].get(scope, {}).get(name)
            now = metrics.get(scope, {}).get(name)
            if before is None or now is None:
                continue
            worse = (before - now) if "min" in bound else (now - before)
            if worse > tolerance + 1e-9:
                failures.append(
                    f"{scope}.{name} regressed from {before:.4f} to {now:.4f} (allowed {tolerance})"
                )
    return failures, notes


def render_markdown(report: dict[str, Any], failures: list[str], notes: list[str]) -> str:
    cols = [
        "faithfulness",
        "citation_precision",
        "citation_hit_rate",
        "refusal_accuracy",
        "unsafe_answer_rate",
        "false_refusal_rate",
        "retrieval_recall",
    ]
    lines = [
        "# Eval report",
        "",
        f"Generated {report['generated_at']}",
        "",
        "Config: " + ", ".join(f"{k}={v}" for k, v in report["config"].items()),
        "",
        "| scope | items | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for scope, m in {**report["metrics"], **report["by_category"]}.items():
        lines.append(
            f"| {scope} | {m['items']} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |"
        )
    lines += ["", "## Gate", ""]
    lines += [f"- FAIL: {f}" for f in failures] or ["- PASS"]
    lines += [f"- note: {n}" for n in notes]
    lines += [
        "",
        "## Items that were not fully right",
        "",
        "| id | split | expected | got | cited | gold | issue |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in report["items"]:
        wrong_decision = (it["status"] == "answered") != it["answerable"]
        imprecise = it["precision"] is not None and it["precision"] < 1.0
        if not (wrong_decision or imprecise or it["unsupported"]):
            continue
        issue = (
            "wrong refusal decision"
            if wrong_decision
            else "imprecise citation"
            if imprecise
            else "; ".join(it["unsupported"])
        )
        expected = "answer" if it["answerable"] else "refuse"
        lines.append(
            f"| {it['id']} | {it['split']} | {expected} | {it['status']}"
            f"{' (' + it['refusal_code'] + ')' if it['refusal_code'] else ''} | "
            f"{', '.join(it['cited'])} | {', '.join(it['gold'])} | {issue} |"
        )
    return "\n".join(lines) + "\n"


def run_eval(
    pipeline: Pipeline,
    *,
    dataset: Path = DATASET,
    split: str = "all",
    out_dir: Path = REPORT_DIR,
    gate_path: Path = GATE,
    baseline_path: Path = BASELINE,
    update_baseline: bool = False,
    workers: int = 1,
) -> tuple[dict[str, Any], list[str], list[str]]:
    items = load_dataset(dataset)
    if split != "all":
        items = [i for i in items if i.split == split]
    report = build_report(pipeline, run_items(pipeline, items, workers=workers))
    gate = json.loads(gate_path.read_text())
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    failures, notes = check_gate(report, gate, baseline)
    report["gate"] = {"passed": not failures, "failures": failures, "notes": notes}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "latest.md").write_text(render_markdown(report, failures, notes))
    if update_baseline:
        if split != "all":
            raise ValueError("update the baseline from a full run (split=all)")
        baseline_path.write_text(
            json.dumps({"config": report["config"], "metrics": report["metrics"]}, indent=2) + "\n"
        )
    return report, failures, notes
