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

from cited_answers.evals.metrics import (
    EvalItem,
    FaithfulnessScorer,
    ItemOutcome,
    QuoteScorer,
    aggregate,
    score_item,
)
from cited_answers.pipeline.graph import Pipeline
from cited_answers.providers.replay import Cassette

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
    pipeline: Pipeline,
    items: Sequence[EvalItem],
    *,
    workers: int = 1,
    scorer: FaithfulnessScorer | None = None,
) -> list[ItemOutcome]:
    """Answer and score every item. Workers > 1 overlap live model calls; order is kept."""

    def one(it: EvalItem) -> ItemOutcome:
        result = pipeline.answer(it.question, debug=True)
        return score_item(pipeline.corpus, it, result, scorer)

    if workers <= 1:
        return [one(it) for it in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def build_report(
    pipeline: Pipeline,
    outcomes: Sequence[ItemOutcome],
    scorer: FaithfulnessScorer | None = None,
) -> dict[str, Any]:
    scopes: dict[str, Any] = {"all": aggregate(outcomes)}
    for split in sorted({o.item.split for o in outcomes}):
        scopes[split] = aggregate([o for o in outcomes if o.item.split == split])
    categories = {
        cat: aggregate([o for o in outcomes if o.item.category == cat])
        for cat in sorted({o.item.category for o in outcomes})
    }
    return {
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config_key(pipeline, scorer),
        "metrics": scopes,
        "by_category": categories,
        "items": [o.to_json() for o in outcomes],
    }


def config_key(pipeline: Pipeline, scorer: FaithfulnessScorer | None = None) -> dict[str, str]:
    return {
        "model": pipeline.model.name,
        "embedder": pipeline.embedder_name,
        "store": pipeline.retriever.name,
        "scorer": (scorer or QuoteScorer()).name,
        "corpus_version": pipeline.corpus.version,
    }


CONFIG_DEFAULTS = {"scorer": "quote"}  # for baselines recorded before the key existed


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
    # Targets: reported, not blocking (a bar the system does not meet yet). Still
    # regression-checked below, so a target cannot quietly get further away.
    for scope in gate["scopes"]:
        if scope not in metrics:
            continue
        for name, bound in gate.get("targets", {}).items():
            value = metrics[scope][name]
            if ("min" in bound and value < bound["min"]) or (
                "max" in bound and value > bound["max"]
            ):
                limit = bound.get("min", bound.get("max"))
                notes.append(f"target not met: {scope}.{name} = {value:.4f} (target {limit})")
    if baseline is None:
        notes.append("no baseline file: regression check skipped")
        return failures, notes
    keys = ("model", "embedder", "store", "scorer")
    same = all(
        baseline["config"].get(k, CONFIG_DEFAULTS.get(k))
        == report["config"].get(k, CONFIG_DEFAULTS.get(k))
        for k in keys
    )
    if not same:
        notes.append(
            "baseline was produced with another model/embedder/store/scorer: only the floors apply"
        )
        return failures, notes
    if baseline["config"].get("corpus_version") != report["config"]["corpus_version"]:
        notes.append("corpus changed since the baseline; re-baseline after reviewing the diff")
    tolerance = gate["max_regression"]
    checked = {**gate.get("targets", {}), **gate["metrics"]}
    for scope in gate["scopes"]:
        for name, bound in checked.items():
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


def cassette_failures(cassette: Cassette | None, config: dict[str, str]) -> list[str]:
    """Replay must have seen every request; a recording must have no failed calls."""
    if cassette is None:
        return []
    if cassette.recording:
        if not cassette.errors:
            return []
        return [
            f"{len(cassette.errors)} live call(s) failed and were not recorded (e.g. rate "
            "limits), so this recording is incomplete. Run it again: every successful call is "
            "reused and only the missing ones are paid for.",
            *[f"  failed: {e[:120]}" for e in cassette.errors[:5]],
        ]
    if not cassette.misses:
        return []
    out = [
        f"{len(cassette.misses)} model/judge request(s) were not in the cassette {cassette.path}: "
        "the model's inputs changed since recording. Re-record it (make record-live) and "
        "review the diff."
    ]
    recorded = cassette.meta.get("embedder")
    if recorded and recorded != config.get("embedder"):
        out.append(
            f"the cassette was recorded with embedder {recorded}, this run uses "
            f"{config.get('embedder')}"
        )
    out += [f"  missing: {m[:100]}" for m in cassette.misses[:5]]
    return out


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
    scorer: FaithfulnessScorer | None = None,
    cassette: Cassette | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    items = load_dataset(dataset)
    if split != "all":
        items = [i for i in items if i.split == split]
    outcomes = run_items(pipeline, items, workers=workers, scorer=scorer)
    report = build_report(pipeline, outcomes, scorer)
    gate = json.loads(gate_path.read_text())
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    failures, notes = check_gate(report, gate, baseline)
    failures += cassette_failures(cassette, report["config"])
    judge_errors = list(getattr(scorer, "errors", []))
    if judge_errors and not (cassette is not None and cassette.misses):
        failures.append(
            f"{len(judge_errors)} judge call(s) failed, so faithfulness is not trustworthy: "
            f"{judge_errors[0][:120]}"
        )
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
