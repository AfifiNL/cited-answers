"""Command line: ask, eval, index, ingest, serve.

cited-answers ask "What is the right to erasure?" [--json] [--debug]
cited-answers eval [--split dev|test|all] [--update-baseline] [--workers N] [--out DIR]
                   [--gate FILE] [--baseline FILE]              exit 1 when the gate fails
cited-answers judge-eval [--workers N]                          test the LLM judge itself
cited-answers index                                             sync corpus into pgvector
cited-answers ingest [--download]                               rebuild data/corpus
cited-answers serve [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from cited_answers.config import Settings


def _ask(args: argparse.Namespace) -> int:
    from cited_answers.factory import build_pipeline

    result = build_pipeline().answer(args.question, debug=args.debug)
    if args.json:
        print(result.model_dump_json(indent=2, exclude_none=True))
        return 0
    if result.status == "refused" and result.refusal:
        print(f"REFUSED ({result.refusal.code}): {result.refusal.message}")
        if result.refusal.detail:
            print(f"  why: {result.refusal.detail}")
        return 0
    print(textwrap.fill(result.answer or "", 100))
    print()
    for c in result.citations:
        print(f"[{c.ref}] {c.label}, {c.title}\n    {c.url}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    from cited_answers.evals.runner import BASELINE, GATE, REPORT_DIR, run_eval
    from cited_answers.factory import build_pipeline, make_scorer, open_cassette

    settings = Settings()
    cassette = open_cassette(settings)
    out_dir = Path(args.out) if args.out else REPORT_DIR
    report, failures, notes = run_eval(
        build_pipeline(settings, cassette=cassette),
        split=args.split,
        update_baseline=args.update_baseline,
        out_dir=out_dir,
        workers=args.workers,
        scorer=make_scorer(settings, cassette),
        cassette=cassette,
        gate_path=Path(args.gate) if args.gate else GATE,
        baseline_path=Path(args.baseline) if args.baseline else BASELINE,
    )
    if cassette is not None and cassette.recording:
        cassette.save()
        print(
            f"recorded {len(cassette.entries)} calls to {cassette.path} "
            f"({cassette.reused} reused from the previous recording)"
        )
    cols = [
        "faithfulness",
        "citation_precision",
        "citation_hit_rate",
        "refusal_accuracy",
        "unsafe_answer_rate",
        "false_refusal_rate",
        "retrieval_recall",
    ]
    print("config:", ", ".join(f"{k}={v}" for k, v in report["config"].items()))
    print(f"{'scope':<8}{'items':>6}  " + "  ".join(f"{c[:18]:>18}" for c in cols))
    for scope, m in report["metrics"].items():
        print(f"{scope:<8}{m['items']:>6}  " + "  ".join(f"{m[c]:>18.3f}" for c in cols))
    for n in notes:
        print(f"note: {n}")
    print(f"report: {out_dir / 'latest.md'}")
    if args.update_baseline:
        print(f"baseline updated: review and commit {args.baseline or 'evals/baseline.json'}")
    if failures:
        print("EVAL GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("EVAL GATE: PASS")
    return 0


def _judge_eval(args: argparse.Namespace) -> int:
    """Test the judge itself on generated claims whose verdict is known (evals/judge.py)."""
    from cited_answers.corpus import Corpus
    from cited_answers.evals.judge import JudgeScorer, load_judge_set, run_judge_eval
    from cited_answers.evals.runner import EVAL_DIR, REPORT_DIR
    from cited_answers.factory import make_judge, open_cassette

    settings = Settings(faithfulness_scorer="judge")
    cassette = open_cassette(settings)
    corpus = Corpus.load(settings.corpus_dir)
    scorer = JudgeScorer(make_judge(settings, cassette))
    report = run_judge_eval(corpus, load_judge_set(), scorer, workers=args.workers)
    gate_path = Path(args.gate) if args.gate else EVAL_DIR / "gate_judge.json"
    gate = json.loads(gate_path.read_text())["judge_set"]
    failures = []
    if report["with_judge"]["catch_rate"] < gate["min_catch_rate"]:
        failures.append(f"judge catch rate below {gate['min_catch_rate']}")
    if report["with_judge"]["false_alarm_rate"] > gate["max_false_alarm_rate"]:
        failures.append(f"judge false alarm rate above {gate['max_false_alarm_rate']}")
    if report["judge_errors"]:
        failures.append(
            f"{report['judge_errors']} judge call(s) failed and have no verdict "
            "(left out of the judge's rates); run again"
        )
    if cassette is not None and not cassette.recording and cassette.misses:
        failures.append(f"{len(cassette.misses)} judge request(s) not in the cassette: re-record")
    report["gate"] = {"passed": not failures, "failures": failures}
    out_dir = Path(args.out) if args.out else REPORT_DIR / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if cassette is not None and cassette.recording:
        cassette.save()
        print(
            f"recorded {len(cassette.entries)} calls to {cassette.path} "
            f"({cassette.reused} reused from the previous recording)"
        )
    print(f"judge: {report['scorer']}  cases: {report['cases']}  errors: {report['judge_errors']}")
    print(f"{'':16}{'accuracy':>10}{'catch rate':>12}{'false alarms':>14}")
    for name in ("quote_only", "with_judge"):
        r = report[name]
        print(
            f"{name:16}{r['accuracy']:>10.3f}{r['catch_rate']:>12.3f}{r['false_alarm_rate']:>14.3f}"
        )
    print(f"{'kind':16}{'expected':>15}{'cases':>7}{'quote passes':>14}{'judge passes':>14}")
    for kind, k in report["by_kind"].items():
        print(
            f"{kind:16}{k['expected']:>15}{k['cases']:>7}"
            f"{k['quote_passes']:>14}{k['judge_passes']:>14}"
        )
    print(f"report: {out_dir / 'latest.json'}")
    if failures:
        print("JUDGE GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("JUDGE GATE: PASS")
    return 0


def _index(_: argparse.Namespace) -> int:
    from cited_answers.corpus import Corpus
    from cited_answers.factory import make_embedder
    from cited_answers.retrieval.pgvector import PgVectorStore

    settings = Settings()
    if not settings.database_url:
        print("set CA_DATABASE_URL (see .env.example)", file=sys.stderr)
        return 2
    store = PgVectorStore(
        settings.database_url, Corpus.load(settings.corpus_dir), make_embedder(settings)
    )
    print(json.dumps(store.sync()))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from cited_answers.ingest.build import build, download
    from cited_answers.laws import LAWS

    if args.download:
        for law in LAWS:
            print(f"downloading {law} ...")
            download(law)
    print(json.dumps(build(), indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from cited_answers.api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cited-answers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(fn=_ask)
    p = sub.add_parser("eval", help="run the eval suite and the gate")
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="parallel questions (live models)")
    p.add_argument("--out", help="report directory (default: evals/reports)")
    p.add_argument("--gate", help="gate file (default: evals/gate.json)")
    p.add_argument("--baseline", help="baseline file (default: evals/baseline.json)")
    p.set_defaults(fn=_eval)
    p = sub.add_parser("judge-eval", help="test the LLM judge on claims with known verdicts")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", help="report directory (default: evals/reports/judge)")
    p.add_argument("--gate", help="gate file (default: evals/gate_judge.json)")
    p.set_defaults(fn=_judge_eval)
    p = sub.add_parser("index", help="sync the corpus into PostgreSQL/pgvector")
    p.set_defaults(fn=_index)
    p = sub.add_parser("ingest", help="rebuild data/corpus from the EUR-Lex snapshots")
    p.add_argument("--download", action="store_true", help="fetch fresh snapshots first")
    p.set_defaults(fn=_ingest)
    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_serve)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
