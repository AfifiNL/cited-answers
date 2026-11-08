"""Command line: ask, eval, index, ingest, serve.

cited-answers ask "What is the right to erasure?" [--json] [--debug]
cited-answers eval [--split dev|test|all] [--update-baseline] [--workers N] [--out DIR]
                                                                exit 1 when the gate fails
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
    from cited_answers.evals.runner import REPORT_DIR, run_eval
    from cited_answers.factory import build_pipeline

    out_dir = Path(args.out) if args.out else REPORT_DIR
    report, failures, notes = run_eval(
        build_pipeline(),
        split=args.split,
        update_baseline=args.update_baseline,
        out_dir=out_dir,
        workers=args.workers,
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
        print("baseline updated: review and commit evals/baseline.json")
    if failures:
        print("EVAL GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("EVAL GATE: PASS")
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
    p.set_defaults(fn=_eval)
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
