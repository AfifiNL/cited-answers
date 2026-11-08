"""The eval metrics and the gate, including proof that the gate catches a regression."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.evals.metrics import EvalItem, aggregate, score_item
from cited_answers.evals.runner import (
    BASELINE,
    GATE,
    check_gate,
    load_dataset,
    run_eval,
    run_items,
)
from cited_answers.pipeline.graph import Pipeline
from cited_answers.pipeline.models import AnswerClaim, AnswerResult, Citation
from cited_answers.providers.mock import ExtractiveAnswerModel
from cited_answers.retrieval.base import Hit
from cited_answers.retrieval.memory import MemoryStore
from cited_answers.text import TermStats

ITEM = EvalItem("x1", "dev", "gdpr", "What is erasure?", True, ["gdpr:art17"])


def answered(corpus: Corpus, chunk_id: str, text: str, quote: str) -> AnswerResult:
    c = corpus.by_id[chunk_id]
    return AnswerResult(
        status="answered",
        question=ITEM.question,
        answer=text,
        claims=[AnswerClaim(text=text, quote=quote, ref=1)],
        citations=[
            Citation(
                ref=1,
                chunk_id=c.id,
                article_key=c.article_key,
                label=c.label,
                title=c.title,
                url=c.url,
            )
        ],
    )


def test_dataset_is_valid_and_gold_articles_exist(corpus: Corpus) -> None:
    items = load_dataset()
    assert len(items) >= 80
    assert {i.split for i in items} == {"dev", "test"}
    assert sum(not i.answerable for i in items) >= 20
    for item in items:
        for key in item.gold:
            assert key in corpus.by_article, f"{item.id}: {key} not in corpus"


def test_faithful_claim_scores_one(corpus: Corpus) -> None:
    quote = "the right to obtain from the controller the erasure of personal data"
    out = score_item(corpus, ITEM, answered(corpus, "gdpr:art17:p1", quote, quote))
    assert (out.claims_supported, out.claims_total, out.precision) == (1, 1, 1.0)


def test_quote_cited_to_the_wrong_article_is_unfaithful(corpus: Corpus) -> None:
    quote = "the right to obtain from the controller the erasure of personal data"
    out = score_item(corpus, ITEM, answered(corpus, "gdpr:art18:p1", quote, quote))
    assert out.claims_supported == 0
    assert "not found verbatim in gdpr:art18" in out.unsupported[0]
    assert out.precision == 0.0


def test_invented_figure_is_unfaithful(corpus: Corpus) -> None:
    quote = "the right to obtain from the controller the erasure of personal data"
    out = score_item(
        corpus,
        ITEM,
        answered(corpus, "gdpr:art17:p1", "Erasure must happen within 30 days.", quote),
    )
    assert out.claims_supported == 0


def test_aggregate_definitions(corpus: Corpus) -> None:
    quote = "the right to obtain from the controller the erasure of personal data"
    ok = score_item(corpus, ITEM, answered(corpus, "gdpr:art17:p1", quote, quote))
    off = EvalItem("u1", "dev", "out_of_scope", "bread?", False, [])
    unsafe = score_item(corpus, off, answered(corpus, "gdpr:art17:p1", quote, quote))
    refused = score_item(corpus, ITEM, AnswerResult(status="refused", question="q"))
    m = aggregate([ok, unsafe, refused])
    assert m["refusal_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["unsafe_answer_rate"] == 1.0
    assert m["false_refusal_rate"] == 0.5
    assert m["citation_precision"] == 1.0


GATE_CFG = {
    "scopes": ["all"],
    "metrics": {"faithfulness": {"min": 0.9}, "unsafe_answer_rate": {"max": 0.1}},
    "max_regression": 0.02,
}


def report(faith: float, unsafe: float, model: str = "m") -> dict[str, object]:
    return {
        "config": {"model": model, "embedder": "e", "store": "s", "corpus_version": "v"},
        "metrics": {"all": {"faithfulness": faith, "unsafe_answer_rate": unsafe}},
    }


def test_gate_floors_and_regressions() -> None:
    base = report(1.0, 0.0)
    assert check_gate(report(1.0, 0.0), GATE_CFG, base)[0] == []
    fails, _ = check_gate(report(0.97, 0.0), GATE_CFG, base)
    assert fails == ["all.faithfulness regressed from 1.0000 to 0.9700 (allowed 0.02)"]
    fails, _ = check_gate(report(0.85, 0.2), GATE_CFG, None)
    assert len(fails) == 2
    fails, notes = check_gate(report(0.95, 0.0, model="other"), GATE_CFG, base)
    assert fails == [] and "only the floors apply" in notes[0]


class ShuffledRetriever:
    """A broken retriever: right candidates, ranking reversed (a plausible refactor bug)."""

    name = "memory"

    def __init__(self, inner: MemoryStore):
        self.inner = inner

    def search(self, query: str, *, k: int, laws: Sequence[str] | None = None) -> list[Hit]:
        return list(reversed(self.inner.search(query, k=k * 4, laws=laws)))[:k]

    def ping(self) -> bool:
        return True


def test_committed_baseline_passes_the_gate(pipeline: Pipeline, tmp_path: Path) -> None:
    report_, failures, _ = run_eval(pipeline, out_dir=tmp_path)
    assert failures == [], failures
    assert json.loads(BASELINE.read_text())["metrics"]["all"] == report_["metrics"]["all"]


def test_parallel_run_matches_sequential_run(pipeline: Pipeline) -> None:
    items = load_dataset()[:12]
    seq = run_items(pipeline, items)
    par = run_items(pipeline, items, workers=4)
    assert [o.item.id for o in par] == [i.id for i in items]
    assert [(o.status, o.cited) for o in par] == [(o.status, o.cited) for o in seq]


def test_gate_fails_on_a_retrieval_regression(
    corpus: Corpus, store: MemoryStore, settings: Settings, tmp_path: Path
) -> None:
    model = ExtractiveAnswerModel(TermStats.build(c.index_text for c in corpus.chunks))
    broken = Pipeline(corpus, ShuffledRetriever(store), model, settings, embedder_name="hash-768")
    _, failures, _ = run_eval(broken, out_dir=tmp_path)
    assert failures, "a reversed ranking must fail the eval gate"
    assert any("citation_precision" in f for f in failures)


def test_gate_fails_when_the_evidence_gate_is_disabled(
    corpus: Corpus, store: MemoryStore, tmp_path: Path
) -> None:
    lax = Settings(min_evidence_coverage=0.0, _env_file=None)  # type: ignore[call-arg]
    model = ExtractiveAnswerModel(TermStats.build(c.index_text for c in corpus.chunks))
    pipe = Pipeline(corpus, store, model, lax, embedder_name="hash-768")
    _, failures, _ = run_eval(pipe, out_dir=tmp_path)
    assert any("unsafe_answer_rate" in f or "refusal_accuracy" in f for f in failures)


def test_gate_file_is_sane() -> None:
    gate = json.loads(GATE.read_text())
    assert set(gate["metrics"]) == {
        "faithfulness",
        "citation_precision",
        "refusal_accuracy",
        "unsafe_answer_rate",
    }
    assert 0 < gate["max_regression"] <= 0.05


def test_targets_are_reported_and_regression_checked() -> None:
    gate = {
        "scopes": ["all"],
        "metrics": {"unsafe_answer_rate": {"max": 0.1}},
        "targets": {"faithfulness": {"min": 0.9}},
        "max_regression": 0.02,
    }

    def rep(faith: float) -> dict:
        config = {"model": "m", "embedder": "e", "store": "s", "scorer": "q", "corpus_version": "v"}
        metrics = {"all": {"faithfulness": faith, "unsafe_answer_rate": 0.0}}
        return {"config": config, "metrics": metrics}

    failures, notes = check_gate(rep(0.85), gate, None)
    assert failures == [] and any("target not met: all.faithfulness" in n for n in notes)
    base = rep(0.85)
    failures, _ = check_gate(rep(0.80), gate, base)
    assert failures == ["all.faithfulness regressed from 0.8500 to 0.8000 (allowed 0.02)"]
