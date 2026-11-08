"""The LLM-judge scorer and its own test set, with a scripted judge: no network, no cost."""

from __future__ import annotations

from collections import Counter

from cited_answers.corpus import Corpus
from cited_answers.evals.judge import (
    JudgeScorer,
    build_judge_set,
    load_judge_set,
    run_judge_eval,
)
from cited_answers.evals.metrics import ClaimToScore, QuoteScorer
from cited_answers.providers.base import JudgeRequest
from cited_answers.providers.mock import ScriptedJudge

QUOTE = "The controller shall communicate any rectification or erasure of personal data"
CLAIM = ClaimToScore(
    "The controller must tell recipients about any erasure.", QUOTE, "gdpr:art19", "GDPR Article 19"
)


def test_committed_judge_set_matches_the_generator(corpus: Corpus) -> None:
    assert load_judge_set() == build_judge_set(corpus)


def test_judge_set_is_balanced_and_well_formed(corpus: Corpus) -> None:
    cases = build_judge_set(corpus)
    kinds = Counter(c.kind for c in cases)
    assert kinds["verbatim"] == kinds["shall_to_must"] == kinds["shall_to_may"] == 20
    assert kinds["negation"] == 20 and kinds["actor_swap"] >= 5 and kinds["number_change"] >= 5
    assert len({c.id for c in cases}) == len(cases)
    quote = QuoteScorer()
    for c in cases:
        claim = ClaimToScore(c.claim, c.quote, c.article_key, c.label)
        if c.expected:  # every claim that should pass survives the deterministic check
            assert quote.score(corpus, claim).supported, c.id
        if c.kind != "verbatim":
            assert c.claim != c.quote, c.id


def test_judge_scorer_asks_only_after_the_quote_check(corpus: Corpus) -> None:
    judge = ScriptedJudge(lambda r: "supported")
    scorer = JudgeScorer(judge)
    bad_quote = ClaimToScore(CLAIM.text, "words that are not in the article", "gdpr:art19", "x")
    verdict = scorer.score(corpus, bad_quote)
    assert not verdict.supported and "not found verbatim" in verdict.reason
    assert judge.calls == []  # no judge call for a claim the quote check already fails
    assert scorer.score(corpus, CLAIM).supported
    request: JudgeRequest = judge.calls[0]
    assert request.provision == corpus.article_text("gdpr:art19")  # the full article
    assert scorer.name == "judge:scripted-judge"


def test_judge_verdicts_and_errors_become_unsupported(corpus: Corpus) -> None:
    partial = JudgeScorer(ScriptedJudge(lambda r: "partially_supported")).score(corpus, CLAIM)
    assert not partial.supported and partial.reason.startswith("judge: partially_supported")
    failing = JudgeScorer(ScriptedJudge(lambda r: RuntimeError("down")))
    broken = failing.score(corpus, CLAIM)
    assert not broken.supported and "judge error" in broken.reason
    assert failing.errors  # listed, so the eval gate can refuse to trust the numbers


def test_judge_eval_rates(corpus: Corpus) -> None:
    cases = load_judge_set()
    by_claim = {c.claim: c.expected for c in cases}
    perfect = JudgeScorer(
        ScriptedJudge(lambda r: "supported" if by_claim[r.claim] else "not_supported")
    )
    report = run_judge_eval(corpus, cases, perfect, workers=4)
    assert report["with_judge"] == {"accuracy": 1.0, "catch_rate": 1.0, "false_alarm_rate": 0.0}
    assert report["judge_mistakes"] == []
    quote_only = report["quote_only"]
    # The deterministic check passes "shall" -> "may" and negations: that is the judge's job.
    assert quote_only["catch_rate"] < 0.3 and quote_only["false_alarm_rate"] == 0.0
    # A changed number is usually caught by the number check, unless the new number happens
    # to appear elsewhere in the quote ("paragraph 1 ... paragraph 2").
    numbers = report["by_kind"]["number_change"]
    assert numbers["quote_passes"] < numbers["cases"] / 2
    lazy = run_judge_eval(corpus, cases, JudgeScorer(ScriptedJudge(lambda r: "supported")))
    assert lazy["with_judge"]["catch_rate"] == quote_only["catch_rate"]


def test_judge_errors_are_not_verdicts(corpus: Corpus) -> None:
    cases = load_judge_set()
    by_claim = {c.claim: c.expected for c in cases}
    flaky_ids = {c.claim for c in cases if c.kind == "verbatim"}

    def decide(r: JudgeRequest) -> str | Exception:
        if r.claim in flaky_ids:
            return RuntimeError("HTTP 429")
        return "supported" if by_claim[r.claim] else "not_supported"

    report = run_judge_eval(corpus, cases, JudgeScorer(ScriptedJudge(decide)))
    assert report["judge_errors"] == 20
    # The failed cases are left out, not counted as false alarms.
    assert report["with_judge"]["false_alarm_rate"] == 0.0
    assert report["by_kind"]["verbatim"]["judge_errors"] == 20
