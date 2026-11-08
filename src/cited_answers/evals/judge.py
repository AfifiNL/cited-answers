"""LLM-as-judge faithfulness, behind the same FaithfulnessScorer interface as the quote check.

JudgeScorer runs the deterministic QuoteScorer first: a quote that is not in the cited article,
or a number the quote does not contain, fails without asking anyone. Only claims that pass go
to the judge, which decides whether the claim says what the provision says. Word overlap cannot
see that "shall" became "may", that an exception was dropped, or that the controller's duty was
given to the processor; a reader can.

A judge is itself a model and needs its own test. `build_judge_set` makes one from the corpus
with edits whose effect on meaning is known in advance:

    verbatim       the sentence as written                         -> supported
    shall_to_must  "shall" rewritten as "must" (plain English)      -> supported
    shall_to_may   an obligation turned into a permission           -> not supported
    negation       "shall" turned into "shall not"                  -> not supported
    actor_swap     the duty moved to another actor                  -> not supported
    number_change  the first number increased by one                -> not supported

The quote stays the original sentence in every case, which is the realistic failure: a correct
quote with a claim that misstates it. `run_judge_eval` scores the set with the quote check
alone and with quote check + judge, so the report shows what the judge adds and what it costs
in false alarms. The set is generated, never hand-picked; `tests/test_judge.py` checks the
committed file against the generator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cited_answers.corpus import Corpus
from cited_answers.evals.metrics import ClaimToScore, ClaimVerdict, FaithfulnessScorer, QuoteScorer
from cited_answers.pipeline.definitions import DEFINITION_ARTICLES
from cited_answers.providers.base import JudgeModel, JudgeRequest, ProviderError
from cited_answers.text import split_segments

ROOT = Path(__file__).resolve().parents[3]
JUDGE_SET = ROOT / "evals" / "judge_set.jsonl"
SENTENCES_PER_LAW = 10


class JudgeScorer:
    def __init__(self, judge: JudgeModel):
        self.judge = judge
        self.name = f"judge:{judge.name}"
        self._quote = QuoteScorer()
        # A failed judge call is not a verdict. It is scored as unsupported (never as
        # supported), listed here, and fails the eval gate so the numbers are not trusted.
        self.errors: list[str] = []

    def score(self, corpus: Corpus, claim: ClaimToScore) -> ClaimVerdict:
        base = self._quote.score(corpus, claim)
        if not base.supported:
            return base
        request = JudgeRequest(
            claim.text, claim.quote, claim.label, corpus.article_text(claim.article_key)
        )
        try:
            judgement = self.judge.judge(request)
        except ProviderError as exc:
            self.errors.append(f"{claim.label}: {exc}")
            return ClaimVerdict(False, f"judge error: {exc}")
        if judgement.verdict == "supported":
            return ClaimVerdict(True)
        return ClaimVerdict(False, f"judge: {judgement.verdict}: {judgement.reason}")


# ── the judge's own test set ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class JudgeCase:
    id: str
    kind: str
    expected: bool  # True: the claim is supported by the quote
    claim: str
    quote: str
    article_key: str
    label: str


# "shall" not already followed by a negation, so that negating it reads naturally.
_SHALL = re.compile(r"\bshall\b(?! (?:not|neither|nor|no|in no)\b)")
_LABEL = re.compile(r"^\(?[0-9a-z]{1,4}[.)]\s+")  # "3. ", "(a) ": numbering, not content
_ACTOR_PAIRS = (
    ("controller", "processor"),
    ("provider", "deployer"),
    ("importer", "distributor"),
    ("Commission", "Board"),
)
_NUMBER = re.compile(r"\b\d+\b")


def _swap_actor(sentence: str) -> str | None:
    for a, b in _ACTOR_PAIRS:
        has_a = re.search(rf"\b{a}s?\b", sentence)
        has_b = re.search(rf"\b{b}s?\b", sentence)
        if bool(has_a) != bool(has_b):  # exactly one of the pair, so the swap is unambiguous
            src, dst = (a, b) if has_a else (b, a)
            return re.sub(rf"\b{src}(s?)\b", rf"{dst}\1", sentence, count=1)
    return None


def _variants(sentence: str) -> list[tuple[str, bool, str]]:
    out = [("verbatim", True, sentence)]
    out.append(("shall_to_must", True, _SHALL.sub("must", sentence, count=1)))
    out.append(("shall_to_may", False, _SHALL.sub("may", sentence, count=1)))
    out.append(("negation", False, _SHALL.sub("shall not", sentence, count=1)))
    swapped = _swap_actor(sentence)
    if swapped:
        out.append(("actor_swap", False, swapped))
    m = _NUMBER.search(sentence)
    if m:
        bumped = sentence[: m.start()] + str(int(m.group()) + 1) + sentence[m.end() :]
        out.append(("number_change", False, bumped))
    return out


def _candidate_sentences(corpus: Corpus, law: str) -> list[tuple[str, str, str]]:
    """(sentence, article_key, label): one obligation sentence per article, in corpus order."""
    seen: set[str] = set()
    found = []
    for chunk in corpus.chunks:
        key = chunk.article_key
        if chunk.law != law or chunk.kind != "article" or key in seen:
            continue
        if key == f"{law}:{DEFINITION_ARTICLES[law]}":
            continue
        for seg in split_segments(chunk.text):
            sentence = _LABEL.sub("", seg.strip())
            if (
                80 <= len(sentence) <= 280
                and sentence[:1].isupper()
                and _SHALL.search(sentence)
                and not re.search(r"\b(?:neither|nor)\b", sentence)  # negation would garble it
                and sentence.endswith((".", ";"))
            ):
                found.append((sentence, key, chunk.label))
                seen.add(key)
                break
    return found


def build_judge_set(corpus: Corpus, per_law: int = SENTENCES_PER_LAW) -> list[JudgeCase]:
    cases: list[JudgeCase] = []
    for law in ("gdpr", "ai_act"):
        pool = _candidate_sentences(corpus, law)
        step = max(1, len(pool) // per_law)
        for n, (sentence, key, label) in enumerate(pool[::step][:per_law], start=1):
            for kind, expected, claim in _variants(sentence):
                cases.append(
                    JudgeCase(f"{law}-{n:02d}-{kind}", kind, expected, claim, sentence, key, label)
                )
    return cases


def write_judge_set(cases: Sequence[JudgeCase], path: Path = JUDGE_SET) -> None:
    lines = [json.dumps(asdict(c), ensure_ascii=False) for c in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_judge_set(path: Path = JUDGE_SET) -> list[JudgeCase]:
    return [JudgeCase(**json.loads(line)) for line in path.read_text().splitlines() if line]


def _rates(cases: Sequence[JudgeCase], passed: Sequence[bool]) -> dict[str, Any]:
    bad = [p for c, p in zip(cases, passed, strict=True) if not c.expected]
    good = [p for c, p in zip(cases, passed, strict=True) if c.expected]
    right = sum(p == c.expected for c, p in zip(cases, passed, strict=True))
    return {
        "accuracy": round(right / len(cases), 4) if cases else 0.0,
        "catch_rate": round(sum(not p for p in bad) / len(bad), 4) if bad else 1.0,
        "false_alarm_rate": round(sum(not p for p in good) / len(good), 4) if good else 0.0,
    }


def run_judge_eval(
    corpus: Corpus,
    cases: Sequence[JudgeCase],
    scorer: FaithfulnessScorer,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    quote = QuoteScorer()

    def one(case: JudgeCase) -> tuple[ClaimVerdict, ClaimVerdict]:
        claim = ClaimToScore(case.claim, case.quote, case.article_key, case.label)
        return quote.score(corpus, claim), scorer.score(corpus, claim)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(one, cases))
    else:
        verdicts = [one(c) for c in cases]
    # Cases whose judge call failed have no verdict: they are left out of the judge's rates
    # (and the CLI fails the run), instead of being counted as "not supported".
    failed = {i for i, (_, j) in enumerate(verdicts) if j.reason.startswith("judge error")}
    quote_pass = [q.supported for q, _ in verdicts]
    judge_pass = [j.supported for _, j in verdicts]
    judged = [i for i in range(len(cases)) if i not in failed]
    kinds = sorted({c.kind for c in cases}, key=lambda k: [c.kind for c in cases].index(k))
    by_kind = {}
    for kind in kinds:
        idx = [i for i, c in enumerate(cases) if c.kind == kind]
        by_kind[kind] = {
            "cases": len(idx),
            "expected": "supported" if cases[idx[0]].expected else "not supported",
            "quote_passes": sum(quote_pass[i] for i in idx),
            "judge_passes": sum(judge_pass[i] for i in idx if i not in failed),
            "judge_errors": sum(i in failed for i in idx),
        }
    return {
        "scorer": scorer.name,
        "cases": len(cases),
        "judge_errors": len(failed),
        "quote_only": _rates(cases, quote_pass),
        "with_judge": _rates([cases[i] for i in judged], [judge_pass[i] for i in judged]),
        "by_kind": by_kind,
        "judge_mistakes": [
            {"id": c.id, "expected": c.expected, "claim": c.claim, "reason": j.reason}
            for i, (c, (_, j)) in enumerate(zip(cases, verdicts, strict=True))
            if j.supported != c.expected and i not in failed
        ],
    }
