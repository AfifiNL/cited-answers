"""Eval metrics. Deliberately re-implemented here rather than imported from the pipeline.

The pipeline's verifier checks a quote against the *chunk* it was shown. The faithfulness
metric below checks the final answer against the *full article text* that the citation
resolves to, through the citation's article key. If a refactor ever maps S2 to the wrong
chunk, or a citation to the wrong article, the verifier may still pass but this metric drops.

Definitions (all over the items of one run):
  faithfulness        supported claims / all claims in answered items. Who decides "supported"
                      is a FaithfulnessScorer. The default, QuoteScorer: the quote is verbatim
                      in the cited article, every number the claim states appears in that quote
                      (or in the citation label), and at least half of its content words appear
                      in the quote. JudgeScorer (evals/judge.py) adds an LLM judge on top, for
                      what word overlap cannot see: "shall" turned into "may", a dropped
                      exception, the controller's duty given to the processor.
  citation_precision  mean over answered, answerable items of |cited ∩ gold| / |cited|,
                      at article level (GDPR Article 17(1) and 17(3) both count as gdpr:art17).
  citation_hit_rate   share of answered, answerable items that cite at least one gold article.
  refusal_accuracy    share of all items where the system answered an answerable question or
                      refused an unanswerable one.
  unsafe_answer_rate  share of unanswerable items that were answered (the costly error).
  false_refusal_rate  share of answerable items that were refused.
  retrieval_recall    share of answerable items with a gold article among the retrieved hits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cited_answers.corpus import Corpus
from cited_answers.pipeline.models import AnswerResult
from cited_answers.text import content_terms, normalize, numbers


@dataclass(frozen=True)
class EvalItem:
    id: str
    split: str
    category: str
    question: str
    answerable: bool
    gold: list[str]


@dataclass
class ItemOutcome:
    item: EvalItem
    status: str
    refusal_code: str | None
    cited: list[str]
    retrieved: list[str]
    claims_total: int
    claims_supported: int
    unsupported: list[str] = field(default_factory=list)

    @property
    def correct_refusal_decision(self) -> bool:
        return (self.status == "answered") == self.item.answerable

    @property
    def precision(self) -> float | None:
        if self.status != "answered" or not self.item.answerable or not self.cited:
            return None
        return len(set(self.cited) & set(self.item.gold)) / len(set(self.cited))

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "split": self.item.split,
            "category": self.item.category,
            "question": self.item.question,
            "answerable": self.item.answerable,
            "gold": self.item.gold,
            "status": self.status,
            "refusal_code": self.refusal_code,
            "cited": self.cited,
            "retrieved_articles": self.retrieved[:10],
            "precision": self.precision,
            "claims_total": self.claims_total,
            "claims_supported": self.claims_supported,
            "unsupported": self.unsupported,
        }


def claim_supported(corpus: Corpus, article_key: str, label: str, text: str, quote: str) -> str:
    """Return '' when supported, else a short reason."""
    article = corpus.article_text(article_key)
    if not article:
        return f"cited article {article_key} does not exist"
    if len(quote.strip()) < 12 or normalize(quote) not in normalize(article):
        return f"quote not found verbatim in {article_key}"
    allowed = numbers(quote) | numbers(label)
    stray = numbers(text) - allowed
    if stray:
        return f"number(s) {sorted(stray)} not in quote"
    claim_terms = set(content_terms(text))
    if not claim_terms:
        return "claim has no content words"
    overlap = len(claim_terms & set(content_terms(quote))) / len(claim_terms)
    if overlap < 0.5:
        return f"claim shares only {overlap:.0%} of its words with the quote"
    return ""


@dataclass(frozen=True)
class ClaimToScore:
    text: str
    quote: str
    article_key: str  # "gdpr:art17", resolved from the answer's citation
    label: str  # "GDPR Article 17(1)"


@dataclass(frozen=True)
class ClaimVerdict:
    supported: bool
    reason: str = ""  # empty when supported


@runtime_checkable
class FaithfulnessScorer(Protocol):
    name: str

    def score(self, corpus: Corpus, claim: ClaimToScore) -> ClaimVerdict: ...


class QuoteScorer:
    """Deterministic: verbatim quote in the cited article, numbers, word overlap."""

    name = "quote"

    def score(self, corpus: Corpus, claim: ClaimToScore) -> ClaimVerdict:
        reason = claim_supported(corpus, claim.article_key, claim.label, claim.text, claim.quote)
        return ClaimVerdict(not reason, reason)


def score_item(
    corpus: Corpus,
    item: EvalItem,
    result: AnswerResult,
    scorer: FaithfulnessScorer | None = None,
) -> ItemOutcome:
    scorer = scorer or QuoteScorer()
    by_ref = {c.ref: c for c in result.citations}
    supported = 0
    unsupported: list[str] = []
    for claim in result.claims:
        cit = by_ref.get(claim.ref)
        if cit is None:
            verdict = ClaimVerdict(False, "claim has no citation")
        else:
            verdict = scorer.score(
                corpus, ClaimToScore(claim.text, claim.quote, cit.article_key, cit.label)
            )
        if verdict.supported:
            supported += 1
        else:
            unsupported.append(verdict.reason)
    retrieved: list[str] = []
    for entry in result.trace or []:
        if entry.get("node") == "retrieve":
            retrieved.extend(corpus.by_id[hit["id"]].article_key for hit in entry["hits"])
    return ItemOutcome(
        item=item,
        status=result.status,
        refusal_code=result.refusal.code if result.refusal else None,
        cited=result.cited_articles,
        retrieved=list(dict.fromkeys(retrieved)),
        claims_total=len(result.claims),
        claims_supported=supported,
        unsupported=unsupported,
    )


def _ratio(num: float, den: float, empty: float = 1.0) -> float:
    return round(num / den, 4) if den else empty


def aggregate(outcomes: Sequence[ItemOutcome]) -> dict[str, Any]:
    answerable = [o for o in outcomes if o.item.answerable]
    unanswerable = [o for o in outcomes if not o.item.answerable]
    answered_ok = [o for o in answerable if o.status == "answered"]
    precisions = [p for o in answered_ok if (p := o.precision) is not None]
    claims = sum(o.claims_total for o in outcomes)
    supported = sum(o.claims_supported for o in outcomes)
    return {
        "items": len(outcomes),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "faithfulness": _ratio(supported, claims),
        "citation_precision": _ratio(sum(precisions), len(precisions), empty=0.0),
        "citation_hit_rate": _ratio(
            sum(1 for o in answered_ok if set(o.cited) & set(o.item.gold)),
            len(answered_ok),
            empty=0.0,
        ),
        "refusal_accuracy": _ratio(
            sum(o.correct_refusal_decision for o in outcomes), len(outcomes)
        ),
        "unsafe_answer_rate": _ratio(
            sum(o.status == "answered" for o in unanswerable), len(unanswerable), empty=0.0
        ),
        "false_refusal_rate": _ratio(
            sum(o.status == "refused" for o in answerable), len(answerable), empty=0.0
        ),
        "retrieval_recall": _ratio(
            sum(1 for o in answerable if set(o.retrieved) & set(o.item.gold)),
            len(answerable),
            empty=0.0,
        ),
        "claims": claims,
    }
