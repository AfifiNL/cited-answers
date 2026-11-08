"""The two deterministic gates: evidence before generation, citations after it."""

from __future__ import annotations

from cited_answers.corpus import Corpus
from cited_answers.pipeline.evidence import assess_evidence
from cited_answers.pipeline.verify import verify_draft
from cited_answers.providers.base import Claim, Draft, Source
from cited_answers.retrieval.base import Hit
from cited_answers.retrieval.memory import MemoryStore
from cited_answers.text import TermStats

SRC = Source(
    label="S1",
    chunk_id="gdpr:art33:p1",
    citation="GDPR Article 33(1)",
    title="Notification of a personal data breach to the supervisory authority",
    text=(
        "1. In the case of a personal data breach, the controller shall without undue delay "
        "and, where feasible, not later than 72 hours after having become aware of it, notify "
        "the personal data breach to the supervisory authority."
    ),
)
QUOTE = "not later than 72 hours after having become aware of it"


def check(*claims: Claim):  # type: ignore[no-untyped-def]
    return verify_draft(
        Draft(refuse=False, claims=list(claims)),
        {"S1": SRC},
        min_support=0.5,
        min_quote_chars=12,
        max_claims=6,
    )


def test_supported_claim_passes() -> None:
    report = check(
        Claim(
            text="The controller must notify within 72 hours after becoming aware of it.",
            source="S1",
            quote=QUOTE,
        )
    )
    assert report.ok, report.problems


def test_typographic_differences_do_not_fail_a_real_quote() -> None:
    report = check(
        Claim(
            text="Notify not later than 72 hours after having become aware.",
            source="S1",
            quote="NOT later than 72  hours after having become aware",
        )
    )
    assert report.ok, report.problems


def test_unknown_source_label() -> None:
    report = check(Claim(text="Notify within 72 hours.", source="S9", quote=QUOTE))
    assert any(p.startswith("unknown_source") for p in report.problems)


def test_fabricated_quote() -> None:
    report = check(
        Claim(
            text="Notify within 72 hours.",
            source="S1",
            quote="not later than 72 hours after the breach occurred",
        )
    )
    assert any(p.startswith("quote_not_found") for p in report.problems)


def test_changed_number() -> None:
    report = check(
        Claim(text="The controller must notify within 48 hours.", source="S1", quote=QUOTE)
    )
    assert any(p.startswith("unsupported_number") for p in report.problems)


def test_number_from_the_citation_label_is_allowed() -> None:
    report = check(
        Claim(
            text="Article 33 says to notify not later than 72 hours after having become aware.",
            source="S1",
            quote=QUOTE,
        )
    )
    assert not any(p.startswith("unsupported_number") for p in report.problems)


def test_claim_unrelated_to_its_quote() -> None:
    report = check(
        Claim(text="Processors may charge a reasonable fee for copies.", source="S1", quote=QUOTE)
    )
    assert any(p.startswith("weak_support") for p in report.problems)


def test_short_quote_and_empty_draft() -> None:
    assert not check(Claim(text="controller", source="S1", quote="controller")).ok
    empty = verify_draft(
        Draft(refuse=False), {"S1": SRC}, min_support=0.5, min_quote_chars=12, max_claims=6
    )
    assert not empty.ok


def test_feedback_lists_every_problem() -> None:
    report = check(Claim(text="Notify within 48 hours.", source="S9", quote="nope"))
    fb = report.feedback()
    assert "unknown_source" in fb and "short_quote" in fb and "unsupported_number" in fb


def test_evidence_gate(corpus: Corpus, store: MemoryStore) -> None:
    stats = TermStats.build(c.index_text for c in corpus.chunks)

    def gate(q: str):  # type: ignore[no-untyped-def]
        return assess_evidence(q, store.search(q, k=8), stats, min_coverage=0.45, top_n=3)

    covered = gate("When must a personal data breach be notified?")
    assert covered.sufficient and covered.coverage >= 0.45
    off_topic = gate("How do I bake sourdough bread?")
    assert not off_topic.sufficient
    assert set(off_topic.missing_terms) >= {"sourdough", "bread"}


def test_evidence_gate_accepts_named_provision(corpus: Corpus) -> None:
    stats = TermStats.build(c.index_text for c in corpus.chunks)
    pinned = [Hit(corpus.by_id["gdpr:art12:p1"], 1.0, pinned=True)]
    assert assess_evidence("Article 12?", pinned, stats, min_coverage=0.9, top_n=3).sufficient


def test_evidence_gate_with_nothing_retrieved(corpus: Corpus) -> None:
    stats = TermStats.build(c.index_text for c in corpus.chunks)
    assert not assess_evidence("anything", [], stats, min_coverage=0.1, top_n=3).sufficient
