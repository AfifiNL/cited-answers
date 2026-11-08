"""End-to-end behaviour of the LangGraph pipeline, including a misbehaving model."""

from __future__ import annotations

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.pipeline.graph import Pipeline
from cited_answers.providers.base import Claim, Draft, DraftRequest, ProviderError
from cited_answers.providers.mock import ScriptedAnswerModel
from cited_answers.retrieval.memory import MemoryStore

BREACH_Q = "When must a controller notify a personal data breach to the supervisory authority?"


def make(corpus: Corpus, store: MemoryStore, settings: Settings, model: object) -> Pipeline:
    return Pipeline(corpus, store, model, settings, embedder_name="hash-768")  # type: ignore[arg-type]


def first_source_quote(request: DraftRequest) -> Draft:
    src = request.sources[0]
    line = next(seg for seg in src.text.split("\n") if len(seg) > 40)
    return Draft(refuse=False, claims=[Claim(text=line, source=src.label, quote=line)])


def test_answer_has_article_citations(pipeline: Pipeline) -> None:
    result = pipeline.answer(BREACH_Q)
    assert result.status == "answered"
    assert "gdpr:art33" in result.cited_articles
    assert "72 hours" in (result.answer or "")
    assert all(c.url.startswith("https://eur-lex.europa.eu/") for c in result.citations)
    assert all(f"[{c.ref}]" in (result.answer or "") for c in result.claims)


def test_out_of_scope_question_is_refused(pipeline: Pipeline) -> None:
    result = pipeline.answer("How do I bake sourdough bread?")
    assert result.status == "refused"
    assert result.refusal is not None and result.refusal.code == "no_evidence"
    assert result.citations == [] and result.answer is None


def test_empty_question(pipeline: Pipeline) -> None:
    result = pipeline.answer("   ")
    assert result.refusal is not None and result.refusal.code == "empty_question"


def test_named_article_is_retrieved_first(pipeline: Pipeline) -> None:
    result = pipeline.answer("What does Article 14 of the AI Act say?", debug=True)
    assert result.cited_articles == ["ai_act:art14"]
    retrieve = next(t for t in result.trace or [] if t["node"] == "retrieve")
    assert retrieve["hits"][0]["pinned"] is True
    assert retrieve["laws"] == ["ai_act"]


def test_question_naming_one_law_searches_only_that_law(pipeline: Pipeline) -> None:
    result = pipeline.answer("What are the penalties under the GDPR?", debug=True)
    retrieve = next(t for t in result.trace or [] if t["node"] == "retrieve")
    assert all(h["id"].startswith("gdpr:") for h in retrieve["hits"])


def test_debug_trace_covers_every_node(pipeline: Pipeline) -> None:
    result = pipeline.answer(BREACH_Q, debug=True)
    assert [t["node"] for t in result.trace or []] == [
        "retrieve",
        "assess_evidence",
        "generate",
        "verify",
        "finalize",
    ]
    assert pipeline.answer(BREACH_Q).trace is None


def test_fabricated_quote_is_retried_with_feedback(
    corpus: Corpus, store: MemoryStore, settings: Settings
) -> None:
    bad = Draft(
        refuse=False,
        claims=[
            Claim(
                text="Notify within 24 hours.", source="S1", quote="within 24 hours of the breach"
            )
        ],
    )
    model = ScriptedAnswerModel([bad, first_source_quote])
    result = make(corpus, store, settings, model).answer(BREACH_Q)
    assert result.status == "answered"
    assert len(model.calls) == 2
    assert "quote_not_found" in model.calls[1].feedback
    assert "24 hours" not in (result.answer or "")


def test_model_that_keeps_hallucinating_gets_a_refusal(
    corpus: Corpus, store: MemoryStore, settings: Settings
) -> None:
    bad = Draft(
        refuse=False,
        claims=[Claim(text="Fines are EUR 99.", source="S7", quote="fines of EUR 99 apply")],
    )
    model = ScriptedAnswerModel([bad])
    result = make(corpus, store, settings, model).answer(BREACH_Q)
    assert result.status == "refused"
    assert result.refusal is not None and result.refusal.code == "unverified"
    assert len(model.calls) == settings.max_attempts
    assert result.claims == [] and result.citations == []


def test_model_refusal_is_passed_through(
    corpus: Corpus, store: MemoryStore, settings: Settings
) -> None:
    model = ScriptedAnswerModel([Draft(refuse=True, refusal_reason="not in sources")])
    result = make(corpus, store, settings, model).answer(BREACH_Q)
    assert result.refusal is not None and result.refusal.code == "model_declined"
    assert result.refusal.detail == "not in sources"


def test_provider_error_is_a_refusal_not_a_crash(
    corpus: Corpus, store: MemoryStore, settings: Settings
) -> None:
    def boom(_: DraftRequest) -> Draft:
        raise ProviderError("OpenAI call failed: RateLimitError (HTTP 429)")

    result = make(corpus, store, settings, ScriptedAnswerModel([boom])).answer(BREACH_Q)
    assert result.refusal is not None and result.refusal.code == "provider_error"


def test_model_is_not_called_when_evidence_is_missing(
    corpus: Corpus, store: MemoryStore, settings: Settings
) -> None:
    model = ScriptedAnswerModel([first_source_quote])
    result = make(corpus, store, settings, model).answer("Who won the 2022 World Cup?")
    assert result.status == "refused" and model.calls == []


def test_sources_shown_to_the_model_are_labelled_and_capped(
    corpus: Corpus, store: MemoryStore, settings: Settings
) -> None:
    model = ScriptedAnswerModel([first_source_quote])
    make(corpus, store, settings, model).answer(BREACH_Q)
    labels = [s.label for s in model.calls[0].sources]
    assert labels == [f"S{i}" for i in range(1, len(labels) + 1)]
    assert len(labels) <= settings.max_sources + settings.max_linked_sources
