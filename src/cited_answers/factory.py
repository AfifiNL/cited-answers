"""Wire settings to concrete implementations. The only place that knows every vendor."""

from __future__ import annotations

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.evals.metrics import FaithfulnessScorer, QuoteScorer
from cited_answers.pipeline.graph import Pipeline
from cited_answers.providers.base import AnswerModel, Embedder, JudgeModel
from cited_answers.providers.mock import EMBED_DIM, ExtractiveAnswerModel, HashEmbedder
from cited_answers.providers.replay import (
    Cassette,
    RecordingAnswerModel,
    RecordingJudge,
    ReplayAnswerModel,
    ReplayJudge,
)
from cited_answers.retrieval.base import Retriever
from cited_answers.text import TermStats


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder(EMBED_DIM)
    if settings.embedder == "openai":
        from cited_answers.providers.openai_provider import OpenAIEmbedder

        return OpenAIEmbedder(
            settings.openai_embedding_model, dim=EMBED_DIM, timeout_s=settings.request_timeout_s
        )
    from cited_answers.providers.gemini_provider import GeminiEmbedder

    return GeminiEmbedder(
        settings.gemini_embedding_model,
        dim=EMBED_DIM,
        vertex=settings.gemini_use_vertex,
        location=settings.gemini_embedding_location,
        timeout_s=settings.request_timeout_s,
    )


def open_cassette(settings: Settings) -> Cassette | None:
    """The cassette this run records to or replays from, if any."""
    if settings.record_cassette:
        return Cassette.record(settings.record_cassette, {})
    replaying = settings.provider == "replay" or (
        settings.faithfulness_scorer == "judge" and settings.judge_provider == "replay"
    )
    if not replaying:
        return None
    if not settings.cassette:
        raise ValueError("replay needs CA_CASSETTE (a recorded cassette file)")
    return Cassette.load(settings.cassette)


def _live_model(settings: Settings, corpus: Corpus) -> AnswerModel:
    if settings.provider == "mock":
        return ExtractiveAnswerModel(TermStats.build(c.index_text for c in corpus.chunks))
    if settings.provider == "openai":
        from cited_answers.providers.openai_provider import OpenAIAnswerModel

        return OpenAIAnswerModel(settings.openai_model, timeout_s=settings.request_timeout_s)
    from cited_answers.providers.gemini_provider import GeminiAnswerModel

    return GeminiAnswerModel(
        settings.gemini_model,
        vertex=settings.gemini_use_vertex,
        location=settings.gemini_location,
        timeout_s=settings.request_timeout_s,
        thinking_level=settings.gemini_thinking_level,
    )


def make_model(settings: Settings, corpus: Corpus, cassette: Cassette | None = None) -> AnswerModel:
    if settings.provider == "replay":
        if cassette is None:
            raise ValueError("CA_PROVIDER=replay needs a cassette")
        return ReplayAnswerModel(cassette)
    model = _live_model(settings, corpus)
    if cassette is not None and cassette.recording and settings.provider != "mock":
        cassette.meta["model"] = model.name
        return RecordingAnswerModel(model, cassette)
    return model


def make_judge(settings: Settings, cassette: Cassette | None = None) -> JudgeModel:
    if settings.judge_provider == "replay":
        if cassette is None:
            raise ValueError("CA_JUDGE_PROVIDER=replay needs a cassette")
        return ReplayJudge(cassette)
    judge: JudgeModel
    if settings.judge_provider == "openai":
        from cited_answers.providers.openai_provider import OpenAIJudgeModel

        judge = OpenAIJudgeModel(
            settings.judge_model or settings.openai_model, timeout_s=settings.request_timeout_s
        )
    else:
        from cited_answers.providers.gemini_provider import GeminiJudgeModel

        judge = GeminiJudgeModel(
            settings.judge_model or settings.gemini_model,
            vertex=settings.gemini_use_vertex,
            location=settings.judge_location or settings.gemini_location,
            timeout_s=settings.request_timeout_s,
            thinking_level=settings.judge_thinking_level,
        )
    if cassette is not None and cassette.recording:
        cassette.meta["judge"] = judge.name
        return RecordingJudge(judge, cassette)
    return judge


def make_scorer(settings: Settings, cassette: Cassette | None = None) -> FaithfulnessScorer:
    if settings.faithfulness_scorer == "quote":
        return QuoteScorer()
    from cited_answers.evals.judge import JudgeScorer

    return JudgeScorer(make_judge(settings, cassette))


def make_retriever(settings: Settings, corpus: Corpus, embedder: Embedder) -> Retriever:
    if settings.store == "memory":
        from cited_answers.retrieval.memory import MemoryStore

        return MemoryStore(corpus, embedder, cache_dir=settings.cache_dir / "embeddings")
    if not settings.database_url:
        raise ValueError("CA_STORE=pgvector needs CA_DATABASE_URL")
    from cited_answers.retrieval.pgvector import PgVectorStore

    return PgVectorStore(settings.database_url, corpus, embedder)


def build_pipeline(
    settings: Settings | None = None,
    corpus: Corpus | None = None,
    cassette: Cassette | None = None,
) -> Pipeline:
    settings = settings or Settings()
    corpus = corpus or Corpus.load(settings.corpus_dir)
    if cassette is None and settings.provider == "replay":
        cassette = open_cassette(settings)
    embedder = make_embedder(settings)
    retriever = make_retriever(settings, corpus, embedder)
    if cassette is not None and cassette.recording:
        cassette.meta.update(
            embedder=embedder.name, store=retriever.name, corpus_version=corpus.version
        )
    return Pipeline(
        corpus,
        retriever,
        make_model(settings, corpus, cassette),
        settings,
        embedder_name=embedder.name,
    )
