"""Wire settings to concrete implementations. The only place that knows every vendor."""

from __future__ import annotations

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.pipeline.graph import Pipeline
from cited_answers.providers.base import AnswerModel, Embedder
from cited_answers.providers.mock import EMBED_DIM, ExtractiveAnswerModel, HashEmbedder
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


def make_model(settings: Settings, corpus: Corpus) -> AnswerModel:
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


def make_retriever(settings: Settings, corpus: Corpus, embedder: Embedder) -> Retriever:
    if settings.store == "memory":
        from cited_answers.retrieval.memory import MemoryStore

        return MemoryStore(corpus, embedder, cache_dir=settings.cache_dir / "embeddings")
    if not settings.database_url:
        raise ValueError("CA_STORE=pgvector needs CA_DATABASE_URL")
    from cited_answers.retrieval.pgvector import PgVectorStore

    return PgVectorStore(settings.database_url, corpus, embedder)


def build_pipeline(settings: Settings | None = None, corpus: Corpus | None = None) -> Pipeline:
    settings = settings or Settings()
    corpus = corpus or Corpus.load(settings.corpus_dir)
    embedder = make_embedder(settings)
    return Pipeline(
        corpus,
        make_retriever(settings, corpus, embedder),
        make_model(settings, corpus),
        settings,
        embedder_name=embedder.name,
    )
