"""Integration tests against a real PostgreSQL with pgvector (marked `pg`)."""

from __future__ import annotations

import psycopg
import pytest

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.factory import build_pipeline
from cited_answers.providers.mock import HashEmbedder
from cited_answers.retrieval.pgvector import PgVectorStore, _or_query

pytestmark = pytest.mark.pg


@pytest.fixture(scope="module")
def pg_store(pg_dsn: str, corpus: Corpus) -> PgVectorStore:
    with psycopg.connect(pg_dsn) as conn:  # start from an empty schema
        conn.execute("DROP TABLE IF EXISTS chunk_embeddings, chunks, corpus_meta CASCADE")
    store = PgVectorStore(pg_dsn, corpus, HashEmbedder())
    first = store.sync()
    assert first["embeddings_written"] == len(corpus)
    return store


def test_sync_is_idempotent(pg_store: PgVectorStore, corpus: Corpus) -> None:
    assert pg_store.sync() == {"chunks_upserted": 0, "chunks_deleted": 0, "embeddings_written": 0}
    assert pg_store.loaded_version() == corpus.version
    assert pg_store.ping()


def test_hnsw_index_exists(pg_dsn: str, pg_store: PgVectorStore) -> None:
    with psycopg.connect(pg_dsn) as conn:
        indexdef = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunk_embeddings_hnsw'"
        ).fetchone()
    assert indexdef and "hnsw" in indexdef[0] and "vector_cosine_ops" in indexdef[0]


@pytest.mark.parametrize(
    ("query", "article"),
    [
        ("right to erasure", "gdpr:art17"),
        ("notify a personal data breach to the supervisory authority", "gdpr:art33"),
        ("prohibited AI practices", "ai_act:art5"),
    ],
)
def test_pg_search_finds_the_article(pg_store: PgVectorStore, query: str, article: str) -> None:
    assert article in [h.chunk.article_key for h in pg_store.search(query, k=5)]


def test_pg_law_filter(pg_store: PgVectorStore) -> None:
    hits = pg_store.search("supervisory authority", k=10, laws=["ai_act"])
    assert hits and all(h.chunk.law == "ai_act" for h in hits)
    assert all(set(h.ranks) <= {"lexical", "vector"} for h in hits)


def test_query_text_cannot_inject_sql(pg_store: PgVectorStore) -> None:
    assert _or_query("erasure'); DROP TABLE chunks; --") == "erasure | drop | table | chunks"
    pg_store.search("erasure'); DROP TABLE chunks; --", k=3)
    assert pg_store.sync()["chunks_upserted"] == 0  # table still there and unchanged


def test_sync_removes_chunks_that_left_the_corpus(pg_dsn: str, corpus: Corpus) -> None:
    smaller = Corpus([c for c in corpus.chunks if c.id != "gdpr:art17:p1"])
    store = PgVectorStore(pg_dsn, smaller, HashEmbedder())
    assert store.sync()["chunks_deleted"] == 1
    full = PgVectorStore(pg_dsn, corpus, HashEmbedder())
    counts = full.sync()
    assert counts["chunks_upserted"] == 1 and counts["embeddings_written"] == 1


def test_pipeline_on_pgvector(pg_dsn: str, pg_store: PgVectorStore, corpus: Corpus) -> None:
    settings = Settings(store="pgvector", database_url=pg_dsn, _env_file=None)  # type: ignore[call-arg]
    pipe = build_pipeline(settings, corpus)
    result = pipe.answer("How quickly must a controller notify a personal data breach?")
    assert result.status == "answered" and "gdpr:art33" in result.cited_articles
    assert result.meta["store"] == "pgvector"
    assert pipe.answer("How do I bake sourdough bread?").status == "refused"
