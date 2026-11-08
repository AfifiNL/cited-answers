from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.factory import build_pipeline
from cited_answers.pipeline.graph import Pipeline
from cited_answers.providers.mock import HashEmbedder
from cited_answers.retrieval.memory import MemoryStore


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.load()


@pytest.fixture(scope="session")
def settings() -> Settings:
    # Explicit offline settings: tests never depend on the caller's environment.
    return Settings(provider="mock", embedder="hash", store="memory", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def store(corpus: Corpus) -> MemoryStore:
    return MemoryStore(corpus, HashEmbedder())


@pytest.fixture(scope="session")
def pipeline(settings: Settings, corpus: Corpus) -> Pipeline:
    return build_pipeline(settings, corpus)


@pytest.fixture(scope="session")
def pg_dsn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A PostgreSQL + pgvector DSN: CA_TEST_DATABASE_URL (CI service container) or a local
    throwaway server from the optional `pgserver` package. Skips when neither is available."""
    dsn = os.environ.get("CA_TEST_DATABASE_URL")
    if dsn:
        yield dsn
        return
    pgserver = pytest.importorskip("pgserver", reason="no CA_TEST_DATABASE_URL and no pgserver")
    server = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="delete")
    try:
        yield server.get_uri()
    finally:
        server.cleanup()
