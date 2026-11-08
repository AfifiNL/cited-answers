"""Settings from environment variables (prefix CA_). Defaults are offline and free."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CA_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    # Which implementations to use. "mock" and "hash" never touch the network.
    provider: Literal["mock", "openai", "gemini", "replay"] = "mock"
    embedder: Literal["hash", "openai", "gemini"] = "hash"
    store: Literal["memory", "pgvector"] = "memory"
    database_url: str | None = None

    # Model ids are configuration, not code. Check current names and prices before a live run.
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_use_vertex: bool = False  # True: Vertex AI with GOOGLE_CLOUD_PROJECT
    # Vertex locations; unset means GOOGLE_CLOUD_LOCATION. Chat and embedding models are not
    # offered in the same locations, so they are set separately.
    gemini_location: str | None = None
    gemini_embedding_location: str | None = None
    # Gemini 3 only: minimal | low | medium | high. Unset keeps the model's default.
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    request_timeout_s: float = 60.0

    # Faithfulness scoring in evals: "quote" (deterministic) or "judge" (quote check + LLM judge).
    faithfulness_scorer: Literal["quote", "judge"] = "quote"
    judge_provider: Literal["openai", "gemini", "replay"] = "gemini"
    judge_model: str | None = None  # unset: the provider's answer model
    judge_location: str | None = None  # Vertex location for the judge; unset: gemini_location
    judge_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    # Record/replay (providers/replay.py). Replay reads `cassette`; a live run with
    # `record_cassette` set stores every model and judge call there.
    cassette: Path | None = None
    record_cassette: Path | None = None

    # Retrieval
    top_k: int = Field(default=8, ge=1, le=50)
    per_article_cap: int = Field(default=3, ge=1)
    max_sources: int = Field(default=6, ge=1, le=20)
    # Same-article context added to the sources (pipeline/linking.py): lead paragraphs and
    # exceptions to the paragraphs shown.
    max_linked_sources: int = Field(default=3, ge=0, le=6)

    # Gates (calibrated offline on the dev split; recalibrate for a semantic embedder)
    min_evidence_coverage: float = Field(default=0.45, ge=0.0, le=1.0)
    evidence_top_n: int = Field(default=3, ge=1)
    min_claim_support: float = Field(default=0.5, ge=0.0, le=1.0)
    min_quote_chars: int = Field(default=12, ge=1)
    max_claims: int = Field(default=6, ge=1)
    max_attempts: int = Field(default=2, ge=1, le=5)

    corpus_dir: Path = ROOT / "data" / "corpus"
    cache_dir: Path = ROOT / ".cache"
