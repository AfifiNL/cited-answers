"""One interface for every model vendor.

The pipeline only talks to two small protocols:
  * Embedder       turns text into unit vectors of a fixed dimension;
  * AnswerModel    turns a question plus numbered sources into a structured Draft.
The evals add a third:
  * JudgeModel     decides whether one claim says what the provision it cites says.

A Draft is a list of claims. Each claim names exactly one source label (S1, S2, ...) and
carries a verbatim quote from that source. The model never writes citations as free text,
so a citation cannot be malformed, and the verifier can check every claim mechanically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str = Field(description="One sentence of the answer, in plain English.")
    source: str = Field(description="Label of the single source that supports it, e.g. 'S2'.")
    quote: str = Field(description="A verbatim span copied from that source that supports it.")


class Draft(BaseModel):
    refuse: bool = Field(description="True when the sources do not answer the question.")
    refusal_reason: str = Field(default="", description="Why the sources are insufficient.")
    claims: list[Claim] = Field(default_factory=list)


@dataclass(frozen=True)
class Source:
    """A retrieved chunk as shown to the model."""

    label: str  # "S1"
    chunk_id: str
    citation: str  # "GDPR Article 17(1)"
    title: str
    text: str


@dataclass(frozen=True)
class DraftRequest:
    question: str
    sources: Sequence[Source]
    feedback: str = ""  # verifier findings from a previous attempt, if any


@runtime_checkable
class Embedder(Protocol):
    # Stable id, part of the index key, e.g. "hash-768" or "openai:text-embedding-3-small:768".
    name: str
    dim: int
    semantic: bool  # False for the lexical hashing embedder used offline

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class AnswerModel(Protocol):
    name: str

    def draft(self, request: DraftRequest) -> Draft: ...


@dataclass(frozen=True)
class JudgeRequest:
    claim: str
    quote: str  # the quote the answer gave as support
    citation: str  # "GDPR Article 17(1)"
    provision: str  # full text of the cited article


class Judgement(BaseModel):
    # The reason comes first so that a model writes its reasoning before the verdict.
    reason: str = Field(
        description="One or two sentences on what the provision does or does not back."
    )
    verdict: Literal["supported", "partially_supported", "not_supported"]


@runtime_checkable
class JudgeModel(Protocol):
    name: str

    def judge(self, request: JudgeRequest) -> Judgement: ...


class ProviderError(RuntimeError):
    """A vendor call failed or returned something that does not fit the Draft schema."""
