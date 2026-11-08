"""Public result types (also the API response schema)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RefusalCode = Literal[
    "empty_question",  # nothing to answer
    "no_evidence",  # evidence gate: the texts do not cover the question
    "model_declined",  # the model said the sources do not answer it
    "unverified",  # the model's claims failed the citation checker on every attempt
    "provider_error",  # the model vendor failed; nothing unverified is returned
]

REFUSAL_MESSAGES: dict[str, str] = {
    "empty_question": "Please ask a question.",
    "no_evidence": (
        "I can't answer this from the EU AI Act or the GDPR: no provision in either text covers it."
    ),
    "model_declined": "The retrieved provisions do not answer this question.",
    "unverified": (
        "I could not produce an answer in which every statement is backed by a quote from "
        "the cited article, so I am not giving one."
    ),
    "provider_error": "The language model could not be reached; no answer was produced.",
}


class Citation(BaseModel):
    ref: int = Field(description="Number used in the answer text, e.g. [1].")
    chunk_id: str
    article_key: str = Field(description="Article-level key, e.g. 'gdpr:art17'.")
    label: str = Field(description="Human-readable citation, e.g. 'GDPR Article 17(1)'.")
    title: str
    url: str


class AnswerClaim(BaseModel):
    text: str
    quote: str
    ref: int


class Refusal(BaseModel):
    code: RefusalCode
    message: str
    detail: str = ""


class AnswerResult(BaseModel):
    status: Literal["answered", "refused"]
    question: str
    answer: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    refusal: Refusal | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None

    @property
    def cited_articles(self) -> list[str]:
        return list(dict.fromkeys(c.article_key for c in self.citations))
