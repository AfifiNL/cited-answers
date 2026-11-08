"""HTTP API.

POST /v1/answer                 answer a question, or refuse, with article-level citations
GET  /v1/articles/{law}/{num}   the full text of a cited article or annex, to check a quote
GET  /v1/sources                where the texts come from (EUR-Lex, CELEX, retrieval date)
GET  /healthz                   process is up
GET  /readyz                    corpus loaded and store reachable
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from cited_answers import __version__
from cited_answers.config import Settings
from cited_answers.corpus import article_key
from cited_answers.factory import build_pipeline
from cited_answers.laws import LAWS
from cited_answers.pipeline.graph import Pipeline
from cited_answers.pipeline.models import AnswerResult

LawKey = Literal["gdpr", "ai_act"]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["What is the right to erasure?"])
    laws: list[LawKey] | None = Field(
        default=None, description="Restrict to these texts; default: as named in the question."
    )
    debug: bool = Field(default=False, description="Include the per-node pipeline trace.")


class ArticleChunk(BaseModel):
    id: str
    label: str
    text: str


class ArticleResponse(BaseModel):
    law: str
    article_key: str
    label: str
    title: str
    chapter: str
    url: str
    chunks: list[ArticleChunk]


def create_app(pipeline: Pipeline | None = None, settings: Settings | None = None) -> FastAPI:
    pipe = pipeline or build_pipeline(settings)
    app = FastAPI(
        title="cited-answers",
        version=__version__,
        description="Answers about the EU AI Act and the GDPR, each claim cited to its "
        "article, or a refusal when the texts do not support an answer. "
        "Not legal advice.",
    )
    manifest_path = pipe.settings.corpus_dir / "MANIFEST.json"

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, object]:
        ok = pipe.retriever.ping()
        if not ok:
            raise HTTPException(status_code=503, detail="retrieval store is not reachable")
        return {
            "status": "ready",
            "chunks": len(pipe.corpus),
            "store": pipe.retriever.name,
            "corpus_version": pipe.corpus.version,
        }

    @app.post("/v1/answer", response_model=AnswerResult, response_model_exclude_none=True)
    def answer(req: AnswerRequest) -> AnswerResult:
        return pipe.answer(req.question, laws=req.laws, debug=req.debug)

    @app.get("/v1/articles/{law}/{number}", response_model=ArticleResponse)
    def article(
        law: LawKey,
        number: Annotated[str, Path(pattern=r"^(\d{1,3}[a-z]?|annex-[IVXLC]{1,6})$")],
    ) -> ArticleResponse:
        kind = "annex" if number.startswith("annex-") else "article"
        num = number.removeprefix("annex-")
        key = article_key(law, kind, num)
        chunks = pipe.corpus.by_article.get(key)
        if not chunks:
            raise HTTPException(status_code=404, detail=f"{LAWS[law].short} has no {kind} {num}")
        first = chunks[0]
        return ArticleResponse(
            law=law,
            article_key=key,
            label=first.article_label,
            title=first.title,
            chapter=first.chapter,
            url=first.url,
            chunks=[ArticleChunk(id=c.id, label=c.label, text=c.text) for c in chunks],
        )

    @app.get("/v1/sources")
    def sources() -> dict[str, object]:
        manifest: dict[str, object] = json.loads(manifest_path.read_text())
        return manifest

    return app
