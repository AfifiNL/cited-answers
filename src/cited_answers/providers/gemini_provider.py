"""Gemini adapter (optional extra: `pip install cited-answers[gemini]`).

Two ways to authenticate, both handled by the google-genai SDK from the environment:
  * Gemini Developer API: GEMINI_API_KEY (or GOOGLE_API_KEY);
  * Vertex AI (CA_GEMINI_USE_VERTEX=true): GOOGLE_CLOUD_PROJECT and Application Default
    Credentials. The location comes from CA_GEMINI_LOCATION / CA_GEMINI_EMBEDDING_LOCATION, or
    GOOGLE_CLOUD_LOCATION when those are unset. Models are not offered in every location: on the
    day this was tested, Gemini 3.x chat models answered in `eu` and `global` but not in
    `europe-west4`, and `gemini-embedding-001` answered in `europe-west4` and `global` but not
    in `eu`. That is why chat and embeddings have separate location settings.
This module never reads, logs or returns credentials.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from cited_answers.providers.base import Draft, DraftRequest, ProviderError
from cited_answers.providers.prompts import SYSTEM_PROMPT, build_user_prompt

# Rate limits and transient server errors are retried with exponential backoff by the SDK.
RETRY_STATUS = (408, 429, 500, 502, 503, 504)


def _client(client: Any, vertex: bool, location: str | None, timeout_s: float) -> Any:
    if client is not None:
        return client
    from google import genai
    from google.genai import types

    http = types.HttpOptions(
        timeout=int(timeout_s * 1000),
        retry_options=types.HttpRetryOptions(attempts=4, http_status_codes=list(RETRY_STATUS)),
    )
    if vertex:
        return genai.Client(vertexai=True, location=location or None, http_options=http)
    return genai.Client(http_options=http)


def _describe(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {code})" if code else "")


class GeminiAnswerModel:
    def __init__(
        self,
        model: str,
        *,
        vertex: bool = False,
        location: str | None = None,
        timeout_s: float = 60.0,
        temperature: float | None = None,
        thinking_level: str | None = None,
        client: Any = None,
    ):
        self.model = model
        self.name = f"gemini:{model}" + (
            f":thinking-{thinking_level.lower()}" if thinking_level else ""
        )
        # None keeps the model's defaults. Google advises against lowering temperature for
        # Gemini 3. thinking_level (minimal|low|medium|high) trades latency for reasoning.
        self.temperature = temperature
        self.thinking_level = thinking_level.upper() if thinking_level else None
        self._client = _client(client, vertex, location, timeout_s)

    def draft(self, request: DraftRequest) -> Draft:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Draft,
            temperature=self.temperature,
            thinking_config=(
                types.ThinkingConfig(thinking_level=types.ThinkingLevel(self.thinking_level))
                if self.thinking_level
                else None
            ),
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=build_user_prompt(request), config=config
            )
        except Exception as exc:
            raise ProviderError(f"Gemini call failed: {_describe(exc)}") from None
        try:
            return Draft.model_validate_json(resp.text or "")
        except ValidationError:
            raise ProviderError("Gemini output did not match the answer schema") from None


class GeminiEmbedder:
    semantic = True

    def __init__(
        self,
        model: str,
        *,
        dim: int,
        vertex: bool = False,
        location: str | None = None,
        timeout_s: float = 60.0,
        client: Any = None,
        batch: int = 50,
    ):
        self.model = model
        self.dim = dim
        self.batch = batch
        self.name = f"gemini:{model}:{dim}"
        self._client = _client(client, vertex, location, timeout_s)

    def _embed(self, texts: Sequence[str], task: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            part = list(texts[i : i + self.batch])
            try:
                resp = self._client.models.embed_content(
                    model=self.model,
                    contents=part,
                    config=types.EmbedContentConfig(output_dimensionality=self.dim, task_type=task),
                )
            except Exception as exc:
                raise ProviderError(f"Gemini embedding failed: {_describe(exc)}") from None
            rows = [list(e.values or []) for e in (resp.embeddings or [])]
            if len(rows) != len(part) or any(len(r) != self.dim for r in rows):
                raise ProviderError("Gemini returned embeddings of the wrong count or size")
            # Vectors with reduced dimensionality are not unit length; cosine needs them to be.
            for r in rows:
                norm = math.sqrt(sum(v * v for v in r)) or 1.0
                out.append([v / norm for v in r])
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
