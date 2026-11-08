"""OpenAI adapter (optional extra: `pip install cited-answers[openai]`).

The API key is read by the SDK from OPENAI_API_KEY; this module never reads, logs or
returns it. Errors are reported by type and status code only, never with the raw message,
because some vendor messages echo part of the key.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from cited_answers.providers.base import (
    Draft,
    DraftRequest,
    Judgement,
    JudgeRequest,
    ProviderError,
)
from cited_answers.providers.prompts import (
    DRAFT_JSON_SCHEMA,
    JUDGE_JSON_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_judge_prompt,
    build_user_prompt,
)


def _client(client: Any, timeout_s: float) -> Any:
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI(timeout=timeout_s, max_retries=2)


def _describe(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _OpenAIStructured:
    """Shared by the answer model and the judge: one strict-JSON-schema chat call."""

    def __init__(self, model: str, *, client: Any = None, timeout_s: float = 60.0):
        self.model = model
        self.name = f"openai:{model}"
        self._client = _client(client, timeout_s)

    def _call(self, system: str, user: str, schema_name: str, schema: dict[str, Any]) -> Any:
        """Return the message, or None when the model refused (OpenAI's refusal field)."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            )
        except Exception as exc:  # the SDK raises many error types; none may leak details
            raise ProviderError(f"OpenAI call failed: {_describe(exc)}") from None
        message = resp.choices[0].message
        return None if getattr(message, "refusal", None) else message


class OpenAIAnswerModel(_OpenAIStructured):
    def draft(self, request: DraftRequest) -> Draft:
        message = self._call(SYSTEM_PROMPT, build_user_prompt(request), "draft", DRAFT_JSON_SCHEMA)
        if message is None:
            return Draft(refuse=True, refusal_reason="The model declined to answer.")
        try:
            return Draft.model_validate_json(message.content or "")
        except ValidationError:
            raise ProviderError("OpenAI output did not match the answer schema") from None


class OpenAIJudgeModel(_OpenAIStructured):
    def judge(self, request: JudgeRequest) -> Judgement:
        message = self._call(
            JUDGE_SYSTEM_PROMPT, build_judge_prompt(request), "judgement", JUDGE_JSON_SCHEMA
        )
        if message is None:
            raise ProviderError("OpenAI declined to judge the claim")
        try:
            return Judgement.model_validate_json(message.content or "")
        except ValidationError:
            raise ProviderError("OpenAI output did not match the judge schema") from None


class OpenAIEmbedder:
    semantic = True

    def __init__(
        self,
        model: str,
        *,
        dim: int,
        client: Any = None,
        timeout_s: float = 60.0,
        batch: int = 128,
    ):
        self.model = model
        self.dim = dim
        self.batch = batch
        self.name = f"openai:{model}:{dim}"
        self._client = _client(client, timeout_s)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            part = list(texts[i : i + self.batch])
            try:
                resp = self._client.embeddings.create(
                    model=self.model, input=part, dimensions=self.dim
                )
            except Exception as exc:
                raise ProviderError(f"OpenAI embedding failed: {_describe(exc)}") from None
            rows = sorted(resp.data, key=lambda d: d.index)
            if len(rows) != len(part) or any(len(r.embedding) != self.dim for r in rows):
                raise ProviderError("OpenAI returned embeddings of the wrong count or size")
            out.extend(_normalise(r.embedding) for r in rows)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
