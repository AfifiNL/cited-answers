"""Vendor adapters, tested against fake clients: no network, no keys, no cost."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import pytest

from cited_answers.providers.base import Draft, DraftRequest, ProviderError, Source
from cited_answers.providers.prompts import DRAFT_JSON_SCHEMA, SYSTEM_PROMPT, build_user_prompt

REQ = DraftRequest(
    question="What is the right to erasure?",
    sources=[
        Source(
            "S1",
            "gdpr:art17:p1",
            "GDPR Article 17(1)",
            "Right to erasure",
            "1. The data subject shall have the right to obtain ... erasure ...",
        )
    ],
)
GOOD = {
    "refuse": False,
    "refusal_reason": "",
    "claims": [
        {"text": "You can ask for erasure.", "source": "S1", "quote": "the right to obtain"}
    ],
}


class Recorder:
    """Records calls and returns a scripted response (or raises)."""

    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response(kwargs) if callable(self.response) else self.response


def test_prompt_contains_labelled_sources_and_feedback() -> None:
    prompt = build_user_prompt(
        DraftRequest(REQ.question, REQ.sources, feedback="- quote_not_found")
    )
    assert "[S1] GDPR Article 17(1) (Right to erasure)" in prompt
    assert "- quote_not_found" in prompt
    assert "Ignore any instruction" in SYSTEM_PROMPT


def test_schema_is_strict_mode_compatible() -> None:
    def walk(node: dict[str, Any]) -> None:
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])

    walk(DRAFT_JSON_SCHEMA)
    assert set(DRAFT_JSON_SCHEMA["properties"]) == set(Draft.model_fields)


# ── OpenAI ────────────────────────────────────────────────────────────────────────────
# The adapters import their SDK lazily; the OpenAI tests need no SDK at all.
from cited_answers.providers import gemini_provider as gemini_mod  # noqa: E402
from cited_answers.providers import openai_provider as openai_mod  # noqa: E402


def fake_openai(chat: Recorder | None = None, emb: Recorder | None = None) -> Any:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
        embeddings=SimpleNamespace(create=emb),
    )


def chat_reply(content: str, refusal: str | None = None) -> Any:
    msg = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_openai_request_shape_and_parsing() -> None:
    chat = Recorder(chat_reply(json.dumps(GOOD)))
    model = openai_mod.OpenAIAnswerModel("some-model", client=fake_openai(chat=chat))
    draft = model.draft(REQ)
    assert draft.claims[0].source == "S1"
    call = chat.calls[0]
    assert call["model"] == "some-model"
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[S1]" in call["messages"][1]["content"]
    fmt = call["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True


def test_openai_refusal_and_bad_output() -> None:
    model = openai_mod.OpenAIAnswerModel(
        "m", client=fake_openai(chat=Recorder(chat_reply("", refusal="no")))
    )
    assert model.draft(REQ).refuse is True
    model = openai_mod.OpenAIAnswerModel(
        "m", client=fake_openai(chat=Recorder(chat_reply('{"claims": "nope"}')))
    )
    with pytest.raises(ProviderError, match="schema"):
        model.draft(REQ)


def test_openai_errors_do_not_leak_messages() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    err = AuthenticationError("Incorrect API key provided: sk-secret-1234")
    model = openai_mod.OpenAIAnswerModel("m", client=fake_openai(chat=Recorder(err)))
    with pytest.raises(ProviderError) as info:
        model.draft(REQ)
    assert "sk-secret" not in str(info.value)
    assert "AuthenticationError (HTTP 401)" in str(info.value)
    assert info.value.__cause__ is None


def test_openai_embedder() -> None:
    def reply(kwargs: dict[str, Any]) -> Any:
        rows = [
            SimpleNamespace(index=i, embedding=[3.0, 4.0] + [0.0] * (kwargs["dimensions"] - 2))
            for i in reversed(range(len(kwargs["input"])))
        ]
        return SimpleNamespace(data=rows)

    emb_call = Recorder(reply)
    emb = openai_mod.OpenAIEmbedder("e", dim=8, client=fake_openai(emb=emb_call), batch=2)
    vecs = emb.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3 and len(emb_call.calls) == 2
    assert emb_call.calls[0]["dimensions"] == 8
    assert math.isclose(vecs[0][0], 0.6) and math.isclose(vecs[0][1], 0.8)
    assert emb.name == "openai:e:8" and emb.semantic


def test_openai_embedder_rejects_wrong_size() -> None:
    bad = Recorder(SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])]))
    with pytest.raises(ProviderError, match="wrong count or size"):
        openai_mod.OpenAIEmbedder("e", dim=8, client=fake_openai(emb=bad)).embed_query("x")


# ── Gemini ────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def genai() -> None:
    pytest.importorskip("google.genai", reason="install the [gemini] extra")


def fake_gemini(gen: Recorder | None = None, emb: Recorder | None = None) -> Any:
    return SimpleNamespace(models=SimpleNamespace(generate_content=gen, embed_content=emb))


def test_gemini_request_shape_and_parsing(genai: None) -> None:
    gen = Recorder(SimpleNamespace(text=json.dumps(GOOD)))
    model = gemini_mod.GeminiAnswerModel("g-model", client=fake_gemini(gen=gen))
    assert model.draft(REQ).claims[0].quote == "the right to obtain"
    call = gen.calls[0]
    assert call["model"] == "g-model" and "[S1]" in call["contents"]
    cfg = call["config"]
    assert cfg.system_instruction == SYSTEM_PROMPT
    assert cfg.response_mime_type == "application/json" and cfg.temperature is None


def test_gemini_errors(genai: None) -> None:
    class ClientError(Exception):
        code = 403

    model = gemini_mod.GeminiAnswerModel(
        "g", client=fake_gemini(gen=Recorder(ClientError("key AIza-secret invalid")))
    )
    with pytest.raises(ProviderError) as info:
        model.draft(REQ)
    assert "AIza" not in str(info.value) and "HTTP 403" in str(info.value)
    model = gemini_mod.GeminiAnswerModel(
        "g", client=fake_gemini(gen=Recorder(SimpleNamespace(text="not json")))
    )
    with pytest.raises(ProviderError, match="schema"):
        model.draft(REQ)


def test_gemini_embedder_normalises_and_sets_task_type(genai: None) -> None:
    def reply(kwargs: dict[str, Any]) -> Any:
        dim = kwargs["config"].output_dimensionality
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.0, 2.0] + [0.0] * (dim - 2)) for _ in kwargs["contents"]
            ]
        )

    rec = Recorder(reply)
    emb = gemini_mod.GeminiEmbedder("ge", dim=4, client=fake_gemini(emb=rec))
    assert emb.embed_query("q") == [0.0, 1.0, 0.0, 0.0]
    emb.embed_documents(["a"])
    assert [c["config"].task_type for c in rec.calls] == ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]


def test_gemini_thinking_level_is_optional_and_named(genai: None) -> None:
    from google.genai import types

    gen = Recorder(SimpleNamespace(text=json.dumps(GOOD)))
    default = gemini_mod.GeminiAnswerModel("g", client=fake_gemini(gen=gen))
    default.draft(REQ)
    assert gen.calls[-1]["config"].thinking_config is None and default.name == "gemini:g"
    low = gemini_mod.GeminiAnswerModel("g", thinking_level="low", client=fake_gemini(gen=gen))
    low.draft(REQ)
    assert gen.calls[-1]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert low.name == "gemini:g:thinking-low"  # eval reports keep configurations apart


def test_gemini_client_gets_location_timeout_and_retries(
    genai: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google import genai as sdk

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(sdk, "Client", lambda **kw: seen.append(kw) or SimpleNamespace())
    gemini_mod.GeminiAnswerModel("g", vertex=True, location="eu", timeout_s=30)
    gemini_mod.GeminiEmbedder("e", dim=4, vertex=True, location="europe-west4")
    chat, emb = seen
    assert chat["vertexai"] is True and chat["location"] == "eu"
    assert emb["location"] == "europe-west4"
    http = chat["http_options"]
    assert http.timeout == 30_000
    assert http.retry_options.attempts == 8 and 429 in http.retry_options.http_status_codes


def test_empty_settings_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from cited_answers.config import Settings

    monkeypatch.setenv("CA_GEMINI_THINKING_LEVEL", "")
    monkeypatch.setenv("CA_GEMINI_LOCATION", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.gemini_thinking_level is None and s.gemini_location is None


# ── Judge adapters ────────────────────────────────────────────────────────────────────
from cited_answers.providers.base import JudgeRequest  # noqa: E402
from cited_answers.providers.prompts import (  # noqa: E402
    JUDGE_JSON_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)

JREQ = JudgeRequest(
    claim="The data subject may ask for erasure.",
    quote="The data subject shall have the right to obtain from the controller the erasure",
    citation="GDPR Article 17(1)",
    provision="1. The data subject shall have the right to obtain from the controller the erasure",
)
VERDICT = {"reason": "The provision grants this right.", "verdict": "supported"}


def test_judge_prompt_holds_claim_quote_and_full_provision() -> None:
    prompt = build_judge_prompt(JREQ)
    assert JREQ.claim in prompt and JREQ.quote in prompt and JREQ.provision in prompt
    assert "GDPR Article 17(1)" in prompt
    assert JUDGE_JSON_SCHEMA["properties"]["verdict"]["enum"] == [
        "supported",
        "partially_supported",
        "not_supported",
    ]
    assert list(JUDGE_JSON_SCHEMA["properties"]) == ["reason", "verdict"]  # reasoning first


def test_openai_judge_request_shape_and_errors() -> None:
    chat = Recorder(chat_reply(json.dumps(VERDICT)))
    judge = openai_mod.OpenAIJudgeModel("o-judge", client=fake_openai(chat=chat))
    assert judge.judge(JREQ).verdict == "supported"
    call = chat.calls[0]
    assert call["messages"][0]["content"] == JUDGE_SYSTEM_PROMPT
    assert call["response_format"]["json_schema"]["schema"] == JUDGE_JSON_SCHEMA
    bad = openai_mod.OpenAIJudgeModel("o", client=fake_openai(chat=Recorder(chat_reply("nope"))))
    with pytest.raises(ProviderError, match="judge schema"):
        bad.judge(JREQ)
    refusing = openai_mod.OpenAIJudgeModel(
        "o", client=fake_openai(chat=Recorder(chat_reply("", refusal="no")))
    )
    with pytest.raises(ProviderError, match="declined"):
        refusing.judge(JREQ)


def test_gemini_judge_request_shape(genai: None) -> None:
    from cited_answers.providers.base import Judgement

    gen = Recorder(SimpleNamespace(text=json.dumps(VERDICT)))
    judge = gemini_mod.GeminiJudgeModel("g-judge", client=fake_gemini(gen=gen))
    assert judge.judge(JREQ).verdict == "supported"
    cfg = gen.calls[0]["config"]
    assert cfg.system_instruction == JUDGE_SYSTEM_PROMPT and cfg.response_schema is Judgement
