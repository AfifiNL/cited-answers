from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cited_answers.api.app import create_app
from cited_answers.pipeline.graph import Pipeline


@pytest.fixture(scope="module")
def client(pipeline: Pipeline) -> TestClient:
    return TestClient(create_app(pipeline))


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["chunks"] > 1000


def test_answer(client: TestClient) -> None:
    r = client.post("/v1/answer", json={"question": "What is the right to data portability?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert {c["article_key"] for c in body["citations"]} == {"gdpr:art20"}
    assert "trace" not in body and "refusal" not in body


def test_refusal(client: TestClient) -> None:
    body = client.post(
        "/v1/answer", json={"question": "What is the speed limit on motorways?"}
    ).json()
    assert body["status"] == "refused"
    assert body["refusal"]["code"] == "no_evidence"
    assert body["citations"] == []


def test_debug_trace(client: TestClient) -> None:
    body = client.post(
        "/v1/answer", json={"question": "What is AI literacy?", "debug": True}
    ).json()
    assert body["trace"][0]["node"] == "retrieve"


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "x" * 1001}, {"question": "ok?", "laws": ["dsa"]}, {}],
)
def test_invalid_requests(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/v1/answer", json=payload).status_code == 422


def test_article_lookup(client: TestClient) -> None:
    body = client.get("/v1/articles/gdpr/17").json()
    assert body["label"] == "GDPR Article 17"
    assert body["title"].startswith("Right to erasure")
    assert body["url"].endswith("#art_17")
    annex = client.get("/v1/articles/ai_act/annex-III").json()
    assert annex["article_key"] == "ai_act:anxIII"


def test_article_lookup_errors(client: TestClient) -> None:
    assert client.get("/v1/articles/gdpr/100").status_code == 404
    assert client.get("/v1/articles/gdpr/annex-I").status_code == 404
    assert client.get("/v1/articles/gdpr/17;drop").status_code == 422


def test_sources_manifest(client: TestClient) -> None:
    body = client.get("/v1/sources").json()
    assert {d["celex"] for d in body["documents"]} == {"32016R0679", "32024R1689"}
    assert "Decision 2011/833/EU" in body["notice"]
