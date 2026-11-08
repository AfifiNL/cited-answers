from __future__ import annotations

import json

import pytest

from cited_answers.cli import main


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CA_PROVIDER", "CA_EMBEDDER", "CA_STORE", "CA_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_ask_prints_answer_and_citations(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "What is the right to data portability?"]) == 0
    out = capsys.readouterr().out
    assert "[1] GDPR Article 20" in out and "eur-lex.europa.eu" in out


def test_ask_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "How do I reset my password?"]) == 0
    assert capsys.readouterr().out.startswith("REFUSED (no_evidence)")


def test_ask_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "--json", "What does AI literacy mean?"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "answered" and body["meta"]["model"] == "mock-extractive"


def test_index_needs_a_database(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index"]) == 2
    assert "CA_DATABASE_URL" in capsys.readouterr().err
