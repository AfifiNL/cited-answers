"""Record/replay: a recorded run replays identically, offline; anything new fails loudly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.evals.judge import JudgeScorer
from cited_answers.evals.runner import load_dataset, run_eval
from cited_answers.factory import build_pipeline, make_scorer, open_cassette
from cited_answers.pipeline.graph import Pipeline
from cited_answers.providers.base import Draft, DraftRequest, ProviderError, Source
from cited_answers.providers.mock import ExtractiveAnswerModel, ScriptedJudge
from cited_answers.providers.replay import (
    Cassette,
    RecordingAnswerModel,
    RecordingJudge,
    ReplayAnswerModel,
    draft_key,
)
from cited_answers.retrieval.memory import MemoryStore
from cited_answers.text import TermStats

SRC = [Source("S1", "gdpr:art17:p1", "GDPR Article 17(1)", "Right to erasure", "text")]


def offline(**kw: object) -> Settings:
    values: dict[str, object] = {"provider": "mock", "embedder": "hash", "store": "memory", **kw}
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_key_covers_everything_the_model_sees() -> None:
    base = DraftRequest("q", SRC)
    assert draft_key(base) == draft_key(DraftRequest("q", SRC))
    assert draft_key(base) != draft_key(DraftRequest("q2", SRC))
    assert draft_key(base) != draft_key(DraftRequest("q", SRC, feedback="fix S1"))
    other = [Source("S1", "gdpr:art17:p1", "GDPR Article 17(1)", "Right to erasure", "text!")]
    assert draft_key(base) != draft_key(DraftRequest("q", other))


class Flaky:
    """A live model stand-in: counts calls, fails on 'boom'."""

    def __init__(self, name: str = "m"):
        self.name = name
        self.calls = 0

    def draft(self, request: DraftRequest) -> Draft:
        self.calls += 1
        if request.question == "boom":
            raise ProviderError("vendor down (HTTP 429)")
        return Draft(refuse=True, refusal_reason=request.question)


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    rec = Cassette.record(path, {"model": "m"})
    model = RecordingAnswerModel(Flaky(), rec)
    assert model.draft(DraftRequest("a", SRC)).refusal_reason == "a"
    with pytest.raises(ProviderError):
        model.draft(DraftRequest("boom", SRC))
    assert rec.errors and "HTTP 429" in rec.errors[0]  # reported...
    rec.save()
    first = path.read_text()
    assert "boom" not in first  # ...but a failed call is not an answer, so it is not recorded
    rec.save()
    assert path.read_text() == first  # stable file: re-saving changes nothing
    replay = ReplayAnswerModel(Cassette.load(path))
    assert replay.name == "replay:m"
    assert replay.draft(DraftRequest("a", SRC)).refusal_reason == "a"
    with pytest.raises(ProviderError, match="re-record"):
        replay.draft(DraftRequest("never seen", SRC))
    assert replay.cassette.misses == ["draft: never seen"]


def test_re_recording_reuses_successful_calls_of_the_same_model(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    first = Cassette.record(path, {"model": "m"})
    live = Flaky()
    RecordingAnswerModel(live, first).draft(DraftRequest("a", SRC))
    RecordingAnswerModel(live, first).draft(DraftRequest("stale", SRC))
    first.save()
    assert live.calls == 2
    again = Cassette.record(path, {"model": "m"})
    RecordingAnswerModel(live, again).draft(DraftRequest("a", SRC))  # reused, not called
    RecordingAnswerModel(live, again).draft(DraftRequest("b", SRC))  # new, called
    again.save()
    assert live.calls == 3 and again.reused == 1
    text = path.read_text()
    assert '"about": "b"' in text and "stale" not in text  # unused entries are dropped
    other = Cassette.record(path, {"model": "another-model"})
    RecordingAnswerModel(Flaky("another-model"), other).draft(DraftRequest("a", SRC))
    assert other.reused == 0  # another model's answers are never reused


def _recorded_run(corpus: Corpus, store: MemoryStore, path: Path) -> dict:
    """Record an eval run of the mock model and a scripted judge, as a live run would be."""
    cassette = Cassette.record(path, {})
    model = ExtractiveAnswerModel(TermStats.build(c.index_text for c in corpus.chunks))
    cassette.meta["model"] = model.name
    settings = offline()
    pipe = Pipeline(
        corpus, store, RecordingAnswerModel(model, cassette), settings, embedder_name="hash-768"
    )
    judge = RecordingJudge(ScriptedJudge(lambda r: "supported"), cassette)
    cassette.meta.update(judge=judge.name, embedder="hash-768")
    report, _, _ = run_eval(
        pipe, out_dir=path.parent / "rec", scorer=JudgeScorer(judge), cassette=cassette
    )
    cassette.save()
    return report


def test_replayed_eval_matches_the_recorded_one(
    corpus: Corpus, store: MemoryStore, tmp_path: Path
) -> None:
    path = tmp_path / "run.jsonl"
    recorded = _recorded_run(corpus, store, path)
    settings = offline(
        provider="replay", cassette=path, faithfulness_scorer="judge", judge_provider="replay"
    )
    cassette = open_cassette(settings)
    pipe = build_pipeline(settings, corpus, cassette=cassette)
    scorer = make_scorer(settings, cassette)
    replayed, failures, _ = run_eval(
        pipe, out_dir=tmp_path / "rep", scorer=scorer, cassette=cassette
    )
    assert failures == [] and cassette is not None and cassette.misses == []
    assert replayed["metrics"] == recorded["metrics"]
    assert replayed["config"]["model"] == "replay:mock-extractive"
    assert replayed["config"]["scorer"] == "judge:replay:scripted-judge"
    assert [i["cited"] for i in replayed["items"]] == [i["cited"] for i in recorded["items"]]


def test_changed_model_inputs_fail_the_gate(
    corpus: Corpus, store: MemoryStore, tmp_path: Path
) -> None:
    path = tmp_path / "run.jsonl"
    _recorded_run(corpus, store, path)
    # A retrieval change: one fewer source per question changes every prompt.
    settings = offline(provider="replay", cassette=path, max_sources=5)
    cassette = open_cassette(settings)
    pipe = build_pipeline(settings, corpus, cassette=cassette)
    _, failures, _ = run_eval(pipe, out_dir=tmp_path / "rep", cassette=cassette)
    assert any("not in the cassette" in f for f in failures), failures


def test_replay_needs_a_cassette(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CA_CASSETTE"):
        open_cassette(offline(provider="replay"))
    with pytest.raises(FileNotFoundError):
        open_cassette(offline(provider="replay", cassette=tmp_path / "missing.jsonl"))


def test_committed_cassette_replays_without_misses(corpus: Corpus, tmp_path: Path) -> None:
    """The cassette CI replays must still match the pipeline exactly (re-record if not)."""
    path = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "live-answers.jsonl"
    if not path.exists():
        pytest.skip("no recorded cassette yet")
    meta = json.loads(path.read_text().splitlines()[0])["meta"]
    settings = offline(
        provider="replay", cassette=path, faithfulness_scorer="judge", judge_provider="replay"
    )
    cassette = open_cassette(settings)
    pipe = build_pipeline(settings, corpus, cassette=cassette)
    items = load_dataset()[:10]
    from cited_answers.evals.runner import run_items

    run_items(pipe, items, scorer=make_scorer(settings, cassette))
    assert cassette is not None and cassette.misses == [], (meta, cassette.misses[:3])
