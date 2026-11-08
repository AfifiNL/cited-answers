"""Record a live run once; replay it in CI at zero cost.

A cassette maps a hash of exactly what a model was sent (system prompt, user prompt and output
schema) to what it answered. `Recording*` wraps a live model and stores every call.
`Replay*` serves the stored answers without any network access.

Replay is strict. A request the cassette has not seen means the model's input changed: a
different prompt, retrieval returning other chunks, or a different verifier message on a
retry. The recorded answers then say nothing about the new behaviour, so the miss is counted
and the eval gate fails with a request to re-record (`make record-live`). It never falls back
to a live call or to a guess.

Only successful calls are recorded. A failed live call (a rate limit, a timeout) is not an
answer the model gave, so it is reported and the recording run fails; running it again reuses
every successful call already in the file (same request, same model) and pays only for the
rest. Entries the new run did not use are dropped when the file is saved.

What replay does gate: everything downstream of the model (verification, retries, refusals,
citation mapping, the metrics and the judge's verdicts), and that the model's inputs did not
change unnoticed. What it does not: how a newer model version or a changed prompt would behave.
That needs a new recording.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from cited_answers.providers.base import (
    AnswerModel,
    Draft,
    DraftRequest,
    Judgement,
    JudgeModel,
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

MISS_HINT = "no recorded response for this request; re-record the cassette (make record-live)"


def request_key(kind: str, system: str, user: str, schema: dict[str, Any]) -> str:
    blob = json.dumps(
        {"kind": kind, "system": system, "user": user, "schema": schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def draft_key(request: DraftRequest) -> str:
    return request_key("draft", SYSTEM_PROMPT, build_user_prompt(request), DRAFT_JSON_SCHEMA)


def judge_key(request: JudgeRequest) -> str:
    return request_key("judge", JUDGE_SYSTEM_PROMPT, build_judge_prompt(request), JUDGE_JSON_SCHEMA)


class Cassette:
    """A JSONL file: one `{"meta": ...}` line, then one line per recorded call, sorted."""

    def __init__(self, path: Path, *, recording: bool, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.recording = recording
        self.meta: dict[str, Any] = dict(meta or {})
        self.entries: dict[str, dict[str, Any]] = {}
        self.misses: list[str] = []  # replay: requests the cassette did not have
        self.errors: list[str] = []  # recording: live calls that failed (not recorded)
        self.reused = 0  # recording: calls served from the previous recording
        self._prior: dict[str, dict[str, Any]] = {}
        self._prior_meta: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path) -> Cassette:
        if not Path(path).exists():
            raise FileNotFoundError(f"cassette {path} does not exist; record it first")
        cassette = cls(path, recording=False)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "meta" in row:
                cassette.meta = row["meta"]
            else:
                cassette.entries[row["key"]] = row
        return cassette

    @classmethod
    def record(cls, path: Path, meta: dict[str, Any]) -> Cassette:
        stamped = {**meta, "recorded_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        cassette = cls(path, recording=True, meta=stamped)
        if Path(path).exists():
            prior = cls.load(path)
            cassette._prior = prior.entries
            cassette._prior_meta = prior.meta
        return cassette

    def get(self, key: str, what: str) -> dict[str, Any]:
        entry = self.entries.get(key)
        if entry is None:
            with self._lock:
                self.misses.append(what)
            raise ProviderError(MISS_HINT)
        return entry

    def reuse(self, key: str, role: str) -> dict[str, Any] | None:
        """A previous recording of this exact request by the same model ('model' or 'judge')."""
        entry = self._prior.get(key)
        if entry is None or "response" not in entry:
            return None
        if self._prior_meta.get(role) != self.meta.get(role):
            return None
        with self._lock:
            self.entries[key] = entry
            self.reused += 1
        return entry

    def put(self, key: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self.entries[key] = {"key": key, **entry}

    def failed(self, what: str) -> None:
        with self._lock:
            self.errors.append(what)

    def save(self) -> None:
        rows = sorted(
            self.entries.values(),
            key=lambda e: (e["kind"], e.get("about", ""), e.get("attempt", ""), e["key"]),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"meta": self.meta}, ensure_ascii=False, sort_keys=True)]
        lines += [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(
    cassette: Cassette, key: str, role: str, entry: dict[str, Any], call: Any, parse: Any
) -> Any:
    prior = cassette.reuse(key, role)
    if prior is not None:
        return parse(prior["response"])
    try:
        result = call()
    except ProviderError as exc:
        cassette.failed(f"{entry['kind']}: {str(entry['about'])[:80]}: {exc}")
        raise
    cassette.put(key, {**entry, "response": result.model_dump()})
    return result


class RecordingAnswerModel:
    def __init__(self, inner: AnswerModel, cassette: Cassette):
        self.inner = inner
        self.name = inner.name
        self.cassette = cassette

    def draft(self, request: DraftRequest) -> Draft:
        entry = {
            "kind": "draft",
            "about": request.question,
            "attempt": "retry" if request.feedback else "first",
            "sources": [s.chunk_id for s in request.sources],
        }
        result: Draft = _record(
            self.cassette,
            draft_key(request),
            "model",
            entry,
            lambda: self.inner.draft(request),
            Draft.model_validate,
        )
        return result


class ReplayAnswerModel:
    def __init__(self, cassette: Cassette):
        self.cassette = cassette
        self.name = f"replay:{cassette.meta.get('model', 'unknown')}"

    def draft(self, request: DraftRequest) -> Draft:
        entry = self.cassette.get(draft_key(request), f"draft: {request.question}")
        return Draft.model_validate(entry["response"])


class RecordingJudge:
    def __init__(self, inner: JudgeModel, cassette: Cassette):
        self.inner = inner
        self.name = inner.name
        self.cassette = cassette

    def judge(self, request: JudgeRequest) -> Judgement:
        entry = {"kind": "judge", "about": request.citation, "claim": request.claim}
        result: Judgement = _record(
            self.cassette,
            judge_key(request),
            "judge",
            entry,
            lambda: self.inner.judge(request),
            Judgement.model_validate,
        )
        return result


class ReplayJudge:
    def __init__(self, cassette: Cassette):
        self.cassette = cassette
        self.name = f"replay:{cassette.meta.get('judge', 'unknown')}"

    def judge(self, request: JudgeRequest) -> Judgement:
        entry = self.cassette.get(judge_key(request), f"judge: {request.claim}")
        return Judgement.model_validate(entry["response"])
