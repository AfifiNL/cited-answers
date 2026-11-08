"""Retrieval contracts shared by the in-memory and PostgreSQL stores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cited_answers.corpus import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused (RRF) score; only comparable within one query
    ranks: dict[str, int] = field(default_factory=dict)  # per-retriever rank, 1-based
    pinned: bool = False  # added because the question named this article explicitly


class Retriever(Protocol):
    name: str

    def search(self, query: str, *, k: int, laws: Sequence[str] | None = None) -> list[Hit]: ...

    def ping(self) -> bool: ...


def rrf(rankings: dict[str, list[str]], *, k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)).

    RRF needs no score calibration between a lexical and a vector retriever, which is why it
    is the default way to combine them here. Ties break on chunk id so results are stable.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[name] = rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(cid, score, ranks[cid]) for cid, score in ordered]
