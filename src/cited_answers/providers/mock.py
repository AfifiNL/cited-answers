"""Deterministic, offline providers. They cost nothing and give the same output every run.

HashEmbedder        feature hashing of stemmed words and word pairs into a fixed-size unit
                    vector. It is lexical, not semantic: it exists so the vector path (and
                    pgvector) runs in tests and CI without a model.
ExtractiveAnswerModel
                    a stand-in for an LLM that answers by selecting the source sentences that
                    best cover the question's informative words, quoting them verbatim. It
                    produces the same Draft structure a real model must produce, so the whole
                    pipeline (gates, verifier, citations, API) is exercised end to end.
ScriptedAnswerModel returns pre-written drafts; tests use it to play a misbehaving model.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import pairwise

from cited_answers.glossary import is_covered
from cited_answers.pipeline.references import find_references, strip_references
from cited_answers.providers.base import Claim, Draft, DraftRequest
from cited_answers.text import TermStats, content_terms, split_segments

EMBED_DIM = 768


class HashEmbedder:
    semantic = False

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vector(self, text: str) -> list[float]:
        terms = content_terms(text)
        feats = Counter(terms)
        feats.update(f"{a}_{b}" for a, b in pairwise(terms))
        vec = [0.0] * self.dim
        for feat, tf in feats.items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if h >> 63 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class ExtractiveAnswerModel:
    """Quotes the source segments that best cover the question, from a single provision.

    Scoring per segment: IDF-weighted share of the question's informative words found in the
    segment or its article title (a real model sees the title too), lay words counting via
    the glossary, with a small preference for higher-ranked sources. A provision the
    question names explicitly ("Article 14") wins, as it would for an instruction-following
    model. Claims are verbatim segments, so this model never paraphrases or invents.
    """

    name = "mock-extractive"

    def __init__(
        self,
        stats: TermStats,
        *,
        max_sources: int = 4,
        max_claims: int = 2,
        min_coverage: float = 0.25,
        second_ratio: float = 0.75,
        min_segment_chars: int = 20,
    ):
        self.stats = stats
        self.min_segment_chars = min_segment_chars
        self.max_sources = max_sources
        self.max_claims = max_claims
        self.min_coverage = min_coverage
        self.second_ratio = second_ratio

    def draft(self, request: DraftRequest) -> Draft:
        refs = find_references(request.question)
        named = {f"art{a}" for a in refs.articles} | {f"anx{x}" for x in refs.annexes}
        topic = strip_references(request.question)
        weights = self.stats.weights(content_terms(topic, drop_corpus_words=True))
        total = sum(weights.values())
        if total == 0 and not named:
            return Draft(refuse=True, refusal_reason="The question has no searchable terms.")
        candidates: list[tuple[float, int, int, str, str, str]] = []
        for rank, src in enumerate(request.sources[: self.max_sources]):
            article = src.chunk_id.split(":")[1]
            title_terms = set(content_terms(src.title))
            bonus = 1.0 if article in named else 0.0
            for pos, seg in enumerate(split_segments(src.text)):
                if len(seg) < self.min_segment_chars:
                    continue  # a bare '(a)' or 'and' is not a quotable statement
                available = set(content_terms(seg)) | title_terms
                covered = sum(w for t, w in weights.items() if is_covered(t, available))
                coverage = covered / total if total else 0.0
                score = coverage * (1.0 - 0.05 * rank) + bonus
                candidates.append((score, rank, pos, seg, src.label, article))
        if not candidates:
            return Draft(refuse=True, refusal_reason="No sources were provided.")
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        best = candidates[0]
        if best[0] < self.min_coverage:
            return Draft(refuse=True, refusal_reason="No source sentence addresses the question.")
        chosen = [best]
        for cand in candidates[1:]:
            if len(chosen) >= self.max_claims or cand[0] < best[0] * self.second_ratio:
                break
            if cand[5] == best[5] and cand[3] not in {c[3] for c in chosen}:
                chosen.append(cand)
        return Draft(
            refuse=False,
            claims=[Claim(text=c[3], source=c[4], quote=c[3]) for c in chosen],
        )


class ScriptedAnswerModel:
    """Replays drafts (or functions of the request) in order; the last one repeats."""

    name = "scripted"

    def __init__(self, script: Sequence[Draft | Callable[[DraftRequest], Draft]]):
        if not script:
            raise ValueError("script must not be empty")
        self.script = list(script)
        self.calls: list[DraftRequest] = []

    def draft(self, request: DraftRequest) -> Draft:
        step = self.script[min(len(self.calls), len(self.script) - 1)]
        self.calls.append(request)
        return step(request) if callable(step) else step
