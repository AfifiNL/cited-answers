"""Okapi BM25 over stemmed content terms (k1 = 1.2, b = 0.75)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from cited_answers.text import TermStats, content_terms


class BM25:
    def __init__(self, docs: Sequence[str], *, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_terms = [Counter(content_terms(d)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.doc_terms]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.stats = TermStats.build(docs)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, counts in enumerate(self.doc_terms):
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((i, tf))

    def scores(self, query: str, allowed: set[int] | None = None) -> dict[int, float]:
        out: dict[int, float] = {}
        for term in dict.fromkeys(content_terms(query)):
            idf = self.stats.idf(term)
            for i, tf in self.postings.get(term, ()):
                if allowed is not None and i not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return out

    def top(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scored = self.scores(query, allowed)
        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
