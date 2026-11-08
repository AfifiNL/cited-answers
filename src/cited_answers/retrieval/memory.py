"""In-process hybrid store: BM25 + dense vectors, fused with RRF.

Used by tests, the offline eval gate and the default local server. Same contract as the
PostgreSQL store, so the pipeline does not know which one it is talking to.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cited_answers.corpus import Corpus
from cited_answers.providers.base import Embedder
from cited_answers.retrieval.base import Hit, rrf
from cited_answers.retrieval.bm25 import BM25


class MemoryStore:
    name = "memory"

    def __init__(self, corpus: Corpus, embedder: Embedder, *, cache_dir: Path | None = None):
        self.corpus = corpus
        self.embedder = embedder
        self.chunks = corpus.chunks
        texts = [c.index_text for c in self.chunks]
        self.bm25 = BM25(texts)
        self.vectors = self._load_vectors(texts, cache_dir)
        self._law_rows: dict[str, set[int]] = {}
        for i, c in enumerate(self.chunks):
            self._law_rows.setdefault(c.law, set()).add(i)

    def _load_vectors(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        cache = None
        if cache_dir is not None and self.embedder.semantic:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedder.name)
            cache = cache_dir / f"{safe}-{self.corpus.version}.npy"
            if cache.exists():
                arr = np.load(cache)
                if arr.shape == (len(texts), self.embedder.dim):
                    return arr
        arr = np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)
        return arr

    def ping(self) -> bool:
        return True

    def search(self, query: str, *, k: int, laws: Sequence[str] | None = None) -> list[Hit]:
        allowed: set[int] | None = None
        if laws:
            allowed = set().union(*(self._law_rows.get(law, set()) for law in laws))
        depth = max(4 * k, 40)
        lexical = [self.chunks[i].id for i, _ in self.bm25.top(query, depth, allowed)]
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        sims = self.vectors @ q
        if allowed is not None:
            mask = np.full(len(self.chunks), -np.inf, dtype=np.float32)
            mask[list(allowed)] = 0.0
            sims = sims + mask
        order = np.argsort(-sims, kind="stable")[:depth]
        vector = [self.chunks[int(i)].id for i in order if np.isfinite(sims[int(i)])]
        fused = rrf({"lexical": lexical, "vector": vector})[:k]
        return [Hit(self.corpus.by_id[cid], score, ranks) for cid, score, ranks in fused]
