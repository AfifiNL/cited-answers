from __future__ import annotations

import math

import pytest

from cited_answers.providers.mock import HashEmbedder
from cited_answers.retrieval.base import rrf
from cited_answers.retrieval.memory import MemoryStore


def test_rrf_by_hand() -> None:
    fused = rrf({"a": ["x", "y"], "b": ["y", "z"]}, k=60)
    scores = {cid: s for cid, s, _ in fused}
    assert [cid for cid, _, _ in fused] == ["y", "x", "z"]
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0][2] == {"a": 2, "b": 1}


def test_rrf_ties_break_on_id() -> None:
    assert [c for c, _, _ in rrf({"a": ["b"], "b": ["a"]})] == ["a", "b"]


def test_hash_embedder_is_deterministic_unit_vectors() -> None:
    emb = HashEmbedder(768)
    v1, v2 = emb.embed_query("right to erasure"), emb.embed_query("right to erasure")
    assert v1 == v2 and len(v1) == 768
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)
    near = sum(a * b for a, b in zip(v1, emb.embed_query("erasure right"), strict=True))
    far = sum(a * b for a, b in zip(v1, emb.embed_query("conformity assessment"), strict=True))
    assert near > far


@pytest.mark.parametrize(
    ("query", "article"),
    [
        ("right to erasure", "gdpr:art17"),
        ("notify a personal data breach to the supervisory authority", "gdpr:art33"),
        ("prohibited AI practices", "ai_act:art5"),
        ("AI literacy of staff", "ai_act:art4"),
        ("data protection officer designation", "gdpr:art37"),
    ],
)
def test_memory_store_finds_the_article(store: MemoryStore, query: str, article: str) -> None:
    keys = [h.chunk.article_key for h in store.search(query, k=5)]
    assert article in keys


def test_law_filter_is_respected(store: MemoryStore) -> None:
    hits = store.search("supervisory authority complaint", k=10, laws=["ai_act"])
    assert hits and all(h.chunk.law == "ai_act" for h in hits)


def test_search_is_deterministic(store: MemoryStore) -> None:
    a = [h.chunk.id for h in store.search("human oversight", k=8)]
    b = [h.chunk.id for h in store.search("human oversight", k=8)]
    assert a == b
