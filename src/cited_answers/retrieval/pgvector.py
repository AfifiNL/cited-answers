"""PostgreSQL + pgvector store: the same hybrid retrieval as MemoryStore, in the database.

Tables (created idempotently by `migrate`):
  chunks            one row per chunk, with a generated `tsvector` for full-text search
  chunk_embeddings  (chunk_id, embedder) -> vector(768), HNSW index for cosine distance
  corpus_meta       which corpus version is loaded

`sync` makes the database match the committed corpus: it upserts changed chunks, deletes
chunks that no longer exist, and embeds only chunks whose content hash changed for the
current embedder, so re-indexing after a small corpus change costs only the changed rows.

Search runs a vector query and a full-text query and fuses both rankings with the same RRF
function as the in-memory store. Postgres ranks text with ts_rank, not BM25, so rankings
differ slightly between stores; the eval baseline is recorded per store for that reason.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg

from cited_answers.corpus import Chunk, Corpus
from cited_answers.providers.base import Embedder
from cited_answers.providers.mock import EMBED_DIM
from cited_answers.retrieval.base import Hit, rrf
from cited_answers.text import STOPWORDS, tokens

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    law         text NOT NULL,
    kind        text NOT NULL,
    number      text NOT NULL,
    paragraph   text,
    point       text,
    title       text NOT NULL,
    chapter     text NOT NULL,
    body        text NOT NULL,
    anchor      text NOT NULL,
    index_text  text NOT NULL,
    content_sha text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', index_text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_law_idx ON chunks (law);
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedder    text NOT NULL,
    content_sha text NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    PRIMARY KEY (chunk_id, embedder)
);
CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS corpus_meta (key text PRIMARY KEY, value text NOT NULL);
"""

_COLUMNS = "id, law, kind, number, paragraph, point, title, chapter, body, anchor"


def _sha(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.index_text.encode()).hexdigest()[:16]


def _vec(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.7f}" for v in values) + "]"


def _or_query(text: str) -> str | None:
    """'right to erasure' -> 'right | erasure' for to_tsquery. Only [a-z0-9] reach SQL."""
    words = [t for t in dict.fromkeys(tokens(text)) if t not in STOPWORDS]
    safe = [w for w in (re.sub(r"[^a-z0-9]", "", w) for w in words) if w]
    return " | ".join(safe) if safe else None


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str, corpus: Corpus, embedder: Embedder, *, ef_search: int = 100):
        if embedder.dim != EMBED_DIM:
            raise ValueError(f"embedder dim {embedder.dim} != table dim {EMBED_DIM}")
        self.dsn = dsn
        self.corpus = corpus
        self.embedder = embedder
        self.ef_search = ef_search

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection[Any]]:
        with psycopg.connect(self.dsn) as conn:
            yield conn

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def migrate(self) -> None:
        with self._conn() as conn:
            conn.execute(SCHEMA)

    def sync(self, *, batch: int = 64) -> dict[str, int]:
        """Make the database match the corpus for this embedder. Returns counts."""
        self.migrate()
        wanted = {c.id: c for c in self.corpus.chunks}
        with self._conn() as conn:
            existing = dict(conn.execute("SELECT id, content_sha FROM chunks").fetchall())
            stale = [cid for cid in existing if cid not in wanted]
            if stale:
                conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (stale,))
            changed = [c for c in self.corpus.chunks if existing.get(c.id) != _sha(c)]
            with conn.cursor() as cur:
                cur.executemany(
                    f"""INSERT INTO chunks ({_COLUMNS}, index_text, content_sha)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                          law = EXCLUDED.law, kind = EXCLUDED.kind, number = EXCLUDED.number,
                          paragraph = EXCLUDED.paragraph, point = EXCLUDED.point,
                          title = EXCLUDED.title, chapter = EXCLUDED.chapter,
                          body = EXCLUDED.body, anchor = EXCLUDED.anchor,
                          index_text = EXCLUDED.index_text,
                          content_sha = EXCLUDED.content_sha""",
                    [
                        (
                            c.id,
                            c.law,
                            c.kind,
                            c.number,
                            c.paragraph,
                            c.point,
                            c.title,
                            c.chapter,
                            c.text,
                            c.anchor,
                            c.index_text,
                            _sha(c),
                        )
                        for c in changed
                    ],
                )
            have = dict(
                conn.execute(
                    "SELECT chunk_id, content_sha FROM chunk_embeddings WHERE embedder = %s",
                    (self.embedder.name,),
                ).fetchall()
            )
            todo = [c for c in self.corpus.chunks if have.get(c.id) != _sha(c)]
            for i in range(0, len(todo), batch):
                part = todo[i : i + batch]
                vectors = self.embedder.embed_documents([c.index_text for c in part])
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO chunk_embeddings (chunk_id, embedder, content_sha, embedding)
                           VALUES (%s, %s, %s, %s::vector)
                           ON CONFLICT (chunk_id, embedder) DO UPDATE SET
                             content_sha = EXCLUDED.content_sha,
                             embedding = EXCLUDED.embedding""",
                        [
                            (c.id, self.embedder.name, _sha(c), _vec(v))
                            for c, v in zip(part, vectors, strict=True)
                        ],
                    )
            conn.execute(
                """INSERT INTO corpus_meta (key, value) VALUES ('corpus_version', %s)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                (self.corpus.version,),
            )
        return {
            "chunks_upserted": len(changed),
            "chunks_deleted": len(stale),
            "embeddings_written": len(todo),
        }

    def loaded_version(self) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM corpus_meta WHERE key = 'corpus_version'"
            ).fetchone()
        return row[0] if row else None

    def search(self, query: str, *, k: int, laws: Sequence[str] | None = None) -> list[Hit]:
        depth = max(4 * k, 40)
        law_list = list(laws) if laws else None
        qvec = _vec(self.embedder.embed_query(query))
        tsq = _or_query(query)
        with self._conn() as conn:
            # Session setting: candidate list size for the HNSW scan (exactness vs speed).
            conn.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")
            vector_ids = [
                r[0]
                for r in conn.execute(
                    """SELECT e.chunk_id
                     FROM chunk_embeddings e JOIN chunks c ON c.id = e.chunk_id
                    WHERE e.embedder = %(embedder)s
                      AND (%(laws)s::text[] IS NULL OR c.law = ANY(%(laws)s::text[]))
                    ORDER BY e.embedding <=> %(qvec)s::vector, e.chunk_id
                    LIMIT %(depth)s""",
                    {
                        "embedder": self.embedder.name,
                        "laws": law_list,
                        "qvec": qvec,
                        "depth": depth,
                    },
                ).fetchall()
            ]
            lexical_ids: list[str] = []
            if tsq:
                lexical_ids = [
                    r[0]
                    for r in conn.execute(
                        """SELECT c.id
                         FROM chunks c, to_tsquery('english', %(tsq)s) q
                        WHERE c.tsv @@ q
                          AND (%(laws)s::text[] IS NULL OR c.law = ANY(%(laws)s::text[]))
                        ORDER BY ts_rank(c.tsv, q, 1) DESC, c.id
                        LIMIT %(depth)s""",
                        {"tsq": tsq, "laws": law_list, "depth": depth},
                    ).fetchall()
                ]
        fused = rrf({"lexical": lexical_ids, "vector": vector_ids})[:k]
        hits = []
        for cid, score, ranks in fused:
            chunk = self.corpus.by_id.get(cid)
            if chunk is not None:  # the database may hold rows from another corpus version
                hits.append(Hit(chunk, score, ranks))
        return hits
