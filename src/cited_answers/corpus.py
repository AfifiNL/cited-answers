"""Corpus model: chunks of legal text with article-level citation metadata.

A chunk is the unit of retrieval (an article paragraph, a definition point or an annex
point). A citation always resolves to an *article* (or annex) so that every claim can be
checked against a stable, human-readable reference such as "GDPR Article 17(1)".
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

from cited_answers.laws import LAWS

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass(frozen=True)
class Chunk:
    id: str  # e.g. "gdpr:art17:p1", "ai_act:art3:pt12", "ai_act:anx3:pt4"
    law: str  # key in LAWS
    kind: str  # "article" | "annex"
    number: str  # "17", "III"
    paragraph: str | None  # "1" for Article 17(1)
    point: str | None  # "a" for Article 5(1)(a), "12" for Article 3, point (12)
    title: str  # article or annex title
    chapter: str  # e.g. "CHAPTER III Rights of the data subject"
    text: str  # the legal text, verbatim apart from whitespace and footnote markers
    anchor: str  # EUR-Lex HTML anchor, e.g. "art_17"

    @property
    def article_key(self) -> str:
        """Article-level citation key used by evals: 'gdpr:art17' or 'ai_act:anxIII'."""
        prefix = "art" if self.kind == "article" else "anx"
        return f"{self.law}:{prefix}{self.number}"

    @property
    def article_label(self) -> str:
        law = LAWS[self.law].short
        word = "Article" if self.kind == "article" else "Annex"
        return f"{law} {word} {self.number}"

    @property
    def label(self) -> str:
        """Finest citation label, e.g. 'GDPR Article 17(1)', 'AI Act Article 3, point (12)'."""
        base = self.article_label
        if self.kind == "annex":  # `paragraph` holds the annex section, e.g. "Section A"
            parts = [base]
            if self.paragraph:
                parts.append(self.paragraph)
            if self.point:
                pt = f"({self.point})" if self.point.isalpha() else self.point
                parts.append(f"point {pt}")
            return ", ".join(parts)
        if self.paragraph and self.point:
            return f"{base}({self.paragraph})({self.point})"
        if self.paragraph:
            return f"{base}({self.paragraph})"
        if self.point:
            return f"{base}, point ({self.point})"
        return base

    @property
    def url(self) -> str:
        return LAWS[self.law].anchor_url(self.anchor)

    @property
    def index_text(self) -> str:
        """Text used for retrieval: a contextual header (law, article, title, chapter) + body.

        The header lets a question phrased in the vocabulary of an article title match
        paragraphs that do not repeat that vocabulary. Quotes and citations always use
        `text`, never the header.
        """
        return f"{self.article_label}. {self.title}. {self.chapter}.\n{self.text}"

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def article_key(law: str, kind: str, number: str) -> str:
    return f"{law}:{'art' if kind == 'article' else 'anx'}{number}"


class Corpus:
    """In-memory view of the committed corpus files (a few thousand chunks, ~2 MB)."""

    def __init__(self, chunks: Sequence[Chunk]):
        if not chunks:
            raise ValueError("empty corpus")
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids in corpus")
        self.chunks: tuple[Chunk, ...] = tuple(chunks)
        self.by_id: dict[str, Chunk] = {c.id: c for c in chunks}
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            grouped[c.article_key].append(c)
        self.by_article: dict[str, tuple[Chunk, ...]] = {k: tuple(v) for k, v in grouped.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def article_text(self, key: str) -> str:
        return "\n".join(c.text for c in self.by_article.get(key, ()))

    def article_title(self, key: str) -> str:
        chunks = self.by_article.get(key)
        return chunks[0].title if chunks else ""

    @cached_property
    def version(self) -> str:
        """Content hash of the corpus; index caches and eval reports are keyed on it."""
        h = hashlib.sha256()
        for c in self.chunks:
            h.update(json.dumps(c.to_json(), sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    @classmethod
    def load(cls, corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> Corpus:
        corpus_dir = Path(corpus_dir)
        files = sorted(corpus_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no corpus files in {corpus_dir}; run `make ingest`")
        chunks: list[Chunk] = []
        for f in files:
            chunks.extend(read_jsonl(f))
        return cls(chunks)


def read_jsonl(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def write_jsonl(path: Path, chunks: Iterable[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
