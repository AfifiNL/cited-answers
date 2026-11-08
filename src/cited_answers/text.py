"""Deterministic text helpers shared by retrieval, the evidence gate, the verifier and evals."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Numbers as written in the Official Journal: '35 000 000', '7 %', '72', '10^25', '2,5'.
_ENUM_LABEL = re.compile(r"^\(?[0-9a-z]{1,4}[.)]$", re.I)
_NUMBER = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?![\d])|\d+(?:[.,]\d+)?(?:\^\d+)?")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "either",
        "else",
        "ever",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "let",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "shall",
        "must",
        "may",
        "might",
        "us",
        "one",
        "per",
        "whether",
        "within",
        "without",
        "upon",
        "thereof",
        "therein",
        "whereas",
        "tell",
        "say",
        "says",
        "said",
        "according",
        "need",
        "needs",
    ]
)

# Words that name the corpus rather than a topic; they say nothing about which provision fits.
CORPUS_WORDS = frozenset(
    [
        "gdpr",
        "regulation",
        "act",
        "article",
        "articl",
        "artificial",
        "intelligence",
        "general",
        "data",
        "protection",
        "eu",
        "european",
        "union",
        "law",
        "laws",
        "rule",
        "rules",
        "provision",
        "provisions",
        "ai",
    ]
)


def normalize(text: str) -> str:
    """Lower-case, fold typographic quotes/dashes and collapse whitespace (for matching)."""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    return str(_STEMMER.stemWord(word))


def content_terms(text: str, *, drop_corpus_words: bool = False) -> list[str]:
    """Stemmed content words in order (stopwords and pure punctuation removed)."""
    out = []
    for tok in tokens(text):
        if tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        if drop_corpus_words and tok in CORPUS_WORDS:
            continue
        out.append(stem(tok))
    return out


def numbers(text: str) -> set[str]:
    """Canonical numbers in a text: '35 000 000' -> '35000000', '2,5' -> '2.5'."""
    found = set()
    for m in _NUMBER.finditer(text):
        value = re.sub(r"[   ]", "", m.group(0)).replace(",", ".")
        found.add(value)
    return found


def split_segments(text: str) -> list[str]:
    """Split legal text into quotable segments: lines (points) and sentences within them."""
    segments: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+(?=[A-Z(])", line) if p.strip()]
        merged: list[str] = []
        for part in parts:
            if merged and _ENUM_LABEL.match(merged[-1]):
                merged[-1] = f"{merged[-1]} {part}"  # keep "1." / "(a)" with its text
            else:
                merged.append(part)
        segments.extend(p for p in merged if len(p) >= 3)
    return segments


@dataclass(frozen=True)
class TermStats:
    """Document frequencies over the corpus, used for IDF weighting."""

    n_docs: int
    df: dict[str, int]

    @classmethod
    def build(cls, docs: Iterable[str]) -> TermStats:
        df: Counter[str] = Counter()
        n = 0
        for doc in docs:
            n += 1
            df.update(set(content_terms(doc)))
        return cls(n_docs=n, df=dict(df))

    def idf(self, term: str) -> float:
        """BM25 IDF; a term absent from the corpus gets the maximum weight."""
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def weights(self, terms: Sequence[str]) -> dict[str, float]:
        return {t: self.idf(t) for t in dict.fromkeys(terms)}
