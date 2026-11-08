"""Explicit references in a question: which law it names and which articles it points at.

"What does Article 17 GDPR say?" should retrieve Article 17 of the GDPR whatever its
wording; a question that names only one law should search only that law. Both are cheap,
deterministic and testable, so they run before any vector search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LAW_PATTERNS = {
    "gdpr": re.compile(r"\b(gdpr|general data protection regulation|2016/679)\b", re.I),
    "ai_act": re.compile(
        r"\b(ai act|artificial intelligence act|2024/1689|eu ai regulation)\b", re.I
    ),
}
_ARTICLE = re.compile(r"\b(?:article|art\.?)\s*(\d{1,3}[a-z]?)\b", re.I)
_ANNEX = re.compile(r"\bannex\s+([IVXLC]{1,6})\b", re.I)


@dataclass(frozen=True)
class References:
    laws: tuple[str, ...]  # laws the question names explicitly
    articles: tuple[str, ...]  # article numbers, e.g. ("17",)
    annexes: tuple[str, ...]  # annex numerals, e.g. ("III",)

    def article_keys(self, corpus_laws: tuple[str, ...]) -> list[str]:
        laws = self.laws or corpus_laws
        keys = [f"{law}:art{n}" for n in self.articles for law in laws]
        keys += [f"{law}:anx{n.upper()}" for n in self.annexes for law in laws]
        return keys


def find_references(question: str) -> References:
    laws = tuple(k for k, pat in _LAW_PATTERNS.items() if pat.search(question))
    articles = tuple(dict.fromkeys(m.group(1).lower() for m in _ARTICLE.finditer(question)))
    annexes = tuple(dict.fromkeys(m.group(1).upper() for m in _ANNEX.finditer(question)))
    return References(laws=laws, articles=articles, annexes=annexes)


def strip_references(question: str) -> str:
    """The question without its law names and article/annex numbers (they are not topics)."""
    text = _ARTICLE.sub(" ", question)
    text = _ANNEX.sub(" ", text)
    for pat in _LAW_PATTERNS.values():
        text = pat.sub(" ", text)
    return " ".join(text.split())
