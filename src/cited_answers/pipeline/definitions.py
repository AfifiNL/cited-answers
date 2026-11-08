"""Definition lookup: "How does the GDPR define personal data?" -> GDPR Article 4, point (1).

Ranked retrieval often misses the right definition. The definitions articles (GDPR Article 4,
AI Act Article 3) have 26 and 68 points that share most of their words ("data", "AI system"),
and only three chunks per article are kept. So when a question asks for the definition of a
term that one of the texts defines, the point that defines it is pinned, the same way an
explicit "Article 17" reference is.

The term has to sit where a definition question puts it: after "define", "definition of",
"meaning of" or "what is meant by", between "what does" and "mean", before "is defined", or as
the whole of "What is/are [a|an|the] X?" (optionally naming the law). It must match a defined
term word for word after stemming, as a whole phrase:

  * "How does the AI Act define an AI system?"          -> 'AI system'
  * "How does the AI Act define high-risk AI systems?"  -> nothing ('high-risk AI system' is
    not a defined term; 'AI system' is only part of the phrase)
  * "What does data protection by design mean for controllers?" -> nothing (the question is
    about data protection by design; 'controller' is not what it asks to define)

The longest matching term wins ("general-purpose AI system" over 'AI system'). No thresholds
and no tuning: the rule is checked against every definition in both texts
(`tests/test_definitions.py`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from cited_answers.corpus import Corpus
from cited_answers.pipeline.references import strip_references
from cited_answers.text import STOPWORDS, stem, tokens

DEFINITION_ARTICLES = {"gdpr": "art4", "ai_act": "art3"}

_LEAD = re.compile(r"^\((\d+)\)\s*‘([^’]+)’")  # "(1) ‘personal data’ means ..."
_INLINE = re.compile(r"\(‘([^’]+)’\)")  # "... natural person (‘data subject’)"

# Where a definition question names its term. The term must start the slot.
_SLOTS = (
    re.compile(r"\bdefin(?:e|es|ing)\s+(?P<slot>.+)$", re.I),
    re.compile(r"\bdefinitions?\s+of\s+(?P<slot>.+)$", re.I),
    re.compile(r"\bmeaning\s+of\s+(?P<slot>.+)$", re.I),
    re.compile(r"\bwhat\s+is\s+meant\s+by\s+(?P<slot>.+)$", re.I),
    re.compile(r"\bwhat\s+(?:does|do)\s+(?P<slot>.+?)\s+mean\b", re.I),
    re.compile(r"\b(?:how|where)\s+(?:is|are)\s+(?P<slot>.+?)\s+defined\b", re.I),
)
# "What is X?" must be nothing but the term (after removing the law's name).
_WHAT_IS = re.compile(r"^\s*what\s+(?:is|are)\s+(?P<slot>.+?)[\s?.!]*$", re.I)
_LAW_TAIL = re.compile(
    r"\s+(?:under|in|according to|for the purposes of|as per)(?:\s+the)?(?:\s+eu)?\s*$", re.I
)
_DETERMINERS = frozenset({"a", "an", "the"})
# Words that may follow a matched term without being part of its phrase.
_BOUNDARY = STOPWORDS | {"mean", "means", "gdpr", "act"}


@dataclass(frozen=True)
class Definition:
    term: str
    law: str
    chunk_id: str
    stems: tuple[str, ...]


def _skip_determiners(words: list[str]) -> list[str]:
    i = 0
    while i < len(words) and words[i] in _DETERMINERS:
        i += 1
    return words[i:]


def asks_for_definition(question: str) -> bool:
    return any(p.search(question) for p in _SLOTS) or bool(
        _WHAT_IS.match(strip_references(question))
    )


class DefinitionIndex:
    def __init__(self, definitions: Sequence[Definition]):
        self.definitions = list(definitions)

    @classmethod
    def build(cls, corpus: Corpus) -> DefinitionIndex:
        found: list[Definition] = []
        for law, article in DEFINITION_ARTICLES.items():
            for chunk in corpus.by_article.get(f"{law}:{article}", ()):
                body = chunk.text.split("\n", 1)[-1]
                m = _LEAD.match(body)
                if not m:
                    continue
                for term in (m.group(2), *_INLINE.findall(body)):
                    found.append(Definition(term, law, chunk.id, tuple(map(stem, tokens(term)))))
        return cls(found)

    def lookup(self, question: str, laws: Sequence[str], *, limit: int = 2) -> list[Definition]:
        """Definitions the question asks for (one term, possibly defined in both laws)."""
        candidates = [d for d in self.definitions if d.law in laws]
        what_is = _WHAT_IS.match(strip_references(question))
        if what_is:
            words = _skip_determiners(tokens(_LAW_TAIL.sub("", what_is.group("slot"))))
            target = tuple(map(stem, words))
            exact = [d for d in candidates if d.stems == target]
            if exact:
                return exact[:limit]
        for pattern in _SLOTS:
            m = pattern.search(question)
            if m:
                found = self._at_start(_skip_determiners(tokens(m.group("slot"))), candidates)
                if found:
                    return found[:limit]
        return []

    @staticmethod
    def _at_start(words: list[str], candidates: Sequence[Definition]) -> list[Definition]:
        stems = tuple(map(stem, words))
        matches = [
            d
            for d in candidates
            if stems[: len(d.stems)] == d.stems
            and (len(words) == len(d.stems) or words[len(d.stems)] in _BOUNDARY)
        ]
        if not matches:
            return []
        longest = max(len(d.stems) for d in matches)
        return [d for d in matches if len(d.stems) == longest]
