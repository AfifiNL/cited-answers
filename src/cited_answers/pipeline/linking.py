"""Structural context: paragraphs a shown paragraph depends on travel with it.

Two links, both read from the structure of the texts, not fitted to eval questions:

1. Lead paragraph. When later paragraphs of an article are shown but its paragraph 1 is not,
   paragraph 1 is added: it usually states the rule the others qualify ("A data protection
   impact assessment shall be carried out where ..." in GDPR Article 35(1), with (2)-(5) on how).
   Without it the model saw only the details and, correctly, declined to answer.

2. Exceptions, described below.

Legal rules and their exceptions often sit in different paragraphs of one article:

    GDPR Article 17(1)  the right to erasure
    GDPR Article 17(3)  "Paragraphs 1 and 2 shall not apply to the extent that processing is
                        necessary: (a) for exercising the right of freedom of expression ..."

Retrieval ranks paragraphs by how well they match the question, and an exception rarely matches
as well as the rule it limits. The model then states the rule without its exception, which is a
true quote and a misleading answer. The LLM judge, which reads the whole article, flagged this
as the most common reason for "partially supported" claims.

So for every retrieved paragraph, the paragraphs of the same article that carve out an exception
to it are added to the sources. An exception paragraph is recognised by its first sentence: it
names paragraph numbers of its own article ("Paragraphs 1 to 4", "paragraph 1 of this Article")
and says they do not apply, are not required, or are derogated from. No thresholds, and nothing
fitted to eval questions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from cited_answers.corpus import Chunk, Corpus

_LABEL = re.compile(r"^\(?[0-9a-z]{1,4}[.)]\s+")
_FIRST_SENTENCE = re.compile(r"^(.*?)(?::|\.\s|\.$|$)", re.S)
_CARVE_OUT = re.compile(
    r"shall not apply|shall not be required|shall not be applicable|does not apply"
    r"|derogation from paragraph|shall not be considered",
    re.I,
)
_PARAGRAPHS = re.compile(r"paragraphs?\s+(\d+(?:\s*(?:,|and|or|to)\s*\d+)*)", re.I)
_OF_OTHER_ARTICLE = re.compile(r"\s*(?:\(\w+\)\s*)?of\s+(?:Article|Regulation|Directive)\b", re.I)


def _numbers(spec: str) -> set[str]:
    """'1, 2 and 3' -> {1,2,3}; '1 to 4' -> {1,2,3,4}."""
    out: set[str] = set()
    for a, b in re.findall(r"(\d+)\s*to\s*(\d+)", spec):
        out.update(str(n) for n in range(int(a), int(b) + 1))
    out.update(re.findall(r"\d+", spec))
    return out


def carved_out_paragraphs(chunk: Chunk) -> set[str]:
    """Paragraph numbers of its own article that this chunk makes an exception to."""
    body = _LABEL.sub("", chunk.text.strip())
    m = _FIRST_SENTENCE.match(body)
    sentence = m.group(1) if m else body
    if not _CARVE_OUT.search(sentence):
        return set()
    targets: set[str] = set()
    for m in _PARAGRAPHS.finditer(sentence):
        if _OF_OTHER_ARTICLE.match(sentence, m.end()):
            continue  # "paragraph 2 of Article 6": another article's paragraph
        targets |= _numbers(m.group(1))
    targets.discard(chunk.paragraph or "")
    return targets


class LinkIndex:
    def __init__(
        self,
        links: dict[tuple[str, str], list[Chunk]],
        leads: dict[str, Chunk] | None = None,
    ):
        self.links = links  # (article_key, paragraph) -> exception chunks
        self.leads = leads or {}  # article_key -> first chunk of its paragraph 1

    @classmethod
    def build(cls, corpus: Corpus) -> LinkIndex:
        links: dict[tuple[str, str], list[Chunk]] = {}
        leads: dict[str, Chunk] = {}
        seen: set[tuple[str, str | None]] = set()
        for chunk in corpus.chunks:
            if chunk.kind != "article":
                continue
            if chunk.paragraph == "1" and chunk.article_key not in leads:
                leads[chunk.article_key] = chunk
            # One chunk per exception paragraph: a long list of exceptions is split into
            # points, and its first chunk carries the lead-in that says what it limits.
            if (chunk.article_key, chunk.paragraph) in seen:
                continue
            targets = carved_out_paragraphs(chunk)
            if not targets:
                continue
            seen.add((chunk.article_key, chunk.paragraph))
            for p in sorted(targets):
                links.setdefault((chunk.article_key, p), []).append(chunk)
        return cls(links, leads)

    def leads_for(self, chunks: Iterable[Chunk], *, limit: int) -> list[Chunk]:
        """Paragraph 1 of each article that is shown only through later paragraphs."""
        chosen = list(chunks)
        shown_p1 = {c.article_key for c in chosen if c.paragraph == "1"}
        have = {c.id for c in chosen}
        out: list[Chunk] = []
        for chunk in chosen:
            lead = self.leads.get(chunk.article_key)
            if (
                lead is not None
                and chunk.paragraph not in (None, "1")
                and chunk.article_key not in shown_p1
                and lead.id not in have
                and len(out) < limit
            ):
                out.append(lead)
                have.add(lead.id)
                shown_p1.add(chunk.article_key)
        return out

    def for_chunks(self, chunks: Iterable[Chunk], *, limit: int) -> list[Chunk]:
        """Exception chunks for the given chunks, not already among them, at most `limit`."""
        chosen = list(chunks)
        have = {c.id for c in chosen}
        out: list[Chunk] = []
        for chunk in chosen:
            for exc in self.links.get((chunk.article_key, chunk.paragraph or ""), ()):
                if exc.id not in have and len(out) < limit:
                    out.append(exc)
                    have.add(exc.id)
        return out

    def __len__(self) -> int:
        return len({c.id for chunks in self.links.values() for c in chunks})


def exception_ids(index: LinkIndex, chunks: Sequence[Chunk]) -> list[str]:
    return [c.id for c in index.for_chunks(chunks, limit=99)]
