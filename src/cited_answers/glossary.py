"""Lay-to-legal vocabulary and law aliases, for query expansion.

People ask "can I make a company delete my data?"; the GDPR says "the controller shall erase".
With a lexical retriever that mismatch is the main cause of misses and false refusals, so
the question is expanded with the legal wording before retrieval, and a lay word counts as
covered by the evidence when its legal counterpart is present.

Rules for adding entries: map everyday words to the words the texts actually use; keep it
small and general; never add an entry to fix a single eval question. A semantic embedder
needs this much less, but it stays as a cheap, explainable fallback.

The texts never use their own acronyms ("GDPR" appears in neither), so a question naming a
law is also expanded with the regulation's number, which is how the other text refers to it.
"""

from __future__ import annotations

from functools import cache

from cited_answers.text import content_terms, stem, tokens

LAY_TO_LEGAL: dict[str, str] = {
    # rights and actions
    "delete": "erase erasure",
    "deletion": "erasure",
    "remove": "erase erasure",
    "ban": "prohibited prohibition",
    "banned": "prohibited prohibition",
    "forbidden": "prohibited prohibition",
    "allowed": "lawful prohibited",
    "legal": "lawful lawfulness",
    "define": "definitions means",
    "definition": "definitions means",
    "explain": "explanation",
    # actors
    "company": "undertaking controller",
    "business": "undertaking controller",
    "firm": "undertaking",
    "organisation": "controller undertaking",
    "people": "natural persons",
    "person": "natural person",
    "individual": "natural person data subject",
    "kid": "child",
    "minor": "child",
    "employer": "employment",
    "employee": "employment worker",
    # events and things
    "hack": "personal data breach",
    "hacked": "personal data breach",
    "leak": "personal data breach",
    "leaked": "personal data breach",
    "chatbot": "interact natural persons",
    "deepfake": "deep fake",
    "facial": "biometric",
    "abroad": "third country",
    "outside": "established third country",
    "penalty": "penalties fines",
    # acronyms the texts spell out
    "dpo": "data protection officer",
    "dpia": "data protection impact assessment",
    "gpai": "general-purpose model",
    "llm": "general-purpose model",
    "fria": "fundamental rights impact assessment",
}

LAW_ALIASES: dict[str, str] = {
    "gdpr": "Regulation (EU) 2016/679",
    "ai_act": "Regulation (EU) 2024/1689",
}


@cache
def _stem_map() -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for lay, legal in LAY_TO_LEGAL.items():
        out.setdefault(stem(lay), set()).update(content_terms(legal))
    return {k: frozenset(v) for k, v in out.items()}


def expansions(term: str) -> frozenset[str]:
    """Stemmed legal terms that may stand in for a stemmed lay term."""
    return _stem_map().get(term, frozenset())


def is_covered(term: str, available: set[str] | frozenset[str]) -> bool:
    return term in available or bool(expansions(term) & available)


def expand_query(question: str, laws: tuple[str, ...] = ()) -> str:
    """The question plus the legal wording of its lay words and the named laws' numbers."""
    extra: list[str] = []
    for tok in tokens(question):
        legal = LAY_TO_LEGAL.get(tok) or LAY_TO_LEGAL.get(tok.rstrip("s"))
        if legal:
            extra.append(legal)
    if len(laws) > 1:  # cross-law question: each text cites the other by its number
        extra += [LAW_ALIASES[law] for law in laws if law in LAW_ALIASES]
    return question if not extra else f"{question} {' '.join(extra)}"
