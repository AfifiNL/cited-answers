"""Evidence gate: decide *before* calling a model whether the sources can answer at all.

Signal: IDF-weighted coverage of the question's informative words by the top retrieved
chunks. Words that name the corpus ("GDPR", "Article", "AI") are ignored; words that never
occur in the corpus get the highest weight, so a question about something the texts do not
mention ("corporate tax rate in the Netherlands") fails the gate even if some generic words
match. A question that names an article explicitly passes, since that article is evidence
by definition. Lay words count as covered when their legal wording is present (glossary.py).

This is deliberately simple and explainable: the report lists the missing words, and the
threshold is calibrated on the eval dev split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from cited_answers.glossary import is_covered
from cited_answers.retrieval.base import Hit
from cited_answers.text import TermStats, content_terms


@dataclass(frozen=True)
class EvidenceReport:
    sufficient: bool
    coverage: float
    reason: str
    missing_terms: list[str] = field(default_factory=list)


def assess_evidence(
    question: str,
    hits: Sequence[Hit],
    stats: TermStats,
    *,
    min_coverage: float,
    top_n: int,
) -> EvidenceReport:
    if not hits:
        return EvidenceReport(False, 0.0, "no passages retrieved")
    if any(h.pinned for h in hits):
        return EvidenceReport(True, 1.0, "the question names a provision that exists")
    weights = stats.weights(content_terms(question, drop_corpus_words=True))
    total = sum(weights.values())
    if total == 0:
        return EvidenceReport(False, 0.0, "the question has no informative words")
    evidence_terms: set[str] = set()
    for h in hits[:top_n]:
        evidence_terms.update(content_terms(h.chunk.index_text))
    covered = sum(w for t, w in weights.items() if is_covered(t, evidence_terms))
    coverage = covered / total
    missing = [t for t in weights if not is_covered(t, evidence_terms)]
    if coverage >= min_coverage:
        return EvidenceReport(True, round(coverage, 4), "sources cover the question", missing)
    return EvidenceReport(
        False,
        round(coverage, 4),
        f"sources cover {coverage:.0%} of the question's informative words "
        f"(minimum {min_coverage:.0%})",
        missing,
    )
