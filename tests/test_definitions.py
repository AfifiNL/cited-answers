"""Definition lookup, checked against every definition in both texts (not against eval items)."""

from __future__ import annotations

import pytest

from cited_answers.corpus import Corpus
from cited_answers.laws import LAWS
from cited_answers.pipeline.definitions import DefinitionIndex, asks_for_definition
from cited_answers.pipeline.graph import Pipeline


@pytest.fixture(scope="module")
def index(corpus: Corpus) -> DefinitionIndex:
    return DefinitionIndex.build(corpus)


def test_index_holds_every_definition(index: DefinitionIndex) -> None:
    by_law: dict[str, int] = {}
    for d in index.definitions:
        by_law[d.law] = by_law.get(d.law, 0) + 1
    # GDPR Art. 4 has 26 points plus 'data subject', defined inside point (1).
    assert by_law == {"gdpr": 27, "ai_act": 68}
    assert {d.chunk_id for d in index.definitions if d.term == "data subject"} == {"gdpr:art4:pt1"}


@pytest.mark.parametrize(
    "template",
    [
        "How does the {law} define {term}?",
        "What is the definition of '{term}' in the {law}?",
        "What does {term} mean under the {law}?",
        "What is meant by {term} in the {law}?",
        "What is {term} under the {law}?",
    ],
)
def test_every_defined_term_is_found(index: DefinitionIndex, template: str) -> None:
    misses = []
    for d in index.definitions:
        question = template.format(law=LAWS[d.law].short, term=d.term)
        found = index.lookup(question, [d.law])
        if not found or found[0].chunk_id != d.chunk_id:
            misses.append((question, [f.chunk_id for f in found]))
    assert not misses, misses[:5]


def test_every_definition_is_pinned_and_cited(pipeline: Pipeline) -> None:
    wrong = []
    for d in pipeline.definitions.definitions:
        question = f"How does the {LAWS[d.law].short} define {d.term}?"
        result = pipeline.answer(question, debug=True)
        retrieve = next(t for t in result.trace or [] if t["node"] == "retrieve")
        first = retrieve["hits"][0]
        if first["id"] != d.chunk_id or first["ranks"] != {"definition": 1}:
            wrong.append((question, first["id"]))
        elif (
            result.status != "answered" or d.chunk_id.rsplit(":", 1)[0] not in result.cited_articles
        ):
            wrong.append((question, result.status, result.cited_articles))
    assert not wrong, wrong[:5]


@pytest.mark.parametrize(
    "question",
    [
        "What is the highest administrative fine under the GDPR?",
        "Can a provider place a high-risk AI system on the market without a conformity assessment?",
        "Which AI practices are prohibited?",
        "How does the AI Act define high-risk AI systems?",  # 'risk' is defined, 'high-risk' is not
        "What does the AI Act say about cryptocurrency mining?",
        "How do I bake sourdough bread?",
        "What is AI?",
        "What does data protection by design and by default mean for controllers?",
    ],
)
def test_no_definition_without_a_definition_question(index: DefinitionIndex, question: str) -> None:
    assert index.lookup(question, ["gdpr", "ai_act"]) == []


def test_longest_term_wins(index: DefinitionIndex) -> None:
    found = index.lookup("How does the AI Act define a general-purpose AI system?", ["ai_act"])
    assert [d.term for d in found] == ["general-purpose AI system"]


def test_law_filter_and_both_laws(index: DefinitionIndex) -> None:
    gdpr_only = index.lookup("How does the GDPR define personal data?", ["gdpr"])
    assert [d.chunk_id for d in gdpr_only] == ["gdpr:art4:pt1"]
    both = index.lookup("What is the definition of personal data?", ["gdpr", "ai_act"])
    assert {d.law for d in both} == {"gdpr", "ai_act"}


def test_plural_articles_and_passive_match(index: DefinitionIndex) -> None:
    assert [d.term for d in index.lookup("What are AI systems?", ["ai_act"])] == ["AI system"]
    assert [d.term for d in index.lookup("What is a data subject?", ["gdpr"])] == ["data subject"]
    found = index.lookup("How is personal data defined in the GDPR?", ["gdpr"])
    assert [d.chunk_id for d in found] == ["gdpr:art4:pt1"]


def test_cue_detection() -> None:
    assert asks_for_definition("How does the GDPR define consent?")
    assert asks_for_definition("What does 'deployer' mean?")
    assert asks_for_definition("What is a deployer under the AI Act?")
    assert asks_for_definition("How is personal data defined in the GDPR?")
    assert not asks_for_definition(
        "Must a deployer inform workers before using a high-risk system?"
    )
