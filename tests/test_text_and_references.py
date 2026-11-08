from __future__ import annotations

import pytest

from cited_answers.glossary import expand_query, is_covered
from cited_answers.pipeline.references import find_references, strip_references
from cited_answers.text import TermStats, content_terms, normalize, numbers, split_segments


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("up to EUR 35 000 000 or 7 % of turnover", {"35000000", "7"}),
        ("not later than 72 hours", {"72"}),
        ("greater than 10^25", {"10^25"}),
        ("2,5 % or 1.5", {"2.5", "1.5"}),
        ("no figures here", set()),
    ],
)
def test_numbers_are_canonical(text: str, expected: set[str]) -> None:
    assert numbers(text) == expected


def test_normalize_folds_typography() -> None:
    assert normalize("‘right  to be\nforgotten’ — yes") == "'right to be forgotten' - yes"


def test_content_terms_stem_and_drop_stopwords() -> None:
    assert content_terms("The erasure of personal data") == ["erasur", "person", "data"]
    assert "gdpr" not in content_terms("Under the GDPR, erasure", drop_corpus_words=True)


def test_split_segments_keeps_points_apart() -> None:
    text = "1. Lead-in text:\n(a) first point; (b) second point.\nTrailing sentence. Another one."
    assert split_segments(text) == [
        "1. Lead-in text:",
        "(a) first point;",
        "(b) second point.",
        "Trailing sentence.",
        "Another one.",
    ]


def test_idf_of_unknown_term_is_maximal() -> None:
    stats = TermStats.build(["data subject rights", "data controller duties"])
    assert stats.idf("sourdough") > stats.idf("subject") > stats.idf("data")


def test_find_references() -> None:
    refs = find_references("What does Article 17 GDPR say, and art. 5 of the AI Act, Annex III?")
    assert refs.laws == ("gdpr", "ai_act")
    assert refs.articles == ("17", "5")
    assert refs.annexes == ("III",)
    assert refs.article_keys(("gdpr", "ai_act"))[:2] == ["gdpr:art17", "ai_act:art17"]


def test_no_false_law_detection() -> None:
    assert find_references("What is the average processing time?").laws == ()


def test_strip_references() -> None:
    assert strip_references("What does Article 14 of the AI Act say?") == "What does of the say?"


def test_glossary_expansion_and_coverage() -> None:
    q = expand_query("Can a company delete my data?")
    assert "erase" in q and "undertaking" in q
    assert is_covered("delet", {"erasur"})
    assert not is_covered("sourdough", {"erasur"})


def test_law_aliases_only_for_cross_law_questions() -> None:
    assert "2016/679" not in expand_query("What is erasure?", ("gdpr",))
    assert "2016/679" in expand_query("Does the AI Act affect the GDPR?", ("gdpr", "ai_act"))
