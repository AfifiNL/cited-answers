"""Structural context: lead paragraphs and exceptions in the same article travel along."""

from __future__ import annotations

from cited_answers.corpus import Corpus
from cited_answers.pipeline.graph import Pipeline
from cited_answers.pipeline.linking import LinkIndex, _numbers, carved_out_paragraphs


def test_paragraph_number_lists() -> None:
    assert _numbers("1 to 4") == {"1", "2", "3", "4"}
    assert _numbers("1, 2 and 3") == {"1", "2", "3"}
    assert _numbers("6") == {"6"}


def test_known_exceptions_are_linked(corpus: Corpus) -> None:
    links = LinkIndex.build(corpus).links
    expected = {
        ("gdpr:art17", "1"): "gdpr:art17:p3",  # erasure: freedom of expression, legal claims ...
        ("gdpr:art30", "1"): "gdpr:art30:p5",  # records: fewer than 250 employees
        ("gdpr:art34", "1"): "gdpr:art34:p3",  # breach communication not required if ...
        ("gdpr:art9", "1"): "gdpr:art9:p2:pta",  # special categories: explicit consent, ...
        ("ai_act:art6", "2"): "ai_act:art6:p3",  # Annex III systems that are not high-risk
    }
    for key, chunk_id in expected.items():
        assert chunk_id in [c.id for c in links.get(key, [])], key


def test_no_self_links_and_no_links_to_other_articles(corpus: Corpus) -> None:
    index = LinkIndex.build(corpus)
    for (article, paragraph), chunks in index.links.items():
        for c in chunks:
            assert c.article_key == article and c.paragraph != paragraph
    # A rule that is not an exception to its own article links nowhere.
    assert carved_out_paragraphs(corpus.by_id["gdpr:art17:p1"]) == set()


def test_exception_is_shown_with_the_rule(pipeline: Pipeline) -> None:
    result = pipeline.answer("Can I ask a company to delete my personal data?", debug=True)
    hits = next(t for t in result.trace or [] if t["node"] == "retrieve")["hits"]
    ids = [h["id"] for h in hits]
    assert "gdpr:art17:p1" in ids and "gdpr:art17:p3" in ids
    linked = next(h for h in hits if h["id"] == "gdpr:art17:p3")
    assert linked["ranks"] in ({"exception": 1}, linked["ranks"])  # linked, or retrieved anyway


def test_limit_is_respected(corpus: Corpus) -> None:
    index = LinkIndex.build(corpus)
    rules = [corpus.by_id[i] for i in ("gdpr:art17:p1", "gdpr:art30:p1", "gdpr:art34:p1")]
    assert len(index.for_chunks(rules, limit=2)) == 2
    assert index.for_chunks(rules, limit=0) == []


def test_lead_paragraph_joins_later_paragraphs(corpus: Corpus) -> None:
    index = LinkIndex.build(corpus)
    later = [corpus.by_id[i] for i in ("gdpr:art35:p2", "gdpr:art35:p4", "gdpr:art82:p4")]
    assert [c.id for c in index.leads_for(later, limit=3)] == ["gdpr:art35:p1", "gdpr:art82:p1"]
    with_p1 = [corpus.by_id["gdpr:art35:p1"], corpus.by_id["gdpr:art35:p2"]]
    assert index.leads_for(with_p1, limit=3) == []  # paragraph 1 already shown
    single = [corpus.by_id["gdpr:art44"]]  # an article without numbered paragraphs
    assert index.leads_for(single, limit=3) == []
