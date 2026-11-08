"""The corpus is exactly what the parser makes of the committed EUR-Lex snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cited_answers.corpus import Corpus, read_jsonl
from cited_answers.laws import LAWS

pytest.importorskip("bs4")
from cited_answers.ingest.build import CORPUS_DIR, RAW_DIR, parse_law, raw_path, read_raw
from cited_answers.ingest.eurlex import parse_document


@pytest.mark.parametrize("law", list(LAWS))
def test_committed_corpus_matches_parser_output(law: str) -> None:
    committed = read_jsonl(CORPUS_DIR / f"{law}.jsonl")
    assert parse_law(law) == committed


@pytest.mark.parametrize("law", list(LAWS))
def test_snapshot_hash_matches_metadata_and_manifest(law: str) -> None:
    html = read_raw(law).encode("utf-8")
    meta = json.loads(raw_path(law).with_suffix("").with_suffix(".meta.json").read_text())
    manifest = json.loads((CORPUS_DIR / "MANIFEST.json").read_text())
    doc = next(d for d in manifest["documents"] if d["law"] == law)
    digest = hashlib.sha256(html).hexdigest()
    assert digest == meta["sha256"] == doc["raw_sha256"]
    assert doc["source_url"] == LAWS[law].html_url
    assert doc["source_url"].startswith("https://eur-lex.europa.eu/")


def test_manifest_version_matches_corpus(corpus: Corpus) -> None:
    manifest = json.loads((CORPUS_DIR / "MANIFEST.json").read_text())
    assert manifest["corpus_version"] == corpus.version


def test_structure_counts(corpus: Corpus) -> None:
    def count(law: str, kind: str) -> int:
        return len({c.number for c in corpus.chunks if c.law == law and c.kind == kind})

    assert count("gdpr", "article") == 99
    assert count("gdpr", "annex") == 0
    assert count("ai_act", "article") == 113
    assert count("ai_act", "annex") == 13


@pytest.mark.parametrize(
    ("chunk_id", "label", "snippet"),
    [
        ("gdpr:art17:p1", "GDPR Article 17(1)", "right to obtain from the controller the erasure"),
        ("gdpr:art4:pt1", "GDPR Article 4, point (1)", "‘personal data’ means any information"),
        ("ai_act:art5:p1:pta", "AI Act Article 5(1)(a)", "subliminal techniques"),
        ("ai_act:art3:pt1", "AI Act Article 3, point (1)", "‘AI system’ means a machine-based"),
        ("ai_act:art99:p3", "AI Act Article 99(3)", "EUR 35 000 000"),
        ("ai_act:art51:p2", "AI Act Article 51(2)", "greater than 10^25"),
        ("ai_act:anxIII:pt4", "AI Act Annex III, point 4", "recruitment or selection"),
        ("ai_act:anxVIII:sA.pt1", "AI Act Annex VIII, Section A, point 1", "name, address"),
    ],
)
def test_known_provisions(corpus: Corpus, chunk_id: str, label: str, snippet: str) -> None:
    chunk = corpus.by_id[chunk_id]
    assert chunk.label == label
    assert snippet in chunk.text


def test_footnote_markers_are_removed(corpus: Corpus) -> None:
    # Article 78(1)(a) cites Directive (EU) 2016/943 with an OJ footnote marker "(57)".
    text = corpus.article_text("ai_act:art78")
    assert "Directive (EU) 2016/943 of the European Parliament and of the Council;" in text
    assert "(57)" not in text


def test_deep_links_point_at_existing_anchors(corpus: Corpus) -> None:
    html = {law: read_raw(law) for law in LAWS}
    for key in ["gdpr:art17", "ai_act:art5", "ai_act:anxIII"]:
        chunk = corpus.by_article[key][0]
        assert chunk.url.endswith(f"#{chunk.anchor}")
        assert f'id="{chunk.anchor}"' in html[chunk.law]


SYNTHETIC = """<html><body>
<div id="cpt_I"><p class="oj-ti-section-1">CHAPTER I</p>
<div class="eli-title"><p class="oj-ti-section-2">General</p></div>
<div class="eli-subdivision" id="art_1">
  <p class="oj-ti-art">Article 1</p>
  <div class="eli-title"><p class="oj-sti-art">Scope</p></div>
  <div id="001.001"><p class="oj-normal">1.   Short rule with a note <a href="#n"
    >(<span class="oj-super oj-note-tag">3</span>)</a>.</p></div>
  <div id="001.002"><p class="oj-normal">2.   Thresholds above 10<span
    class="oj-super">25</span>:</p>
    <table><tbody><tr><td><p class="oj-normal">(a)</p></td>
      <td><p class="oj-normal">first point;</p></td></tr></tbody></table>
    <table><tbody><tr><td><p class="oj-normal">—</p></td>
      <td><p class="oj-normal">dash item.</p></td></tr></tbody></table>
  </div>
</div></div>
</body></html>"""


def test_parser_on_synthetic_markup() -> None:
    chunks = parse_document(SYNTHETIC, "gdpr")
    assert [c.id for c in chunks] == ["gdpr:art1:p1", "gdpr:art1:p2"]
    first, second = chunks
    assert first.text == "1. Short rule with a note."
    assert first.chapter == "CHAPTER I General"
    assert first.title == "Scope"
    assert second.text == "2. Thresholds above 10^25:\n(a) first point;\n— dash item."


def test_long_paragraph_is_split_per_point_with_lead_in(corpus: Corpus) -> None:
    points = [c for c in corpus.by_article["ai_act:art3"] if c.point]
    assert len(points) >= 60  # Article 3 has 68 definitions
    assert all(c.text.startswith("For the purposes of this Regulation") for c in points)


def test_raw_dir_contains_only_expected_files() -> None:
    names = sorted(p.name for p in Path(RAW_DIR).iterdir())
    expected = sorted(
        n
        for law in LAWS
        for n in (raw_path(law).name, raw_path(law).with_suffix("").with_suffix(".meta.json").name)
    )
    assert names == expected
