"""Build the committed corpus from EUR-Lex snapshots.

  download  one polite GET per law (identifying User-Agent, pause between requests); the
            HTML is stored gzipped under data/raw/ with a sidecar recording URL, date and
            SHA-256, so the corpus can be rebuilt and checked without the network.
  build     parse every snapshot into data/corpus/<law>.jsonl and write MANIFEST.json.

The tests re-run `build` in memory and compare it with the committed files, so a parser
change that alters the corpus cannot slip through unnoticed.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from cited_answers.corpus import Chunk, write_jsonl
from cited_answers.laws import LAWS

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
CORPUS_DIR = ROOT / "data" / "corpus"
USER_AGENT = "cited-answers-ingest/0.1 (one-off snapshot of public EUR-Lex texts)"


def raw_path(law: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{law}_{LAWS[law].celex}.html.gz"


def download(law: str, raw_dir: Path = RAW_DIR, pause_s: float = 4.0) -> Path:
    spec = LAWS[law]
    req = urllib.request.Request(spec.html_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = resp.status
        html = resp.read()
    if status != 200 or b'class="oj-ti-art"' not in html:
        raise RuntimeError(f"{law}: unexpected response (HTTP {status}); not a legal text page")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_path(law, raw_dir)
    with gzip.GzipFile(out, "wb", mtime=0) as fh:  # mtime=0 keeps the archive reproducible
        fh.write(html)
    meta = {
        "law": law,
        "url": spec.html_url,
        "retrieved_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "http_status": status,
        "bytes": len(html),
        "sha256": hashlib.sha256(html).hexdigest(),
    }
    out.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    time.sleep(pause_s)
    return out


def read_raw(law: str, raw_dir: Path = RAW_DIR) -> str:
    with gzip.open(raw_path(law, raw_dir), "rb") as fh:
        return fh.read().decode("utf-8")


def parse_law(law: str, raw_dir: Path = RAW_DIR) -> list[Chunk]:
    from cited_answers.ingest.eurlex import parse_document  # needs the [ingest] extra

    return parse_document(read_raw(law, raw_dir), law)


def build(raw_dir: Path = RAW_DIR, corpus_dir: Path = CORPUS_DIR) -> dict[str, object]:
    documents = []
    for law, spec in LAWS.items():
        chunks = parse_law(law, raw_dir)
        write_jsonl(corpus_dir / f"{law}.jsonl", chunks)
        meta_file = raw_path(law, raw_dir).with_suffix("").with_suffix(".meta.json")
        meta = json.loads(meta_file.read_text())
        documents.append(
            {
                "law": law,
                "title": spec.title,
                "celex": spec.celex,
                "eli": spec.eli,
                "official_journal": spec.oj_reference,
                "source_url": meta["url"],
                "retrieved_at": meta["retrieved_at"],
                "raw_sha256": meta["sha256"],
                "articles": len({c.number for c in chunks if c.kind == "article"}),
                "annexes": len({c.number for c in chunks if c.kind == "annex"}),
                "chunks": len(chunks),
            }
        )
    from cited_answers.corpus import Corpus

    manifest: dict[str, object] = {
        "corpus_version": Corpus.load(corpus_dir).version,
        "documents": documents,
        "notice": (
            "Text: © European Union, https://eur-lex.europa.eu. Reuse is authorised under "
            "Commission Decision 2011/833/EU. Only the Official Journal version is authentic."
        ),
    }
    (corpus_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
