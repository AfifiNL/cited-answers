# Sources

| | GDPR | AI Act |
|---|---|---|
| Act | Regulation (EU) 2016/679 of the European Parliament and of the Council | Regulation (EU) 2024/1689 of the European Parliament and of the Council |
| Official Journal | OJ L 119, 4.5.2016, p. 1 | OJ L, 2024/1689, 12.7.2024 |
| CELEX | 32016R0679 | 32024R1689 |
| ELI | http://data.europa.eu/eli/reg/2016/679/oj | http://data.europa.eu/eli/reg/2024/1689/oj |
| Fetched from | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679 | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689 |
| Retrieved (UTC) | 2026-11-08T16:24:53Z | 2026-11-08T16:24:57Z |
| SHA-256 of the HTML | `ec79902acd1c853751c8dff1a7bb74e146b20ba9ac62137c97472670b0ea6ec2` | `edb42bea61f1834b1a296ce489c513c1f022a386338c71aa4f7381acd6683911` |
| Parsed | 99 articles, 511 chunks | 113 articles, 13 annexes, 817 chunks |

Each page was fetched once with an identifying User-Agent. The snapshots are stored gzipped
in `data/raw/` with a `.meta.json` sidecar. `data/corpus/MANIFEST.json` records the same facts
plus the corpus version (a hash of all chunks).

What was kept: article and annex text, verbatim apart from whitespace, Official Journal
footnote markers (such as "(57)") and exponents written as `^` (`10^25`). What was left out:
recitals, footnotes, signatures. A stray backtick in the AI Act's Article 1 title
("Subject matter`") is in the source and was kept as is.

Not included: corrigenda and later amendments. The texts are the versions published in the
Official Journal on the dates above.

Reuse: © European Union, https://eur-lex.europa.eu. Reuse of EUR-Lex documents is authorised
under Commission Decision 2011/833/EU, provided the source is acknowledged. Only the Official
Journal of the European Union in its printed (or, since 1 July 2013, electronic) edition is
authentic.

Rebuild: `make ingest` re-parses the snapshots. `cited-answers ingest --download` fetches fresh
copies first (two requests, a pause between them); review the corpus diff before committing.
