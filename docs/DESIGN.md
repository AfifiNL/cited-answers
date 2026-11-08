# Design notes

## Goals, in priority order

1. Never state something the texts do not say. An unsupported answer is worse than a refusal.
2. Make every statement checkable by a reader: article-level citation, a deep link, a verbatim quote.
3. Make quality measurable and regressions visible in CI, at zero cost per run.
4. Keep the model vendor replaceable.

## Decisions

**The model fills one node; code decides every edge.** Retrieval, the evidence gate, the
verifier and the routing between them are deterministic. That makes behaviour testable with a
scripted model (`tests/test_pipeline.py` plays a model that invents quotes, changes figures,
cites unknown sources, refuses or crashes) and keeps the graph's safety properties independent
of the vendor.

**Claims carry quotes; citations are labels, not text.** The model returns
`{text, source: "S2", quote}` per claim. The verifier checks the quote against the source the
label points to (verbatim after whitespace and typographic normalisation), checks that every
number in the claim appears in the quote, and checks that the claim shares enough content words
with its quote. A paraphrase is fine; a paraphrase with a figure the quote does not contain is
not. Article numbers in the claim are allowed when they appear in the citation label.

**Strict on failure.** A failed check goes back to the model once, with the findings listed. If
the second draft also fails, the result is a refusal. Dropping the failing sentences and keeping
the rest was considered and rejected: in legal text the dropped sentence may be the exception
that changes the meaning of the kept one.

**Evidence gate before generation.** IDF-weighted coverage of the question's informative words
by the top three chunks. Words that only name the corpus ("GDPR", "Article", "AI") are ignored.
Words absent from the corpus get the highest weight. A question that names an existing article
passes. The gate is cheap, explainable (the trace lists the missing words) and runs before any
token is spent.

**Lay-to-legal glossary.** People say "delete", "banned", "company"; the texts say "erase",
"prohibited", "controller"/"undertaking". With a lexical retriever this mismatch caused most
false refusals on `dev`. `glossary.py` holds a short list of everyday words mapped to the
texts' own wording, used for query expansion and for the coverage check. Rule: general mappings
only, never an entry to fix one eval question. Neither text uses the acronym "GDPR"; a question
that names both laws is also expanded with the regulation numbers, which is how each text
refers to the other.

**Search deep, then cap per article.** Candidates are retrieved at 4 × top_k, then at most three
chunks per article are kept. Without the deeper pool, the cap starved the source list when one
long article (GDPR Article 83) dominated the ranking.

**One vector dimension for every vendor.** OpenAI and Gemini both support shortened embeddings,
so all embedders produce 768-dimensional unit vectors. One `vector(768)` column and one HNSW
index serve all of them. Rows are keyed by `(chunk_id, embedder)` with a content hash, so
switching embedder or editing a chunk re-embeds only what changed.

**Fusion in Python, not SQL.** Both stores return two rankings (lexical, vector) and share one
RRF implementation. A single SQL statement could fuse them too, but two implementations of the
same formula is a place for drift. Postgres ranks text with `ts_rank`, not BM25, so the stores'
rankings differ slightly; the eval baseline is recorded for the in-memory store, and the
pgvector store is checked by integration tests and an eval run (see below).

**An extractive mock, not a random one.** The mock answers by quoting the source segments that
best cover the question, from a single provision, and defers to a provision the question names.
It produces the same `Draft` structure a real model must. That gives offline evals a meaningful
signal for everything except the model's own wording.

**Eval faithfulness is re-implemented, not imported.** The verifier checks quotes against the
chunk the model was shown. The eval checks the final answer against the full text of the
article the citation resolves to. A bug that maps a source label to the wrong chunk, or a
citation to the wrong article, passes the first and fails the second
(`test_quote_cited_to_the_wrong_article_is_unfaithful`).

## Calibration (dev split only)

| threshold | value | evidence |
|---|---|---|
| `min_evidence_coverage` | 0.45 | On `dev`, the highest coverage of an unanswerable question was 0.37 and the lowest of an answerable one 0.49. The threshold sits nearer the answerable side because a wrong answer costs more than a wrong refusal. |
| `min_claim_support` | 0.50 | Half of a claim's content words must appear in its quote: loose enough for a paraphrase, tight enough to reject a claim attached to an unrelated quote. Not tuned on data. |
| gate floors | see `evals/gate.json` | Written before the test split was first run, and not changed after it. |

## Live run (Gemini on Vertex AI, 2026-11-08)

Results and reports are in the README and `evals/live/2026-11-08/`. Decisions taken on `dev`:

**Evidence threshold re-checked, kept at 0.45.** With `gemini-embedding-001`, retrieval pulls in
chunks that are semantically close to an off-topic question, so the lexical coverage of those
chunks rises. On `dev`, the lowest coverage of an answerable question stayed at 0.49, but the
highest coverage of an unanswerable one rose from 0.37 to 0.60 (*"Which programming language
should I use to build a chatbot?"*, matched by the AI Act's chatbot provisions). No threshold
separates the two without refusing answerable questions, and moving it for one question would
be tuning to the dataset. The threshold stays; the live model declines that question, and the
offline CI configuration keeps its clean separation.

**Thinking level "low".** On `dev`, `gemini-3.6-flash` with thinking level `low` scored at least
as well as the default on every metric (citation precision 0.984 vs 0.952, the same refusal
decisions) and ran 4.5 times faster (21 s vs 95 s for 44 questions, 6 in parallel; about 2–3 s
per answered question). The task is mostly reading and quoting; deep reasoning bought nothing.

**Temperature at the model default.** Google advises against lowering the temperature for
Gemini 3, so the adapter does not set it. The cost is run-to-run variation; one `dev` question
was refused as unverified in one run and answered in a re-run.

**Separate locations for chat and embeddings.** On the test day, Gemini 3.x chat models answered
in the `eu` multi-region and `global` but not in `europe-west4`, and `gemini-embedding-001`
answered in `europe-west4` and `global` but not in `eu`. Rather than fall back to `global`, the
adapter takes one location per model kind, which keeps both calls in the EU.

**Retries in the SDK.** 408, 429 and 5xx responses are retried up to four attempts with
exponential backoff by the google-genai client, so a rate limit during a parallel eval does not
show up as a `provider_error` refusal.

**Failure analysis.** Every live false refusal was the model declining because the retrieved
paragraphs did not contain the answer: the GDPR's fine amounts (Art. 83(4)–(5) not retrieved,
only 83(2)) and the definitions of "AI system" and "personal data" (the definitions articles
have dozens of points; the per-article cap kept three others). In each case the model said what
was missing instead of answering from memory. The fix belongs in retrieval (a definition lookup
that pins the point whose defined term the question names), not in the prompt. It was not built
in this round: only one `dev` question exercises it, which is too little evidence to tune on.

## Experiments that were rejected

- **Article-title field as a third RRF ranking.** Legal titles look like precise topic labels, so
  each article's title was indexed separately and fused as a third list. On `dev`, citation
  precision fell from 0.853 to 0.813, one out-of-scope question was answered and two answerable
  ones were refused. Titles are short and shared across many articles ("Procedure",
  "Independence", "Tasks"), so title matches were noisy. Reverted.

## Threats to validity

- 87 questions written by the author of the system, in English. Gold labels are article-level
  and were checked against the text, but they reflect one reading of which article answers
  a question.
- Offline scores measure retrieval and gating with a lexical embedder. The live run measures
  one model (Gemini on Vertex AI), once per configuration; live results vary between runs.
- The test split has 10 unanswerable questions, so one mistake moves the unsafe answer rate by
  0.10.

## pgvector check

The same eval, run against PostgreSQL 16 + pgvector 0.6 (local `pgserver`), on 2026-11-08:

| scope | items | faithfulness | citation precision | refusal accuracy | unsafe answers | false refusals | retrieval recall |
|---|---|---|---|---|---|---|---|
| all | 87 | 1.000 | 0.894 | 0.977 | 0.050 | 0.015 | 0.970 |
| dev | 44 | 1.000 | 0.824 | 1.000 | 0.000 | 0.000 | 0.941 |
| test | 43 | 1.000 | 0.969 | 0.954 | 0.100 | 0.030 | 1.000 |

Close to the in-memory store; the differences come from `ts_rank` versus BM25 (one extra false
refusal on `test`). The gate passes on its floors. The regression check applies only to the
in-memory baseline. CI runs the pgvector integration tests against `pgvector/pgvector:pg17`.
