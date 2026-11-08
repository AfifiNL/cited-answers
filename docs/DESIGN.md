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

## Round 2: definitions, an LLM judge, record/replay (2026-11-08)

**Definition lookup** (`pipeline/definitions.py`). The live run's clearest gap: definition
questions were declined because the definitions articles (26 and 68 points) share their words,
and the per-article cap kept the wrong points. A defined term is pinned when it sits where a
definition question puts it ("define X", "definition of X", "what does X mean", "X is defined",
or the whole of "What is X?") and matches a defined term as a whole phrase. A first version that
matched any defined term anywhere in a "define/mean" question pinned 'controller' for "What does
data protection by design mean for controllers?" (`dev`), so the term must now fill that slot.
The rule is tested against all 95 definitions with five phrasings each, plus near misses
("high-risk AI systems", "What is AI?"), not against eval questions. Offline it changed one
`test` item: *"What are AI regulatory sandboxes?"* now cites the definition (Art. 3(55))
instead of Art. 57, which its gold label names. The definition does answer the question, but
the gold label was not edited after seeing a `test` result; the offline baseline was
re-recorded with that one change.

**An LLM judge behind the same metric interface** (`evals/judge.py`). Faithfulness is computed
by a `FaithfulnessScorer`. The default `QuoteScorer` is the deterministic check: a verbatim quote
in the cited article, numbers that match, and word overlap. It cannot read. `JudgeScorer` runs
that check first, so a fake quote never reaches the judge, then asks an LLM whether the claim
says what the cited article says: "supported", "partially supported" (a dropped condition or
exception, "shall" turned into "may", the wrong actor, a rule stated as general that the text
limits) or "not supported". Only "supported" counts. Choices:

- The judge is a different, larger model than the one answering (`gemini-3.1-pro-preview`
  judging `gemini-3.6-flash`), so a model does not grade its own phrasing. It reads the full
  cited article, not only the chunk the answer saw, because that is what a reader would check
  against. (The preview model was only offered in Vertex's `global` location. The judge sees
  public legal text and generated claims, nothing personal.)
- A failed judge call is not a verdict. It is listed, kept out of the judge's rates and fails
  the run. The first recording counted rate-limit errors as "not supported", which inflated the
  judge's catch rate on its own test set. The error count exposed it: 41 errors, not the 15 that
  showed as false alarms.
- The judge has its own test set, generated from the corpus: 20 obligation sentences with edits
  whose effect is known (verbatim and "shall" -> "must" should pass; "shall" -> "may", negation,
  another actor and a changed number should fail). Result, 93 cases:

  | scorer | accuracy | catches meaning changes | false alarms on faithful claims |
  |---|---|---|---|
  | quote check only | 0.473 | 4 of 53 (0.075) | 0 of 40 |
  | quote check + judge | 1.000 | 53 of 53 | 0 of 40 |

  This tests sensitivity to clear distortions. It does not test subtle paraphrase; the claims
  that should pass are close to the source text.

**What the judge found.** Scored by the judge, the recorded live run's faithfulness was 0.80:
26 of 131 claims were "partially supported", although every one passed the quote check. Two
patterns: a rule stated without an exception in another paragraph of the same article (the
right to erasure without Art. 17(3)), and a qualifier dropped from the quoted sentence ("where
possible", "when technically feasible", "and" turned into "or"). The fixes came from the `dev`
items only and are general:

1. *Exception linking* (`pipeline/linking.py`): when a shown paragraph has an exception
   elsewhere in its article ("Paragraphs 1 and 2 shall not apply ...", "By derogation from
   paragraph 2 ..."), that paragraph is added to the sources. 16 such paragraphs, found from
   their first sentence.
2. *A prompt rule*: keep the conditions that limit a rule, keep "and" versus "or", and give a
   sourced exception as its own claim.
3. *Lead-paragraph linking*: fixes 1 and 2 made the model decline more often. In three of the
   new `dev` refusals it named the missing paragraph itself ("Article 35(1) is missing"): the
   sources held paragraphs 2 to 5 of the right article but not paragraph 1, the rule the others
   qualify. Paragraph 1 is now added when only later paragraphs of an article are shown.

Recorded live runs (offline hashing embedder, judge scoring, all 87 questions):

| version | `dev` faithfulness | `dev` refusal accuracy | `test` faithfulness | `test` refusal accuracy | unsafe answers |
|---|---|---|---|---|---|
| before | 0.797 | 0.909 | 0.806 | 0.930 | 0 |
| + exception linking, prompt rule | 0.838 | 0.841 | 0.942 | 0.930 | 0 |
| + lead paragraph | 0.824 | 0.932 | 0.926 | 0.977 | 0 |

The judge-scored faithfulness target (0.90) is met on `test` and not on `dev`. The remaining 22
rejected claims (of 172) show the same two patterns: an exception or a condition left out. In
several cases the exception was among the sources (GDPR Art. 30(5), AI Act Art. 53(2)), so the
model does not always follow the new rule; in others it sits inside the quoted point itself
(the medical and safety exception in AI Act Art. 5(1)(f)). Iteration stopped here rather than
keep tuning to `dev`; each round costs a live recording.

The 0.90 floor for judge-scored faithfulness was written before any judge result existed. With
the system at 0.87, a blocking floor would keep CI red on every push and hide other
regressions, so it became a tracked *target* in `evals/gate_judge.json`: reported on every run,
regression-checked against the replay baseline, not blocking. That was the owner's decision,
made after seeing the result, and the gate file says so. Likely next steps: let the verifier
reject a claim whose source paragraph has a linked exception that no claim mentions, and
feed that back to the model like any other verifier finding.

**Record/replay** (`providers/replay.py`). A live run is recorded once into a cassette: a hash
of exactly what each model was sent (system prompt, user prompt, output schema) mapped to what it
answered. CI replays it with no network. Choices:

- Strict replay. A request the cassette has not seen means the model's input changed (a prompt
  edit, retrieval returning other chunks, a different verifier message on a retry). The recorded
  answers then say nothing about the new code, so a miss fails the gate. Replay never falls back
  to a live call or to a guess. The test suite also checks that the committed cassette still
  replays without misses, so a pull request that changes model inputs fails fast and says
  "re-record".
- Recorded with the offline hashing embedder, so replay needs no stored vectors and no network.
  The cost: the replayed run is not the best configuration (semantic retrieval scores higher).
  Its job is to catch regressions in everything around the model, not to be the benchmark.
- Only successful calls are recorded; a failed live call is not something the model said.
  Re-recording reuses every stored answer to the same request from the same model and pays only
  for what changed (31 of 246 calls in the last round). Entries a run no longer uses are dropped.
- The replay of a recording reproduces its metrics exactly (checked), and its baseline
  (`evals/baseline_replay.json`) is regression-checked like the offline one.

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
- Round 2's fixes were chosen from `dev` failures only, but `test` results were visible in
  each recording. `test` is no longer a fully untouched hold-out; a fresh question set would be.
- The judge is a model too. Its own test set shows it catches clear distortions without false
  alarms; it is not validated on subtle paraphrase, and its "partially supported" verdicts on
  live claims were spot-read, not checked by a lawyer.
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
