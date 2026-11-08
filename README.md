# cited-answers

Question answering over the **EU AI Act** and the **GDPR** where every statement in an answer
cites the article it comes from, and carries a verbatim quote from that article. When the two
texts do not support an answer, the service refuses instead of guessing.

- **FastAPI** service, **PostgreSQL + pgvector** hybrid retrieval, a **LangGraph** pipeline.
- **OpenAI** and **Gemini** (API key or Vertex AI) behind one provider interface, plus a
  deterministic **mock** model so tests, evals and CI run offline and cost nothing.
- An **eval suite** (faithfulness, citation precision, refusal correctness) that **fails CI**
  when a score drops.
- An **LLM judge** behind the same metric interface, tested on its own generated set, and a
  **recorded live run** that CI replays offline, so model behaviour is gated at zero cost.
- **Live results** (Gemini on Vertex AI, all 87 questions): no answer to an out-of-scope
  question, refusal accuracy 0.98, citation precision 0.94, judge-scored faithfulness 0.88
  ([details](#live-run-round-2-final-code-2026-11-08)).

> Not legal advice. The corpus is a snapshot of the texts as published in the Official Journal
> and does not include later corrigenda or amendments. Only the Official Journal is authentic.

## What it does

```text
$ cited-answers ask "When must a controller notify a personal data breach?"
2. The processor shall notify the controller without undue delay after becoming aware of a personal
data breach. [1] 1. In the case of a personal data breach, the controller shall without undue delay
and, where feasible, not later than 72 hours after having become aware of it, notify the personal
data breach to the supervisory authority competent in accordance with Article 55, unless the
personal data breach is unlikely to result in a risk to the rights and freedoms of natural persons.
[2]

[1] GDPR Article 33(2), Notification of a personal data breach to the supervisory authority
    https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679#art_33
[2] GDPR Article 33(1), Notification of a personal data breach to the supervisory authority
    https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679#art_33

$ cited-answers ask "What is the corporate tax rate in the Netherlands?"
REFUSED (no_evidence): I can't answer this from the EU AI Act or the GDPR: no provision in either text covers it.
  why: sources cover 14% of the question's informative words (minimum 45%)
```

The output above comes from the offline mock model, which answers by quoting source sentences
verbatim (paragraph numbers included).
With a live model, the claim text is a plain-English sentence, but it still has to carry a
verbatim quote that passes the same checks.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install          # .venv with all extras and dev tools
make test             # 157 tests; pgvector tests run when a database is available
make eval             # offline eval gate (mock model): prints scores, exits 1 on a regression
make eval-replay      # replay of a recorded live run (model + LLM judge), still offline and free
make serve            # API on http://127.0.0.1:8000, OpenAPI docs at /docs
```

```bash
curl -s localhost:8000/v1/answer -H 'content-type: application/json' \
  -d '{"question": "Which AI practices are prohibited?", "debug": false}'
```

PostgreSQL instead of the in-memory store:

```bash
make db-up                                     # pgvector/pgvector:pg17 in Docker, localhost only
export CA_STORE=pgvector CA_DATABASE_URL=postgresql://cited:cited@127.0.0.1:5432/cited
make index                                     # loads 1,328 chunks + embeddings (idempotent)
make serve
```

No Docker? `make install-localpg` installs a pip-packaged PostgreSQL 16 with pgvector that the
integration tests start on their own (`make test-pg`).

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/answer` | `{question, laws?, debug?}` → answer with claims, quotes and citations, or a refusal with a code |
| `GET` | `/v1/articles/{law}/{number}` | full text of a cited article (`gdpr/17`, `ai_act/annex-III`) so a reader can check a quote |
| `GET` | `/v1/sources` | provenance: CELEX numbers, source URLs, retrieval time, SHA-256 of each snapshot |
| `GET` | `/healthz`, `/readyz` | liveness; readiness checks the corpus and the store |

Refusal codes: `no_evidence` (the texts do not cover the question), `model_declined`,
`unverified` (the model's claims failed the citation checker twice), `provider_error`,
`empty_question`.

## How it works

```text
question
   │
   ▼
retrieve ──► explicit references ("Article 17 GDPR", "Annex III") are pinned first, and so is
   │         the defining point when a question asks for a definition ("define X");
   │         a question naming one law searches only that law;
   │         lay words are expanded with the texts' own wording ("delete" → "erase");
   │         hybrid search: BM25 + dense vectors, fused with Reciprocal Rank Fusion;
   │         at most 3 chunks per article so one long article cannot crowd out the rest
   ▼
assess_evidence ──insufficient──► refuse (no model call)
   │  IDF-weighted share of the question's informative words found in the top chunks
   ▼
generate ── the model returns structured claims: {text, source label, verbatim quote}
   │
   ▼
verify ──fails──► generate again with the checker's findings ──fails again──► refuse
   │  per claim: label exists · quote is verbatim in that source · every number in the
   │  claim appears in its quote · the claim shares enough words with its quote
   ▼
finalize ── answer text with [n] markers, citations resolved to article, title and EUR-Lex link
```

Each box is a node in a LangGraph `StateGraph`; every edge is decided by deterministic code.
The model only fills in `generate`. With `debug: true` the API returns a per-node trace
(retrieved chunks with their per-retriever ranks, evidence coverage and missing words, verifier
findings, timings).

Techniques, and why:

- **Article-aware chunking.** The EUR-Lex HTML is parsed into one chunk per numbered paragraph.
  Long lists (68 definitions, the prohibited practices) are split per point and keep their
  lead-in sentence. Every chunk knows its law, article, paragraph, point, title and chapter, so a
  citation reads like `AI Act Article 5(1)(f)` and links to the article's anchor on EUR-Lex.
- **Contextual chunk headers.** Retrieval indexes `law + article + title + chapter + text`,
  while quotes and citations use the text alone. A paragraph that never repeats its article's
  topic can still be found by that topic.
- **Hybrid retrieval with RRF.** Lexical and vector rankings are fused by rank, so no score
  calibration between them is needed. The same fusion code serves both stores.
- **Definition lookup.** "How does the GDPR define personal data?" pins GDPR Article 4, point
  (1). The definitions articles have 26 and 68 points that share most of their words, so ranked
  retrieval kept picking the wrong ones. A term only counts when it sits where a definition
  question puts it ("define X", "what does X mean", "What is X?") and matches a defined term as
  a whole phrase, so "high-risk AI systems" does not pull in the definition of 'AI system'.
  Tested against all 95 definitions in both texts, not against the eval questions.
- **Two deterministic gates around one model call.** The evidence gate stops out-of-scope
  questions before any tokens are spent. The citation verifier stops unsupported claims after
  generation. A failed check is fed back to the model once; a second failure is a refusal, never
  a partially checked answer.
- **Structured output instead of free-text citations.** The model cannot produce a malformed
  citation. It can only name a source label it was shown, and the verifier rejects anything else.
- **Vendor-neutral providers.** `Embedder` and `AnswerModel` are small protocols. OpenAI
  (Chat Completions with a strict JSON schema) and Gemini (`response_schema`, API key or Vertex
  AI) share one prompt and one schema. Vendor errors are reported by type and status code only,
  because some vendor messages echo part of the key.

## Evals

`evals/dataset.jsonl` holds 87 questions: 67 answerable, each labelled with the article(s) that
answer it, and 20 that the two texts cannot answer (other laws, everyday questions, near misses
such as "What does the AI Act say about cryptocurrency mining?"). Every gold label was checked
against the article text. The set is split into `dev` (44) and `test` (43). The split was fixed
before any tuning, all tuning used `dev` only, and `test` was run once at the end.

| metric | definition |
|---|---|
| faithfulness | share of claims whose quote is verbatim in the **cited article** (looked up independently of the pipeline), whose numbers all appear in that quote, and which share at least half their words with it. With `CA_FAITHFULNESS_SCORER=judge`, an LLM judge must also find the claim fully supported by the article (below) |
| citation precision | over answered, answerable questions: share of cited articles that are gold articles |
| refusal accuracy | share of questions where the system answered an answerable one or refused an unanswerable one |
| unsafe answer rate | share of unanswerable questions that were answered (the costly error) |
| false refusal rate, retrieval recall | diagnostics, not gated |

Current offline scores (mock model, hashing embedder, in-memory store):

| scope | items | faithfulness | citation precision | refusal accuracy | unsafe answers | false refusals | retrieval recall |
|---|---|---|---|---|---|---|---|
| all | 87 | 1.000 | 0.881 | 0.989 | 0.050 | 0.000 | 0.970 |
| dev | 44 | 1.000 | 0.853 | 1.000 | 0.000 | 0.000 | 0.941 |
| test | 43 | 1.000 | 0.909 | 0.977 | 0.100 | 0.000 | 1.000 |

The **gate** (`evals/gate.json`) fails the build when, in any scope, faithfulness < 0.98, citation
precision < 0.75, refusal accuracy < 0.90 or unsafe answer rate > 0.10, **or** when any of these
moves more than 0.02 in the wrong direction against `evals/baseline.json`. The mock is
deterministic, so any change is real. Accepting a lower score means re-recording the baseline
(`make eval-baseline`), which shows up as a reviewed diff. The tests prove the gate works: a
pipeline whose retriever ranking is reversed, and one whose evidence gate is switched off, both
fail it.

**What the offline eval measures, and what it does not.** With the mock model, faithfulness is
1.0 by construction: the mock only quotes. The offline numbers therefore measure retrieval,
reference handling, the evidence gate, citation mapping and the checkers, and they catch
regressions in any of them. They do not measure how well a live model paraphrases or follows
the refusal rule. The live run below measures that.

Known offline failures, all visible in `evals/reports/latest.md`:

- *"What interest rate does the European Central Bank charge?"* is answered (test split). The
  AI Act mentions the European Central Bank, so the lexical evidence gate passes and the mock
  quotes an unrelated sentence. The live model declines it (below).
- Seven answerable questions cite a neighbouring article instead of the gold one. Examples:
  emotion recognition at work cites the transparency duty (Art. 50) instead of the prohibition
  (Art. 5(1)(f)); "legal grounds for processing" misses Art. 6 because the text never says
  "grounds". These are vocabulary gaps that a hashing embedder cannot bridge.

### Live run, round 1 (Gemini on Vertex AI, 2026-11-08)

Same dataset, same metrics, same gate. Chat model `gemini-3.6-flash`, embeddings
`gemini-embedding-001` (768 dimensions), both in EU locations. The configuration was chosen on
`dev`; `test` was then run once. Reports: [`evals/live/2026-11-08/`](evals/live/2026-11-08/).

`dev` (44 questions), in-memory store:

| model | embedder | citation precision | refusal accuracy | unsafe answers | false refusals | retrieval recall | wall time* |
|---|---|---|---|---|---|---|---|
| mock (offline) | hashing (offline) | 0.853 | 1.000 | 0.000 | 0.000 | 0.941 | |
| mock | gemini-embedding-001 | 0.882 | 0.977 | 0.100 | 0.000 | 1.000 | |
| gemini-3.6-flash | hashing | 0.928 | 0.909 | 0.000 | 0.118 | 0.941 | 69 s |
| gemini-3.6-flash | gemini-embedding-001 | 0.952 | 0.932 | 0.000 | 0.088 | 1.000 | 95 s |
| **gemini-3.6-flash, thinking low** | **gemini-embedding-001** | **0.984** | **0.932** | **0.000** | **0.088** | **1.000** | **21 s** |

\* 44 questions, 6 in parallel. Faithfulness was 1.000 in every row: every answered claim's
quote was found verbatim in the cited article by the eval's independent check.

The chosen configuration on the held-out split, run once:

| scope | items | faithfulness | citation precision | refusal accuracy | unsafe answers | false refusals | retrieval recall |
|---|---|---|---|---|---|---|---|
| test, offline (mock + hashing) | 43 | 1.000 | 0.939 | 0.977 | 0.100 | 0.000 | 1.000 |
| **test, live** | 43 | 1.000 | 0.953 | 0.977 | **0.000** | 0.030 | 1.000 |
| all, live on PostgreSQL + pgvector | 87 | 1.000 | 0.976 | 0.954 | 0.000 | 0.060 | 1.000 |

What changed with a live model:

- **No unsafe answers.** The European Central Bank question is now refused by the model. With
  semantic retrieval, *"Which programming language should I use to build a chatbot?"* passes
  the evidence gate (the AI Act's chatbot provisions cover its words), and the model declines
  it. The two layers catch different things.
- **The price is some over-refusal.** The model declines when the retrieved paragraphs do not
  contain the answer, even when it knows the answer. *"What is the highest administrative fine
  under the GDPR?"* retrieved the criteria in Art. 83(2), not the amounts in Art. 83(4)–(5), and
  the model said so instead of quoting the amounts from memory. Both definition questions
  (*"How does the AI Act define an AI system?"*, *"How does the GDPR define personal data?"*)
  were declined because the definitions articles have dozens of points and the right one was not
  among the three kept per article. That was a retrieval gap; round 2 adds a definition lookup.
- **Results vary between runs.** Temperature is left at the model default, so a borderline
  question can flip: one `dev` question was refused as unverified in one run and answered with
  four verified claims in a re-run. Each configuration was run once; treat differences of one
  question (0.03 on 34 answerable items) as noise.

To reproduce, with your own Vertex AI project and Application Default Credentials (this costs
money; see `.env.example` for the variables):

```bash
CA_PROVIDER=gemini CA_EMBEDDER=gemini CA_GEMINI_USE_VERTEX=true \
CA_GEMINI_MODEL=gemini-3.6-flash CA_GEMINI_THINKING_LEVEL=low \
CA_GEMINI_LOCATION=eu CA_GEMINI_EMBEDDING_LOCATION=europe-west4 \
uv run cited-answers eval --split dev --workers 6 --out evals/reports/live
```

### Live run, round 2 (final code, 2026-11-08)

All 87 questions with the code in this repository: definition lookup, same-article linking and
the conditions rule (below), `gemini-3.6-flash` (thinking low) and `gemini-embedding-001`,
faithfulness scored by the LLM judge.

| scope | items | faithfulness (judge) | citation precision | refusal accuracy | unsafe answers | false refusals | retrieval recall |
|---|---|---|---|---|---|---|---|
| all | 87 | 0.884 | 0.939 | 0.977 | 0.000 | 0.030 | 1.000 |
| dev | 44 | 0.827 | 0.938 | 0.955 | 0.000 | 0.059 | 1.000 |
| test | 43 | 0.946 | 0.939 | 1.000 | 0.000 | 0.000 | 1.000 |

- Both definition questions are now answered from the defining point and judged supported.
- Two false refusals are left, both on `dev`: the GDPR's fine amounts (Art. 83(4)–(5) still not
  retrieved) and one answer that failed verification twice.
- One judge call failed on a rate limit. That claim counts as unsupported and the run was
  marked as failed for it, as designed; faithfulness may be one claim higher.
- Round 1's faithfulness of 1.000 used the quote check; these numbers use the judge, which is
  stricter. They are not comparable.

Reports: [`evals/live/2026-11-08/`](evals/live/2026-11-08/) (`round2-*`).

### LLM judge

The quote check proves that a quote is real and that the numbers match, but it cannot read.
`CA_FAITHFULNESS_SCORER=judge` keeps it and adds an LLM judge behind the same
`FaithfulnessScorer` interface (`evals/judge.py`). The judge gets the claim, its quote and the
full cited article, and answers "supported", "partially supported" (a dropped condition or
exception, "shall" turned into "may", the wrong actor) or "not supported". Only "supported"
counts. It is a different, larger model than the one answering (`gemini-3.1-pro-preview` judging
`gemini-3.6-flash`). A failed judge call is not a verdict: it fails the run.

A judge needs its own test. `evals/judge_set.jsonl` is generated from the corpus: 20 obligation
sentences, each edited in ways whose effect is known in advance (`cited-answers judge-eval`):

| kind | expected | cases | quote check passes | judge passes |
|---|---|---|---|---|
| verbatim | supported | 20 | 20 | 20 |
| "shall" -> "must" | supported | 20 | 20 | 20 |
| "shall" -> "may" | not supported | 20 | 20 | **0** |
| negation | not supported | 20 | 20 | **0** |
| another actor | not supported | 8 | 8 | **0** |
| changed number | not supported | 5 | 1 | 0 |

The quote check catches 4 of the 53 meaning changes; with the judge, 53 of 53, with no false
alarm on the 40 faithful claims. This shows sensitivity to clear distortions, not to subtle
paraphrase.

**What it found.** On a recorded live run, the judge rejected 26 of 131 claims that had all
passed the quote check (faithfulness 0.80). Most stated a rule without an exception from another
paragraph of the same article (the right to erasure without Art. 17(3)), or dropped a qualifier
("where possible", "and" turned into "or"). Three general fixes, chosen from `dev` failures:
exception paragraphs travel with the paragraph they limit, a paragraph 1 travels with the later
paragraphs that qualify it (`pipeline/linking.py`), and the prompt asks to keep conditions and to
give a sourced exception as its own claim. Judge-scored faithfulness rose to 0.87 (`dev` 0.82,
`test` 0.93) and refusal accuracy to 0.95, with no unsafe answers. The 0.90 target is not met on
`dev`; see [docs/DESIGN.md](docs/DESIGN.md) for what is left. In `evals/gate_judge.json` it is
a tracked target: every run reports "target not met" and it may not regress, but it does not
block. Citation precision, refusal accuracy and unsafe answers stay blocking floors.

### Record and replay in CI

A live run is recorded once into a cassette (`evals/fixtures/`): a hash of exactly what each
model was sent, mapped to what it answered. `make eval-replay` replays the answers and the
judge's verdicts with no network and no cost, and CI runs it on every push. It reproduces the
recorded metrics exactly and is regression-checked against `evals/baseline_replay.json`.

Replay is strict. If a change alters what the model would be sent (a prompt edit, retrieval
returning other chunks), the request is not in the cassette and the gate fails with "re-record"
instead of guessing. `make record-live` re-records; it reuses every stored answer to an
unchanged request and pays only for the rest. A failed live call (a rate limit) is never
stored as an answer; it fails the recording run, and running it again fills the gap. The cassettes are recorded with the offline hashing embedder, so replay needs no stored
vectors; the semantic-embedding numbers are in the live-run tables.

## Configuration

Environment variables (prefix `CA_`, see `.env.example`): `PROVIDER` (mock|openai|gemini|replay),
`EMBEDDER` (hash|openai|gemini), `STORE` (memory|pgvector), `DATABASE_URL`, model ids, and the
gate thresholds (`MIN_EVIDENCE_COVERAGE`, `MIN_CLAIM_SUPPORT`, ...), the eval scorer
(`FAITHFULNESS_SCORER`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_LOCATION`) and record/replay
(`RECORD_CASSETTE`, `CASSETTE`). API keys are read by the
vendor SDKs from their usual variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, or Vertex AI
Application Default Credentials). The application never reads or logs them.

All vendors embed into 768 dimensions (OpenAI `dimensions=768`, Gemini
`output_dimensionality=768`, vectors re-normalised), so one `vector(768)` column and one HNSW
index serve every embedder. Rows are keyed by `(chunk, embedder)`.

On Vertex AI, chat and embedding models are not offered in the same locations, so they are set
separately: `CA_GEMINI_LOCATION` and `CA_GEMINI_EMBEDDING_LOCATION` (unset means
`GOOGLE_CLOUD_LOCATION`). The Gemini adapter retries 408/429/5xx with exponential backoff, uses
`CA_REQUEST_TIMEOUT_S`, leaves temperature at the model default (Google advises against
lowering it for Gemini 3), and can set `CA_GEMINI_THINKING_LEVEL`.

## Data and provenance

| text | CELEX | source |
|---|---|---|
| Regulation (EU) 2016/679 (GDPR), OJ L 119, 4.5.2016 | 32016R0679 | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679 |
| Regulation (EU) 2024/1689 (AI Act), OJ L, 12.7.2024 | 32024R1689 | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689 |

Both pages were fetched once on 2026-11-08 and are kept gzipped in `data/raw/` with their URL,
retrieval time and SHA-256. `data/corpus/` is the parsed result (99 + 113 articles, 13 annexes,
1,328 chunks). Recitals are left out because they are not citable provisions. A test rebuilds
the corpus from the snapshots and fails if the result differs from the committed files. See
[data/SOURCES.md](data/SOURCES.md).

Legal texts: © European Union, https://eur-lex.europa.eu. Reuse is authorised under Commission
Decision 2011/833/EU.

## Layout

```text
src/cited_answers/
  ingest/        EUR-Lex HTML parser and snapshot builder
  retrieval/     BM25, in-memory hybrid store, PostgreSQL + pgvector store, RRF
  providers/     protocols, mock/scripted models, OpenAI and Gemini adapters, shared prompt
  pipeline/      LangGraph graph, evidence gate, citation verifier, reference resolution
  evals/         metrics (independent of the pipeline's own checks), runner and gate
  api/           FastAPI app
data/            raw snapshots, parsed corpus, provenance
evals/           dataset, gates, baselines, recorded cassettes (fixtures/), judge test set,
                 live-run reports (live/)
docs/DESIGN.md   design decisions, calibration, experiments that were rejected
```

## Limitations and next steps

- Live runs so far: Gemini on Vertex AI only. The OpenAI adapters (answer and judge) are tested
  against fake clients, not yet against the live API.
- CI gates the offline mock run and the replay of one recorded live run. A new live recording
  is manual (`make record-live`) and costs money.
- With semantic retrieval, the lexical evidence gate no longer separates `dev` cleanly (see
  [docs/DESIGN.md](docs/DESIGN.md)); the model-side refusal covers the gap.
- Judge-scored faithfulness is below its 0.90 target on `dev`: the model still sometimes states a
  rule without an exception or condition that the sources contain.
- English only; corrigenda, amendments and recitals are not included.
- Next: let the verifier flag a claim whose paragraph has a linked exception that no claim
  mentions; a fresh, independently written question set (round 2 saw `test` results); judge
  validation on subtle paraphrases, labelled by a lawyer.

## License

Code: [Apache-2.0](LICENSE). Legal texts: © European Union, reused under Commission Decision
2011/833/EU (see [data/SOURCES.md](data/SOURCES.md)).
