# Everything here runs offline and costs nothing, except `record-live`, which calls the live
# models on purpose. `make check` is what CI runs.
UV      ?= uv
OFFLINE := CA_PROVIDER=mock CA_EMBEDDER=hash CA_STORE=memory

# Record/replay of a live run (providers/replay.py). Recorded with the offline hashing embedder,
# so replay needs no vectors and no network.
CASSETTE       := evals/fixtures/live-answers.jsonl
JUDGE_CASSETTE := evals/fixtures/judge-set.jsonl
REPLAY := CA_PROVIDER=replay CA_EMBEDDER=hash CA_STORE=memory CA_CASSETTE=$(CASSETTE) \
          CA_FAITHFULNESS_SCORER=judge CA_JUDGE_PROVIDER=replay
REPLAY_GATE := --gate evals/gate_judge.json --baseline evals/baseline_replay.json
# The live configuration that `record-live` records. Needs Vertex AI credentials (ADC) and
# GOOGLE_CLOUD_PROJECT in the environment, and costs money.
LIVE ?= CA_PROVIDER=gemini CA_GEMINI_USE_VERTEX=true CA_GEMINI_MODEL=gemini-3.6-flash \
        CA_GEMINI_THINKING_LEVEL=low CA_GEMINI_LOCATION=eu \
        CA_FAITHFULNESS_SCORER=judge CA_JUDGE_PROVIDER=gemini \
        CA_JUDGE_MODEL=gemini-3.1-pro-preview CA_JUDGE_LOCATION=global

.PHONY: help install install-localpg lint typecheck test test-pg eval eval-baseline check \
        eval-replay eval-replay-baseline record-live serve ask ingest db-up index

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:         ## create .venv with all extras and dev tools
	$(UV) sync --all-extras --group dev

install-localpg: ## also install a pip-packaged PostgreSQL+pgvector for `make test-pg` without Docker
	$(UV) sync --all-extras --group dev --group localpg

lint:            ## ruff lint + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

typecheck:       ## mypy
	$(UV) run mypy

test:            ## all tests (pgvector tests skip unless a database is available)
	$(OFFLINE) $(UV) run pytest

test-pg:         ## only the PostgreSQL/pgvector integration tests
	$(OFFLINE) $(UV) run pytest -m pg

eval:            ## offline eval gate: exits 1 when a score is below its floor or regressed
	$(OFFLINE) $(UV) run cited-answers eval

eval-baseline:   ## re-record evals/baseline.json (review the diff before committing it)
	$(OFFLINE) $(UV) run cited-answers eval --update-baseline

eval-replay:     ## replay the recorded live run (model + judge) offline; fails on any change
	$(REPLAY) $(UV) run cited-answers eval $(REPLAY_GATE) --out evals/reports/replay
	CA_JUDGE_PROVIDER=replay CA_CASSETTE=$(JUDGE_CASSETTE) $(UV) run cited-answers judge-eval

eval-replay-baseline: ## re-record evals/baseline_replay.json from the cassette (review the diff)
	$(REPLAY) $(UV) run cited-answers eval $(REPLAY_GATE) --out evals/reports/replay --update-baseline

record-live:     ## re-record both cassettes with the live models (costs money, needs credentials)
	$(LIVE) CA_EMBEDDER=hash CA_STORE=memory CA_RECORD_CASSETTE=$(CASSETTE) \
	  $(UV) run cited-answers eval --workers 3 $(REPLAY_GATE) --out evals/reports/record
	$(LIVE) CA_RECORD_CASSETTE=$(JUDGE_CASSETTE) $(UV) run cited-answers judge-eval --workers 2

check: lint typecheck test eval eval-replay  ## everything CI runs

serve:           ## HTTP API on http://127.0.0.1:8000 (docs at /docs)
	$(UV) run cited-answers serve

ask:             ## make ask Q="What is the right to erasure?"
	$(UV) run cited-answers ask "$(Q)"

ingest:          ## rebuild data/corpus from the committed EUR-Lex snapshots
	$(UV) run cited-answers ingest

db-up:           ## local PostgreSQL 17 + pgvector in Docker
	docker compose up -d db

index:           ## load the corpus into PostgreSQL (needs CA_DATABASE_URL)
	$(UV) run cited-answers index
