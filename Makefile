# Everything here runs offline and costs nothing unless you set CA_PROVIDER/CA_EMBEDDER to a
# live vendor yourself. `make check` is what CI runs.
UV      ?= uv
OFFLINE := CA_PROVIDER=mock CA_EMBEDDER=hash CA_STORE=memory

.PHONY: help install install-localpg lint typecheck test test-pg eval eval-baseline check \
        serve ask ingest db-up index

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

check: lint typecheck test eval  ## everything CI runs

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
