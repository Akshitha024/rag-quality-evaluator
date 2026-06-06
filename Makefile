.PHONY: help install lint typecheck test eval drift plots clean

DATA ?= tests/fixtures/synthetic.jsonl
PROVIDER ?= heuristic

help:
	@echo "make install                       - install deps via uv"
	@echo "make lint / typecheck / test       - quality gates"
	@echo "make eval DATA=path PROVIDER=name  - score a JSONL of (q, a, contexts, gold)"
	@echo "  PROVIDER choices: heuristic, anthropic, openai"
	@echo "make drift                         - compare two run snapshots for drift"
	@echo "make plots                         - regenerate the 6 chart types"

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not slow and not needs_provider"

eval:
	uv run rev score --data $(DATA) --provider $(PROVIDER) --out-dir results

drift:
	uv run rev drift --baseline results/baseline.json --candidate results/latest.json

plots:
	uv run rev plots --out-dir results/figures

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
