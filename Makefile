# Phase 1 scaffold. Targets are wired up now; the scripts they call arrive in later phases.
# The project is runnable without Docker: use the venv targets below.

PYTHON ?= python3.13
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

.PHONY: help venv install-dev install-runtime install-all lint test \
	serve-baseline serve-llama serve-gateway bench quality log profile plots regress clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

venv:  ## Create the Python 3.13 virtual environment (no project packages)
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install-dev: venv  ## Install dev tooling only (pytest, ruff)
	$(PIP) install -e '.[dev]'

install-runtime: venv  ## Install serving + benchmark dependencies
	$(PIP) install -e .

install-all: venv  ## Install every declared extra (dev, mlflow, matplotlib)
	$(PIP) install -e '.[all]'

lint:  ## Run ruff (implemented in Phase 8)
	$(VENV)/bin/ruff check src tests benchmarks scripts

test:  ## Run pytest (tests written in Phases 3 and 8; needs deps installed)
	$(VENV)/bin/pytest

serve-baseline:  ## Start the HF baseline server (needs deps + model; ask before installing)
	$(PY) -m llm_serve_bench.baseline.server

serve-llama:  ## Start llama.cpp via the gateway (needs llama-server + GGUF model)
	LLM_SERVE_CONFIG=configs/llama_cpp_q4.yaml $(PY) -m llm_serve_bench.gateway.app

serve-gateway:  ## Start the gateway with the safe mock backend
	$(PY) -m llm_serve_bench.gateway.app

bench:  ## Benchmark a live gateway; pass ARGS="--backend-label ... --quantization ..."
	$(PY) benchmarks/run_benchmark.py $(ARGS)

log:  ## Log a saved result JSON to MLflow; pass ARGS="--result results/... --artifact results/....png"
	$(PY) scripts/log_result.py $(ARGS)

profile:  ## Profile the in-process gateway with cProfile; pass ARGS="--config ... --requests ..."
	$(PY) scripts/profile_gateway.py $(ARGS)

plots:  ## Generate PNGs from saved result JSON; pass ARGS="--result results/... --result results/..."
	$(PY) scripts/make_plots.py $(ARGS)

quality:  ## Run the fixed QA check against a live gateway; pass ARGS="--backend-label ... --quantization ..."
	$(PY) scripts/quality_check.py $(ARGS)

regress:  ## Compare fresh vs stored p95 from saved JSON; pass ARGS="--fresh results/..."
	$(PY) scripts/check_regression.py $(ARGS)

clean:  ## Remove local caches and the venv (never deletes results/)
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
