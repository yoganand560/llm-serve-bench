# llm-serve-bench

**A Mac-first, production-style LLM serving and benchmarking platform — built to measure the full serving path, gateway included.**

`llm-serve-bench` puts a FastAPI gateway in front of swappable inference backends and drives that gateway with an async benchmark harness, so every latency and throughput number reflects the real request path — not just the model.

---

## Why this project exists

Most "LLM benchmark" repos measure a model in isolation. Real serving includes the gateway: validation, timeouts, error mapping, metrics, and streaming overhead. This project measures **end-to-end**, with a few deliberate design choices:

- **Pluggable backends** behind one interface — swap llama.cpp for vLLM without touching the harness.
- **Gateway-driven benchmarking** — gateway overhead is *inside* the measurement, not hidden.
- **Mock backend** — the gateway and tests run with zero model downloads.
- **Honest results policy** — a mock run validates the harness only; it is never reported as model performance.

---

## Architecture

| Layer | Responsibility |
|---|---|
| **FastAPI gateway** | Pydantic validation, timeouts, error mapping, `/metrics` |
| **`ServeBackend` interface** | One contract, three implementations (real / stub / mock) |
| **Benchmark harness** | Async load generation, metric computation, run orchestration |
| **Tracking** | MLflow logging per run |
| **Quality** | Fixed QA set for output sanity checks |
| **CI** | ruff + pytest on push; CPU-only latency-regression gate |

**Config-driven:** `configs/default.yaml` selects the mock backend (exercise the gateway before any download); `configs/llama_cpp_*.yaml` select the real Metal backend.

---

## Environment

Designed and tested on the following hardware:

| Check | Result |
|---|---|
| Architecture | `arm64` |
| Chip | Apple M2 — 8 cores (4P + 4E) |
| Unified memory | 8 GB |
| macOS | 27.2.0 |
| Python (venv) | 3.13 (`/opt/homebrew/bin/python3.13`) |
| `nvidia-smi` / `nvcc` | not available |
| Docker | not installed |
| `llama-server` / `cmake` | not installed |

These constraints shape the design: **no NVIDIA GPU** (vLLM ships as a documented stub + a future rented-GPU path), **8 GB unified memory** (high-concurrency runs may OOM — a real OOM is recorded as a failure, not hidden), and **no Metal inside Docker** (`llama-server` runs natively; only the gateway is containerized).

---

## Quickstart

```bash
# 1. Create and activate a Python 3.13 venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. List available commands
make help

# 3. Exercise the gateway with the mock backend (no model download)
make run CONFIG=configs/default.yaml

# 4. Run a real benchmark on llama.cpp / Metal
make run CONFIG=configs/llama_cpp_q4.yaml
Repository structure
text
llm-serve-bench/
├── .github/workflows/ci.yml
├── configs/            # backend, model, seed, concurrency, thresholds
├── docker/             # gateway-only Dockerfile + compose
├── docs/               # methodology, profiling analysis, GPU run instructions
├── benchmarks/         # benchmark CLI, fixed prompt set, regression baseline template
├── results/            # raw JSON + charts from real runs
├── scripts/            # regression check, plotting, profiling, quality check
├── src/llm_serve_bench/
│   ├── backends/       # ServeBackend ABC + llama.cpp / vLLM stub / mock adapters
│   ├── gateway/        # FastAPI app, schemas, metrics, errors
│   ├── bench/          # load generator, metric computation, run orchestration
│   ├── tracking/       # MLflow logging
│   └── quality/        # fixed QA set
└── tests/              # config, gateway, regression, metrics, backends
Tech stack
Serving: FastAPI, Uvicorn, llama.cpp (llama-server, Metal)

Benchmarking: async load generator, fixed prompt set, concurrency 1–32, three repeats

Tracking: MLflow

Quality: fixed QA set

Tooling: Python 3.13, ruff, pytest, Makefile, Docker, GitHub Actions

Roadmap
☑ Pluggable backend interface (real / stub / mock)
☑ FastAPI gateway with validation, timeouts, error mapping, /metrics
☑ Async benchmarking harness and metric computation
☑ MLflow tracking and fixed QA set
☑ CI: ruff + pytest + CPU-only latency-regression gate
☑ Gateway-only Dockerfile and compose
□ First real llama.cpp / Metal benchmark on the M2 → commit JSON to results/
□ Quantization comparison (F16 / Q8_0 / Q4_K_M)
□ vLLM run on a rented NVIDIA GPU per docs/GPU_PENDING.md
□ Generate charts from real result JSON

