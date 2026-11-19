# llm-serve-bench

Mac-first, production-style LLM serving and benchmarking platform for MLOps / LLM-infra portfolios.

**Status: Phase 9 of 9 — all code written; no real runs yet.** No dependencies installed, no models downloaded, no servers started, no benchmarks run, no charts generated, no Docker images built.

This project was designed on an Apple Silicon Mac. The serving backend is a pluggable interface: the real, runnable backend here is llama.cpp `llama-server` (Metal); an NVIDIA/vLLM adapter exists as a documented stub with the same interface for a future rented-GPU run. No vLLM, continuous-batching or paged-attention results are claimed.

## Environment facts (measured, not assumed)

| Check | Result |
|---|---|
| Architecture | `arm64` |
| Chip | Apple M2, 8 cores (4 performance + 4 efficiency) |
| Unified memory | 8 GB |
| macOS | 27.2.0 |
| Python for venv | 3.13 (`/opt/homebrew/bin/python3.13`) |
| `nvidia-smi` / `nvcc` | not available |
| Docker | not installed |
| `llama-server` / `cmake` | not installed |
| Free disk | 42 GiB at check time |
| Installed by this project | nothing |

## What has and hasn't been run

- **Run:** environment checks only (`uname -m`, `sysctl hw.memsize`, `system_profiler`, Python/tool availability, disk space).
- **Not run:** model downloads, any server, any benchmark, MLflow logging, profiling, lint, tests, Docker builds, CI.

## Architecture (planned)

```
client -> FastAPI gateway -> ServeBackend (abstract interface)
                              |-- LlamaCppBackend  (real, Metal via llama-server, streaming)
                              |-- VLLMBackendStub  (documented stub; raises NotImplementedError)
                              `-- MockBackend      (deterministic, seeded; tests + CI)
```

The gateway handles Pydantic validation, timeouts, error mapping and `/metrics`. The benchmark harness drives the gateway (not the backend directly), so gateway overhead is included in measurements. `configs/default.yaml` selects the mock backend so the gateway can be exercised before any model download; `configs/llama_cpp_*.yaml` select the real Metal backend.

## Benchmark results

**PENDING — no runs yet.** Every cell below is a placeholder. Numbers will be filled only from raw JSON saved in `results/` by a real run. A run with `configs/default.yaml` uses the mock backend and only validates the harness; it is never reported as model performance.

### Mac / llama.cpp / Metal

| Backend | Quantization | Concurrency | TTFT p50 | p95 E2E | tokens/s | requests/s | peak RSS proxy |
|---|---|---|---|---|---|---|---|
| baseline | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| llama.cpp | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

"peak RSS proxy" is peak process resident memory, *not* NVIDIA VRAM. Fixing 8 GB unified memory means a high-concurrency run may OOM; a real failure will be recorded as such.

### Quantization

**PENDING.** See `docs/quantization.md` for the F16 / Q8_0 / Q4_K_M comparison skeleton. All cells there are placeholders until real runs save JSON under `results/`.

### GPU / vLLM

**PENDING: to be run on an NVIDIA GPU machine.** No numbers are claimed in this section. See `docs/GPU_PENDING.md` for the exact commands to produce them.

## Docker

Docker is not installed on this machine. The Dockerfile and compose file were written in Phase 3 but are **not built or tested here**. Metal GPU is not available inside Docker, so `llama-server` runs natively on the Mac and only the gateway would be containerised. Nothing here should be read as a claim that the container works.

## CI

`.github/workflows/ci.yml` runs ruff and pytest on every push. The latency gate reads two saved result JSON files (`results/regression_baseline.json` and `results/regression_latest.json`) and needs no GPU, model or server. It skips with a notice until real results are committed; `benchmarks/regression_baseline.example.json` is a null-valued placeholder that the check explicitly rejects rather than treating as a pass.

## Repo structure

```
llm-serve-bench/
├── .github/workflows/ci.yml
├── configs/            # backend, model, seed, concurrency and threshold settings
├── docker/             # gateway-only Dockerfile + compose (untested here)
├── docs/               # methodology, profiling analysis, GPU run instructions
├── benchmarks/         # benchmark CLI, fixed prompt set, regression baseline template
├── results/            # raw JSON + charts from real runs only
├── scripts/            # regression check, plotting, profiling, quality check
├── src/llm_serve_bench/
│   ├── backends/       # ServeBackend ABC + llama.cpp / vLLM stub / mock adapters
│   ├── gateway/        # FastAPI app, schemas, metrics, errors
│   ├── bench/          # load generator, metric computation, run orchestration
│   ├── tracking/       # MLflow logging
│   └── quality/        # fixed QA set
└── tests/              # config, gateway, regression, metrics, backends
```

## Quickstart

Available after later phases; the project will run entirely through the Makefile and a Python 3.13 venv, with no Docker required. `make help` lists the targets.

## Charts

**PENDING — no PNGs generated, because no real result JSON exists yet.** After real runs, generate them with `make plots ARGS="--result results/baseline.json --result results/llama_cpp_q4.json"` and embed here:

<!--
![Throughput vs concurrency](results/throughput_vs_concurrency.png)
![p95 latency vs concurrency](results/p95_latency_vs_concurrency.png)
![Peak memory vs concurrency](results/memory_vs_concurrency.png)
-->

## Why vLLM might be faster (background, not a measured result)

**PENDING GPU run.** vLLM's continuous batching and paged attention are design features that often help throughput under high concurrency, but this project has **no vLLM measurements**. This paragraph is background only; replace it with an explanation grounded in `results/` only after the NVIDIA run described in `docs/GPU_PENDING.md` is complete.

## Resume bullets

**PENDING — fill `Z` only from real numbers in `results/` after runs complete.**

- Built a pluggable FastAPI LLM gateway with streaming, validation, timeouts and latency telemetry for a Qwen2.5-1.5B server on Apple Silicon, resulting in PENDING.
- Implemented an async benchmarking harness with fixed prompts, concurrency 1–32 and three repeats to compare a Hugging Face baseline with llama.cpp F16/Q8/Q4, resulting in PENDING.
- Instrumented every run with MLflow and added a CPU-only CI latency-regression gate over saved results, resulting in PENDING.
