# Quantization experiment — F16 vs Q8_0 vs Q4_K_M

**PENDING — no runs yet.** Every cell below is a placeholder. Fill it only from raw JSON
saved under `results/` by real runs of the benchmark and quality-check CLIs.

## Method

- Same model (Qwen/Qwen2.5-1.5B-Instruct), prompt set, seed, max tokens, concurrency
  levels and repeat count for every quantization variant.
- `peak_rss_mb` is process RSS sampled from `llama-server` — a unified-memory proxy on
  Apple Silicon, **not** NVIDIA VRAM.
- Quality uses a small fixed QA set with keyword matching (10 items). It detects obvious
  degradation but is not a formal accuracy evaluation.
- Qwen2.5-1.5B-Instruct GGUF files are not downloaded yet: F16 ≈ 3.1 GB, Q8_0 ≈ 1.7 GB,
  Q4_K_M ≈ 1.0 GB. These sizes are download expectations, not measured results.

## Results

| Quantization | Peak RSS (MiB) | Throughput at concurrency 1 (tok/s) | Throughput at concurrency 8 (tok/s) | p95 E2E at concurrency 8 (ms) | QA passed / total |
|---|---|---|---|---|---|
| F16 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Q8_0 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Q4_K_M | PENDING | PENDING | PENDING | PENDING | PENDING |

## Trade-off analysis

PENDING: write after real runs. Describe observed memory reduction, speed change and
any QA regressions, including whether the QA sample is too small to be conclusive.
