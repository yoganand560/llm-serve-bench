# GPU / vLLM work — PENDING

**Nothing in this file has been run.** This project was developed on an Apple Silicon
Mac (M2, 8 GB) with no NVIDIA GPU, no CUDA toolkit and no Docker. The `vllm` adapter
in `src/llm_serve_bench/backends/vllm_stub.py` implements the same interface as the real
adapters but intentionally raises `NotImplementedError`.

There are **no vLLM results, no continuous-batching results and no paged-attention
results** in this repository. Do not interpret any placeholder as a measurement.

## Procedure for a future NVIDIA host (not executed)

1. Provision a Linux host with an NVIDIA GPU and the CUDA drivers.
2. Install Docker with NVIDIA container toolkit, or install vLLM into a Python env.
3. Start vLLM with the configured model, e.g.
   `vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --port 8000`.
4. Use `configs/vllm_nvidia.example.yaml` as a starting point and select it with
   `LLM_SERVE_CONFIG=...`.
5. Replace the vLLM stub with a real adapter implementing `ServeBackend` (streaming
   from `/v1/completions`, and `memory_stats()` from `torch.cuda.max_memory_allocated()`).
6. Run the same benchmark CLI against both backends and save raw JSON under `results/`.
7. Only then fill the GPU section of the README and the resume bullets with real numbers.

Until those steps are completed on real hardware, the GPU section stays marked
**PENDING: to be run on an NVIDIA GPU machine**.
