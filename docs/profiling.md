# Gateway profiling (cProfile)

## What is profiled

- The in-process FastAPI gateway: routing, Pydantic validation, backend client calls,
  SSE encoding and the metrics endpoint.
- **Not** the native llama-server process. If a real backend is used, its compute time
  appears in this profile only as time spent waiting on an HTTP call.
- A profile with `configs/default.yaml` uses the mock backend and validates the tooling;
  it is not model performance.

## How to reproduce

With dependencies installed:

```bash
make profile ARGS="--config configs/llama_cpp_q4.yaml --requests 20"
```

This writes `results/gateway_profile.prof` (trace) and
`results/gateway_profile_summary.txt` (top functions sorted by cumulative time).

## Analysis

**PENDING — no profile has been run yet.** After a real profile, write a short paragraph
here naming the top cumulative-time functions from the saved summary and explaining what
they mean. Do not speculate about hotspots before reading that output.
