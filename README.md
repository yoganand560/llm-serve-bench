# LLM Serve Bench 🚀
### LLM Inference Serving & Performance Benchmarking Platform

A modular framework for exploring, serving, and benchmarking Large Language Model (LLM) inference workloads. Built with Python and FastAPI, the project provides a configurable serving architecture and benchmarking pipeline designed to evaluate latency, throughput, concurrency, and memory consumption across inference backends.

The project focuses on understanding the systems engineering challenges involved in deploying and optimizing LLM inference workloads.

---

## 🎯 Project Overview

Deploying an LLM involves more than loading a model and generating text. An efficient inference system must handle concurrent requests, minimize response latency, manage memory resources, and maintain consistent performance under different workloads.

**LLM Serve Bench** explores these challenges through a modular inference-serving architecture and a configurable benchmarking framework.

The platform is designed to:

- Provide a centralized FastAPI inference gateway.
- Support interchangeable model-serving backends.
- Handle request validation and streaming-oriented responses.
- Benchmark inference workloads under different concurrency levels.
- Measure latency, throughput, and memory consumption.
- Track experiments and benchmark metadata using MLflow.
- Support benchmark-result comparison and regression analysis.

> **Project status:** Prototype. The architecture and benchmarking components are under development and require further execution and validation with real LLM workloads.

---

## 🏗️ System Architecture

```text
                 Client / Benchmark Runner
                           |
                           v
                  FastAPI Gateway
                           |
                 Request Validation
                    (Pydantic)
                           |
                 Request Management
                           |
                  Backend Interface
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      llama.cpp           vLLM          Mock Backend
      Backend           Adapter          Backend
          |                |                |
          v                v                v
      Local LLM       NVIDIA GPU       Test Responses
      Inference       (Future)
          |
          v
    Generated Responses
          |
          v
   Performance Monitoring
          |
     +----+---------+
     |              |
     v              v
 Benchmark       MLflow
 Results         Tracking
     |
     v
 Performance Analysis
```

### Architecture Components

| Component | Description |
|---|---|
| FastAPI Gateway | Centralized API layer for inference requests |
| Pydantic | Request and response validation |
| Backend Interface | Modular abstraction for inference engines |
| llama.cpp | Intended local LLM inference integration |
| vLLM | Planned NVIDIA GPU serving integration |
| Mock Backend | Deterministic backend for testing |
| Benchmark Engine | Configurable workload generation and performance measurement |
| MLflow | Experiment tracking and benchmark metadata |
| Regression Checks | Comparison of saved benchmark results |

---

## ⚙️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| Data Validation | Pydantic |
| Model Serving | llama.cpp |
| GPU Serving | vLLM (planned integration) |
| Experiment Tracking | MLflow |
| Benchmarking | Python-based benchmark harness |
| Testing | pytest |
| Configuration | YAML / JSON |
| Development | Git, GitHub |

---

## 📊 Performance Metrics

The benchmarking framework is designed to evaluate the following metrics:

### 1. Time to First Token (TTFT)
Measures the time between submitting a prompt and receiving the first generated token.

### 2. End-to-End Latency
Measures the total time required to complete an inference request.

### 3. Token Throughput
Measures the number of generated tokens per second.

### 4. Request Throughput
Measures the number of completed inference requests per second.

### 5. P50 and P95 Latency
Provides latency distribution measurements to understand typical and slower requests.

### 6. Memory Consumption
Tracks serving-process memory usage during inference workloads.

---

## 📁 Project Structure

```text
llm-serve-bench/
│
├── .github/
│   └── workflows/
│
├── benchmarks/
│
├── configs/
│
├── docker/
│
├── docs/
│
├── results/
│
├── scripts/
│
├── src/
│   └── llm_serve_bench/
│       ├── backends/
│       ├── gateway/
│       ├── bench/
│       ├── tracking/
│       └── quality/
│
├── tests/
│
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 💻 Installation

### Prerequisites

- Python 3.10+
- Git
- pip
- macOS or Linux
- Additional model-serving dependencies depending on the selected backend

### 1. Clone the repository

```bash
git clone https://github.com/yoganand560/llm-serve-bench.git

cd llm-serve-bench
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

Use the dependencies declared in the project configuration:

```bash
pip install -e .
```

### 4. Inspect available commands

```bash
make help
```

Follow the available Makefile targets and project configuration to start the gateway and benchmark workflows.

---

## 🧪 Benchmarking Methodology

The benchmarking workflow is designed around configurable workloads.

A typical evaluation process involves:

1. Configure the selected inference backend.
2. Select the model and generation parameters.
3. Configure the benchmark prompt dataset.
4. Define the request concurrency level.
5. Execute inference requests through the gateway.
6. Capture response latency and throughput metrics.
7. Store benchmark results.
8. Analyze performance across configurations.

### Example Evaluation Scenarios

| Scenario | Objective |
|---|---|
| Single-request inference | Establish baseline latency |
| Concurrent requests | Evaluate performance under parallel workloads |
| Increasing concurrency | Analyze throughput and latency trade-offs |
| Backend comparison | Compare serving implementations |
| Memory profiling | Understand resource consumption |

Actual measurements must be generated before publishing performance comparisons.

---

## 📈 Benchmark Results

Real model benchmark results have not yet been validated.

The following table is the planned reporting format:

| Metric | Result |
|---|---|
| TTFT (p50) | Pending |
| End-to-End Latency (p95) | Pending |
| Token Throughput | Pending |
| Request Throughput | Pending |
| Peak Memory Usage | Pending |

Results will be added after executing real inference workloads.

---

## 🔬 Engineering Focus

This project explores several important areas of LLM infrastructure engineering:

- Modular inference-serving architecture.
- API-based model serving.
- Concurrent request processing.
- Latency and throughput measurement.
- Memory-efficiency analysis.
- Experiment tracking and reproducibility.
- Benchmark regression detection.
- Local inference on Apple Silicon.

---

## 🚧 Current Limitations

- Real llama.cpp inference benchmarks have not yet been validated.
- The vLLM backend is currently a planned integration / adapter stub.
- Benchmark performance results are not yet available.
- Docker deployment requires validation.
- MLflow experiment execution requires validation.

---

## 🛣️ Future Improvements

- Complete and validate llama.cpp integration with Apple Metal.
- Implement a functional vLLM serving backend.
- Add dynamic batching and request scheduling.
- Introduce GPU and memory profiling.
- Generate automated benchmark visualizations.
- Add reproducible performance comparisons.
- Expand CI testing and Docker deployment.
- Investigate inference optimization techniques such as quantization and KV-cache management.

---

## 👨‍💻 Author

**Yoganand Gajanan Patgar**

Machine Learning Engineer | LLM Systems | AI Infrastructure

- GitHub: https://github.com/yoganand560
- LinkedIn: https://www.linkedin.com/in/yoganand-/

---

## ⭐ Project Goal

The goal of LLM Serve Bench is to explore the engineering principles behind efficient LLM inference serving and develop practical experience with model-serving infrastructure, performance benchmarking, and scalable AI systems.
