"""Factory that maps a backend name from config to its adapter."""

from __future__ import annotations

from ..config import BackendConfig
from .base import ServeBackend


def build_backend(config: BackendConfig) -> ServeBackend:
    """Construct (but do not start) the configured backend adapter."""
    if config.name == "mock":
        from .mock import MockBackend

        return MockBackend(config)
    if config.name == "llama_cpp":
        from .llama_cpp import LlamaCppBackend

        return LlamaCppBackend(config)
    if config.name == "baseline_http":
        from .baseline_http import BaselineHttpBackend

        return BaselineHttpBackend(config)
    if config.name == "vllm":
        from .vllm_stub import VLLMBackendStub

        return VLLMBackendStub(config)
    raise ValueError(f"unknown backend '{config.name}'")
