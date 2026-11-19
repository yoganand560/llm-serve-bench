"""Real Apple Silicon backend: llama.cpp ``llama-server`` over its OpenAI-compatible API.

Metal acceleration is selected with ``--n-gpu-layers -1``. Memory is sampled from the
server process with psutil and labelled as a unified-memory proxy, never as VRAM.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import AsyncIterator

import httpx
import psutil

from ..config import BackendConfig
from .base import MemoryStats, ServeBackend, TokenChunk


class LlamaCppBackend(ServeBackend):
    """Launch and stream from a local llama-server process."""

    name = "llama_cpp"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._sampler: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._peak_rss_mb = 0.0

    @property
    def _base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def start(self) -> None:
        """Start llama-server, wait for health, then begin sampling memory."""
        model = self.config.model_path or self.config.model
        command = [
            "llama-server",
            "--model",
            model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--n-gpu-layers",
            str(self.config.n_gpu_layers),
            "--ctx-size",
            str(self.config.ctx_size),
            "--no-webui",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_until_ready(timeout_s=60.0)
        self._sampler = threading.Thread(target=self._sample_memory, daemon=True)
        self._sampler.start()

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("llama-server exited during startup; check the model path")
            try:
                if httpx.get(f"{self._base_url}/health", timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.25)
        raise TimeoutError("llama-server did not become healthy within the startup timeout")

    def _sample_memory(self) -> None:
        if self._process is None:
            return
        process = psutil.Process(self._process.pid)
        while not self._stop_event.is_set():
            try:
                rss_mb = process.memory_info().rss / (1024 * 1024)
                self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
            except psutil.Error:
                break
            self._stop_event.wait(0.2)

    def stop(self) -> None:
        """Stop sampling and terminate the server process."""
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2.0)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def health(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            return httpx.get(f"{self._base_url}/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def memory_stats(self) -> MemoryStats:
        return MemoryStats(
            peak_rss_mb=round(self._peak_rss_mb, 2) if self._peak_rss_mb else None,
            source="psutil RSS of llama-server (unified-memory proxy, not VRAM)",
        )

    def reset_memory_peak(self) -> None:
        self._peak_rss_mb = 0.0

    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> AsyncIterator[TokenChunk]:
        """Stream completion text from ``/v1/completions`` as Server-Sent Events."""
        payload: dict[str, object] = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        exact_tokens: int | None = None
        async with httpx.AsyncClient(base_url=self._base_url, timeout=None) as client:
            async with client.stream("POST", "/v1/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line.removeprefix("data:").strip()
                    if body == "[DONE]":
                        break
                    event = json.loads(body)
                    usage = event.get("usage") or {}
                    timings = event.get("timings") or {}
                    if usage.get("completion_tokens") is not None:
                        exact_tokens = int(usage["completion_tokens"])
                    if timings.get("predicted_n") is not None:
                        exact_tokens = int(timings["predicted_n"])
                    choices = event.get("choices") or []
                    text = choices[0].get("text", "") if choices else ""
                    if text:
                        yield TokenChunk(text=text)
        yield TokenChunk(text="", is_last=True, token_count=exact_tokens)
