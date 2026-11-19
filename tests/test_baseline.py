"""Tests for the baseline server using an injected fake generator.

These need FastAPI + pytest only: torch and transformers are never imported.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_serve_bench.baseline.server import GenerateResult, create_app
from llm_serve_bench.baseline.settings import BaselineSettings


class FakeGenerator:
    """Deterministic stand-in for the transformers generator."""

    @property
    def is_loaded(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> GenerateResult:
        return GenerateResult(text=f"answer to {prompt}", generated_tokens=3)


def _client() -> TestClient:
    return TestClient(create_app(BaselineSettings(), FakeGenerator()))


def test_health_reports_loaded_model() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "baseline"
    assert payload["model_loaded"] is True
    assert payload["model"] == BaselineSettings().model_name


def test_generate_returns_text_and_token_count() -> None:
    response = _client().post("/generate", json={"prompt": "hi", "max_new_tokens": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "answer to hi"
    assert payload["generated_tokens"] == 3
    assert payload["backend"] == "baseline"


def test_generate_rejects_empty_prompt() -> None:
    response = _client().post("/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_generate_enforces_configured_token_limit() -> None:
    settings = BaselineSettings(default_max_new_tokens=2, max_max_new_tokens=4)
    client = TestClient(create_app(settings, FakeGenerator()))
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 5})
    assert response.status_code == 422


def test_generate_returns_503_when_no_model_loaded() -> None:
    client = TestClient(create_app(BaselineSettings()))
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 503
