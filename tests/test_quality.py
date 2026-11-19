"""Tests for QA-set loading, scoring and gateway-backed checking."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from llm_serve_bench.backends.mock import MockBackend
from llm_serve_bench.config import load_config
from llm_serve_bench.gateway.app import create_app
from llm_serve_bench.quality.check import run_quality_check
from llm_serve_bench.quality.qa_set import QAItem, load_qa_set, score_answer

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
QA_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "qa_set.json"


def test_shipped_qa_set_loads() -> None:
    items = load_qa_set(QA_PATH)
    assert len(items) == 10
    assert all(item.expected_keywords for item in items)


def test_invalid_qa_set_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"id": "x", "question": "why", "expected_keywords": []}]), encoding="utf-8")
    try:
        load_qa_set(path)
    except ValueError as exc:
        assert "expected_keywords" in str(exc)
    else:
        raise AssertionError("invalid QA set should raise ValueError")


def test_scoring_is_case_insensitive() -> None:
    assert score_answer("The answer is Paris.", ("Paris",)) == ("Paris",)
    assert score_answer("No idea.", ("Paris",)) == ()


def test_quality_check_uses_gateway_response() -> None:
    config = load_config(CONFIG_PATH)
    app = create_app(config, MockBackend(config.backend))
    items = [QAItem(id="mock", question="hi", expected_keywords=("The",))]

    async def scenario() -> list[object]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return list(await run_quality_check(client, items, max_new_tokens=3, seed=42))

    results = asyncio.run(scenario())
    assert len(results) == 1
    assert getattr(results[0], "passed") is True
