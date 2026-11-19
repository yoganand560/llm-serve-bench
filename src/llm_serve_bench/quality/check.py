"""Run the fixed QA set against a live gateway and score every answer."""

from __future__ import annotations

import httpx

from .qa_set import QAItem, QAResult, score_answer


async def run_quality_check(
    client: httpx.AsyncClient,
    items: list[QAItem],
    *,
    endpoint: str = "/generate",
    max_new_tokens: int,
    seed: int,
) -> list[QAResult]:
    """Ask each question once with greedy decoding and fixed seed."""
    results: list[QAResult] = []
    for item in items:
        response = await client.post(
            endpoint,
            json={
                "prompt": item.question,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "seed": seed,
            },
        )
        response.raise_for_status()
        answer = response.json()["text"]
        matched = score_answer(answer, item.expected_keywords)
        results.append(
            QAResult(
                id=item.id,
                question=item.question,
                answer=answer,
                matched=matched,
                passed=bool(matched),
            )
        )
    return results
