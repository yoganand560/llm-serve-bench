"""Fixed QA items and transparent keyword-based scoring.

This is a small sanity check, not a formal benchmark: it detects obvious quality loss
from quantization but cannot replace a proper evaluation suite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QAItem:
    """One question with case-insensitive expected substrings."""

    id: str
    question: str
    expected_keywords: tuple[str, ...]


@dataclass(frozen=True)
class QAResult:
    """Scored answer for one QA item."""

    id: str
    question: str
    answer: str
    matched: tuple[str, ...]
    passed: bool


def load_qa_set(path: str | Path) -> list[QAItem]:
    """Load and validate a QA set from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("QA set must be a non-empty JSON list")

    items: list[QAItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each QA item must be an object")
        identifier = entry.get("id")
        question = entry.get("question")
        keywords = entry.get("expected_keywords")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("QA item id must be a non-empty string")
        if not isinstance(question, str) or not question:
            raise ValueError("QA item question must be a non-empty string")
        if not isinstance(keywords, list) or not keywords or not all(
            isinstance(keyword, str) and keyword for keyword in keywords
        ):
            raise ValueError("expected_keywords must be a non-empty list of strings")
        items.append(QAItem(id=identifier, question=question, expected_keywords=tuple(keywords)))
    return items


def score_answer(answer: str, expected_keywords: tuple[str, ...]) -> tuple[str, ...]:
    """Return the expected keywords found (case-insensitively) in ``answer``."""
    lowered = answer.lower()
    return tuple(keyword for keyword in expected_keywords if keyword.lower() in lowered)
