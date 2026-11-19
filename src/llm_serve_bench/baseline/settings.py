"""Typed settings for the baseline server.

Phase 2 uses environment variables via ``main()``. A shared YAML config loader is
introduced in Phase 3; these fields are deliberately a subset of that future config.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BaselineSettings(BaseModel):
    """Configuration for the Hugging Face baseline server."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float16", "float32"] = "auto"
    host: str = "127.0.0.1"
    port: int = Field(default=8001, ge=1, le=65535)
    default_max_new_tokens: int = Field(default=64, ge=1)
    max_max_new_tokens: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def _validate_token_limits(self) -> BaselineSettings:
        """Ensure the default request cannot exceed the hard limit."""
        if self.default_max_new_tokens > self.max_max_new_tokens:
            raise ValueError("default_max_new_tokens cannot exceed max_max_new_tokens")
        return self
