"""Baseline generation server: Hugging Face ``transformers.generate()`` behind FastAPI.

Design notes for interviews:

* Heavy libraries (torch, transformers) are imported inside ``TransformersGenerator.load``,
  so this module and its tests can be imported on a machine with none of them installed.
* ``create_app`` accepts any object satisfying ``TextGenerator``. Tests inject a fake;
  production passes :class:`TransformersGenerator` from ``main()``.
* The baseline returns a complete response in one chunk. The benchmark harness will
  therefore report baseline TTFT as equivalent to end-to-end latency and inter-token
  latency as not applicable. This is documented rather than silently estimated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .settings import BaselineSettings


class ModelNotLoadedError(RuntimeError):
    """Raised when a generation request arrives before a model is available."""


@dataclass(frozen=True)
class GenerateResult:
    """Backend-neutral generation output."""

    text: str
    generated_tokens: int


class TextGenerator(Protocol):
    """Minimal interface the baseline endpoint depends on."""

    @property
    def is_loaded(self) -> bool:
        """Whether this generator can serve requests."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> GenerateResult:
        """Generate a completion for one prompt."""
        ...


class GenerateRequest(BaseModel):
    """Validated body for ``POST /generate``."""

    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, description="Optional torch manual seed.")


class GenerateResponse(BaseModel):
    """Response for ``POST /generate``."""

    text: str
    generated_tokens: int
    backend: str = "baseline"
    model: str


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    backend: str
    model: str
    model_loaded: bool


class _UnloadedGenerator:
    """Placeholder used when the app is created without a real generator."""

    @property
    def is_loaded(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> GenerateResult:
        raise ModelNotLoadedError("No model is loaded. Start the server with a generator.")


class TransformersGenerator:
    """Real generator backed by ``AutoModelForCausalLM.generate()``."""

    def __init__(self, settings: BaselineSettings) -> None:
        self._settings = settings
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load tokenizer and model. Called once at startup."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "Baseline server needs torch and transformers. "
                "Install them before running `make serve-baseline`."
            ) from exc

        self._torch = torch
        device_name = self._settings.device
        if device_name == "auto":
            if torch.backends.mps.is_available():
                device_name = "mps"
            elif torch.cuda.is_available():
                device_name = "cuda"
            else:
                device_name = "cpu"
        self._device = torch.device(device_name)

        dtype_name = self._settings.dtype
        if dtype_name == "auto":
            torch_dtype = torch.float16 if device_name in {"mps", "cuda"} else torch.float32
        else:
            torch_dtype = getattr(torch, dtype_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._settings.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._settings.model_name,
            torch_dtype=torch_dtype,
        ).to(self._device)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> GenerateResult:
        """Run one synchronous generation call."""
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded.")

        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        options: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            options["temperature"] = temperature

        with torch.inference_mode():
            output = self._model.generate(**encoded, **options)

        new_ids = output[0][prompt_tokens:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerateResult(text=text, generated_tokens=int(new_ids.shape[-1]))


def create_app(
    settings: BaselineSettings | None = None,
    generator: TextGenerator | None = None,
) -> FastAPI:
    """Build the baseline FastAPI app.

    Args:
        settings: Server settings; defaults are used when omitted.
        generator: Injected generator. Defaults to an unloaded placeholder so
            ``/health`` works without torch, transformers or a model download.
    """
    resolved_settings = settings or BaselineSettings()
    resolved_generator: TextGenerator = generator or _UnloadedGenerator()
    app = FastAPI(title="llm-serve-bench baseline", version="0.1.0")

    @app.exception_handler(ModelNotLoadedError)
    async def _handle_not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if resolved_generator.is_loaded else "degraded",
            backend="baseline",
            model=resolved_settings.model_name,
            model_loaded=resolved_generator.is_loaded,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        if request.max_new_tokens > resolved_settings.max_max_new_tokens:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"max_new_tokens exceeds {resolved_settings.max_max_new_tokens}",
            )
        if not resolved_generator.is_loaded:
            raise ModelNotLoadedError("No model is loaded.")

        try:
            result = resolved_generator.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                seed=request.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - gateway maps backend failures to HTTP
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {exc}",
            ) from exc

        return GenerateResponse(
            text=result.text,
            generated_tokens=result.generated_tokens,
            model=resolved_settings.model_name,
        )

    return app


def main() -> None:
    """CLI entrypoint: ``python -m llm_serve_bench.baseline.server``."""
    import uvicorn

    settings = BaselineSettings(
        model_name=os.getenv("LLM_SERVE_BASELINE_MODEL", BaselineSettings().model_name),
        device=os.getenv("LLM_SERVE_BASELINE_DEVICE", "auto"),  # type: ignore[arg-type]
        dtype=os.getenv("LLM_SERVE_BASELINE_DTYPE", "auto"),  # type: ignore[arg-type]
        host=os.getenv("LLM_SERVE_BASELINE_HOST", "127.0.0.1"),
        port=int(os.getenv("LLM_SERVE_BASELINE_PORT", "8001")),
    )
    app = create_app(settings, TransformersGenerator(settings))
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
