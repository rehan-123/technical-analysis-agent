from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Provider-agnostic result of a single generation call.

    A strongly-typed value object rather than a raw dict, so every backend
    returns the same shape and downstream code never special-cases a provider.
    Deliberately minimal and transport-focused:

      * ``text``     — the model's raw output string. This layer does NOT parse
                       or validate it; interpreting the text (e.g. as the
                       thesis JSON) belongs to a later phase.
      * ``model``    — the model that actually produced the output, echoed back
                       for observability and audit.
      * ``raw``      — the provider's native metadata (token counts, timings,
                       finish reason, …), kept opaque so provider-specific
                       fields are available without leaking into the interface.
      * ``latency_ms`` — wall-clock time for the call, measured by the provider.
    """

    text: str
    model: str
    raw: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class LLMProvider(ABC):
    """Abstract large-language-model backend.

    The single Dependency-Inversion seam of the AI layer: the AI service and
    agent depend only on this interface, never on a concrete backend, so the
    provider can be swapped for any concrete implementation without any change
    above it. This is the direct analog of ``MarketDataProvider`` and
    ``NewsProvider``.

    The interface is intentionally provider-neutral — no backend-specific
    concepts appear here. Concrete providers own their own transport, wire
    format, and configuration.

    Responsibility boundary: a provider is a pure transport adapter. It sends a
    prompt and returns the model's raw text plus metadata. It does NOT build
    prompts, validate or repair the model's output, or make any business
    decision — those are owned by higher layers.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for ``prompt`` and return it as an
        ``LLMResponse``.

        Args:
            prompt: The user prompt. Constructed by a higher layer; the
                provider treats it as opaque text.
            system: Optional system prompt establishing role/output contract.
            schema: Optional JSON schema the caller wants the output to conform
                to. Providers that support structured output may use it as a
                hint; the provider does NOT itself validate the result against
                it — validation is a later-phase concern.
            model: Optional per-call model override; falls back to the
                provider's configured default when omitted.

        Raises:
            LLMProviderError: transport failure or non-success response.
            LLMRateLimitError: the backend signalled rate limiting (HTTP 429).
            LLMResponseError: a successful response whose envelope is unusable.

        Async-first: implementations must not block the event loop.
        """
        raise NotImplementedError
