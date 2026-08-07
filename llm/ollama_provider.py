from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from config.settings import Settings, get_settings
from llm.base import LLMProvider, LLMResponse
from llm.exceptions import LLMProviderError, LLMRateLimitError, LLMResponseError
from utils.logger import get_logger

logger = get_logger(__name__)

# Transient server-side statuses worth retrying. 4xx are deterministic and 429
# is handled separately, so neither is retried.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


class OllamaProvider(LLMProvider):
    """Transport adapter for a local Ollama server.

    Implements *only* the transport concern of ``LLMProvider``: HTTP
    communication with Ollama's ``/api/chat`` endpoint, timeout handling,
    retry policy, envelope parsing, and metadata collection. It builds no
    prompts, validates no task output, and makes no business decision — those
    belong to later phases.

    Dependency Injection: an ``httpx.AsyncClient`` may be supplied by the
    caller (tests inject one backed by ``httpx.MockTransport``, so the provider
    is exercisable with zero network). If none is supplied, the provider
    lazily creates and exclusively owns one — and only ever closes a client it
    created itself. Lifecycle handling is identical to ``FinnhubProvider``.
    """

    PROVIDER_NAME = "ollama"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = self._settings.llm_base_url.rstrip("/")
        self._default_model = self._settings.llm_model
        self._timeout = self._settings.llm_request_timeout
        self._max_retries = max(1, self._settings.llm_max_retries)
        self._retry_backoff = self._settings.llm_retry_backoff
        self._temperature = self._settings.llm_temperature
        self._num_predict = self._settings.llm_num_predict
        self._num_ctx = self._settings.llm_num_ctx

        self._client = client
        self._owns_client = client is None
        self._client_lock = asyncio.Lock()

    # --- Async resource management -------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating one lazily if we own it.

        Lazy creation keeps ``__init__`` free of I/O and event-loop dependency,
        so the provider can be built at wiring time. The lock makes concurrent
        first-use safe — without it, simultaneous requests could each build a
        client and leak all but one.
        """
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
                self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Release the HTTP client if (and only if) this provider owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaProvider":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- Request assembly (wire format only, no prompt construction) ----------

    def _build_payload(
        self,
        prompt: str,
        system: str | None,
        schema: dict[str, Any] | None,
        model: str | None,
    ) -> dict[str, Any]:
        """Assemble the Ollama ``/api/chat`` request body.

        This is wire-format mapping, not prompt construction: ``prompt`` and
        ``system`` are received already-built and placed into Ollama's message
        structure verbatim.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                # Bound the completion. Generation — not prompt ingestion — is
                # the dominant cost on CPU, and Ollama otherwise runs until EOS
                # or the context fills.
                "num_predict": self._num_predict,
                # Sized explicitly so prompt + completion fit. Ollama's default
                # context would silently truncate the prompt instead.
                "num_ctx": self._num_ctx,
            },
        }
        # Ollama accepts ``format: "json"`` or a JSON schema object to bias the
        # model toward structured output. Passed as a transport hint only; this
        # layer does not validate the result against it.
        payload["format"] = schema if schema else "json"
        return payload

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw.strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _log_usage(result: LLMResponse, model: str, latency_ms: float) -> None:
        """Log real token counts and generation timing — never any content.

        Ollama returns ``prompt_eval_count`` / ``eval_count`` (actual tokens in
        and out) and nanosecond durations, which ``_extract`` already preserves
        in ``LLMResponse.raw``. Surfacing them turns "it timed out" into a
        measurable throughput figure, and makes an oversized prompt or an
        unbounded completion visible immediately.
        """
        raw = result.raw or {}
        prompt_tokens = raw.get("prompt_eval_count")
        output_tokens = raw.get("eval_count")
        eval_ns = raw.get("eval_duration")
        tps = None
        if isinstance(output_tokens, int) and isinstance(eval_ns, int) and eval_ns > 0:
            tps = output_tokens / (eval_ns / 1_000_000_000)
        logger.info(
            "Ollama generate ok model=%s prompt_tokens=%s output_tokens=%s "
            "latency=%.0fms throughput=%s",
            model, prompt_tokens, output_tokens, latency_ms,
            f"{tps:.1f} tok/s" if tps else "n/a",
        )

    def _extract(self, data: Any, model: str, latency_ms: float, url: str) -> LLMResponse:
        """Map Ollama's response envelope onto ``LLMResponse``.

        Ollama returns ``{"message": {"content": "..."}, "model": ..., ...}``.
        A missing/!dict envelope or absent content is an unusable response.
        """
        if not isinstance(data, dict):
            raise LLMResponseError(f"expected a JSON object from Ollama, got {type(data).__name__}")
        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise LLMResponseError("Ollama response missing 'message.content'")

        metadata = {k: v for k, v in data.items() if k != "message"}
        return LLMResponse(
            text=str(message.get("content") or ""),
            model=str(data.get("model") or model),
            raw=metadata,
            latency_ms=round(latency_ms, 2),
        )

    # --- LLMProvider contract --------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send one chat completion request to Ollama and return the result.

        Retries transient transport/5xx failures with exponential backoff;
        raises on rate limiting (immediately) and on deterministic 4xx.
        """
        client = await self._get_client()
        url = f"{self._base_url}/api/chat"
        payload = self._build_payload(prompt, system, schema, model)
        resolved_model = payload["model"]

        last_error: LLMProviderError | None = None

        for attempt in range(self._max_retries):
            start = time.perf_counter()
            try:
                response = await client.post(url, json=payload)
            except httpx.ReadTimeout as exc:
                # The server accepted the request and the model was still
                # generating when we gave up. Retrying re-runs the full
                # generation from scratch, multiplying wall-clock time and CPU
                # by max_retries without changing the outcome — so fail fast
                # and let the caller raise the timeout or bound the output.
                raise LLMProviderError(
                    f"generation exceeded the {self._timeout}s timeout "
                    f"(model still generating; not retried): {exc}",
                    provider=self.PROVIDER_NAME,
                    model=resolved_model,
                    url=url,
                ) from exc
            except httpx.TimeoutException as exc:
                # Connect/pool timeout — the request never reached the model,
                # so retrying is cheap and worthwhile.
                last_error = LLMProviderError(
                    f"request timed out after {self._timeout}s: {exc}",
                    provider=self.PROVIDER_NAME, model=resolved_model, url=url,
                )
            except httpx.TransportError as exc:
                last_error = LLMProviderError(
                    f"transport error: {exc}",
                    provider=self.PROVIDER_NAME, model=resolved_model, url=url,
                )
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                status = response.status_code

                if status == 429:
                    raise LLMRateLimitError(
                        "Ollama rate limit exceeded",
                        retry_after=self._parse_retry_after(response),
                        provider=self.PROVIDER_NAME, model=resolved_model, url=url,
                    )
                if status in _RETRYABLE_STATUS:
                    last_error = LLMProviderError(
                        f"upstream server error (HTTP {status})",
                        provider=self.PROVIDER_NAME, model=resolved_model,
                        status_code=status, url=url,
                    )
                elif status >= 400:
                    raise LLMProviderError(
                        f"request rejected with HTTP {status}",
                        provider=self.PROVIDER_NAME, model=resolved_model,
                        status_code=status, url=url,
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise LLMResponseError(
                            f"Ollama response was not valid JSON: {exc}"
                        ) from exc
                    result = self._extract(data, resolved_model, latency_ms, url)
                    self._log_usage(result, resolved_model, latency_ms)
                    return result

            if attempt < self._max_retries - 1:
                delay = self._retry_backoff * (2**attempt)
                logger.warning(
                    "Ollama generate failed (attempt %d/%d): %s — retrying in %.2fs",
                    attempt + 1, self._max_retries, last_error, delay,
                )
                await asyncio.sleep(delay)

        assert last_error is not None  # unreachable: loop runs >= 1 time
        raise last_error
