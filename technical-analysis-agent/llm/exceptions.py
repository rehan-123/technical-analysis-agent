from __future__ import annotations


class LLMError(Exception):
    """Base class for every error raised within the LLM integration layer.

    Catching this catches any LLM-specific failure while letting unrelated
    exceptions propagate. Mirrors the ``NewsAgentError`` base used by the News
    domain, so the two external-capability boundaries stay structurally
    consistent without sharing a module.
    """


class LLMConfigurationError(LLMError):
    """Raised when the LLM layer is misconfigured.

    For example: an unknown provider name, or a cloud backend selected with no
    API key. Raised at construction/wiring time so misconfiguration fails fast
    and loudly (a clean 503 at the API boundary later) rather than surfacing as
    an opaque request failure.
    """


class LLMProviderError(LLMError):
    """Raised when an LLM backend fails to produce a response.

    Wraps transport-level problems (timeouts, connection resets) and
    non-success HTTP responses from the model server. Carries optional context
    for observability: knowing *which* provider, *which* model, and *what* HTTP
    status turns a support ticket into a one-line diagnosis. Context is stored
    as structured attributes (indexable by log processors) and rendered into
    the message (visible in a plain traceback).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        url: str | None = None,
    ) -> None:
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.url = url
        super().__init__(self._render())

    def _render(self) -> str:
        context = {
            "provider": self.provider,
            "model": self.model,
            "status": self.status_code,
            "url": self.url,
        }
        rendered = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
        return f"{self.message} [{rendered}]" if rendered else self.message


class LLMRateLimitError(LLMProviderError):
    """Raised when an LLM backend rejects a request for rate limiting (HTTP 429).

    A distinct subclass because rate limiting is operationally different from a
    generic provider failure: it is expected, transient, and quota-driven
    rather than a fault. Separating it lets callers back off or fail over
    instead of treating it as an outage. Being a subclass of
    ``LLMProviderError`` means existing handlers that catch the parent still
    work unchanged. (Local Ollama rarely rate-limits; this matters chiefly for
    the future cloud backends.)

    ``retry_after`` carries the server's ``Retry-After`` hint in seconds when
    one was supplied, so a caller can honour the server's own guidance.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = 429,
        url: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message, provider=provider, model=model, status_code=status_code, url=url
        )

    def _render(self) -> str:
        base = super()._render()
        return f"{base} (retry_after={self.retry_after}s)" if self.retry_after is not None else base


class LLMResponseError(LLMError):
    """Raised when the backend returns a transport-level success but the body
    is unusable — malformed JSON envelope, or a payload missing the fields this
    layer needs to build an ``LLMResponse``.

    Note this is about the *provider envelope*, not the model's task output:
    validating and repairing the model's answer (the investment-thesis JSON) is
    a later-phase concern for the AI service, not the transport layer.
    """
