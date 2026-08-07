from __future__ import annotations


class NewsAgentError(Exception):
    """Base class for every error raised within the News Agent domain.

    Catching this catches any news-specific failure while letting unrelated
    exceptions propagate. Mirrors the ``TechnicalAgentError`` base used by the
    Technical Analysis Agent, so the two domains stay structurally consistent
    without sharing a module.
    """


class NewsConfigurationError(NewsAgentError):
    """Raised when the News Agent is misconfigured.

    For example: a provider that requires an API key is selected but no key is
    present in settings. Raised at construction/wiring time so misconfiguration
    fails fast and loudly rather than surfacing as an opaque request failure.
    """


class NewsProviderError(NewsAgentError):
    """Raised when a news provider fails to retrieve data.

    Wraps transport-level problems (timeouts, connection resets), non-success
    HTTP responses, and unparseable payloads from an upstream news source. The
    service layer may catch this to fail over to another provider (once a
    fallback chain exists) or to surface a clean error to the caller.

    Carries optional provider context for observability. A bare
    "request failed" message is nearly useless in production logs; knowing
    *which* provider, *which* ticker, and *what* HTTP status turns a support
    ticket into a one-line diagnosis. Context is attached as structured
    attributes (so log processors can index them) and also rendered into the
    message (so it is visible in a plain traceback).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        ticker: str | None = None,
        status_code: int | None = None,
        url: str | None = None,
    ) -> None:
        self.message = message
        self.provider = provider
        self.ticker = ticker
        self.status_code = status_code
        self.url = url
        super().__init__(self._render())

    def _render(self) -> str:
        context = {
            "provider": self.provider,
            "ticker": self.ticker,
            "status": self.status_code,
            "url": self.url,
        }
        rendered = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
        return f"{self.message} [{rendered}]" if rendered else self.message


class NewsRateLimitError(NewsProviderError):
    """Raised when an upstream news API rejects a request for rate limiting
    (HTTP 429).

    A distinct subclass because rate limiting is operationally different from
    a generic provider failure: it is expected, transient, and quota-driven
    rather than a fault. Separating it lets callers react appropriately —
    back off, fail over to another source, or surface a "try again shortly"
    response — instead of treating it as an outage. Being a subclass of
    ``NewsProviderError`` means existing handlers that catch the parent still
    work unchanged.

    ``retry_after`` carries the upstream's ``Retry-After`` hint in seconds
    when the response supplied one, so a caller can honour the server's own
    guidance instead of guessing.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        ticker: str | None = None,
        status_code: int | None = 429,
        url: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message, provider=provider, ticker=ticker, status_code=status_code, url=url
        )

    def _render(self) -> str:
        base = super()._render()
        return f"{base} (retry_after={self.retry_after}s)" if self.retry_after is not None else base


class NewsValidationError(NewsAgentError):
    """Raised when news data is structurally invalid in an unrecoverable way.

    Note: individual malformed *articles* are normally filtered out silently
    (they simply fail ``NewsArticle`` construction and are skipped, per the
    'filter invalid articles' responsibility). This exception is reserved for
    the stronger case where an entire response is malformed enough that no
    meaningful result can be produced.
    """
