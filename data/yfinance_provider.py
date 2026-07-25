from __future__ import annotations

import asyncio
import time

import pandas as pd

from data.base import MarketDataProvider
from utils.exceptions import DataFetchError
from utils.logger import get_logger

logger = get_logger(__name__)


def classify_network_error(exc: Exception) -> str:
    """Turn a raw transport error into an actionable, human-readable cause.

    The generic ``curl: (35) Recv failure: Connection was reset`` tells an
    operator almost nothing. This maps common signatures to the thing they
    actually need to check (proxy, TLS interception, DNS, rate limiting).
    """
    text = str(exc).lower()
    if "recv failure" in text or "connection reset" in text or "reset by peer" in text:
        return (
            "connection reset during TLS/receive — most likely an egress "
            "proxy/firewall or CDN rejecting the curl_cffi browser-impersonation "
            "TLS fingerprint (not Yahoo being down)"
        )
    if "ssl" in text or "certificate" in text or "(35)" in text:
        return (
            "TLS handshake failure — a TLS-inspecting proxy with an untrusted "
            "CA, or an unsupported impersonation target"
        )
    if "could not resolve" in text or "name or service not known" in text or "getaddrinfo" in text:
        return "DNS resolution failed — egress DNS blocked or no outbound network"
    if "timed out" in text or "timeout" in text:
        return "request timed out — slow or blocked egress path"
    if "429" in text or "too many requests" in text or "rate" in text:
        return "rate limited by the upstream (HTTP 429)"
    if "proxy" in text:
        return "proxy error — check HTTP_PROXY/HTTPS_PROXY configuration"
    return "unclassified transport error"


class YFinanceProvider(MarketDataProvider):
    """Fetches historical OHLCV data from Yahoo Finance via ``yfinance``.

    Hardened for production:

    - **Retries with exponential backoff** for transient resets/timeouts.
    - **Configurable curl_cffi impersonation session.** Modern ``yfinance``
      uses ``curl_cffi`` to present a browser-like TLS fingerprint. When that
      fingerprint is what an inspecting proxy/CDN resets, being able to pin or
      change the impersonation target (or supply an explicit session) is the
      difference between working and a blanket connection reset. Built
      defensively so it degrades to plain ``yfinance`` if ``curl_cffi`` or the
      ``session=`` kwarg isn't available in the installed version.
    - **Proxy-aware.** ``curl_cffi``/``requests`` honour ``HTTP(S)_PROXY``
      environment variables automatically; nothing here overrides them.
    - **Actionable errors.** Failures are classified (proxy / TLS / DNS /
      timeout / rate-limit) so the resulting 502 says *why*.

    ``yfinance`` is blocking, so the call runs in a worker thread to keep the
    event loop responsive.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        max_retries: int = 3,
        retry_backoff: float = 0.75,
        impersonate: str = "chrome",
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.impersonate = impersonate

    def _build_session(self):
        """Return a curl_cffi impersonation session, or None if unavailable.

        Isolated and defensive: any import/version issue simply yields None,
        and the fetch falls back to yfinance's own default session.
        """
        if not self.impersonate:
            return None
        try:
            from curl_cffi import requests as cffi_requests  # type: ignore

            return cffi_requests.Session(impersonate=self.impersonate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("curl_cffi session unavailable (%s); using yfinance default", exc)
            return None

    def _fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        session = self._build_session()
        # Try with an explicit session first (works on versions that accept
        # it); fall back to the no-session path on any TypeError/attribute
        # mismatch so we're robust across yfinance versions.
        for use_session in ([True, False] if session is not None else [False]):
            try:
                if use_session:
                    tk = yf.Ticker(ticker, session=session)
                else:
                    tk = yf.Ticker(ticker)
                return tk.history(
                    period=period, interval=interval, auto_adjust=True, timeout=self.timeout
                )
            except TypeError:
                # e.g. this yfinance version doesn't accept session=/timeout=
                continue
        # Last resort: the simplest possible call.
        return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    async def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                df = await asyncio.to_thread(self._fetch, ticker, period, interval)
                if df is not None and not df.empty:
                    return df
                last_exc = DataFetchError(f"No data returned for ticker '{ticker}'")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                cause = classify_network_error(exc)
                logger.warning(
                    "yfinance fetch for %s failed (attempt %d/%d): %s [%s]",
                    ticker, attempt + 1, self.max_retries, exc, cause,
                )
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_backoff * (2**attempt))

        cause = classify_network_error(last_exc) if last_exc else "unknown"
        raise DataFetchError(
            f"yfinance could not fetch '{ticker}' after {self.max_retries} attempts: "
            f"{last_exc} ({cause})"
        ) from last_exc
