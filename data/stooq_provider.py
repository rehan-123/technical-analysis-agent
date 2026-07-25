from __future__ import annotations

import asyncio
import csv
import io
import urllib.error
import urllib.request
from datetime import date, timedelta

import pandas as pd

from data.base import MarketDataProvider
from utils.exceptions import DataFetchError
from utils.logger import get_logger

logger = get_logger(__name__)

# period -> approximate calendar lookback in days
_PERIOD_DAYS = {
    "1mo": 31, "3mo": 93, "6mo": 186, "1y": 372, "2y": 744, "5y": 1830, "10y": 3660, "max": 20000,
}
# interval -> Stooq daily/weekly/monthly code (Stooq's free CSV is not intraday)
_INTERVAL_CODE = {"1d": "d", "1wk": "w", "1mo": "m"}


class StooqProvider(MarketDataProvider):
    """Fallback OHLCV source using Stooq's free CSV endpoint.

    Why this exists and why it's the *fallback of choice*: it fetches over
    the Python standard library (``urllib``) using the system's normal
    OpenSSL TLS stack — a completely ordinary TLS fingerprint, the opposite
    of ``curl_cffi``'s browser impersonation. So when the primary failure is
    an egress proxy/CDN resetting the impersonated ``curl_cffi`` handshake
    (the classic ``curl (35) Recv failure: Connection was reset``), this path
    is unaffected. It also:

    - adds **no new dependencies** (stdlib only),
    - is **proxy-aware** (``urllib`` honours ``HTTP(S)_PROXY`` automatically),
    - covers **daily/weekly/monthly** history for equities and ETFs.

    Limitations (intentional, documented): no intraday intervals, and symbol
    coverage/format differs from Yahoo (US tickers are mapped to Stooq's
    ``<symbol>.us`` convention; unknown mappings surface a clear error so the
    provider chain can move on).
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    @staticmethod
    def _to_stooq_symbol(ticker: str) -> str:
        """Map a Yahoo-style ticker to Stooq's symbol convention.

        Handled: plain US equities/ETFs (``AAPL`` -> ``aapl.us``), symbols
        that already carry a market suffix (``…​.US``, ``…​.DE``) are lowercased
        as-is, and crypto pairs like ``BTC-USD`` -> ``btcusd``.
        """
        t = ticker.strip().lower()
        if "-" in t:  # e.g. BTC-USD -> btcusd (Stooq crypto convention)
            return t.replace("-", "")
        if "." in t:  # already has a market suffix
            return t
        return f"{t}.us"

    def _build_url(self, ticker: str, period: str, interval: str) -> str:
        code = _INTERVAL_CODE.get(interval)
        if code is None:
            raise DataFetchError(
                f"Stooq fallback supports only daily/weekly/monthly intervals, not '{interval}'"
            )
        symbol = self._to_stooq_symbol(ticker)
        days = _PERIOD_DAYS.get(period, 372)
        d2 = date.today()
        d1 = d2 - timedelta(days=days)
        return (
            f"{self.BASE_URL}?s={symbol}&i={code}"
            f"&d1={d1:%Y%m%d}&d2={d2:%Y%m%d}"
        )

    def _fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        url = self._build_url(ticker, period, interval)
        # urllib honours HTTP(S)_PROXY / NO_PROXY via the default opener.
        req = urllib.request.Request(url, headers={"User-Agent": "technical-analysis-agent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise DataFetchError(f"Stooq HTTP {exc.code} for '{ticker}'") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DataFetchError(f"Stooq request failed for '{ticker}': {exc.reason if hasattr(exc, 'reason') else exc}") from exc

        return self._parse_csv(raw, ticker)

    @staticmethod
    def _parse_csv(raw: str, ticker: str) -> pd.DataFrame:
        # Stooq returns the literal string "No data" (or an error line) for
        # unknown symbols / empty ranges rather than an HTTP error.
        stripped = raw.strip()
        if not stripped or stripped.lower().startswith("no data") or "Date,Open" not in raw:
            raise DataFetchError(f"Stooq returned no usable data for '{ticker}'")

        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        if not rows:
            raise DataFetchError(f"Stooq returned an empty series for '{ticker}'")

        records = []
        index = []
        for r in rows:
            try:
                index.append(pd.Timestamp(r["Date"]))
                records.append(
                    {
                        "open": float(r["Open"]),
                        "high": float(r["High"]),
                        "low": float(r["Low"]),
                        "close": float(r["Close"]),
                        "volume": float(r.get("Volume") or 0.0),
                    }
                )
            except (KeyError, ValueError):
                continue  # skip malformed rows (e.g. trailing blanks)

        if not records:
            raise DataFetchError(f"Stooq data for '{ticker}' had no parseable rows")

        df = pd.DataFrame.from_records(records, index=pd.DatetimeIndex(index))
        df.index.name = "Date"
        return df.sort_index()

    async def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        df = await asyncio.to_thread(self._fetch, ticker, period, interval)
        if df is None or df.empty:
            raise DataFetchError(f"No data returned by Stooq for ticker '{ticker}'")
        return df
