from __future__ import annotations

import pandas as pd

from data.base import MarketDataProvider
from utils.exceptions import DataFetchError
from utils.logger import get_logger

logger = get_logger(__name__)


class FallbackDataProvider(MarketDataProvider):
    """Tries an ordered list of real data sources, returning the first that
    succeeds.

    Design decisions:

    - **Real sources only.** The synthetic generator is never part of this
      chain: silently serving fabricated prices into an analysis that emits
      entry/stop/target levels would be actively dangerous. If every real
      source fails, this raises — it does not invent data.
    - **Diagnosable failure.** When all sources fail, the raised
      ``DataFetchError`` includes each source's name and its classified
      reason, so the resulting 502 explains *why* every path failed instead
      of collapsing to one opaque message.
    - **Same contract.** It is itself a ``MarketDataProvider``, so the agent
      and API are unaware there is a chain at all.
    """

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("FallbackDataProvider requires at least one provider")
        self.providers = providers

    async def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        failures: list[str] = []
        for provider in self.providers:
            name = type(provider).__name__
            try:
                df = await provider.get_ohlcv(ticker, period, interval)
                if df is not None and not df.empty:
                    if failures:  # we recovered via a fallback — worth noting
                        logger.info("%s served '%s' after %d source(s) failed", name, ticker, len(failures))
                    return df
                failures.append(f"{name}: empty result")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data source %s failed for '%s': %s", name, ticker, exc)
                failures.append(f"{name}: {exc}")

        raise DataFetchError(
            f"All data sources failed for '{ticker}'. " + " | ".join(failures)
        )
