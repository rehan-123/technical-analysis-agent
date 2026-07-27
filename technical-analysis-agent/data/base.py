from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """Abstract market data source.

    Concrete providers (yfinance, a broker API, a synthetic generator for
    tests/demos) all implement this same contract, so the rest of the
    agent never depends on where the data actually comes from. Swapping
    data sources later (e.g. to a paid real-time feed) means writing one
    new class here — nothing else in the codebase changes.
    """

    @abstractmethod
    async def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Return a raw OHLCV DataFrame indexed by timestamp with columns
        open, high, low, close, volume (case-insensitive)."""
        raise NotImplementedError
