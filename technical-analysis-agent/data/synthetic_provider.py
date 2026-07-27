from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from data.base import MarketDataProvider

_PERIOD_TO_BARS = {"1mo": 22, "3mo": 65, "6mo": 130, "1y": 252, "2y": 504, "5y": 1260}


def _stable_seed(*parts: str | int) -> int:
    """Deterministic seed derived from the given parts.

    Python's built-in `hash()` is randomized per-process (PYTHONHASHSEED)
    for security reasons, so it must never be used for reproducible RNG
    seeding — using it here would silently make "seeded" synthetic data
    different on every process run. This uses a stable hash instead.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


class SyntheticDataProvider(MarketDataProvider):
    """Generates realistic-looking OHLCV data via a seeded geometric random
    walk. Used for offline development, demos, and unit tests where
    hitting a live market data API is undesirable, rate-limited, or
    (as in a network-restricted sandbox) simply unreachable. Implements
    the same ``MarketDataProvider`` contract as ``YFinanceProvider``, so
    it is a drop-in replacement anywhere a data source is needed.
    """

    def __init__(
        self,
        seed: int = 42,
        start_price: float = 100.0,
        drift: float = 0.0004,
        volatility: float = 0.018,
    ) -> None:
        self.seed = seed
        self.start_price = start_price
        self.drift = drift
        self.volatility = volatility

    async def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        bars = _PERIOD_TO_BARS.get(period, 252)
        rng = np.random.default_rng(_stable_seed(ticker, self.seed))

        t = np.arange(bars)

        # Deterministic trend component (in log-space). This guarantees the
        # series trends in the direction of ``drift`` regardless of the noise
        # realisation.
        #
        # The previous implementation was a pure geometric random walk:
        #   returns = normal(drift, volatility); close = start * cumprod(1+returns)
        # There, the noise is a *random walk* whose spread grows with
        # sqrt(time), while the drift grows linearly with time. For the
        # parameters used by the test fixtures (drift ~1e-3, volatility ~2e-2)
        # the accumulated noise over ~252 bars is comparable to, and often
        # larger than, the accumulated drift — so a nominally "bearish"
        # (negative-drift) series could realise as a net *rising* series purely
        # by chance, depending on the ticker/seed RNG stream. That made
        # direction-dependent fixtures unreliable (the root cause of a bearish
        # fixture being correctly classified bullish by the engine, because the
        # data itself had risen).
        trend = self.drift * t

        # Stationary, mean-reverting AR(1) noise in log-space: realistic
        # autocorrelated swings, pullbacks and consolidations that fluctuate
        # *around* the trend without accumulating away from it, so the trend
        # stays dominant and ``drift`` reliably controls direction. Being
        # stationary, its spread does not grow with time (unlike a random
        # walk), which is exactly what keeps the trend from being swamped.
        phi = 0.85  # autocorrelation — gives momentum-like, realistic wiggles
        eps = rng.normal(0.0, self.volatility, bars)
        noise = np.empty(bars)
        noise[0] = eps[0]
        for i in range(1, bars):
            noise[i] = phi * noise[i - 1] + eps[i]

        close = np.exp(np.log(self.start_price) + trend + noise)

        open_ = np.empty(bars)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        # Intrabar extremes bracket BOTH open and close, guaranteeing valid,
        # coherent OHLC bars (high >= max(open, close), low <= min(open, close)).
        hi_wick = np.abs(rng.normal(0, self.volatility / 2, bars))
        lo_wick = np.abs(rng.normal(0, self.volatility / 2, bars))
        high = np.maximum(open_, close) * (1 + hi_wick)
        low = np.minimum(open_, close) * (1 - lo_wick)

        volume = rng.integers(1_000_000, 8_000_000, bars).astype(float)

        index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=bars, freq="B")

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
