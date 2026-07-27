from __future__ import annotations

from typing import Callable

import pandas as pd


class ComputeCache:
    """A tiny per-request memoization cache for shared intermediate series.

    Many indicators need the same building blocks (EMA-N, ATR-N, typical
    price, true range). Recomputing them independently is wasteful once the
    indicator count grows. Engines share a single ``ComputeCache`` bound to
    one OHLCV frame for the duration of one analysis, so each base series is
    computed at most once.

    The cache is intentionally *request-scoped* (constructed fresh per
    analyze() call) — never a process-global — to avoid leaking one
    ticker's series into another's computation.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._store: dict[str, pd.Series | pd.DataFrame] = {}

    def get(self, key: str, producer: Callable[[], pd.Series | pd.DataFrame]) -> pd.Series | pd.DataFrame:
        if key not in self._store:
            self._store[key] = producer()
        return self._store[key]

    # --- Common building blocks -------------------------------------------------

    def ema(self, period: int) -> pd.Series:
        return self.get(
            f"ema_{period}",
            lambda: self.df["close"].ewm(span=period, adjust=False, min_periods=period).mean(),
        )

    def true_range(self) -> pd.Series:
        def _tr() -> pd.Series:
            prev_close = self.df["close"].shift(1)
            return pd.concat(
                [
                    self.df["high"] - self.df["low"],
                    (self.df["high"] - prev_close).abs(),
                    (self.df["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

        return self.get("true_range", _tr)

    def atr(self, period: int) -> pd.Series:
        return self.get(
            f"atr_{period}",
            lambda: self.true_range().ewm(alpha=1 / period, min_periods=period, adjust=False).mean(),
        )

    def typical_price(self) -> pd.Series:
        return self.get(
            "typical_price",
            lambda: (self.df["high"] + self.df["low"] + self.df["close"]) / 3,
        )
