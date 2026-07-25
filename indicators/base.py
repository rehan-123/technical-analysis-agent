from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    """Base contract every indicator must implement.

    Each indicator is a small, independent, stateless unit: it takes a
    validated OHLCV DataFrame and returns a Series (single value per bar)
    or DataFrame (multiple related values per bar, e.g. MACD) aligned to
    the same index. Indicators never mutate the input and never know
    about each other — composition happens one layer up, in
    ``services.indicator_engine.IndicatorEngine``.
    """

    name: str

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """Compute the indicator over the given OHLCV DataFrame."""
        raise NotImplementedError
