from __future__ import annotations

import pytest

from config.settings import Settings
from data.synthetic_provider import SyntheticDataProvider


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def synthetic_provider() -> SyntheticDataProvider:
    """A provider that generates a clearly bullish synthetic series."""
    return SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014)


@pytest.fixture
def bearish_provider() -> SyntheticDataProvider:
    """A provider that generates a clearly bearish synthetic series.

    Relies on the trend-stationary generation in ``SyntheticDataProvider``:
    a negative ``drift`` now reliably produces a net-declining series that is
    below its moving-average stack at the measurement point, regardless of
    the ticker/seed RNG stream. (Previously the generator was a pure random
    walk in which volatility could dominate drift, so this fixture could
    realise as a net-*rising* series — which the engine then correctly, but
    confusingly, labelled bullish.)
    """
    return SyntheticDataProvider(seed=13, start_price=150.0, drift=-0.0012, volatility=0.02)
