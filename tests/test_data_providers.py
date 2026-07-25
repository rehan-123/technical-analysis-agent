from __future__ import annotations

import pandas as pd
import pytest

from data.base import MarketDataProvider
from data.fallback_provider import FallbackDataProvider
from data.stooq_provider import StooqProvider
from data.yfinance_provider import classify_network_error
from utils.exceptions import DataFetchError


# --- Test doubles -------------------------------------------------------------

class _OKProvider(MarketDataProvider):
    def __init__(self, tag="ok"):
        self.tag = tag
        self.called = False

    async def get_ohlcv(self, ticker, period, interval):
        self.called = True
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        return pd.DataFrame(
            {"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3], "volume": [10, 10, 10]},
            index=idx,
        )


class _FailProvider(MarketDataProvider):
    def __init__(self, msg="boom"):
        self.msg = msg
        self.called = False

    async def get_ohlcv(self, ticker, period, interval):
        self.called = True
        raise DataFetchError(self.msg)


class _EmptyProvider(MarketDataProvider):
    async def get_ohlcv(self, ticker, period, interval):
        return pd.DataFrame()


# --- FallbackDataProvider -----------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_returns_first_success_without_calling_later_sources():
    primary = _OKProvider("primary")
    secondary = _OKProvider("secondary")
    chain = FallbackDataProvider([primary, secondary])
    df = await chain.get_ohlcv("AAPL", "1y", "1d")
    assert not df.empty
    assert primary.called is True
    assert secondary.called is False  # short-circuits on first success


@pytest.mark.asyncio
async def test_fallback_recovers_when_primary_fails():
    primary = _FailProvider("connection reset")
    secondary = _OKProvider("secondary")
    chain = FallbackDataProvider([primary, secondary])
    df = await chain.get_ohlcv("AAPL", "1y", "1d")
    assert not df.empty
    assert primary.called and secondary.called


@pytest.mark.asyncio
async def test_fallback_treats_empty_result_as_failure_and_continues():
    chain = FallbackDataProvider([_EmptyProvider(), _OKProvider("secondary")])
    df = await chain.get_ohlcv("AAPL", "1y", "1d")
    assert not df.empty


@pytest.mark.asyncio
async def test_fallback_aggregates_all_reasons_when_everything_fails():
    chain = FallbackDataProvider([_FailProvider("yf reset"), _FailProvider("stooq 404")])
    with pytest.raises(DataFetchError) as ei:
        await chain.get_ohlcv("AAPL", "1y", "1d")
    msg = str(ei.value)
    assert "yf reset" in msg and "stooq 404" in msg  # both causes surfaced


def test_fallback_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        FallbackDataProvider([])


# --- Stooq symbol mapping + CSV parsing (no network) --------------------------

def test_stooq_symbol_mapping():
    assert StooqProvider._to_stooq_symbol("AAPL") == "aapl.us"
    assert StooqProvider._to_stooq_symbol("BTC-USD") == "btcusd"
    assert StooqProvider._to_stooq_symbol("BMW.DE") == "bmw.de"


def test_stooq_parses_valid_csv():
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,10.0,11.0,9.5,10.5,1000\n"
        "2024-01-03,10.5,12.0,10.0,11.8,1500\n"
    )
    df = StooqProvider._parse_csv(csv, "AAPL")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 11.8


def test_stooq_raises_on_no_data_sentinel():
    with pytest.raises(DataFetchError):
        StooqProvider._parse_csv("No data", "ZZZZ")


def test_stooq_rejects_intraday_interval():
    provider = StooqProvider()
    with pytest.raises(DataFetchError):
        provider._build_url("AAPL", "1y", "5m")


# --- Error classification (turns opaque resets into actionable causes) --------

def test_classify_connection_reset():
    cause = classify_network_error(Exception("curl: (35) Recv failure: Connection was reset"))
    assert "curl_cffi" in cause or "reset" in cause


def test_classify_dns():
    assert "DNS" in classify_network_error(Exception("Could not resolve host: query1.finance.yahoo.com"))


def test_classify_rate_limit():
    assert "rate" in classify_network_error(Exception("HTTP Error 429: Too Many Requests")).lower()


def test_classify_timeout():
    assert "timed out" in classify_network_error(Exception("Operation timed out"))
