from __future__ import annotations

import pytest

from scanner.exchange import infer_exchange


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("AAPL", "US"),
        ("aapl", "US"),
        ("RELIANCE.NS", "NSE"),
        ("TATASTEEL.BO", "BSE"),
        ("VOD.L", "LSE"),
        ("SHOP.TO", "TSX"),
        ("BTC-USD", "CRYPTO"),
        ("ETH-USDT", "CRYPTO"),
    ],
)
def test_infer_exchange_known_conventions(ticker, expected):
    assert infer_exchange(ticker) == expected


def test_unrecognized_suffix_returns_unknown():
    assert infer_exchange("FOO.XX") == "UNKNOWN"
