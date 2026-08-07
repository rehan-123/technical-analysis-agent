from __future__ import annotations

#: Ticker-suffix -> exchange, ordered by nothing in particular (lookup is a
#: dict scan, not order-sensitive, and no two suffixes overlap).
_SUFFIX_EXCHANGES: dict[str, str] = {
    ".NS": "NSE",
    ".BO": "BSE",
    ".L": "LSE",
    ".TO": "TSX",
    ".V": "TSXV",
    ".HK": "HKEX",
    ".SS": "SSE",
    ".SZ": "SZSE",
    ".AX": "ASX",
    ".DE": "XETRA",
    ".PA": "EURONEXT_PARIS",
    ".MI": "BORSA_ITALIANA",
    ".SW": "SIX",
    ".T": "TSE",
    ".KS": "KRX",
}

_CRYPTO_QUOTE_SUFFIXES: tuple[str, ...] = ("-USD", "-USDT", "-BTC", "-ETH")


def infer_exchange(ticker: str) -> str:
    """Best-effort, fully deterministic exchange classification from ticker
    formatting conventions alone — no network lookup and no reference-data
    dependency (the platform ingests neither).

    This is explicitly a heuristic, not authoritative listing data: suffix
    conventions are not universal and a handful of exchanges share notation.
    Anything not recognized returns ``"UNKNOWN"`` rather than a guess, so a
    caller never mistakes a heuristic label for a confirmed venue.
    """
    symbol = ticker.strip().upper()
    for suffix, exchange in _SUFFIX_EXCHANGES.items():
        if symbol.endswith(suffix):
            return exchange
    if symbol.endswith(_CRYPTO_QUOTE_SUFFIXES):
        return "CRYPTO"
    if "." not in symbol and "-" not in symbol:
        return "US"
    return "UNKNOWN"
