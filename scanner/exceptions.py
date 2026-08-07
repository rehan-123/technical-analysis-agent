from __future__ import annotations


class ScannerError(Exception):
    """Base class for every error raised within the Market Scanner domain.

    Mirrors ``TechnicalAgentError`` / ``NewsAgentError`` / ``PortfolioError``:
    a per-domain base so callers can catch any Scanner-specific failure
    without a shared cross-domain exception module.
    """


class WatchlistNotFoundError(ScannerError):
    """Raised when a named watchlist does not exist."""


class NoSymbolsProvidedError(ScannerError):
    """Raised when a scan request resolves to an empty symbol set (no
    ``symbols``, no resolvable ``watchlist``, and no ``default`` watchlist)."""


class ScanTooLargeError(ScannerError):
    """Raised when a scan request's symbol count exceeds
    ``scanner_max_symbols_per_scan``."""
