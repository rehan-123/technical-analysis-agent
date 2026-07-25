from __future__ import annotations


class TechnicalAgentError(Exception):
    """Base exception for all Technical Analysis Agent errors."""


class InvalidTickerError(TechnicalAgentError):
    """Raised when a ticker symbol is malformed or unrecognized."""


class DataFetchError(TechnicalAgentError):
    """Raised when historical market data cannot be retrieved."""


class InsufficientDataError(TechnicalAgentError):
    """Raised when there are not enough bars to compute indicators reliably."""


class IndicatorCalculationError(TechnicalAgentError):
    """Raised when an indicator fails to compute on the given data."""
