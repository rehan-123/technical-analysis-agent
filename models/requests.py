from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TechnicalAnalysisRequest(BaseModel):
    """Canonical input contract for the Technical Analysis Agent.

    This is the same shape whether the request arrives over HTTP (FastAPI)
    or in-process from a Chief Decision Agent.
    """

    ticker: str = Field(..., description="Stock or crypto ticker, e.g. AAPL, BTC-USD")
    period: str = Field(default="1y", description="Historical lookback window, e.g. 6mo, 1y, 2y")
    interval: str = Field(default="1d", description="Bar interval, e.g. 1d, 1h, 15m")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker must not be empty")
        return v
