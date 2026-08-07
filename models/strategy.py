from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrategyName(str, Enum):
    """Closed vocabulary of reusable strategies implemented by the Strategy
    Engine. A real ``str, Enum`` (not a ``Literal``) so it is usable as a
    registry key, appears cleanly in the OpenAPI schema, and still
    serializes as a plain JSON string on the wire.
    """

    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    SWING = "swing"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"


class StrategyDirection(str, Enum):
    """The directional call a strategy makes for one ticker."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class StrategySignal(BaseModel):
    """Structured output contract every ``Strategy`` implementation returns.

    Deliberately parallels ``TechnicalAnalysisResult`` / ``NewsAnalysisResult``
    (a ticker, a timestamp, a payload) so the Market Scanner and Ranking Engine
    can consume every strategy uniformly regardless of which one produced it —
    the property that makes strategies interchangeable.

    A strategy is a pure function of an already-computed
    ``TechnicalAnalysisResult``: it performs no data fetching and no indicator
    computation of its own, so identical input always yields an identical
    signal (deterministic outputs, per the Strategy Engine's contract).
    """

    model_config = ConfigDict(frozen=True)

    strategy: StrategyName
    ticker: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    applicable: bool = Field(
        ..., description="Whether this strategy's setup conditions are met for this ticker right now"
    )
    direction: StrategyDirection = StrategyDirection.NONE
    confidence: int = Field(..., ge=0, le=100, description="0-100 conviction in the call")
    score: int = Field(
        ..., ge=0, le=100, description="0-100 graded setup strength; smoothly informs ranking even when not applicable"
    )

    signals: list[str] = Field(default_factory=list, description="Short, specific evidence strings")
    reasoning: list[str] = Field(default_factory=list, description="Human-readable explanation of the call")

    @property
    def is_actionable(self) -> bool:
        """A signal worth surfacing: applicable and directional."""
        return self.applicable and self.direction is not StrategyDirection.NONE
