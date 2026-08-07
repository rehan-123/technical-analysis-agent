from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Signal = Literal["bullish", "bearish", "neutral"]


class IndicatorResult(BaseModel):
    """Uniform, self-describing output for a single indicator at the latest bar.

    Every indicator exposes the same shape so the confluence and confidence
    engines can consume them polymorphically without special-casing:

    - ``value``: the primary scalar (or a small dict of scalars for
      multi-line indicators like MACD/Ichimoku).
    - ``signal``: directional read (bullish/bearish/neutral).
    - ``strength``: 0–100 conviction of *this* indicator's signal.
    - ``interpretation``: one human-readable sentence.
    - ``confidence_contribution``: signed weight this indicator adds to the
      aggregate confidence blend, already scaled by its configured weight.
    """

    name: str
    value: float | dict[str, float | None] | None
    signal: Signal = "neutral"
    strength: int = Field(default=0, ge=0, le=100)
    interpretation: str = ""
    confidence_contribution: float = 0.0
