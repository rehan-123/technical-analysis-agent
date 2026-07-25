from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import Settings
from engines.confluence import ConfluenceResult
from engines.volatility import VolatilityResult


@dataclass
class ConfidenceResult:
    """Confidence is a deterministic, auditable blend — NOT a probability of
    profit and not statistically validated. It answers "how internally
    consistent and one-sided is the current evidence?", nothing more.
    """

    confidence: int  # 0-100
    directional_agreement: float
    dominant_side: str
    components: dict[str, float] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)


class ConfidenceEngine:
    """Derives confidence from three transparent components:

    1. **Separation** — how far apart the bullish and bearish confluence
       scores are (a decisive read scores high; a coin-flip scores low).
    2. **Dominant magnitude** — the absolute strength of the winning side
       (weak-but-one-sided evidence shouldn't read as high confidence).
    3. **Volatility penalty** — high-volatility / exhaustion regimes reduce
       confidence, since any single read is less reliable there.

    Each component is bounded and combined with fixed, documented weights,
    so the same inputs always yield the same confidence.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self, confluence: ConfluenceResult, volatility: VolatilityResult
    ) -> ConfidenceResult:
        bull, bear = confluence.bullish_score, confluence.bearish_score
        dominant_val = max(bull, bear)
        dominant_side = "bullish" if bull >= bear else "bearish"

        separation = abs(bull - bear)  # 0-100
        agreement = separation / max(dominant_val, 1e-9)  # 0-1, how one-sided

        # Component scores (each 0-100).
        c_separation = min(100.0, separation * 1.5)
        c_magnitude = min(100.0, dominant_val * 1.2)
        c_agreement = min(100.0, agreement * 100)

        base = 0.4 * c_separation + 0.35 * c_magnitude + 0.25 * c_agreement

        caveats: list[str] = []
        penalty = 0.0
        if volatility.regime == "high":
            penalty += 10
            caveats.append("Elevated volatility reduces reliability")
        if volatility.trend_exhaustion:
            penalty += 10
            caveats.append("Possible trend exhaustion")
        if confluence.net_bias == "neutral":
            penalty += 15
            caveats.append("Signals are mixed / non-directional")

        confidence = int(max(0, min(100, round(base - penalty))))

        return ConfidenceResult(
            confidence=confidence,
            directional_agreement=round(agreement, 3),
            dominant_side=dominant_side,
            components={
                "separation": round(c_separation, 1),
                "magnitude": round(c_magnitude, 1),
                "agreement": round(c_agreement, 1),
                "volatility_penalty": round(penalty, 1),
            },
            caveats=caveats,
        )
