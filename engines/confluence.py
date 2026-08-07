from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import Settings
from engines.candlestick import CandlestickResult
from engines.market_structure import MarketStructureResult
from engines.smc import SMCResult
from engines.volatility import VolatilityResult
from engines.volume import VolumeResult
from models.indicator_result import IndicatorResult

# Maps each indicator name to a weight category so the confluence engine can
# apply the configured category weights without hardcoding per-indicator numbers.
_INDICATOR_CATEGORY: dict[str, str] = {
    "ema_stack": "trend", "wma": "trend", "vwma": "trend", "vwap": "trend",
    "supertrend": "trend", "parabolic_sar": "trend", "ichimoku": "trend",
    "keltner": "trend", "donchian": "trend",
    "rsi": "momentum", "stoch_rsi": "momentum", "macd": "momentum",
    "adx": "momentum", "cci": "momentum", "roc": "momentum", "momentum": "momentum",
    "mfi": "volume", "cmf": "volume", "obv": "volume",
    "bollinger": "volatility",
    "pivots": "structure",
}


@dataclass
class ConfluenceBreakdown:
    category: str
    bull_contribution: float
    bear_contribution: float
    detail: list[str] = field(default_factory=list)


@dataclass
class ConfluenceResult:
    bullish_score: float  # 0-100
    bearish_score: float  # 0-100
    net_bias: str  # bullish / bearish / neutral
    breakdown: list[ConfluenceBreakdown] = field(default_factory=list)


class ConfluenceEngine:
    """Aggregates every engine's directional read into weighted bull/bear
    scores.

    Each category's raw bullish/bearish strength is first accumulated, then
    **normalized within the category** to a net directional lean before the
    category weight is applied. This is deliberate: if it aggregated raw
    strength directly, a category with many redundant same-direction
    indicators (e.g. five short-term momentum indicators all reacting to a
    single recent bounce) could outvote its configured weight and drown out
    the primary trend. Normalizing within category means each category
    contributes at most its configured weight, scaled by how one-sided its
    own evidence is (`lean`) and how much evidence it actually has
    (`conviction`) — not by how many indicators happened to fire.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._weights = {
            "trend": settings.weight_trend,
            "momentum": settings.weight_momentum,
            "structure": settings.weight_structure,
            "volume": settings.weight_volume,
            "volatility": settings.weight_volatility,
            "candlestick": settings.weight_candlestick,
            "smc": settings.weight_smc,
            "pattern": settings.weight_pattern,
        }

    def evaluate(
        self,
        indicators: dict[str, IndicatorResult],
        structure: MarketStructureResult,
        candles: CandlestickResult,
        volume: VolumeResult,
        volatility: VolatilityResult,
        smc: SMCResult,
    ) -> ConfluenceResult:
        raw_bull: dict[str, float] = {c: 0.0 for c in self._weights}
        raw_bear: dict[str, float] = {c: 0.0 for c in self._weights}
        detail: dict[str, list[str]] = {c: [] for c in self._weights}

        def vote(category: str, signal: str, strength: float, note: str) -> None:
            if signal == "bullish":
                raw_bull[category] += strength
                detail[category].append(note)
            elif signal == "bearish":
                raw_bear[category] += strength
                detail[category].append(note)

        # --- Indicator votes ---
        for name, r in indicators.items():
            category = _INDICATOR_CATEGORY.get(name, "momentum")
            vote(category, r.signal, r.strength, f"{name}: {r.interpretation}")

        # --- Structure vote (with BOS/CHoCH amplifying strength) ---
        struct_strength = 70.0
        if structure.break_of_structure:
            struct_strength += 15
        if structure.change_of_character:
            struct_strength += 15
        vote("structure", structure.signal, struct_strength,
             f"Structure: {structure.structure} ({structure.last_label})")
        if structure.break_of_structure:
            detail["structure"].append("Break of Structure confirmed")
        if structure.change_of_character:
            detail["structure"].append("Change of Character detected")

        # --- Candlestick vote ---
        if candles.signal in ("bullish", "bearish"):
            vote("candlestick", candles.signal, candles.strength, f"Candles: {', '.join(candles.patterns)}")

        # --- Volume vote ---
        if volume.buying_pressure:
            vote("volume", "bullish", volume.strength or 60, "Buying pressure (CMF/OBV)")
        elif volume.selling_pressure:
            vote("volume", "bearish", volume.strength or 60, "Selling pressure (CMF/OBV)")

        # --- SMC vote ---
        if smc.signal in ("bullish", "bearish"):
            vote("smc", smc.signal, smc.strength, f"SMC: {smc.market_zone} zone (heuristic)")

        if volatility.trend_exhaustion:
            detail["volatility"].append("Volatility exhaustion caps conviction")

        # --- Normalize within category, then weight ---
        bull: dict[str, float] = {}
        bear: dict[str, float] = {}
        for c in self._weights:
            total = raw_bull[c] + raw_bear[c]
            weight = self._weights[c]
            if total <= 0:
                bull[c] = bear[c] = 0.0
                continue
            lean = (raw_bull[c] - raw_bear[c]) / total  # [-1, 1] net direction
            conviction = min(1.0, total / 100.0)  # weak categories contribute less
            bull[c] = weight * max(lean, 0.0) * conviction
            bear[c] = weight * max(-lean, 0.0) * conviction

        max_weight = sum(self._weights.values())
        bullish_score = round(min(100.0, sum(bull.values()) / max_weight * 100), 1)
        bearish_score = round(min(100.0, sum(bear.values()) / max_weight * 100), 1)

        if bullish_score - bearish_score >= 10:
            net = "bullish"
        elif bearish_score - bullish_score >= 10:
            net = "bearish"
        else:
            net = "neutral"

        breakdown = [
            ConfluenceBreakdown(
                category=c,
                bull_contribution=round(bull[c] / max_weight * 100, 1),
                bear_contribution=round(bear[c] / max_weight * 100, 1),
                detail=detail[c],
            )
            for c in self._weights
            if bull[c] > 0 or bear[c] > 0
        ]

        return ConfluenceResult(bullish_score, bearish_score, net, breakdown)
