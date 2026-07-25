from __future__ import annotations

from config.settings import Settings
from engines.candlestick import CandlestickResult
from engines.confidence import ConfidenceResult
from engines.confluence import ConfluenceResult
from engines.market_structure import MarketStructureResult
from engines.risk import RiskResult
from engines.smc import SMCResult
from engines.volatility import VolatilityResult
from engines.volume import VolumeResult


class ExplanationEngine:
    """Produces the human-readable reasoning that must accompany every call.

    The agent never returns a bare BUY/SELL; this engine turns the numeric
    evidence into an ordered narrative (structure → momentum/volume →
    volatility → risk) plus an explicit invalidation level, so a human or a
    higher-level agent can audit *why* a bias was reached.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        ticker: str,
        confluence: ConfluenceResult,
        confidence: ConfidenceResult,
        structure: MarketStructureResult,
        candles: CandlestickResult,
        volume: VolumeResult,
        volatility: VolatilityResult,
        smc: SMCResult,
        risk: RiskResult,
    ) -> tuple[str, list[str]]:
        bias = confluence.net_bias
        reasons: list[str] = []

        # Structure
        if structure.structure != "undetermined":
            struct_line = f"Market structure is {structure.structure}"
            if structure.break_of_structure:
                struct_line += " with a confirmed Break of Structure"
            if structure.change_of_character:
                struct_line += " showing a Change of Character (early reversal cue)"
            reasons.append(struct_line)

        # Top contributing categories
        ranked = sorted(
            confluence.breakdown,
            key=lambda b: max(b.bull_contribution, b.bear_contribution),
            reverse=True,
        )
        for b in ranked[:4]:
            side = "bullish" if b.bull_contribution >= b.bear_contribution else "bearish"
            if b.detail:
                reasons.append(f"{b.category.capitalize()} is net {side}: " + "; ".join(b.detail[:2]))

        # Candles
        if candles.patterns:
            reasons.append(f"Candlesticks: {', '.join(candles.patterns[:3])}")

        # Volume
        if volume.buying_pressure:
            reasons.append("Volume confirms buying pressure")
        elif volume.selling_pressure:
            reasons.append("Volume confirms selling pressure")
        if volume.profile:
            reasons.append(f"Point of Control at {volume.profile.point_of_control}")

        # Volatility
        if volatility.bollinger_squeeze:
            reasons.append(f"Volatility squeeze active (breakout probability ~{volatility.breakout_probability}%)")
        if volatility.trend_exhaustion:
            reasons.append("Signs of trend exhaustion at elevated volatility")

        # SMC (flagged heuristic)
        if smc.signal != "neutral":
            reasons.append(f"Smart-money read (heuristic): {smc.market_zone} zone, {smc.signal} lean")

        for c in confidence.caveats:
            reasons.append(f"Caveat: {c}")

        # Assemble summary
        if bias == "neutral":
            headline = f"{ticker}: mixed technical picture with no decisive edge"
        else:
            headline = (
                f"{ticker}: technical structure leans {bias} "
                f"(bull {confluence.bullish_score} / bear {confluence.bearish_score}), "
                f"confidence {confidence.confidence}%"
            )

        plan = (
            f" Plan: {risk.direction} bias, entry ~{risk.entry_zone[0]}–{risk.entry_zone[1]}, "
            f"stop {risk.stop_loss}, targets {risk.targets}, invalidation below {risk.invalidation}."
            if risk.direction != "none"
            else " No directional trade recommended; monitor for confirmation."
        )

        summary = headline + "." + plan
        return summary, reasons
