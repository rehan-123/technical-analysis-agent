from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import Settings
from engines.market_structure import MarketStructureResult
from models.analysis_result import SupportResistanceLevels


@dataclass
class RiskResult:
    direction: str  # long / short / none
    entry_zone: tuple[float, float]
    stop_loss: float
    atr_stop: float
    structure_stop: float | None
    targets: list[float]  # TP1, TP2, TP3
    risk_reward: list[float]  # R:R at each target
    position_size: float  # units, given account + risk-per-trade config
    risk_amount: float  # currency at risk
    expected_value_r: float  # EV in R multiples (illustrative)
    invalidation: float
    risk_tier: str  # Low / Medium / High
    notes: list[str] = field(default_factory=list)


class RiskEngine:
    """Turns a directional bias into a concrete, risk-defined trade plan.

    Stops are the tighter/looser of an ATR-based stop and a structure-based
    stop (last swing). Targets are placed at configurable R-multiples of the
    entry-to-stop distance, then reconciled against structural levels.
    Position size follows fixed-fractional risk: risk no more than
    ``risk_per_trade_pct`` of the account on the entry-to-stop distance.

    The expected-value figure is illustrative only — it uses a configurable
    assumed win rate, NOT a measured edge, and must not be read as a
    profit forecast.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        df: pd.DataFrame,
        bias: str,  # bullish / bearish / neutral
        levels: SupportResistanceLevels,
        structure: MarketStructureResult,
        atr_pct: float,
    ) -> RiskResult:
        s = self.settings
        latest = df.iloc[-1]
        close = float(latest["close"])
        atr = float(latest[f"atr_{s.atr_period}"])

        direction = "long" if bias == "bullish" else "short" if bias == "bearish" else "none"
        long = direction == "long"

        buf = s.entry_zone_buffer_pct
        entry_zone = (round(close * (1 - buf), 2), round(close * (1 + buf), 2))

        # --- Stops ---
        atr_stop = close - s.atr_stop_multiplier * atr if long else close + s.atr_stop_multiplier * atr
        structure_stop: float | None = None
        if long and structure.swing_lows:
            structure_stop = min(structure.swing_lows) * 0.997
        elif not long and structure.swing_highs:
            structure_stop = max(structure.swing_highs) * 1.003

        if direction == "none":
            stop = round(atr_stop, 2)
        elif long:
            candidates = [atr_stop] + ([structure_stop] if structure_stop and structure_stop < close else [])
            stop = round(min(candidates), 2)  # widest protective stop below price
        else:
            candidates = [atr_stop] + ([structure_stop] if structure_stop and structure_stop > close else [])
            stop = round(max(candidates), 2)

        risk_per_unit = abs(close - stop)
        risk_per_unit = max(risk_per_unit, close * 0.001)  # floor to avoid div-by-zero

        # --- Targets at configured R-multiples ---
        targets: list[float] = []
        rr: list[float] = []
        for rmult in s.risk_reward_targets:
            tp = close + rmult * risk_per_unit if long else close - rmult * risk_per_unit
            targets.append(round(tp, 2))
            rr.append(round(rmult, 2))

        # --- Position sizing (fixed fractional) ---
        risk_amount = s.risk_account_size * s.risk_per_trade_pct
        position_size = round(risk_amount / risk_per_unit, 4)

        # --- Illustrative EV in R (assumed win rate, capped reward at TP1 R) ---
        p = s.risk_win_rate_assumption
        reward_r = s.risk_reward_targets[0]
        ev_r = round(p * reward_r - (1 - p) * 1.0, 3)

        # --- Risk tier from ATR% ---
        if atr_pct <= s.risk_low_atr_pct:
            tier = "Low"
        elif atr_pct <= s.risk_medium_atr_pct:
            tier = "Medium"
        else:
            tier = "High"

        invalidation = structure_stop if structure_stop is not None else stop

        notes: list[str] = []
        if direction == "none":
            notes.append("No directional bias — trade plan is indicative only")
        notes.append(f"Sizing assumes {s.risk_per_trade_pct:.1%} account risk on {s.risk_account_size:,.0f}")

        return RiskResult(
            direction=direction,
            entry_zone=entry_zone,
            stop_loss=stop,
            atr_stop=round(atr_stop, 2),
            structure_stop=round(structure_stop, 2) if structure_stop is not None else None,
            targets=targets,
            risk_reward=rr,
            position_size=position_size,
            risk_amount=round(risk_amount, 2),
            expected_value_r=ev_r,
            invalidation=round(float(invalidation), 2),
            risk_tier=tier,
            notes=notes,
        )
