from __future__ import annotations

import pandas as pd

from config.settings import Settings
from models.analysis_result import SupportResistanceLevels


class SupportResistanceService:
    """Detects swing-based support and resistance levels.

    A "swing high" is a local maximum over a symmetric window (a bar
    whose high is the highest in its neighborhood); a "swing low" is the
    mirror image. Nearby swing points are then clustered together (within
    a configurable tolerance) so market noise doesn't produce redundant,
    near-duplicate levels.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _find_swing_points(series: pd.Series, order: int, is_high: bool) -> list[float]:
        points: list[float] = []
        values = series.values
        n = len(values)
        for i in range(order, n - order):
            window = values[i - order : i + order + 1]
            center = values[i]
            if is_high and center == window.max():
                points.append(float(center))
            elif not is_high and center == window.min():
                points.append(float(center))
        return points

    @staticmethod
    def _cluster(levels: list[float], tolerance_pct: float) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: list[list[float]] = [[levels[0]]]
        for level in levels[1:]:
            if abs(level - clusters[-1][-1]) / clusters[-1][-1] <= tolerance_pct:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        return [sum(c) / len(c) for c in clusters]

    def evaluate(self, df: pd.DataFrame) -> SupportResistanceLevels:
        s = self.settings
        recent = df.tail(s.sr_lookback_bars)
        current_price = df["close"].iloc[-1]

        swing_highs = self._find_swing_points(recent["high"], s.sr_swing_order, is_high=True)
        swing_lows = self._find_swing_points(recent["low"], s.sr_swing_order, is_high=False)

        resistance = self._cluster(
            [lvl for lvl in swing_highs if lvl > current_price], s.sr_cluster_tolerance_pct
        )
        support = self._cluster(
            [lvl for lvl in swing_lows if lvl < current_price], s.sr_cluster_tolerance_pct
        )

        # Nearest levels to current price first.
        resistance = sorted(resistance)[: s.sr_max_levels]
        support = sorted(support, reverse=True)[: s.sr_max_levels]

        return SupportResistanceLevels(
            support=[round(v, 2) for v in support],
            resistance=[round(v, 2) for v in resistance],
        )
