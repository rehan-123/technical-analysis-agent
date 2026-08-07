from __future__ import annotations

from typing import Mapping

from config.settings import Settings, get_settings
from models.analysis_result import TechnicalAnalysisResult
from models.strategy import StrategyName, StrategySignal
from strategy.base import Strategy
from strategy.registry import build_strategy_registry, get_strategy


class UnknownStrategyError(KeyError):
    """Raised when a ``StrategyName`` has no registered implementation."""


class StrategyEngine:
    """Evaluates the reusable strategy roster against one ticker's already
    computed ``TechnicalAnalysisResult``.

    This is the "Strategy Selection" stage of the Market Scanner pipeline. It
    performs no data fetching and no indicator computation itself — every
    strategy it holds is a pure function of the ``TechnicalAnalysisResult``
    already produced by the Technical Agent. Strategies are interchangeable
    (they share one contract, ``StrategySignal``), so this engine can add or
    drop a strategy from ``registry.py`` without any caller here changing.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: Mapping[StrategyName, Strategy] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._registry = registry if registry is not None else build_strategy_registry(self._settings)

    @property
    def registry(self) -> Mapping[StrategyName, Strategy]:
        return self._registry

    def evaluate_one(self, technical: TechnicalAnalysisResult, name: StrategyName) -> StrategySignal:
        """Evaluate a single named strategy against ``technical``.

        Raises:
            UnknownStrategyError: if ``name`` has no registered implementation.
        """
        try:
            strategy = get_strategy(name, registry=self._registry)
        except KeyError as exc:
            raise UnknownStrategyError(str(exc)) from exc
        return strategy.evaluate(technical)

    def evaluate_all(self, technical: TechnicalAnalysisResult) -> list[StrategySignal]:
        """Evaluate every registered strategy against ``technical``.

        Returned in the registry's deterministic (sorted-name) order, so
        repeated calls against identical input yield an identical list —
        consistent with each individual strategy's deterministic-output
        contract.
        """
        return [
            strategy.evaluate(technical)
            for strategy in (self._registry[name] for name in sorted(self._registry, key=lambda n: n.value))
        ]

    def best(self, technical: TechnicalAnalysisResult) -> StrategySignal | None:
        """The single best applicable, directional signal for ``technical``,
        ranked by ``score`` (ties broken by ``confidence``, then by strategy
        name for full determinism). Returns ``None`` when no strategy is
        applicable — a symbol with no qualifying setup is a valid outcome,
        not an error.
        """
        actionable = [s for s in self.evaluate_all(technical) if s.is_actionable]
        if not actionable:
            return None
        return max(actionable, key=lambda s: (s.score, s.confidence, s.strategy.value))
