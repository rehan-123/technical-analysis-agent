from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from config.settings import Settings, get_settings
from models.strategy import StrategyName
from strategy.base import Strategy
from strategy.breakout import BreakoutStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.momentum import MomentumStrategy
from strategy.swing import SwingStrategy
from strategy.trend_following import TrendFollowingStrategy

# ---------------------------------------------------------------------------
# Immutable strategy registry: maps a ``StrategyName`` to its shared
# implementation instance. Deliberately mirrors
# ``services.prompt_sections.registry`` (see that module's docstring for the
# full rationale): singleton instances because strategies are stateless,
# deterministic transforms; ``MappingProxyType`` because the set of known
# strategies is fixed at import time and auditable by reading this file; and
# kept separate from ``StrategyEngine`` for the same open/closed reason the
# section registry is kept separate from ``PromptBuilder`` — a new strategy
# is added here (one entry) without touching the engine that evaluates them.
# ---------------------------------------------------------------------------


def _build_registry(settings: Settings) -> Mapping[StrategyName, Strategy]:
    return MappingProxyType(
        {
            TrendFollowingStrategy.name: TrendFollowingStrategy(settings),
            MomentumStrategy.name: MomentumStrategy(settings),
            SwingStrategy.name: SwingStrategy(settings),
            BreakoutStrategy.name: BreakoutStrategy(settings),
            MeanReversionStrategy.name: MeanReversionStrategy(settings),
        }
    )


#: Default registry, built once from the process-global cached settings.
#: Callers that need custom settings (tests, alternate configuration) build
#: their own via :func:`build_strategy_registry` rather than mutating this one.
_DEFAULT_REGISTRY: Final[Mapping[StrategyName, Strategy]] = _build_registry(get_settings())


def build_strategy_registry(settings: Settings | None = None) -> Mapping[StrategyName, Strategy]:
    """Build a fresh, immutable strategy registry.

    Args:
        settings: Optional injected configuration. When omitted, returns the
            shared default registry (no rebuild) — the common case.
    """
    if settings is None:
        return _DEFAULT_REGISTRY
    return _build_registry(settings)


def available_strategies() -> tuple[StrategyName, ...]:
    """Every registered strategy name, sorted for deterministic output."""
    return tuple(sorted(_DEFAULT_REGISTRY, key=lambda name: name.value))


def get_strategy(name: StrategyName, *, registry: Mapping[StrategyName, Strategy] | None = None) -> Strategy:
    """Return the registered strategy for ``name``.

    Raises:
        KeyError: if no strategy is registered for ``name``.
    """
    reg = registry if registry is not None else _DEFAULT_REGISTRY
    try:
        return reg[name]
    except KeyError as exc:
        available = ", ".join(n.value for n in available_strategies())
        raise KeyError(f"No strategy registered for {name!r}. Available: {available}") from exc
