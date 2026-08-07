from __future__ import annotations

import pytest

from config.settings import Settings
from models.strategy import StrategyName
from strategy.base import Strategy
from strategy.registry import available_strategies, build_strategy_registry, get_strategy


def test_available_strategies_lists_all_five():
    names = available_strategies()
    assert set(names) == set(StrategyName)
    assert len(names) == 5


def test_available_strategies_is_sorted_and_deterministic():
    assert available_strategies() == tuple(sorted(available_strategies(), key=lambda n: n.value))
    assert available_strategies() == available_strategies()


@pytest.mark.parametrize("name", list(StrategyName))
def test_get_strategy_returns_matching_implementation(name):
    strategy = get_strategy(name)
    assert isinstance(strategy, Strategy)
    assert strategy.name is name


def test_get_strategy_unknown_name_raises():
    with pytest.raises(KeyError):
        get_strategy("not_a_real_strategy")  # type: ignore[arg-type]


def test_default_registry_is_immutable():
    registry = build_strategy_registry()
    with pytest.raises(TypeError):
        registry[StrategyName.MOMENTUM] = None  # type: ignore[index]


def test_build_strategy_registry_with_custom_settings_returns_fresh_mapping():
    registry = build_strategy_registry(Settings())
    assert set(registry) == set(StrategyName)
    for name, strategy in registry.items():
        assert strategy.name is name
