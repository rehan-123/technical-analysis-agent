from __future__ import annotations

import pytest

from models.analysis_result import (
    AnalysisMetadata,
    IndicatorSnapshot,
    PatternFlags,
    SupportResistanceLevels,
    TechnicalAnalysisResult,
)
from services.prompt_sections.base import RenderedSection
from services.prompt_sections.technical_section import TechnicalSectionRenderer


def _indicators(**overrides) -> IndicatorSnapshot:
    base = dict(
        close=190.0, ema_20=188.0, ema_50=185.0, ema_200=180.0, sma=186.0,
        rsi=61.0, macd_line=1.2, macd_signal=0.9, macd_histogram=0.3,
        atr=3.1, atr_pct=1.6, bb_upper=195.0, bb_middle=188.0, bb_lower=181.0,
        bb_percent_b=0.7, bb_bandwidth=0.07, volume=1_000_000.0,
        volume_sma=900_000.0, relative_volume=1.11,
    )
    base.update(overrides)
    return IndicatorSnapshot(**base)


def _result(**overrides) -> TechnicalAnalysisResult:
    base = dict(
        agent="technical_analysis_agent",
        ticker="AAPL",
        trend="Bullish",
        strength=72,
        signals=["EMA20 > EMA50", "RSI rising", "MACD bullish crossover"],
        entry_zone=(188.5, 190.2),
        stop_loss=184.0,
        targets=[195.0, 200.0, 205.0],
        risk="Medium",
        confidence=78,
        summary="Uptrend intact with improving momentum.",
        indicators=_indicators(),
        levels=SupportResistanceLevels(support=[184.0, 180.0], resistance=[195.0, 200.0]),
        patterns=PatternFlags(
            breakout=True, pullback=False, trend_reversal=False,
            consolidation=False, high_volatility=True,
        ),
        indicator_suite={"rsi": {"value": 61}},
        market_structure={"hh_hl": True},
        volume_analysis={"poc": 188.0},
        volatility={"atr": 3.1},
        smc={"order_block": 187.0},
        confluence={"score": 0.8},
        confidence_breakdown={"trend": 30},
        risk_plan={"r_multiple": 2.0},
        reasoning=["engine reasoning step 1", "engine reasoning step 2"],
        metadata=AnalysisMetadata(
            execution_ms=42.7, bars_analyzed=250, period="1y", interval="1d",
            warnings=[], validation_errors=[], heuristic_components=["smc"],
        ),
    )
    base.update(overrides)
    return TechnicalAnalysisResult(**base)


R = TechnicalSectionRenderer()


class TestBasicRendering:
    def test_returns_a_rendered_section_with_correct_kind_and_title(self):
        section = R.render(_result(), max_items=8)
        assert isinstance(section, RenderedSection)
        assert section.kind == "technical"
        assert section.title == "Technical Analysis"

    def test_headline_fields_are_present(self):
        body = R.render(_result(), max_items=8).body
        assert "Bullish" in body
        assert "72" in body            # strength
        assert "78" in body            # confidence
        assert "Medium" in body        # risk

    def test_trade_plan_and_levels_present(self):
        body = R.render(_result(), max_items=8).body
        assert "188.5" in body and "190.2" in body   # entry zone
        assert "184.0" in body                        # stop loss
        assert "195.0" in body and "200.0" in body    # targets / resistance
        assert "Support levels" in body and "Resistance levels" in body

    def test_signals_and_active_patterns_present(self):
        body = R.render(_result(), max_items=8).body
        assert "RSI rising" in body
        assert "breakout" in body and "high_volatility" in body
        # inactive patterns must NOT appear
        assert "pullback" not in body and "trend_reversal" not in body

    def test_summary_present(self):
        assert "Uptrend intact" in R.render(_result(), max_items=8).body


class TestDeterminism:
    def test_same_input_yields_identical_text(self):
        a = R.render(_result(), max_items=8)
        b = R.render(_result(), max_items=8)
        assert a.body == b.body
        assert a == b

    def test_output_contains_no_timestamp_or_execution_metrics(self):
        """Non-deterministic / diagnostic values must never appear."""
        body = R.render(_result(), max_items=8).body
        assert "execution_ms" not in body and "42.7" not in body
        assert "bars_analyzed" not in body and "250" not in body


class TestWhitelistEnforcement:
    def test_excluded_engine_dicts_do_not_leak(self):
        body = R.render(_result(), max_items=8).body
        for leaked in ("order_block", "poc", "r_multiple", "hh_hl", "confidence_breakdown", "0.8"):
            assert leaked not in body, f"excluded field leaked: {leaked}"

    def test_raw_indicator_values_do_not_leak(self):
        """Raw RSI/MACD/Bollinger numbers are intentionally excluded."""
        body = R.render(_result(), max_items=8).body
        assert "macd_line" not in body and "bb_upper" not in body
        assert "relative_volume" not in body

    def test_engine_reasoning_is_not_echoed(self):
        body = R.render(_result(), max_items=8).body
        assert "engine reasoning" not in body

    def test_a_new_model_field_does_not_appear_in_output(self):
        """Whitelist stability: content the renderer doesn't project is absent.
        (Simulated by asserting an excluded field's contents never appear.)"""
        result = _result(smc={"SECRET_NEW_SIGNAL": 123.456})
        assert "SECRET_NEW_SIGNAL" not in R.render(result, max_items=8).body
        assert "123.456" not in R.render(result, max_items=8).body

    def test_no_raw_model_dump(self):
        """The body must be curated text, not a JSON/dict dump of the model."""
        body = R.render(_result(), max_items=8).body
        assert "{" not in body and "}" not in body
        assert "indicator_suite" not in body


class TestEmptyAndOptional:
    def test_empty_levels_render_a_placeholder_not_an_error(self):
        result = _result(levels=SupportResistanceLevels(support=[], resistance=[]))
        body = R.render(result, max_items=8).body
        assert "none identified" in body

    def test_no_active_patterns_omits_the_patterns_line(self):
        result = _result(patterns=PatternFlags(
            breakout=False, pullback=False, trend_reversal=False,
            consolidation=False, high_volatility=False,
        ))
        assert "Patterns:" not in R.render(result, max_items=8).body

    def test_empty_summary_omits_the_summary_line(self):
        assert "Summary:" not in R.render(_result(summary=""), max_items=8).body

    def test_metadata_none_is_handled(self):
        section = R.render(_result(metadata=None), max_items=8)
        assert section.kind == "technical"


class TestItemCountAndTruncation:
    def test_item_count_reflects_signals_shown(self):
        section = R.render(_result(signals=["a", "b", "c"]), max_items=8)
        assert section.item_count == 3

    def test_not_truncated_when_within_cap(self):
        assert R.render(_result(), max_items=8).truncated is False

    def test_truncated_flag_set_when_signals_exceed_cap(self):
        result = _result(signals=[f"sig{i}" for i in range(10)])
        section = R.render(result, max_items=3)
        assert section.truncated is True
        assert section.item_count == 3
        assert "sig9" not in section.body  # clipped entries absent

    def test_truncated_flag_set_when_levels_exceed_cap(self):
        result = _result(levels=SupportResistanceLevels(
            support=[1.0, 2.0, 3.0, 4.0], resistance=[10.0, 11.0],
        ))
        assert R.render(result, max_items=2).truncated is True

    def test_truncation_is_deterministic_leading_entries(self):
        result = _result(signals=["first", "second", "third", "fourth"])
        body = R.render(result, max_items=2).body
        assert "first" in body and "second" in body
        assert "third" not in body and "fourth" not in body
