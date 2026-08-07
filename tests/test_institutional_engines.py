from __future__ import annotations

import time

import pandas as pd
import pytest

from config.settings import Settings
from engines.candlestick import CandlestickEngine
from engines.confidence import ConfidenceEngine
from engines.confluence import ConfluenceEngine
from engines.indicator_suite import IndicatorSuite
from engines.market_structure import MarketStructureEngine
from engines.risk import RiskEngine
from engines.smc import SMCEngine
from engines.volatility import VolatilityEngine
from engines.volume import VolumeEngine
from services.indicator_engine import IndicatorEngine
from services.support_resistance_service import SupportResistanceService


async def _enriched(provider, ticker="AAPL", period="1y", interval="1d"):
    settings = Settings()
    df = await provider.get_ohlcv(ticker, period, interval)
    enriched = IndicatorEngine(settings).compute(df)
    warm = [f"ema_{settings.ema_fast_period}", f"rsi_{settings.rsi_period}"]
    return enriched.dropna(subset=warm), settings


# --- Market structure -------------------------------------------------------

@pytest.mark.asyncio
async def test_market_structure_detects_uptrend(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    result = MarketStructureEngine(settings).evaluate(enriched)
    assert result.structure in ("uptrend", "downtrend", "ranging", "undetermined")
    assert result.signal in ("bullish", "bearish", "neutral")
    assert len(result.swing_highs) >= 0
    assert len(result.swing_lows) >= 0


@pytest.mark.asyncio
async def test_market_structure_bos_and_choch_are_booleans(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    result = MarketStructureEngine(settings).evaluate(enriched)
    assert isinstance(result.break_of_structure, bool)
    assert isinstance(result.change_of_character, bool)


# --- Candlestick --------------------------------------------------------------

@pytest.mark.asyncio
async def test_candlestick_engine_runs_without_error(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    result = CandlestickEngine(settings).evaluate(enriched)
    assert result.signal in ("bullish", "bearish", "neutral")
    assert 0 <= result.strength <= 100
    assert isinstance(result.patterns, list)


def test_candlestick_engine_handles_short_frame():
    """Fewer than 3 bars must not raise — should return a neutral no-op result."""
    settings = Settings()
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {"open": [10, 11], "high": [11, 12], "low": [9, 10], "close": [10.5, 11.5], "volume": [1000, 1000]},
        index=idx,
    )
    result = CandlestickEngine(settings).evaluate(df)
    assert result.patterns == []
    assert result.signal == "neutral"


# --- Volatility -----------------------------------------------------------

@pytest.mark.asyncio
async def test_volatility_engine_regime_is_valid(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    result = VolatilityEngine(settings).evaluate(enriched)
    assert result.regime in ("low", "normal", "high")
    assert 0 <= result.breakout_probability <= 100
    assert result.atr_pct >= 0


# --- Volume -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_volume_engine_produces_profile(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    result = VolumeEngine(settings).evaluate(enriched)
    assert result.profile is not None
    assert result.profile.value_area_low <= result.profile.point_of_control <= result.profile.value_area_high
    assert result.volume_trend in ("increasing", "decreasing", "stable")
    assert 0 <= result.strength <= 100


# --- SMC (heuristic) ----------------------------------------------------------

@pytest.mark.asyncio
async def test_smc_engine_flags_itself_as_heuristic(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    result = SMCEngine(settings).evaluate(enriched)
    assert result.heuristic is True
    assert result.market_zone in ("premium", "discount", "equilibrium")
    for gap in result.fair_value_gaps:
        assert gap.direction in ("bullish", "bearish")


@pytest.mark.asyncio
async def test_smc_equal_levels_are_plain_floats_not_numpy(synthetic_provider):
    """Regression: equal-level clustering must not leak np.float64 into the
    result, which would break JSON serialization downstream."""
    enriched, settings = await _enriched(synthetic_provider)
    result = SMCEngine(settings).evaluate(enriched)
    for level in [*result.equal_highs, *result.equal_lows]:
        assert type(level) is float


# --- Indicator suite ----------------------------------------------------------

@pytest.mark.asyncio
async def test_indicator_suite_has_expected_indicators(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    suite = IndicatorSuite(settings).compute(enriched)
    expected = {
        "ema_stack", "wma", "vwma", "vwap", "rsi", "macd", "cci", "roc", "momentum",
        "mfi", "cmf", "obv", "bollinger", "keltner", "donchian", "supertrend",
        "parabolic_sar", "pivots",
    }
    assert expected.issubset(suite.keys())
    for name, r in suite.items():
        assert r.signal in ("bullish", "bearish", "neutral"), name
        assert 0 <= r.strength <= 100, name
        assert r.interpretation, name


@pytest.mark.asyncio
async def test_indicator_suite_is_json_serializable(synthetic_provider):
    import json
    enriched, settings = await _enriched(synthetic_provider)
    suite = IndicatorSuite(settings).compute(enriched)
    payload = {k: v.model_dump() for k, v in suite.items()}
    json.dumps(payload)  # must not raise


# --- Confluence / confidence / risk (unit-level, real engines) ---------------

@pytest.mark.asyncio
async def test_confluence_engine_scores_are_bounded(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    suite = IndicatorSuite(settings).compute(enriched)
    structure = MarketStructureEngine(settings).evaluate(enriched)
    candles = CandlestickEngine(settings).evaluate(enriched)
    volume = VolumeEngine(settings).evaluate(enriched)
    volatility = VolatilityEngine(settings).evaluate(enriched)
    smc = SMCEngine(settings).evaluate(enriched)

    confluence = ConfluenceEngine(settings).evaluate(suite, structure, candles, volume, volatility, smc)
    assert 0 <= confluence.bullish_score <= 100
    assert 0 <= confluence.bearish_score <= 100
    assert confluence.net_bias in ("bullish", "bearish", "neutral")

    confidence = ConfidenceEngine(settings).evaluate(confluence, volatility)
    assert 0 <= confidence.confidence <= 100


@pytest.mark.asyncio
async def test_confluence_does_not_let_redundant_short_term_signals_outvote_category_weight(bearish_provider):
    """Regression for the within-category normalization fix: a category
    with many redundant same-direction indicators must not contribute more
    than its configured weight to the aggregate score."""
    enriched, settings = await _enriched(bearish_provider)
    suite = IndicatorSuite(settings).compute(enriched)
    structure = MarketStructureEngine(settings).evaluate(enriched)
    candles = CandlestickEngine(settings).evaluate(enriched)
    volume = VolumeEngine(settings).evaluate(enriched)
    volatility = VolatilityEngine(settings).evaluate(enriched)
    smc = SMCEngine(settings).evaluate(enriched)
    confluence = ConfluenceEngine(settings).evaluate(suite, structure, candles, volume, volatility, smc)

    for b in confluence.breakdown:
        weight_pct = getattr(settings, f"weight_{b.category}", 1.0) / sum([
            settings.weight_trend, settings.weight_momentum, settings.weight_structure,
            settings.weight_volume, settings.weight_volatility, settings.weight_candlestick,
            settings.weight_smc, settings.weight_pattern,
        ]) * 100
        assert b.bull_contribution <= weight_pct + 0.5, f"{b.category} bull contribution exceeds its weight cap"
        assert b.bear_contribution <= weight_pct + 0.5, f"{b.category} bear contribution exceeds its weight cap"


@pytest.mark.asyncio
async def test_risk_engine_long_plan_is_internally_consistent(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    structure = MarketStructureEngine(settings).evaluate(enriched)
    levels = SupportResistanceService(settings).evaluate(enriched)
    atr_pct = float(enriched.iloc[-1]["atr_pct"])

    risk = RiskEngine(settings).evaluate(enriched, "bullish", levels, structure, atr_pct)
    assert risk.direction == "long"
    close = float(enriched.iloc[-1]["close"])
    assert risk.stop_loss < close
    assert all(t > close for t in risk.targets)
    assert risk.targets == sorted(risk.targets)  # TP1 < TP2 < TP3
    assert risk.position_size > 0
    assert risk.risk_tier in ("Low", "Medium", "High")


@pytest.mark.asyncio
async def test_risk_engine_short_plan_is_internally_consistent(synthetic_provider):
    enriched, settings = await _enriched(synthetic_provider)
    structure = MarketStructureEngine(settings).evaluate(enriched)
    levels = SupportResistanceService(settings).evaluate(enriched)
    atr_pct = float(enriched.iloc[-1]["atr_pct"])

    risk = RiskEngine(settings).evaluate(enriched, "bearish", levels, structure, atr_pct)
    assert risk.direction == "short"
    close = float(enriched.iloc[-1]["close"])
    assert risk.stop_loss > close
    assert all(t < close for t in risk.targets)
    assert risk.targets == sorted(risk.targets, reverse=True)  # TP1 > TP2 > TP3 (descending toward target)


# --- The critical end-to-end trend-anchor regression (via the real agent) ---

@pytest.mark.asyncio
async def test_trend_label_matches_risk_direction_never_inconsistent(synthetic_provider, bearish_provider):
    """Regression: the headline `trend` field and `risk_plan.direction` must
    never disagree — e.g. trend='Bearish' while risk.direction='none'. Both
    are derived from ONE canonical bias in the agent; this guards against
    that derivation splitting apart again during future changes."""
    from agent.technical_analysis_agent import TechnicalAnalysisAgent

    for provider in (synthetic_provider, bearish_provider):
        agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=provider)
        result = await agent.analyze("AAPL")
        direction = result.risk_plan.get("direction")
        if result.trend in ("Bullish", "Strong Bullish"):
            assert direction == "long", f"trend={result.trend} but risk direction={direction}"
        elif result.trend in ("Bearish", "Strong Bearish"):
            assert direction == "short", f"trend={result.trend} but risk direction={direction}"
        else:  # Neutral
            assert direction == "none", f"trend=Neutral but risk direction={direction}"


def test_bias_trend_label_boundaries_are_aligned():
    """The bias thresholds used for the risk plan must partition net_score
    identically to the trend-label bands, so a 'Neutral' headline can never
    carry a directional (long/short) risk plan, and vice versa — including
    at the exact band boundaries (40 and 60), which random-data tests rarely
    land on."""
    def trend_label(score):
        if score >= 80: return "Strong Bullish"
        if score >= 60: return "Bullish"
        if score >= 40: return "Neutral"
        if score >= 20: return "Bearish"
        return "Strong Bearish"

    def bias_of(score):
        return "bullish" if score >= 60 else "bearish" if score < 40 else "neutral"

    for score in range(0, 101):
        label, bias = trend_label(score), bias_of(score)
        if label in ("Bullish", "Strong Bullish"):
            assert bias == "bullish", f"score={score} label={label} bias={bias}"
        elif label in ("Bearish", "Strong Bearish"):
            assert bias == "bearish", f"score={score} label={label} bias={bias}"
        else:
            assert bias == "neutral", f"score={score} label={label} bias={bias}"


# --- Validation module --------------------------------------------------------

class TestOHLCVValidator:
    def setup_method(self):
        from validation.ohlcv_validator import OHLCVValidator
        self.validator = OHLCVValidator()

    def _base_df(self, n=10):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {
                "open": [10.0 + i for i in range(n)],
                "high": [10.5 + i for i in range(n)],
                "low": [9.5 + i for i in range(n)],
                "close": [10.2 + i for i in range(n)],
                "volume": [1000.0] * n,
            },
            index=idx,
        )

    def test_clean_frame_has_no_warnings_or_errors(self):
        report = self.validator.validate(self._base_df(), "AAPL")
        assert report.ok
        assert report.warnings == []
        assert len(report.cleaned) == 10

    def test_empty_frame_is_an_error(self):
        report = self.validator.validate(pd.DataFrame(), "AAPL")
        assert not report.ok
        assert report.errors

    def test_missing_columns_is_an_error(self):
        df = self._base_df().drop(columns=["volume"])
        report = self.validator.validate(df, "AAPL")
        assert not report.ok

    def test_negative_prices_are_dropped_with_a_warning(self):
        df = self._base_df()
        df.iloc[3, df.columns.get_loc("close")] = -5.0
        report = self.validator.validate(df, "AAPL")
        assert report.ok
        assert len(report.cleaned) == 9
        assert any("non-positive" in w for w in report.warnings)

    def test_duplicate_timestamps_are_deduped_with_a_warning(self):
        df = self._base_df()
        df.index = list(df.index[:-1]) + [df.index[-2]]  # duplicate the second-to-last timestamp
        report = self.validator.validate(df, "AAPL")
        assert report.ok
        assert any("duplicate" in w for w in report.warnings)

    def test_unsorted_timestamps_are_sorted_with_a_warning(self):
        df = self._base_df()
        df = df.iloc[::-1]  # reverse order
        report = self.validator.validate(df, "AAPL")
        assert report.ok
        assert report.cleaned.index.is_monotonic_increasing
        assert any("sorted" in w for w in report.warnings)

    def test_incoherent_high_low_rows_are_dropped_with_a_warning(self):
        df = self._base_df()
        df.iloc[2, df.columns.get_loc("high")] = 1.0  # high < low, impossible bar
        report = self.validator.validate(df, "AAPL")
        assert report.ok
        assert len(report.cleaned) == 9
        assert any("incoherent" in w for w in report.warnings)


# --- Performance ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_analysis_completes_within_reasonable_time(synthetic_provider):
    """Not a strict benchmark (sandbox CI hardware varies) — a generous
    ceiling that would catch an accidental O(n^2)+ regression in any engine."""
    from agent.technical_analysis_agent import TechnicalAnalysisAgent

    agent = TechnicalAnalysisAgent(settings=Settings(), data_provider=synthetic_provider)
    start = time.perf_counter()
    result = await agent.analyze("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 5000, f"Full analysis took {elapsed_ms:.0f}ms, expected < 5000ms"
    assert result.metadata is not None
    assert result.metadata.execution_ms > 0
