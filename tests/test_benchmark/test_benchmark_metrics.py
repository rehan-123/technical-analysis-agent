from __future__ import annotations

import pytest

from benchmark import benchmark_metrics as m
from benchmark.benchmark_models import (
    BenchmarkStage,
    StageTiming,
    TickerBenchmark,
    TokenSource,
    TokenUsage,
)


def _run(ticker="AAPL", *, ok=True, total=100.0, conf=80, rec="BUY",
         model="qwen2.5:7b", err=None, tokens=True, source=TokenSource.PROVIDER,
         stages=None):
    st = list(stages or [])
    st.append(StageTiming(stage=BenchmarkStage.TOTAL, duration_ms=total))
    return TickerBenchmark(
        ticker=ticker, succeeded=ok, stages=tuple(st),
        tokens=TokenUsage(prompt_chars=4000, prompt_tokens=1000, output_tokens=600,
                          source=source) if tokens else None,
        recommendation=rec if ok else None, confidence=conf if ok else None,
        model_used=model if ok else None, error_type=err,
    )


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert m.estimate_tokens("") == 0

    def test_scales_with_length(self):
        assert m.estimate_tokens("x" * 400) == 100

    def test_ratio_is_configurable(self):
        assert m.estimate_tokens("x" * 300, chars_per_token=3.0) == 100

    def test_non_positive_ratio_is_safe(self):
        assert m.estimate_tokens("abc", chars_per_token=0) == 0


class TestPercentile:
    def test_empty_is_zero(self):
        assert m.percentile([], 50) == 0.0

    def test_single_value(self):
        assert m.percentile([7.0], 95) == 7.0

    def test_interpolates(self):
        assert m.percentile([1, 2, 3, 4], 50) == 2.5

    def test_bounds(self):
        assert m.percentile([10, 20, 30], 0) == 10
        assert m.percentile([10, 20, 30], 100) == 30

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            m.percentile([1, 2], 101)

    def test_is_order_independent(self):
        assert m.percentile([3, 1, 2], 50) == m.percentile([1, 2, 3], 50)


class TestAggregates:
    def test_mean_of_empty(self):
        assert m.mean([]) == 0.0

    def test_success_rate(self):
        runs = [_run(ok=True), _run(ok=True), _run(ok=False)]
        assert m.success_rate(runs) == pytest.approx(2 / 3)

    def test_success_rate_empty(self):
        assert m.success_rate([]) == 0.0

    def test_throughput_uses_successful_runs_only(self):
        runs = [_run(total=30_000.0), _run(ok=False, total=1000.0)]
        assert m.throughput_per_min(runs) == pytest.approx(2.0)

    def test_throughput_zero_when_no_successes(self):
        assert m.throughput_per_min([_run(ok=False)]) == 0.0

    def test_mean_confidence_ignores_failures(self):
        runs = [_run(conf=60), _run(conf=80), _run(ok=False)]
        assert m.mean_confidence(runs) == 70.0

    def test_mean_confidence_none_when_all_failed(self):
        assert m.mean_confidence([_run(ok=False)]) is None

    def test_count_values_sorted_and_skips_none(self):
        assert m.count_values(["B", "A", "B", None]) == {"A": 1, "B": 2}


class TestStageStatistics:
    def test_skips_stages_never_reached(self):
        runs = [_run(stages=[StageTiming(stage=BenchmarkStage.TECHNICAL, duration_ms=10.0)])]
        stages = {s.stage for s in m.stage_statistics(runs)}
        assert BenchmarkStage.TECHNICAL in stages
        assert BenchmarkStage.INFERENCE not in stages

    def test_emitted_in_fixed_enum_order(self):
        runs = [_run(stages=[
            StageTiming(stage=BenchmarkStage.INFERENCE, duration_ms=50.0),
            StageTiming(stage=BenchmarkStage.TECHNICAL, duration_ms=10.0),
        ])]
        order = [s.stage for s in m.stage_statistics(runs)]
        assert order.index(BenchmarkStage.TECHNICAL) < order.index(BenchmarkStage.INFERENCE)

    def test_computes_expected_aggregates(self):
        runs = [
            _run(stages=[StageTiming(stage=BenchmarkStage.TECHNICAL, duration_ms=10.0)]),
            _run(stages=[StageTiming(stage=BenchmarkStage.TECHNICAL, duration_ms=30.0)]),
        ]
        tech = next(s for s in m.stage_statistics(runs) if s.stage is BenchmarkStage.TECHNICAL)
        assert tech.samples == 2 and tech.mean_ms == 20.0 and tech.max_ms == 30.0


class TestTokenSourceReporting:
    def test_provider_only_when_every_run_measured(self):
        runs = [_run(source=TokenSource.PROVIDER), _run(source=TokenSource.PROVIDER)]
        assert m.dominant_token_source(runs) is TokenSource.PROVIDER

    def test_one_estimate_downgrades_the_whole_report(self):
        runs = [_run(source=TokenSource.PROVIDER), _run(source=TokenSource.ESTIMATED)]
        assert m.dominant_token_source(runs) is TokenSource.ESTIMATED

    def test_no_tokens_defaults_to_estimated(self):
        assert m.dominant_token_source([_run(tokens=False)]) is TokenSource.ESTIMATED
