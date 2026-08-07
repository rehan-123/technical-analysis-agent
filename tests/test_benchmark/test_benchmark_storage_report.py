from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from benchmark.benchmark_models import (
    BenchmarkReport,
    BenchmarkStage,
    BenchmarkSummary,
    StageStats,
    StageTiming,
    TickerBenchmark,
    TokenSource,
    TokenUsage,
)
from benchmark.benchmark_report import render_markdown
from benchmark.benchmark_storage import CSV_COLUMNS, BenchmarkStorage


def _run(ticker="AAPL", ok=True):
    stages = (
        StageTiming(stage=BenchmarkStage.TECHNICAL, duration_ms=12.5),
        StageTiming(stage=BenchmarkStage.NEWS, duration_ms=30.0),
        StageTiming(stage=BenchmarkStage.PROMPT_BUILD, duration_ms=1.5),
        StageTiming(stage=BenchmarkStage.INFERENCE, duration_ms=8000.0),
        StageTiming(stage=BenchmarkStage.PARSE, duration_ms=0.5),
        StageTiming(stage=BenchmarkStage.TOTAL, duration_ms=8045.0),
    )
    if ok:
        return TickerBenchmark(
            ticker=ticker, succeeded=True, stages=stages,
            tokens=TokenUsage(prompt_chars=3846, prompt_tokens=1012,
                              output_tokens=640, source=TokenSource.PROVIDER),
            recommendation="BUY", confidence=78, model_used="qwen2.5:7b",
            prompt_version="1.0", news_articles_included=15,
        )
    return TickerBenchmark(
        ticker=ticker, succeeded=False,
        stages=(StageTiming(stage=BenchmarkStage.TOTAL, duration_ms=42.0),),
        failed_stage=BenchmarkStage.TECHNICAL, error_type="DataFetchError",
    )


def _report(runs=None):
    runs = runs or (_run("AAPL"), _run("MSFT", ok=False))
    summary = BenchmarkSummary(
        run_id="abc123def456",
        generated_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        model="qwen2.5:7b", provider="ollama",
        total_cases=len(runs), succeeded=sum(r.succeeded for r in runs),
        failed=sum(not r.succeeded for r in runs), success_rate=0.5,
        mean_latency_ms=8045.0, p50_latency_ms=8045.0, p95_latency_ms=8045.0,
        max_latency_ms=8045.0, throughput_per_min=7.46,
        mean_prompt_chars=3846.0, mean_prompt_tokens=1012.0, mean_output_tokens=640.0,
        token_source=TokenSource.PROVIDER, mean_confidence=78.0,
        recommendation_counts={"BUY": 1}, model_usage={"qwen2.5:7b": 1},
        error_counts={"DataFetchError": 1}, timeout_count=0,
        stage_stats=(StageStats(stage=BenchmarkStage.INFERENCE, samples=1,
                                mean_ms=8000.0, p50_ms=8000.0, p95_ms=8000.0, max_ms=8000.0),),
    )
    return BenchmarkReport(summary=summary, runs=tuple(runs))


class TestStorage:
    def test_write_all_creates_every_artifact(self, tmp_path):
        paths = BenchmarkStorage(tmp_path).write_all(_report(), summary_markdown="# hi")
        names = {p.name for p in paths}
        assert names == {"benchmark.json", "benchmark.csv", "benchmark_summary.md"}
        assert all(p.exists() for p in paths)

    def test_creates_missing_output_directory(self, tmp_path):
        target = tmp_path / "nested" / "out"
        BenchmarkStorage(target).write_all(_report())
        assert (target / "benchmark.json").exists()

    def test_json_is_valid_and_sorted(self, tmp_path):
        path = BenchmarkStorage(tmp_path).write_json(_report())
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        assert data["summary"]["run_id"] == "abc123def456"
        assert len(data["runs"]) == 2
        top = list(data.keys())
        assert top == sorted(top)

    def test_csv_columns_are_fixed_and_ordered(self, tmp_path):
        path = BenchmarkStorage(tmp_path).write_csv(_report())
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        assert tuple(rows[0]) == CSV_COLUMNS
        assert rows[1][0] == "AAPL" and rows[2][0] == "MSFT"

    def test_csv_row_carries_stage_timings(self, tmp_path):
        path = BenchmarkStorage(tmp_path).write_csv(_report())
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert rows[0]["inference_ms"] == "8000.0"
        assert rows[0]["token_source"] == "provider"

    def test_failed_row_has_blanks_not_crashes(self, tmp_path):
        path = BenchmarkStorage(tmp_path).write_csv(_report())
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        failed = rows[1]
        assert failed["succeeded"] == "False"
        assert failed["recommendation"] == ""
        assert failed["inference_ms"] == ""
        assert failed["failed_stage"] == "technical"

    def test_output_is_byte_identical_across_writes(self, tmp_path):
        report = _report()
        a = BenchmarkStorage(tmp_path / "a"); a.write_all(report)
        b = BenchmarkStorage(tmp_path / "b"); b.write_all(report)
        for name in ("benchmark.json", "benchmark.csv"):
            assert (a.output_dir / name).read_bytes() == (b.output_dir / name).read_bytes()

    def test_uses_lf_newlines(self, tmp_path):
        path = BenchmarkStorage(tmp_path).write_csv(_report())
        assert b"\r\n" not in path.read_bytes()


class TestMarkdownReport:
    def test_contains_headline_sections(self):
        md = render_markdown(_report())
        for heading in ("# AI Pipeline Benchmark", "## Results", "## Latency",
                        "## Stage breakdown", "## Prompt & response size",
                        "## Output quality", "## Reliability", "## Per-ticker"):
            assert heading in md

    def test_reports_run_identity_and_model(self):
        md = render_markdown(_report())
        assert "abc123def456" in md and "qwen2.5:7b" in md and "ollama" in md

    def test_distinguishes_measured_from_estimated_tokens(self):
        assert "measured by provider" in render_markdown(_report())

    def test_estimated_tokens_are_labelled_as_such(self):
        r = _report()
        summary = r.summary.model_copy(update={"token_source": TokenSource.ESTIMATED})
        assert "estimated from characters" in render_markdown(
            BenchmarkReport(summary=summary, runs=r.runs)
        )

    def test_lists_every_ticker(self):
        md = render_markdown(_report())
        assert "| AAPL |" in md and "| MSFT |" in md

    def test_failed_ticker_shows_failing_stage(self):
        assert "technical" in render_markdown(_report())

    def test_handles_absent_confidence(self):
        r = _report()
        summary = r.summary.model_copy(update={"mean_confidence": None})
        assert "n/a" in render_markdown(BenchmarkReport(summary=summary, runs=r.runs))

    def test_reports_clean_run_with_no_errors(self):
        r = _report(runs=(_run("AAPL"),))
        summary = r.summary.model_copy(update={"error_counts": {}, "failed": 0})
        assert "No errors recorded" in render_markdown(
            BenchmarkReport(summary=summary, runs=r.runs)
        )

    def test_render_is_deterministic(self):
        report = _report()
        assert render_markdown(report) == render_markdown(report)

    def test_ends_with_newline(self):
        assert render_markdown(_report()).endswith("\n")
