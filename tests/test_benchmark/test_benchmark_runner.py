from __future__ import annotations

import itertools
from datetime import datetime, timezone

import pytest

from benchmark.benchmark_models import BenchmarkStage, TokenSource
from benchmark.benchmark_runner import BenchmarkRunner
from config.settings import Settings
from models.ai_analysis import AIAnalysisResult, Recommendation
from models.prompt_package import PromptMetadata, PromptPackage
from services.ai_exceptions import ResponseParseError
from services.prompt_sections.exceptions import PromptBuildError
from utils.exceptions import DataFetchError

FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _clock():
    """Monotonic fake clock: every call advances 1 ms. Deterministic."""
    counter = itertools.count(0, 0.001)
    return lambda: next(counter)


class FakeTechnical:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    async def analyze(self, ticker, *, period="1y", interval="1d"):
        self.calls.append((ticker, period, interval))
        if self.error:
            raise self.error
        return None  # AIAnalysisRequest.technical is optional


class FakeNews:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    async def run(self, ticker, **kwargs):
        self.calls.append(ticker)
        if self.error:
            raise self.error
        return None


class FakeBuilder:
    def __init__(self, error=None):
        self.error = error

    def build(self, request):
        if self.error:
            raise self.error
        md = PromptMetadata(ticker=request.ticker, prompt_version="1.0",
                            news_articles_included=3)
        return PromptPackage(system_prompt="SYS", user_prompt="USER",
                             metadata=md, prompt_version="1.0")


class FakeResponse:
    def __init__(self, text='{"ok": true}', model="qwen2.5:7b", raw=None):
        self.text, self.model, self.raw = text, model, raw if raw is not None else {}


class FakeProvider:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response or FakeResponse(), error, []

    async def generate(self, prompt, *, system=None, schema=None, model=None):
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self.error:
            raise self.error
        return self.response


class FakeParser:
    def __init__(self, error=None, confidence=80):
        self.error, self.confidence = error, confidence

    def parse(self, text, *, ticker, model_used=""):
        if self.error:
            raise self.error
        return AIAnalysisResult(
            ticker=ticker, recommendation=Recommendation.BUY,
            confidence=self.confidence, investment_thesis="t", model_used=model_used,
        )


def _runner(**over):
    kwargs = dict(
        technical_agent=FakeTechnical(), news_agent=FakeNews(),
        prompt_builder=FakeBuilder(), provider=FakeProvider(), parser=FakeParser(),
        settings=Settings(llm_model="qwen2.5:7b", llm_provider="ollama"),
        clock=_clock(), now=lambda: FIXED_NOW,
    )
    kwargs.update(over)
    return BenchmarkRunner(**kwargs)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_reports_one_run_per_ticker_in_order(self):
        report = await _runner().run(["AAPL", "MSFT", "NVDA"])
        assert [r.ticker for r in report.runs] == ["AAPL", "MSFT", "NVDA"]
        assert report.summary.total_cases == 3
        assert report.summary.succeeded == 3
        assert report.summary.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_records_every_stage(self):
        report = await _runner().run(["AAPL"])
        stages = {t.stage for t in report.runs[0].stages}
        assert stages == {
            BenchmarkStage.TECHNICAL, BenchmarkStage.NEWS, BenchmarkStage.PROMPT_BUILD,
            BenchmarkStage.INFERENCE, BenchmarkStage.PARSE, BenchmarkStage.TOTAL,
        }

    @pytest.mark.asyncio
    async def test_captures_result_quality_fields(self):
        report = await _runner().run(["AAPL"])
        run = report.runs[0]
        assert run.recommendation == "BUY"
        assert run.confidence == 80
        assert run.prompt_version == "1.0"
        assert run.news_articles_included == 3

    @pytest.mark.asyncio
    async def test_period_and_interval_are_forwarded(self):
        tech = FakeTechnical()
        await _runner(technical_agent=tech).run(["AAPL"], period="6mo", interval="1h")
        assert tech.calls == [("AAPL", "6mo", "1h")]

    @pytest.mark.asyncio
    async def test_prompt_is_passed_to_the_provider_verbatim(self):
        provider = FakeProvider()
        await _runner(provider=provider).run(["AAPL"])
        assert provider.calls[0]["prompt"] == "USER"
        assert provider.calls[0]["system"] == "SYS"


class TestTokenAccounting:
    @pytest.mark.asyncio
    async def test_prefers_provider_reported_counts(self):
        provider = FakeProvider(FakeResponse(raw={"prompt_eval_count": 1012, "eval_count": 640}))
        report = await _runner(provider=provider).run(["AAPL"])
        tokens = report.runs[0].tokens
        assert tokens.source is TokenSource.PROVIDER
        assert tokens.prompt_tokens == 1012 and tokens.output_tokens == 640
        assert tokens.total_tokens == 1652

    @pytest.mark.asyncio
    async def test_falls_back_to_estimates_without_counters(self):
        report = await _runner().run(["AAPL"])
        assert report.runs[0].tokens.source is TokenSource.ESTIMATED

    @pytest.mark.asyncio
    async def test_partial_counters_still_estimate(self):
        provider = FakeProvider(FakeResponse(raw={"prompt_eval_count": 500}))
        report = await _runner(provider=provider).run(["AAPL"])
        assert report.runs[0].tokens.source is TokenSource.ESTIMATED

    @pytest.mark.asyncio
    async def test_prompt_chars_span_system_and_user(self):
        report = await _runner().run(["AAPL"])
        assert report.runs[0].tokens.prompt_chars == len("SYS") + len("USER")


class TestFailureAttribution:
    @pytest.mark.asyncio
    async def test_technical_failure_is_fatal_and_attributed(self):
        r = await _runner(technical_agent=FakeTechnical(DataFetchError("down"))).run(["AAPL"])
        run = r.runs[0]
        assert run.succeeded is False
        assert run.failed_stage is BenchmarkStage.TECHNICAL
        assert run.error_type == "DataFetchError"

    @pytest.mark.asyncio
    async def test_failed_run_still_records_total(self):
        r = await _runner(technical_agent=FakeTechnical(DataFetchError("x"))).run(["AAPL"])
        assert r.runs[0].stage_ms(BenchmarkStage.TOTAL) is not None

    @pytest.mark.asyncio
    async def test_news_failure_is_tolerated(self):
        """News is enrichment — the run continues and still succeeds."""
        r = await _runner(news_agent=FakeNews(RuntimeError("news down"))).run(["AAPL"])
        assert r.runs[0].succeeded is True
        news_timing = next(t for t in r.runs[0].stages if t.stage is BenchmarkStage.NEWS)
        assert news_timing.succeeded is False
        assert news_timing.error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_runs_without_a_news_agent(self):
        r = await _runner(news_agent=None).run(["AAPL"])
        assert r.runs[0].succeeded is True
        assert r.runs[0].stage_ms(BenchmarkStage.NEWS) is None

    @pytest.mark.asyncio
    async def test_prompt_build_failure_attributed(self):
        r = await _runner(prompt_builder=FakeBuilder(PromptBuildError("nothing"))).run(["AAPL"])
        assert r.runs[0].failed_stage is BenchmarkStage.PROMPT_BUILD

    @pytest.mark.asyncio
    async def test_parse_failure_attributed(self):
        r = await _runner(parser=FakeParser(ResponseParseError("bad"))).run(["AAPL"])
        assert r.runs[0].failed_stage is BenchmarkStage.PARSE

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_the_batch(self):
        class Flaky(FakeTechnical):
            async def analyze(self, ticker, *, period="1y", interval="1d"):
                if ticker == "BAD":
                    raise DataFetchError("boom")
                return None

        r = await _runner(technical_agent=Flaky()).run(["AAPL", "BAD", "MSFT"])
        assert [x.succeeded for x in r.runs] == [True, False, True]
        assert r.summary.succeeded == 2 and r.summary.failed == 1

    @pytest.mark.asyncio
    async def test_timeouts_are_counted_separately(self):
        class Timeout(Exception):
            pass
        Timeout.__name__ = "ReadTimeout"
        r = await _runner(provider=FakeProvider(error=Timeout("slow"))).run(["AAPL"])
        assert r.summary.timeout_count == 1


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_identical_inputs_produce_identical_reports(self):
        a = await _runner().run(["AAPL", "MSFT"])
        b = await _runner().run(["AAPL", "MSFT"])
        assert a.model_dump() == b.model_dump()

    @pytest.mark.asyncio
    async def test_run_id_is_stable_and_order_independent(self):
        a = await _runner().run(["AAPL", "MSFT"])
        b = await _runner().run(["MSFT", "AAPL"])
        assert a.summary.run_id == b.summary.run_id

    @pytest.mark.asyncio
    async def test_run_id_changes_with_the_ticker_set(self):
        a = await _runner().run(["AAPL"])
        b = await _runner().run(["AAPL", "MSFT"])
        assert a.summary.run_id != b.summary.run_id

    @pytest.mark.asyncio
    async def test_summary_aggregates_are_populated(self):
        r = await _runner().run(["AAPL", "MSFT"])
        s = r.summary
        assert s.model == "qwen2.5:7b" and s.provider == "ollama"
        assert s.mean_confidence == 80.0
        assert s.recommendation_counts == {"BUY": 2}
        assert s.model_usage == {"qwen2.5:7b": 2}
        assert s.generated_at == FIXED_NOW

    @pytest.mark.asyncio
    async def test_empty_ticker_list_is_safe(self):
        r = await _runner().run([])
        assert r.runs == () and r.summary.total_cases == 0
        assert r.summary.success_rate == 0.0
