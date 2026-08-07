from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from benchmark import benchmark_metrics as metrics
from benchmark.benchmark_models import (
    BenchmarkReport,
    BenchmarkStage,
    BenchmarkSummary,
    StageTiming,
    TickerBenchmark,
    TokenSource,
    TokenUsage,
)
from config.settings import Settings, get_settings
from models.ai_analysis import AIAnalysisRequest
from utils.logger import get_logger

logger = get_logger(__name__)

#: Ollama-style usage counters. Present on backends that report exact token
#: counts; absent ones fall back to estimation.
_PROMPT_TOKEN_KEYS = ("prompt_eval_count", "prompt_tokens")
_OUTPUT_TOKEN_KEYS = ("eval_count", "completion_tokens")


class BenchmarkRunner:
    """Measures the cost and quality of the AI pipeline, stage by stage.

    Drives the *existing* components in the same order the production service
    does — technical agent, news agent, prompt builder, LLM provider, response
    parser — and times each one. It reimplements none of them: no prompt
    formatting, no parsing, no analysis. Every collaborator is injected, so the
    harness runs against fakes with no network at all.

    Cases run **sequentially and in the order supplied**. That is deliberate:
    concurrent runs would contend for the same CPU-bound model and make latency
    figures meaningless, and stable ordering keeps reports diffable.
    """

    def __init__(
        self,
        *,
        technical_agent,
        prompt_builder,
        provider,
        parser,
        news_agent=None,
        settings: Settings | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Args:
        technical_agent / news_agent: input producers. News is optional; when
            absent or failing, the benchmark proceeds technical-only, matching
            the pipeline agent's own tolerance policy.
        prompt_builder / provider / parser: the measured pipeline components.
        clock: monotonic time source, injectable so tests produce fixed timings.
        now: wall-clock source for the report header, injectable for the same
            reason.
        """
        self._technical_agent = technical_agent
        self._news_agent = news_agent
        self._prompt_builder = prompt_builder
        self._provider = provider
        self._parser = parser
        self._settings = settings or get_settings()
        self._clock = clock
        self._now = now or (lambda: datetime.now(timezone.utc))

    async def run(
        self,
        tickers: Sequence[str],
        *,
        period: str = "1y",
        interval: str = "1d",
    ) -> BenchmarkReport:
        """Benchmark every ticker and return the aggregated report."""
        runs: list[TickerBenchmark] = []
        for ticker in tickers:
            result = await self._run_one(ticker, period=period, interval=interval)
            runs.append(result)
            logger.info(
                "benchmark %s: %s in %.0fms",
                ticker, "ok" if result.succeeded else f"failed@{result.failed_stage}", result.total_ms,
            )
        return BenchmarkReport(summary=self._summarize(tuple(runs), tickers), runs=tuple(runs))

    # --- single case ----------------------------------------------------------

    async def _run_one(self, ticker: str, *, period: str, interval: str) -> TickerBenchmark:
        """Run one ticker through every stage, recording timings as it goes.

        On failure the partial timings collected so far are preserved — knowing
        *where* a run died is the point of stage attribution.
        """
        timings: list[StageTiming] = []
        started = self._clock()

        technical, failure = await self._timed(
            timings, BenchmarkStage.TECHNICAL,
            lambda: self._technical_agent.analyze(ticker, period=period, interval=interval),
        )
        if failure is not None:
            return self._failed(ticker, timings, started, BenchmarkStage.TECHNICAL, failure)

        news = None
        if self._news_agent is not None:
            # News is enrichment: a failure is recorded but does not end the run.
            news, _ = await self._timed(
                timings, BenchmarkStage.NEWS, lambda: self._news_agent.run(ticker)
            )

        request = AIAnalysisRequest(ticker=ticker, technical=technical, news=news)

        package, failure = await self._timed(
            timings, BenchmarkStage.PROMPT_BUILD,
            lambda: self._prompt_builder.build(request), is_async=False,
        )
        if failure is not None:
            return self._failed(ticker, timings, started, BenchmarkStage.PROMPT_BUILD, failure)

        response, failure = await self._timed(
            timings, BenchmarkStage.INFERENCE,
            lambda: self._provider.generate(
                package.user_prompt, system=package.system_prompt, model=None
            ),
        )
        if failure is not None:
            return self._failed(ticker, timings, started, BenchmarkStage.INFERENCE, failure)

        result, failure = await self._timed(
            timings, BenchmarkStage.PARSE,
            lambda: self._parser.parse(
                response.text, ticker=ticker, model_used=response.model
            ),
            is_async=False,
        )
        if failure is not None:
            return self._failed(ticker, timings, started, BenchmarkStage.PARSE, failure)

        timings.append(self._total(started))
        return TickerBenchmark(
            ticker=ticker,
            succeeded=True,
            stages=tuple(timings),
            tokens=self._token_usage(package.user_prompt, package.system_prompt, response),
            recommendation=str(result.recommendation.value),
            confidence=result.confidence,
            model_used=result.model_used or response.model,
            prompt_version=package.prompt_version,
            news_articles_included=package.metadata.news_articles_included,
        )

    async def _timed(self, timings, stage, call, *, is_async: bool = True):
        """Execute ``call``, append its timing, and return ``(value, failure)``.

        Failures are returned rather than raised so the caller decides whether a
        stage is fatal — the same asymmetric policy the pipeline agent applies.
        """
        start = self._clock()
        try:
            value = await call() if is_async else call()
        except Exception as exc:  # noqa: BLE001 — recorded, then classified by the caller
            timings.append(StageTiming(
                stage=stage, duration_ms=self._elapsed_ms(start),
                succeeded=False, error_type=type(exc).__name__,
            ))
            return None, exc
        timings.append(StageTiming(stage=stage, duration_ms=self._elapsed_ms(start)))
        return value, None

    def _failed(self, ticker, timings, started, stage, exc) -> TickerBenchmark:
        timings.append(self._total(started))
        return TickerBenchmark(
            ticker=ticker, succeeded=False, stages=tuple(timings),
            failed_stage=stage, error_type=type(exc).__name__,
        )

    def _total(self, started: float) -> StageTiming:
        return StageTiming(stage=BenchmarkStage.TOTAL, duration_ms=self._elapsed_ms(started))

    def _elapsed_ms(self, start: float) -> float:
        return round(max(0.0, (self._clock() - start) * 1000.0), 2)

    def _token_usage(self, user_prompt: str, system_prompt: str, response) -> TokenUsage:
        """Prefer the backend's exact counters; estimate only when absent."""
        raw = getattr(response, "raw", None) or {}
        prompt_tokens = _first_int(raw, _PROMPT_TOKEN_KEYS)
        output_tokens = _first_int(raw, _OUTPUT_TOKEN_KEYS)
        prompt_chars = len(user_prompt) + len(system_prompt)

        if prompt_tokens is not None and output_tokens is not None:
            return TokenUsage(
                prompt_chars=prompt_chars, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens, source=TokenSource.PROVIDER,
            )
        return TokenUsage(
            prompt_chars=prompt_chars,
            prompt_tokens=metrics.estimate_tokens(system_prompt + user_prompt),
            output_tokens=metrics.estimate_tokens(getattr(response, "text", "") or ""),
            source=TokenSource.ESTIMATED,
        )

    # --- aggregation ----------------------------------------------------------

    def _summarize(self, runs: tuple[TickerBenchmark, ...], tickers: Sequence[str]) -> BenchmarkSummary:
        latencies = [r.total_ms for r in runs if r.succeeded]
        token_runs = [r for r in runs if r.tokens is not None]
        errors = metrics.count_values(r.error_type for r in runs)
        return BenchmarkSummary(
            run_id=_run_id(tickers, self._settings.llm_model),
            generated_at=self._now(),
            model=self._settings.llm_model,
            provider=self._settings.llm_provider,
            total_cases=len(runs),
            succeeded=sum(1 for r in runs if r.succeeded),
            failed=sum(1 for r in runs if not r.succeeded),
            success_rate=round(metrics.success_rate(runs), 4),
            mean_latency_ms=round(metrics.mean(latencies), 2),
            p50_latency_ms=round(metrics.percentile(latencies, 50), 2),
            p95_latency_ms=round(metrics.percentile(latencies, 95), 2),
            max_latency_ms=round(max(latencies), 2) if latencies else 0.0,
            throughput_per_min=round(metrics.throughput_per_min(runs), 3),
            mean_prompt_chars=round(metrics.mean([r.tokens.prompt_chars for r in token_runs]), 2),
            mean_prompt_tokens=round(metrics.mean([r.tokens.prompt_tokens for r in token_runs]), 2),
            mean_output_tokens=round(metrics.mean([r.tokens.output_tokens for r in token_runs]), 2),
            token_source=metrics.dominant_token_source(runs),
            mean_confidence=metrics.mean_confidence(runs),
            recommendation_counts=metrics.count_values(r.recommendation for r in runs),
            model_usage=metrics.count_values(r.model_used for r in runs),
            error_counts=errors,
            timeout_count=sum(v for k, v in errors.items() if "Timeout" in k),
            stage_stats=metrics.stage_statistics(runs),
        )


def _first_int(raw: dict, keys: tuple[str, ...]) -> int | None:
    """Return the first integer value found under ``keys``."""
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int):
            return value
    return None


def _run_id(tickers: Sequence[str], model: str) -> str:
    """Stable identifier derived from the benchmark's inputs.

    Deterministic on purpose: the same ticker set against the same model always
    yields the same id, so two reports can be compared directly.
    """
    payload = "|".join([model, *sorted(tickers)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
