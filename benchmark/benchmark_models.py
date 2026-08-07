from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field


class BenchmarkStage(str, Enum):
    """The pipeline stages measured independently.

    ``TOTAL`` is wall-clock around the whole per-ticker sequence, so it also
    captures overhead that falls between stages.
    """

    TECHNICAL = "technical"
    NEWS = "news"
    PROMPT_BUILD = "prompt_build"
    INFERENCE = "inference"
    PARSE = "parse"
    TOTAL = "total"


class TokenSource(str, Enum):
    """Where token counts came from — measured counts beat estimates, and the
    report says which it used rather than blurring them together."""

    PROVIDER = "provider"    # exact counts reported by the backend
    ESTIMATED = "estimated"  # derived from character length


class StageTiming(BaseModel):
    """One stage's measured cost."""

    model_config = ConfigDict(frozen=True)

    stage: BenchmarkStage
    duration_ms: float = Field(..., ge=0.0)
    succeeded: bool = True
    error_type: str | None = Field(default=None, description="Exception class name on failure")


class TokenUsage(BaseModel):
    """Prompt/response size for one benchmarked call."""

    model_config = ConfigDict(frozen=True)

    prompt_chars: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    source: TokenSource = TokenSource.ESTIMATED

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


class TickerBenchmark(BaseModel):
    """Result of benchmarking a single ticker through the pipeline."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., min_length=1)
    succeeded: bool
    stages: tuple[StageTiming, ...] = ()
    tokens: TokenUsage | None = None
    recommendation: str | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    model_used: str | None = None
    prompt_version: str | None = None
    news_articles_included: int = Field(default=0, ge=0)
    failed_stage: BenchmarkStage | None = None
    error_type: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_ms(self) -> float:
        """Wall-clock for this ticker, taken from the TOTAL stage."""
        for timing in self.stages:
            if timing.stage is BenchmarkStage.TOTAL:
                return timing.duration_ms
        return 0.0

    def stage_ms(self, stage: BenchmarkStage) -> float | None:
        """Duration for ``stage``, or ``None`` if it was never reached."""
        for timing in self.stages:
            if timing.stage is stage:
                return timing.duration_ms
        return None


class StageStats(BaseModel):
    """Aggregated timings for one stage across all tickers."""

    model_config = ConfigDict(frozen=True)

    stage: BenchmarkStage
    samples: int = Field(..., ge=0)
    mean_ms: float = Field(..., ge=0.0)
    p50_ms: float = Field(..., ge=0.0)
    p95_ms: float = Field(..., ge=0.0)
    max_ms: float = Field(..., ge=0.0)


class BenchmarkSummary(BaseModel):
    """Aggregate view across every benchmarked ticker."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    generated_at: datetime
    model: str
    provider: str

    total_cases: int = Field(..., ge=0)
    succeeded: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)

    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    p50_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_latency_ms: float = Field(default=0.0, ge=0.0)
    max_latency_ms: float = Field(default=0.0, ge=0.0)
    throughput_per_min: float = Field(default=0.0, ge=0.0)

    mean_prompt_chars: float = Field(default=0.0, ge=0.0)
    mean_prompt_tokens: float = Field(default=0.0, ge=0.0)
    mean_output_tokens: float = Field(default=0.0, ge=0.0)
    token_source: TokenSource = TokenSource.ESTIMATED

    mean_confidence: float | None = None
    recommendation_counts: dict[str, int] = Field(default_factory=dict)
    model_usage: dict[str, int] = Field(default_factory=dict)
    error_counts: dict[str, int] = Field(default_factory=dict)
    timeout_count: int = Field(default=0, ge=0)
    stage_stats: tuple[StageStats, ...] = ()


class BenchmarkReport(BaseModel):
    """The complete, serializable benchmark artifact."""

    model_config = ConfigDict(frozen=True)

    summary: BenchmarkSummary
    runs: tuple[TickerBenchmark, ...] = ()
