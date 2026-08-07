from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from benchmark.benchmark_models import (
    BenchmarkStage,
    StageStats,
    TickerBenchmark,
    TokenSource,
)

#: Average characters per token for English prose with embedded JSON. Used only
#: when the backend does not report exact counts; ``TokenSource`` records which
#: path produced a figure so estimates are never mistaken for measurements.
DEFAULT_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from character length.

    Deliberately simple and dependency-free: a real tokenizer would tie the
    benchmark to one model family, and the backend's own counters are preferred
    whenever they are available.
    """
    if not text or chars_per_token <= 0:
        return 0
    return round(len(text) / chars_per_token)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile of ``values`` (``p`` in 0..100).

    Implemented directly rather than pulling in numpy: it keeps the metrics
    module pure, dependency-free, and trivially deterministic.
    """
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def success_rate(runs: Sequence[TickerBenchmark]) -> float:
    """Fraction of runs that produced a validated result (0.0-1.0)."""
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.succeeded) / len(runs)


def throughput_per_min(runs: Sequence[TickerBenchmark]) -> float:
    """Successful analyses per minute, derived from measured wall-clock.

    Uses summed per-ticker latency rather than a separate stopwatch so the
    figure stays consistent with the latency statistics in the same report.
    """
    successful = [r for r in runs if r.succeeded]
    total_ms = sum(r.total_ms for r in successful)
    if not successful or total_ms <= 0:
        return 0.0
    return len(successful) / (total_ms / 60_000.0)


def stage_statistics(runs: Sequence[TickerBenchmark]) -> tuple[StageStats, ...]:
    """Per-stage aggregates, in the fixed stage order for deterministic reports.

    Stages that were never reached are omitted rather than reported as zero,
    which would understate the mean.
    """
    stats: list[StageStats] = []
    for stage in BenchmarkStage:
        samples = [ms for r in runs if (ms := r.stage_ms(stage)) is not None]
        if not samples:
            continue
        stats.append(
            StageStats(
                stage=stage,
                samples=len(samples),
                mean_ms=round(mean(samples), 2),
                p50_ms=round(percentile(samples, 50), 2),
                p95_ms=round(percentile(samples, 95), 2),
                max_ms=round(max(samples), 2),
            )
        )
    return tuple(stats)


def count_values(values: Iterable[str | None]) -> dict[str, int]:
    """Count non-null values, key-sorted so serialized output is stable."""
    counts = Counter(v for v in values if v)
    return dict(sorted(counts.items()))


def mean_confidence(runs: Sequence[TickerBenchmark]) -> float | None:
    """Mean confidence across successful runs, or ``None`` if none succeeded."""
    values = [float(r.confidence) for r in runs if r.succeeded and r.confidence is not None]
    return round(mean(values), 2) if values else None


def dominant_token_source(runs: Sequence[TickerBenchmark]) -> TokenSource:
    """Report measured counts only when every sampled run supplied them.

    Mixing exact and estimated counts into one average would misrepresent the
    result, so a single estimate downgrades the whole figure.
    """
    sources = [r.tokens.source for r in runs if r.tokens is not None]
    if sources and all(s is TokenSource.PROVIDER for s in sources):
        return TokenSource.PROVIDER
    return TokenSource.ESTIMATED
