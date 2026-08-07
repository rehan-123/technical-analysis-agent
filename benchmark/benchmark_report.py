from __future__ import annotations

from benchmark.benchmark_models import BenchmarkReport, TokenSource


def render_markdown(report: BenchmarkReport) -> str:
    """Render a human-readable benchmark summary.

    Deterministic: every collection is emitted in sorted or measured order, and
    all figures come pre-rounded from the model, so the same report always
    renders the same bytes.
    """
    s = report.summary
    lines: list[str] = [
        "# AI Pipeline Benchmark",
        "",
        f"- **Run ID**: `{s.run_id}`",
        f"- **Generated**: {s.generated_at.isoformat()}",
        f"- **Provider / model**: {s.provider or 'unconfigured'} / {s.model}",
        "",
        "## Results",
        "",
        f"| Cases | Succeeded | Failed | Success rate |",
        f"|---|---|---|---|",
        f"| {s.total_cases} | {s.succeeded} | {s.failed} | {s.success_rate:.1%} |",
        "",
        "## Latency",
        "",
        "| Mean | p50 | p95 | Max | Throughput |",
        "|---|---|---|---|---|",
        f"| {s.mean_latency_ms:.0f} ms | {s.p50_latency_ms:.0f} ms | "
        f"{s.p95_latency_ms:.0f} ms | {s.max_latency_ms:.0f} ms | "
        f"{s.throughput_per_min:.2f}/min |",
        "",
    ]

    if s.stage_stats:
        lines += ["## Stage breakdown", "",
                  "| Stage | Samples | Mean | p50 | p95 | Max |",
                  "|---|---|---|---|---|---|"]
        lines += [
            f"| {st.stage.value} | {st.samples} | {st.mean_ms:.0f} ms | "
            f"{st.p50_ms:.0f} ms | {st.p95_ms:.0f} ms | {st.max_ms:.0f} ms |"
            for st in s.stage_stats
        ]
        lines.append("")

    qualifier = "measured by provider" if s.token_source is TokenSource.PROVIDER else "estimated from characters"
    lines += [
        "## Prompt & response size",
        "",
        f"Token counts are **{qualifier}**.",
        "",
        "| Mean prompt chars | Mean prompt tokens | Mean output tokens |",
        "|---|---|---|",
        f"| {s.mean_prompt_chars:.0f} | {s.mean_prompt_tokens:.0f} | {s.mean_output_tokens:.0f} |",
        "",
        "## Output quality",
        "",
        f"- Mean confidence: {f'{s.mean_confidence:.1f}' if s.mean_confidence is not None else 'n/a'}",
    ]
    lines += _counts_block("Recommendations", s.recommendation_counts)
    lines += _counts_block("Model usage", s.model_usage)

    lines += ["", "## Reliability", "", f"- Timeouts: {s.timeout_count}"]
    if s.error_counts:
        lines += [f"- {name}: {count}" for name, count in s.error_counts.items()]
    else:
        lines.append("- No errors recorded")

    lines += ["", "## Per-ticker", "",
              "| Ticker | OK | Recommendation | Confidence | Total |",
              "|---|---|---|---|---|"]
    lines += [
        f"| {r.ticker} | {'yes' if r.succeeded else 'no'} | "
        f"{r.recommendation or (r.failed_stage.value if r.failed_stage else '-')} | "
        f"{r.confidence if r.confidence is not None else '-'} | {r.total_ms:.0f} ms |"
        for r in report.runs
    ]
    return "\n".join(lines) + "\n"


def _counts_block(title: str, counts: dict[str, int]) -> list[str]:
    """Render a counted breakdown; counts arrive already key-sorted."""
    if not counts:
        return []
    return ["", f"**{title}**", ""] + [f"- {name}: {count}" for name, count in counts.items()]
