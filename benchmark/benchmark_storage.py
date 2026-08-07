from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from benchmark.benchmark_models import BenchmarkReport, BenchmarkStage

#: Fixed CSV column order. Declared explicitly (never derived from dict order)
#: so a schema change is a visible edit rather than a silent column shuffle.
CSV_COLUMNS: tuple[str, ...] = (
    "ticker", "succeeded", "recommendation", "confidence", "model_used",
    "prompt_version", "news_articles_included", "prompt_chars", "prompt_tokens",
    "output_tokens", "token_source", "technical_ms", "news_ms", "prompt_build_ms",
    "inference_ms", "parse_ms", "total_ms", "failed_stage", "error_type",
)


class BenchmarkStorage:
    """Writes benchmark artifacts to disk deterministically.

    Every serializer here is byte-stable for a given report: keys are sorted,
    columns are fixed, floats are rounded at the model boundary, and newlines
    are normalized to ``\\n``. Two runs with identical measurements produce
    identical files, so reports can be diffed and committed.
    """

    JSON_NAME = "benchmark.json"
    CSV_NAME = "benchmark.csv"

    def __init__(self, output_dir: Path | str) -> None:
        self._output_dir = Path(output_dir)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def write_all(self, report: BenchmarkReport, *, summary_markdown: str | None = None) -> tuple[Path, ...]:
        """Write every artifact and return the paths, in a stable order."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        written = [self.write_json(report), self.write_csv(report)]
        if summary_markdown is not None:
            written.append(self.write_markdown(summary_markdown))
        return tuple(written)

    def write_json(self, report: BenchmarkReport) -> Path:
        """Serialize the full report, sorted-key and pretty-printed."""
        path = self._output_dir / self.JSON_NAME
        payload = report.model_dump(mode="json")
        self._write(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        return path

    def write_csv(self, report: BenchmarkReport) -> Path:
        """Serialize one row per ticker, in the order they were benchmarked."""
        path = self._output_dir / self.CSV_NAME
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for run in report.runs:
            writer.writerow(self.row_for(run))
        self._write(path, buffer.getvalue())
        return path

    def write_markdown(self, markdown: str) -> Path:
        path = self._output_dir / "benchmark_summary.md"
        self._write(path, markdown if markdown.endswith("\n") else markdown + "\n")
        return path

    @staticmethod
    def row_for(run) -> dict[str, object]:
        """Flatten one benchmark into its CSV row."""
        tokens = run.tokens
        return {
            "ticker": run.ticker,
            "succeeded": run.succeeded,
            "recommendation": run.recommendation or "",
            "confidence": run.confidence if run.confidence is not None else "",
            "model_used": run.model_used or "",
            "prompt_version": run.prompt_version or "",
            "news_articles_included": run.news_articles_included,
            "prompt_chars": tokens.prompt_chars if tokens else "",
            "prompt_tokens": tokens.prompt_tokens if tokens else "",
            "output_tokens": tokens.output_tokens if tokens else "",
            "token_source": tokens.source.value if tokens else "",
            "technical_ms": _ms(run, BenchmarkStage.TECHNICAL),
            "news_ms": _ms(run, BenchmarkStage.NEWS),
            "prompt_build_ms": _ms(run, BenchmarkStage.PROMPT_BUILD),
            "inference_ms": _ms(run, BenchmarkStage.INFERENCE),
            "parse_ms": _ms(run, BenchmarkStage.PARSE),
            "total_ms": run.total_ms,
            "failed_stage": run.failed_stage.value if run.failed_stage else "",
            "error_type": run.error_type or "",
        }

    @staticmethod
    def _write(path: Path, content: str) -> None:
        """UTF-8, LF newlines — identical bytes on every platform."""
        path.write_text(content, encoding="utf-8", newline="\n")


def _ms(run, stage: BenchmarkStage) -> float | str:
    value = run.stage_ms(stage)
    return value if value is not None else ""
