from __future__ import annotations

from models.analysis_result import TechnicalAnalysisResult
from services.prompt_sections.base import RenderedSection, SectionRenderer

# ---------------------------------------------------------------------------
# Field whitelist
#
# The renderer projects ONLY the fields below — a small, decision-relevant
# subset of TechnicalAnalysisResult. This is an explicit allowlist, never a
# model dump: fields added to TechnicalAnalysisResult in the future will NOT
# leak into the prompt unless this list is deliberately extended.
#
# INCLUDED (and why):
#   ticker, trend, strength, confidence, risk  -> the headline read a thesis
#       must be grounded in.
#   entry_zone, stop_loss, targets             -> the concrete trade plan the
#       upstream analysis already produced.
#   levels.support / levels.resistance         -> the price structure the LLM
#       reasons around.
#   signals                                    -> short, already-computed signal
#       phrases (human-readable, not raw numbers).
#   patterns (the True flags only)             -> qualitative pattern context.
#   summary                                    -> the upstream one-line synopsis.
#
# EXCLUDED (and why):
#   indicators (IndicatorSnapshot)  -> raw numeric indicator values (RSI, MACD,
#       Bollinger, EMAs, volume...). The analysis is already done; feeding raw
#       numbers invites the LLM to re-derive conclusions (re-doing technical
#       analysis), which is explicitly out of scope. The distilled trend /
#       strength / signals already encode them.
#   indicator_suite, market_structure, volume_analysis, volatility, smc,
#   confluence, confidence_breakdown, risk_plan (all dicts)  -> engine-internal
#       breakdowns. Large, verbose, and diagnostic; they would bloat the prompt
#       (cost, context-window, parse reliability) without improving the thesis.
#       risk_plan's *conclusions* are already surfaced via entry_zone/stop/targets.
#   reasoning  -> the technical engine's own reasoning trace. The AI agent
#       forms its OWN reasoning; echoing the engine's would bias/duplicate it.
#   agent, timestamp, metadata (AnalysisMetadata)  -> provenance/diagnostics
#       (execution_ms, bars_analyzed, warnings...). Irrelevant to the thesis,
#       and timestamp/execution_ms are non-deterministic — including them would
#       break the "identical input -> identical text" guarantee.
# ---------------------------------------------------------------------------

_TREND_LINE = "Trend: {trend} (strength {strength}/100)"


class TechnicalSectionRenderer(SectionRenderer):
    """Projects a ``TechnicalAnalysisResult`` into one prompt section.

    A pure formatting layer. The technical analysis has already been performed
    upstream; this renderer only *presents* a whitelisted subset of that result
    as concise, human-readable text for the LLM. It computes no indicators,
    performs no analysis, and makes no BUY/SELL/HOLD decision — it reads fields
    and lays them out.

    Deterministic by construction: it touches only stable, already-computed
    fields (no timestamps, no execution metrics, no randomness), so the same
    ``TechnicalAnalysisResult`` always yields identical text.
    """

    kind = "technical"
    version = "1.0"

    def render(self, model: TechnicalAnalysisResult, *, max_items: int) -> RenderedSection:
        """Render the technical section.

        ``max_items`` bounds how many support/resistance levels and signals are
        shown, so a pathologically long list cannot blow up the prompt. It is
        applied deterministically (the leading ``max_items`` entries, preserving
        the upstream ordering). ``truncated`` is set if any projected list was
        clipped.
        """
        lines: list[str] = []
        truncated = False

        # --- Headline read -------------------------------------------------
        lines.append(_TREND_LINE.format(trend=model.trend, strength=model.strength))
        lines.append(f"Confidence: {model.confidence}/100")
        lines.append(f"Risk: {model.risk}")

        # --- Trade plan (already produced upstream) ------------------------
        entry_low, entry_high = model.entry_zone
        lines.append(f"Entry zone: {entry_low} - {entry_high}")
        lines.append(f"Stop loss: {model.stop_loss}")
        if model.targets:
            shown_targets = model.targets[:max_items]
            truncated = truncated or len(model.targets) > len(shown_targets)
            lines.append("Targets: " + ", ".join(str(t) for t in shown_targets))

        # --- Price structure ----------------------------------------------
        support = model.levels.support[:max_items]
        resistance = model.levels.resistance[:max_items]
        truncated = truncated or len(model.levels.support) > len(support)
        truncated = truncated or len(model.levels.resistance) > len(resistance)
        lines.append("Support levels: " + (", ".join(str(s) for s in support) if support else "none identified"))
        lines.append("Resistance levels: " + (", ".join(str(r) for r in resistance) if resistance else "none identified"))

        # --- Signals (already-computed phrases) ---------------------------
        if model.signals:
            shown_signals = model.signals[:max_items]
            truncated = truncated or len(model.signals) > len(shown_signals)
            lines.append("Signals:")
            lines.extend(f"  - {s}" for s in shown_signals)

        # --- Active patterns (True flags only, stable order) --------------
        active = self._active_patterns(model)
        if active:
            lines.append("Patterns: " + ", ".join(active))

        # --- Upstream summary ---------------------------------------------
        if model.summary:
            lines.append(f"Summary: {model.summary}")

        # item_count reflects the primary payload (the distinct signals shown).
        item_count = len(model.signals[:max_items]) if model.signals else 0

        return RenderedSection(
            kind=self.kind,
            title="Technical Analysis",
            body="\n".join(lines),
            item_count=item_count,
            truncated=truncated,
        )

    @staticmethod
    def _active_patterns(model: TechnicalAnalysisResult) -> list[str]:
        """Return the names of pattern flags that are True, in a fixed order.

        Fixed ordering (not ``dict``/attribute iteration) keeps output
        deterministic and readable. Only active patterns are shown, to keep the
        section concise.
        """
        flags = model.patterns
        ordered = [
            ("breakout", flags.breakout),
            ("pullback", flags.pullback),
            ("trend_reversal", flags.trend_reversal),
            ("consolidation", flags.consolidation),
            ("high_volatility", flags.high_volatility),
        ]
        return [name for name, is_active in ordered if is_active]
