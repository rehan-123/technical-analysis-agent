# Architecture Audit & Improvement Roadmap

_Technical Analysis Agent — audit performed before the institutional-grade upgrade._

This document is deliverable #1 and #2. It audits the pre-upgrade codebase
(1,644 LOC, 22 tests, all green) and lays out a priority-ordered roadmap.
It is deliberately honest about what is genuinely solid, what is a v1
heuristic, and what is out of reach without live infrastructure — because
overclaiming in a system that emits trade parameters is a real hazard, not
a stylistic one.

---

## 1. Current strengths

- **Clean layering.** `data → indicators → services → agent → api` is a
  correct dependency direction with no upward imports. The data source is
  already dependency-inverted behind `MarketDataProvider`.
- **Deterministic indicators** implemented natively in pandas/numpy (no
  TA-Lib build dependency), each behind a uniform `Indicator` interface.
- **Config-driven.** No magic numbers in logic; everything routes through
  `config/settings.py` (Pydantic settings, env-overridable).
- **Orchestration hook exists.** `BaseAgent.run()` is the seam a Chief
  Decision Agent will consume.
- **Tests are offline and deterministic** (synthetic provider with a stable
  hash seed), so CI won't flake or depend on Yahoo Finance.

## 2. Weaknesses & design smells

| # | Issue | Impact |
|---|-------|--------|
| W1 | Indicators return bare Series/DataFrames — no `value / signal / strength / interpretation`. | Downstream agents must re-derive meaning; no uniform signal contract. |
| W2 | Only 7 indicators. No momentum breadth (ADX/StochRSI/CCI/MFI), no volume-flow (OBV/CMF), no channel/trend overlays (Keltner/Donchian/SuperTrend/Ichimoku/PSAR/Pivots). | Thin evidence base for confluence. |
| W3 | Trend + scoring logic is a single hand-tuned additive function. | Not decomposable, hard to weight or explain per-factor. |
| W4 | No market-structure primitives (swings, HH/HL/LH/LL, BOS, CHoCH). | Can't reason about structure, the backbone of the requested SMC/price-action work. |
| W5 | No candlestick engine, no volatility-regime engine, thin volume analysis. | Missing whole evidence categories. |
| W6 | Confidence == strength (circular). | Confidence isn't independently derived. |
| W7 | Risk output has no R:R, position sizing, EV, or explicit invalidation. | Not decision-ready for a Risk/Portfolio agent. |
| W8 | No human-readable explanation. Output is signals + numbers. | Violates the "never just BUY/SELL" requirement. |
| W9 | Validation is minimal (`validate_ohlcv` only checks columns/empties). | No NaN/duplicate/negative-price/monotonic-timestamp guards. |
| W10 | `services/` mixes engine logic and naming; no `engines/` package. | Weak module boundaries as the engine count grows. |

## 3. Performance profile

- Current bottleneck is I/O (yfinance), not compute; indicators are already
  vectorized. As indicator count grows ~4x, the risk is **redundant EMA/ATR
  recomputation** across indicators (MACD, Keltner, SuperTrend all need
  EMA/ATR). Mitigation: a per-request **computation cache** so each base
  series is computed once. Benchmarked in `docs/BENCHMARKS.md`.

## 4. Scalability concerns

- Single shared agent instance is fine (stateless), but the indicator
  engine rebuilds indicator objects per construction. Keep engines
  stateless and inject a per-request cache rather than memoizing globally
  (avoids cross-ticker leakage).

## 5. Gaps vs. the institutional spec

- **Missing indicators:** WMA, VWMA, VWAP, StochRSI, ADX, CCI, ROC,
  Momentum, MFI, OBV, CMF, Keltner, Donchian, SuperTrend, Ichimoku, PSAR,
  Pivot Points.
- **Missing engines:** market structure, candlesticks, volume intelligence,
  volatility regime, SMC, confluence, confidence, risk, explanation.
- **Missing infra:** dedicated validation module, structured/timed logging,
  performance tests, architecture/API/indicator docs.

## 6. Honesty ledger (what is *not* claimed to be institutional-grade)

These are built as **transparent v1 heuristics** and labeled as such in code
and output. They are useful and deterministic, but there is no canonical,
proven algorithm for them, and they should not be represented otherwise:

- **Smart Money Concepts** — Fair Value Gaps and equal highs/lows are
  mechanically well-defined and implemented faithfully. Order blocks,
  breaker/mitigation blocks, liquidity sweeps, inducement, and displacement
  are subjective in the trading literature; this pass implements FVG,
  equal-H/L, premium/discount, and a basic order-block heuristic only, each
  emitting an explicit `heuristic: true` flag and a confidence.
- **Chart patterns** (H&S, cup-and-handle, wedges, the full zoo) — reliable
  algorithmic detection is an open problem. This pass ships the tractable
  subset (double top/bottom, channels/triangles via trendline regression,
  flag continuation) and marks the rest as roadmap.
- **Confidence score** — a deterministic weighted blend of engine outputs.
  It is a *transparency device*, **not** a probability of profit and not
  statistically validated. Documented explicitly in the API and README.
- **Live multi-timeframe** — the MTF *merge architecture* is built and
  tested against resampled data; genuine multi-TF requires a live feed
  (unavailable in the build sandbox), so it's wired but not benchmarked
  end-to-end.

## 7. Roadmap (priority-ordered)

**Phase 1 — Signal contract & indicator breadth** _(this pass)_
1. `IndicatorResult` contract: `value, signal, strength, interpretation, confidence_contribution`.
2. Per-request computation cache to kill redundant EMA/ATR work.
3. 17 new indicators, each emitting an `IndicatorResult`.

**Phase 2 — Analytical engines** _(this pass)_
4. Market Structure (swings, HH/HL/LH/LL, BOS, CHoCH).
5. Candlestick engine (13 classic patterns).
6. Volatility engine (ATR expansion/compression, BB squeeze, regime).
7. Volume engine (rvol, spike, climax, dry-up, pressure, volume-profile POC/HVN/LVN).
8. SMC engine (FVG, equal H/L, premium/discount, basic order blocks — heuristic).

**Phase 3 — Decision layer** _(this pass)_
9. Confluence engine (weighted bull/bear scoring across all engines).
10. Confidence engine (mathematically derived, with per-engine breakdown).
11. Risk engine (entry, SL, TP1–3, R:R, position size, EV, invalidation).
12. Explanation engine (human-readable narrative).

**Phase 4 — Hardening** _(this pass)_
13. Dedicated `validation/` module (NaN, dupes, negatives, monotonic ts, gaps).
14. Structured + timed logging.
15. Tests for every new module; performance test.
16. Docs: audit (this file), API, indicator reference, benchmarks.

**Phase 5 — Future** _(roadmapped, not this pass)_
17. Live multi-timeframe fetch + alignment across 1m→1M.
18. Full chart-pattern recognition (H&S, cup-and-handle, wedges).
19. Full SMC suite (breaker/mitigation blocks, liquidity sweeps, inducement, displacement).
20. Learned confidence model to replace the heuristic blend.
21. Delta/footprint from tick data; true volume profile from intraday prints.
