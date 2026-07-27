# Technical Analysis Agent

A standalone, single-responsibility agent that analyzes **only** technical
(price/volume) structure for a stock or crypto ticker. It is the first of
several planned specialist agents (News, Risk, Macro, Options, ...) in a
larger multi-agent investment platform, and is built to be orchestrated by
a future **Chief Decision Agent** — either called in-process as a Python
object, or over HTTP.

## What it does

Given a ticker (`AAPL`, `BTC-USD`, `TSLA`, ...), the agent:

1. Fetches and validates historical OHLCV data (via `yfinance` by default).
2. Computes ~24 indicators — moving averages (SMA/EMA/WMA/VWMA/VWAP), momentum
   (RSI, StochRSI, MACD, ADX, CCI, ROC, Momentum), volume-flow (OBV, CMF, MFI),
   and overlays (Bollinger, Keltner, Donchian, SuperTrend, Parabolic SAR,
   Ichimoku, Pivots) — each emitting a structured signal/strength/interpretation.
3. Runs analytical engines: market structure (swings, HH/HL/LH/LL, BOS, CHoCH),
   candlestick patterns, volatility regime, volume intelligence (with a
   volume-by-price profile / POC), and a heuristic Smart Money Concepts pass.
4. Aggregates everything through a weighted **confluence engine**, derives a
   transparent **confidence** score, and anchors the headline trend to the
   primary (slow) trend so a short-term bounce can't flip the label.
5. Produces an actionable entry zone, ATR+structure stop loss, TP1–3 with R:R,
   position sizing, invalidation, and risk tier — plus a human-readable
   reasoning chain (never a bare BUY/SELL).
6. Returns everything as a single validated `TechnicalAnalysisResult` (JSON),
   whose top-level contract is unchanged from v1 (all richer detail is additive).

> **Not financial advice.** The confidence score is a transparent, deterministic
> measure of how internally consistent and one-sided the current technical
> evidence is — **not** a probability of profit, and not statistically
> validated. SMC detections are labeled heuristics. See
> `docs/ARCHITECTURE_AUDIT.md` for the full honesty ledger and
> `docs/VERIFICATION_STATUS.md` for exactly what has been verified.

## Architecture

```
technical-analysis-agent/
├── agent/                      # Orchestration-ready agent (the "brain")
│   ├── base.py                 #   BaseAgent interface — implemented by every specialist agent
│   └── technical_analysis_agent.py
├── api/                        # Transport layer (HTTP) — thin wrapper around the agent
│   └── routes.py
├── config/
│   └── settings.py             # Every threshold/period/multiplier — no hardcoded values
├── data/                       # Data sources, swappable via a common interface
│   ├── base.py                 #   MarketDataProvider interface
│   ├── yfinance_provider.py    #   Production data source
│   └── synthetic_provider.py   #   Deterministic generator for tests/offline dev
├── indicators/                 # Independent, single-purpose indicator classes
│   ├── base.py                 #   Indicator interface
│   ├── moving_averages.py      #   SMA, EMA
│   ├── rsi.py / macd.py / atr.py / bollinger.py / volume.py
├── models/                     # Pydantic contracts
│   ├── requests.py             #   Input schema
│   ├── analysis_result.py      #   Output schema (the JSON contract)
│   └── market_data.py          #   OHLCV validation
├── services/                   # Business logic composed from indicators
│   ├── indicator_engine.py     #   Runs & merges all indicators
│   ├── trend_service.py        #   Direction + strength score
│   ├── support_resistance_service.py
│   ├── pattern_service.py      #   Breakout / pullback / reversal / consolidation / volatility
│   └── scoring_service.py      #   Entry zone, stop loss, targets, risk, confidence
├── utils/                      # Logging, exceptions
├── tests/                      # pytest suite (22 tests, no network required)
├── main.py                     # FastAPI entrypoint
├── requirements.txt
└── .env.example
```

**Design principles:**

- **Dependency inversion for data**: `agent` and `services` never import
  `yfinance` directly — they depend on the `MarketDataProvider` interface.
  Swapping to a different data source (a broker API, a paid real-time feed)
  means writing one new class in `data/`; nothing else changes.
- **Independent indicators, no duplicated logic**: each indicator in
  `indicators/` is a self-contained class that only knows how to compute
  itself from raw OHLCV. `IndicatorEngine` composes them; adding a new
  indicator means adding one line there.
- **No hardcoded values**: every period, multiplier, and threshold lives in
  `config/settings.py`, overridable via environment variables.
- **Orchestration-ready**: `TechnicalAnalysisAgent` implements `BaseAgent.run()`.
  A Chief Decision Agent can hold a list of `BaseAgent` instances (Technical,
  News, Risk, ...) and call `await agent.run(ticker)` on each, polymorphically,
  without knowing anything about their internals.
- **TA-Lib / pandas-ta**: indicators are implemented natively in pandas/numpy
  rather than via TA-Lib or pandas-ta. This was a deliberate choice — it
  avoids TA-Lib's C-library build dependency and pandas-ta's numpy 2.x
  incompatibilities, while keeping the formulas auditable. The `Indicator`
  interface makes it a drop-in swap if you'd rather delegate to one of
  those libraries later.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional — defaults work out of the box
```

## Running the API

```bash
uvicorn main:app --reload
```

```bash
curl "http://localhost:8000/analyze/AAPL?period=1y&interval=1d"

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BTC-USD", "period": "1y", "interval": "1d"}'

curl http://localhost:8000/health
```

## Using it in-process (how a Chief Decision Agent would call it)

```python
from agent.technical_analysis_agent import TechnicalAnalysisAgent

agent = TechnicalAnalysisAgent()
result = await agent.run("AAPL")   # BaseAgent interface
# result.trend, result.strength, result.signals, result.entry_zone, ...
```

A Chief Decision Agent would typically hold a roster like:

```python
agents: list[BaseAgent] = [TechnicalAnalysisAgent(), NewsAgent(), RiskAgent(), ...]
results = {a.name: await a.run(ticker) for a in agents}
```

## Example output

```json
{
  "agent": "technical_analysis_agent",
  "ticker": "AAPL",
  "timestamp": "2026-07-21T10:00:00Z",
  "trend": "Bullish",
  "strength": 72,
  "signals": ["EMA20 above EMA50", "EMA50 above EMA200", "RSI 55", "Volume Increasing"],
  "entry_zone": [280.72, 282.13],
  "stop_loss": 273.77,
  "targets": [289.09, 296.73, 304.5],
  "risk": "Low",
  "confidence": 72,
  "summary": "Technical structure is bullish. Overall risk assessed as low.",
  "indicators": { "close": 281.43, "ema_20": 278.82, "rsi": 54.66, "...": "..." },
  "levels": { "support": [267.78, 259.57], "resistance": [296.73] },
  "patterns": { "breakout": false, "pullback": false, "trend_reversal": false, "consolidation": false, "high_volatility": false }
}
```

The top-level fields match the minimal contract the platform expects.
`indicators`, `levels`, and `patterns` are additive — richer structured
detail a Chief Decision Agent can cross-reference against other agents'
output without recomputing anything.

## Configuration

All tunables live in `config/settings.py` and can be overridden via `.env`
(see `.env.example`), e.g.:

```
TA_RSI_PERIOD=14
TA_EMA_FAST_PERIOD=20
TA_ATR_STOP_MULTIPLIER=1.5
TA_RISK_MEDIUM_ATR_PCT=0.04
```

## Testing

```bash
pytest -v
```

22 tests, all offline — they use `SyntheticDataProvider` (a seeded random
walk implementing the same `MarketDataProvider` interface as production)
instead of hitting Yahoo Finance, so the suite is fast and deterministic.
Coverage includes: each indicator in isolation, trend classification on
known up/down trends, support/resistance clustering, pattern detection
(breakout, consolidation), the full agent pipeline, and the API layer.

## Extending

- **New indicator**: implement `Indicator` in `indicators/`, register it
  in `IndicatorEngine._build_indicators()`.
- **New pattern**: add a check to `PatternService.evaluate()` and a flag
  to `PatternFlags`.
- **New data source**: implement `MarketDataProvider` in `data/`, pass it
  to `TechnicalAnalysisAgent(data_provider=...)`.
- **New specialist agent** (News, Risk, Macro, Options): implement
  `BaseAgent` from `agent/base.py` following the same pattern as
  `TechnicalAnalysisAgent` — self-contained, own data source, one `run()` method.

## Known limitations

- `yfinance` requires outbound network access to Yahoo Finance; it wasn't
  reachable in the sandbox this was built in, so end-to-end validation used
  `SyntheticDataProvider`. The `YFinanceProvider` code path is standard
  `yfinance` usage and should work as-is in an environment with normal
  internet access.
- Trend strength and confidence scoring are transparent, configurable
  heuristics (not a trained model) — reasonable for a v1, but a natural
  place to later swap in a learned scoring model without touching any
  other layer.
