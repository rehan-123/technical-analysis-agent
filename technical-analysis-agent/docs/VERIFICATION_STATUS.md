# Verification Status

_This document records what has been verified and how. The v2 upgrade was
developed in a network-restricted sandbox that could not install
`pydantic` / `pytest` / `fastapi`; verification was therefore done first via
offline harnesses, then reconciled against a real `pytest` run on the user's
machine._

## The real pytest run, and the bug it surfaced

A full `pip install` + `pytest -q` on a normal machine produced **45 passed,
1 failed**. The single failure was `test_bearish_synthetic_data_trends_bearish`
(expected Bearish/Strong Bearish, got Strong Bullish).

**Root cause (found, not patched around):** the failure was *not* in any
analysis engine. The `SyntheticDataProvider` generated price data with a pure
geometric random walk (`close = start * cumprod(1 + normal(drift, vol))`). For
the fixture parameters (`drift ≈ 1e-3`, `volatility ≈ 2e-2`), the accumulated
random-walk noise over ~252 bars is comparable to or larger than the
accumulated drift, so a nominally "bearish" (negative-drift) series could
realise as a *net-rising* series depending on the ticker/seed RNG stream. For
`ticker="AAPL"` it rose ~17–24%, ending above its entire moving-average stack —
so the engine **correctly** classified genuinely-rising data as bullish. The
generator simply wasn't honouring its own `drift` contract.

**Fix:** `SyntheticDataProvider` was made *trend-stationary* — a deterministic
trend component in the `drift` direction plus **mean-reverting AR(1) noise**
(which, being stationary, does not accumulate away from the trend). Direction
is now reliably controlled by `drift` while the data keeps realistic
autocorrelated swings and valid OHLC. This is a fix to the test double's
generation algorithm, not a change to any assertion or fixture seed (an earlier
seed swap was reverted — that had been patching the symptom).

## Post-fix verification (offline, real code)

With `pytest`/`pydantic` still uninstallable in-sandbox, the fix was verified by
running the **real test files** through a minimal offline runner
(`_diag/run_tests.py` during development; not shipped) that provides a small
`pytest` shim and duck-typed stand-ins for the pydantic models, while executing
the real engine / agent / service / indicator / provider code unmodified:

- **42 of the 46 tests pass**, including the previously-failing
  `test_bearish_synthetic_data_trends_bearish` (now Strong Bearish for the
  original seed=13 / AAPL fixture, via net −24% genuinely-bearish data).
- The remaining **4 tests are in `test_api.py`**, which needs FastAPI's real
  `TestClient`; they assert health status, response shape, ticker
  normalisation, and period rejection — none depend on trend direction, so the
  generator change cannot affect them, and they passed in the real run.
- A faithful full-agent reproduction (real ticker, full 21-indicator suite)
  confirmed the diagnosis and the fix, and a sweep across 8 tickers confirmed
  the generated direction is now reliable (every negative-drift stream ends
  −23% to −30%, every positive-drift stream +18% to +31%).
- Determinism preserved: identical `last_close` across repeated fresh processes
  with randomized `PYTHONHASHSEED`.

## Still owed once network is available

Run the authoritative suite on a normal machine:

```bash
pip install -r requirements.txt && pytest -q   # expect 46 passed
```

The offline runner does not exercise pydantic's own field validation or the
FastAPI HTTP stack (both unchanged by this fix). Everything else — all engine,
agent, service, indicator, and provider logic — was executed directly.

---

## What was verified earlier (engine logic, schema stability)

## Environment constraint

The sandbox's egress proxy rejected every host except `api.anthropic.com`
(confirmed against `pypi.org`, `files.pythonhosted.org`, `github.com`, and via
`apt`), so the runtime dependencies could not be installed. `pandas` and
`numpy` were preinstalled; everything below was verified with those alone.

## What WAS verified (against real production code)

A verification harness imported and executed the **real** engine, indicator,
service, and agent modules — using a small import shim that supplies duck-typed
stand-ins only where a `pydantic`/`Settings` object would sit (the config
values were extracted from the real `Settings` class via AST, so there is no
drift between the stand-in and the real defaults). Under that harness:

- **All 17 new indicators** compute finite, sensible values.
- **All 9 engines** (market structure, candlestick, volatility, volume, SMC,
  confluence, confidence, risk, explanation) run end-to-end.
- **The trend-anchor fix** — the original bug — is resolved: a primary
  downtrend no longer mislabels as Bullish, and a primary uptrend does not
  mislabel as Bearish, verified against the **exact** `SyntheticDataProvider`
  fixtures the real pytest suite uses.
- **Trend/risk-direction consistency**: the headline `trend` and
  `risk_plan.direction` are derived from one canonical bias and can never
  disagree — verified exhaustively across all 101 possible `net_score` values,
  including the band boundaries (a boundary off-by-one was found and fixed this
  way).
- **Confluence within-category normalization**: no category contributes more
  than its configured weight, so redundant short-term indicators can't outvote
  the primary trend.
- **Determinism**: identical inputs produce byte-identical `trend` /
  `net_score` / `confidence` across repeated fresh processes with randomized
  `PYTHONHASHSEED` (the SHA-256 seed fix holds through the full pipeline).
- **OHLCV validator**: clean/empty/missing-column/negative-price/duplicate-
  timestamp/unsorted/incoherent-bar cases all behave as specified.
- **Risk plan**: long and short plans are internally consistent (stop side,
  target ordering, positive position size).
- **SMC** equal-levels are plain Python floats (no `np.float64` leak that would
  break JSON serialization).
- **Schema stability** (AST-level structural diff against the v1 zip): all 15
  original `TechnicalAnalysisResult` fields are present with identical types;
  the 10 new fields are all optional/defaulted (additive, non-breaking); and
  `IndicatorSnapshot`, `SupportResistanceLevels`, `PatternFlags`,
  `models/requests.py`, and `api/routes.py` are byte-for-byte unchanged.

## What was NOT verified (still owed once network returns)

- **The real `pytest` suite has not been run.** New tests were authored
  (`tests/test_institutional_engines.py`) and their assertions were validated
  against real code via the harness, but `pytest` itself never executed. The
  original 22 tests likewise could not be re-run under `pytest`.
- **`pydantic` model instantiation / validation** was not exercised
  (`IndicatorResult`, the extended `TechnicalAnalysisResult`, `AnalysisMetadata`
  were checked structurally via AST, not constructed).
- **FastAPI routes** (`api/routes.py`, `main.py`) were confirmed unchanged and
  import-compatible, but the HTTP layer was not run.
- **`yfinance`** live path — unchanged from v1, still not reachable in-sandbox.

## To finish verification

```bash
pip install -r requirements.txt   # needs network
pytest -q                         # expect the original 22 + new engine tests green
```

If any test fails, the most likely spots are (a) a `pydantic` v2 validation
constraint the AST check couldn't see, or (b) a fixture assumption in the new
tests. Both are quick to resolve; the underlying engine logic has been
exercised directly.
