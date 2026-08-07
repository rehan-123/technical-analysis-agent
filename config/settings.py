from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Technical Analysis Agent.

    Every threshold, period, and multiplier used anywhere in the agent is
    defined here so that no module contains hardcoded "magic numbers".
    Values can be overridden via environment variables (prefixed ``TA_``)
    or a ``.env`` file — see ``.env.example``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TA_", extra="ignore")

    # --- Data fetching ---
    default_period: str = "1y"
    default_interval: str = "1d"
    min_bars_required: int = 60

    # --- Moving averages ---
    ema_fast_period: int = 20
    ema_medium_period: int = 50
    ema_long_period: int = 200
    sma_period: int = 50

    # --- RSI ---
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # --- MACD ---
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- ATR ---
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_target_multipliers: list[float] = Field(default_factory=lambda: [1.5, 3.0, 4.5])

    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # --- Volume ---
    volume_sma_period: int = 20
    volume_spike_multiplier: float = 1.5
    volume_trend_high: float = 1.1
    volume_trend_low: float = 0.9

    # --- Support / Resistance ---
    sr_lookback_bars: int = 120
    sr_swing_order: int = 5
    sr_cluster_tolerance_pct: float = 0.008  # 0.8%
    sr_max_levels: int = 3

    # --- Pattern detection ---
    breakout_buffer_pct: float = 0.003
    consolidation_bandwidth_percentile: float = 0.30
    high_volatility_atr_multiplier: float = 1.5
    reversal_lookback_bars: int = 5
    pullback_rsi_low: float = 40.0
    pullback_rsi_high: float = 55.0

    # --- Risk thresholds (ATR as a % of price) ---
    risk_low_atr_pct: float = 0.02
    risk_medium_atr_pct: float = 0.04

    # --- Scoring ---
    entry_zone_buffer_pct: float = 0.0025
    bullish_strength_threshold: int = 50

    # --- Extended trend/overlay indicators ---
    wma_period: int = 20
    vwma_period: int = 20
    vwap_period: int = 20
    donchian_period: int = 20
    keltner_ema_period: int = 20
    keltner_atr_period: int = 10
    keltner_multiplier: float = 2.0
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    psar_step: float = 0.02
    psar_max_step: float = 0.2
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52

    # --- Extended momentum indicators ---
    roc_period: int = 12
    momentum_period: int = 10
    cci_period: int = 20
    cci_overbought: float = 100.0
    cci_oversold: float = -100.0
    stochrsi_period: int = 14
    stochrsi_smooth_k: int = 3
    stochrsi_smooth_d: int = 3
    adx_period: int = 14
    adx_trend_threshold: float = 25.0

    # --- Volume-flow indicators ---
    mfi_period: int = 14
    mfi_overbought: float = 80.0
    mfi_oversold: float = 20.0
    cmf_period: int = 20

    # --- Market structure ---
    ms_swing_order: int = 5
    ms_lookback_bars: int = 150

    # --- Candlestick ---
    candle_doji_body_pct: float = 0.1
    candle_long_body_pct: float = 0.6
    candle_shadow_ratio: float = 2.0

    # --- Volatility engine ---
    vol_squeeze_lookback: int = 120
    vol_squeeze_percentile: float = 0.25
    vol_expansion_multiplier: float = 1.5
    vol_regime_lookback: int = 100

    # --- Volume profile ---
    volume_profile_bins: int = 24
    volume_profile_value_area_pct: float = 0.70

    # --- Smart Money Concepts (heuristic) ---
    smc_fvg_min_gap_pct: float = 0.001
    smc_equal_level_tolerance_pct: float = 0.0015
    smc_lookback_bars: int = 120

    # --- Confluence engine weights (relative; normalized internally) ---
    weight_trend: float = 0.22
    weight_momentum: float = 0.18
    weight_structure: float = 0.16
    weight_volume: float = 0.12
    weight_volatility: float = 0.08
    weight_candlestick: float = 0.08
    weight_smc: float = 0.10
    weight_pattern: float = 0.06

    # --- Risk engine ---
    risk_account_size: float = 10000.0
    risk_per_trade_pct: float = 0.01
    risk_reward_targets: list[float] = Field(default_factory=lambda: [1.5, 2.5, 4.0])
    risk_win_rate_assumption: float = 0.5  # used only for EV illustration

    # --- Logging ---
    log_level: str = "INFO"
    enable_timing_logs: bool = True

    # --- Data acquisition ---
    data_request_timeout: float = 15.0
    data_max_retries: int = 3
    data_retry_backoff: float = 0.75  # seconds, multiplied by 2**attempt
    # curl_cffi browser-impersonation target used by yfinance. Configurable
    # because a mismatched/blocked TLS fingerprint is a common cause of
    # connection resets against Yahoo behind inspecting proxies/CDNs.
    yfinance_impersonate: str = "chrome"
    # Ordered fallback chain of data sources. Only real sources are listed;
    # the synthetic generator is deliberately excluded so fabricated prices
    # can never silently feed a live analysis. Override via TA_DATA_SOURCES.
    data_sources: list[str] = Field(default_factory=lambda: ["yfinance", "stooq"])

    # --- News Agent ---
    # Centralized here (rather than a separate news_settings module) so the
    # platform keeps one configuration surface. All of these inherit the
    # class-level ``TA_`` env prefix, e.g. TA_NEWS_FINNHUB_API_KEY.
    #
    # Ordered list of news sources, mirroring ``data_sources`` above. Only
    # "finnhub" is implemented today; the list form means adding a source
    # later is a configuration change, not a service-layer change.
    news_sources: list[str] = Field(default_factory=lambda: ["finnhub"])
    news_finnhub_api_key: str = Field(default="", description="Finnhub API key; required when finnhub is enabled")
    news_finnhub_base_url: str = "https://finnhub.io/api/v1"

    # Transport behaviour for news retrieval (kept separate from market-data
    # transport settings so the two can be tuned independently).
    news_request_timeout: float = 10.0
    news_max_retries: int = 3
    news_retry_backoff: float = 0.5  # seconds, multiplied by 2**attempt

    # Defaults applied when a NewsRequest does not specify them.
    news_default_lookback_days: int = Field(default=7, ge=1, le=365)
    news_default_limit: int = Field(default=50, ge=1, le=250)
    news_default_language: str | None = Field(
        default=None, description="Optional ISO-639-1 filter, e.g. 'en'; None disables language filtering"
    )

    # Deterministic pipeline tuning (consumed by NewsService).
    news_deduplicate: bool = True
    # Two articles with the same normalized title published within this window
    # are treated as the same story (catches syndicated reprints across
    # outlets). Only used for the secondary title-based dedup key; exact URL
    # matches are always deduplicated regardless.
    news_dedup_time_window_minutes: int = Field(default=60, ge=1, le=1440)

    # --- AI Analysis Agent (LLM) ---
    # Centralized here, same as the news_* block, so the platform keeps one
    # configuration surface. All inherit the class-level ``TA_`` env prefix,
    # e.g. TA_LLM_BASE_URL. These are consumed only by the (future) llm/
    # package and AI service; nothing in V1.0 reads them.
    #
    # Ordered by concern: which backend, where it lives, how to call it, and
    # how the AI service should bound the request.
    llm_provider: str = Field(default="ollama", description="LLM backend name; '' disables the AI agent")
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    # Cloud backends (OpenAI/Claude/Gemini/Azure) will need a key later; local
    # Ollama does not. Present now so the abstraction is key-aware from day one,
    # but defaults empty so Ollama works with no secret.
    llm_api_key: str = Field(default="", description="API key for cloud LLM backends; unused by local Ollama")

    # Transport behaviour, kept independent of market-data and news transport
    # so the (much slower) LLM call can be tuned on its own.
    llm_request_timeout: float = Field(
        default=120.0,
        description=(
            "Per-request timeout in seconds. Sized for CPU inference: a 7B model "
            "generating a full structured thesis at ~5-15 tok/s needs well over 60s."
        ),
    )
    llm_max_retries: int = 3
    llm_retry_backoff: float = 0.75  # seconds, multiplied by 2**attempt

    # Generation controls. Low temperature favours deterministic,
    # schema-conformant JSON output.
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_num_predict: int = Field(
        default=768,
        ge=64,
        le=8192,
        description=(
            "Hard cap on generated tokens. Without it Ollama generates until the "
            "model emits EOS or the context fills, which is the dominant cost on CPU."
        ),
    )
    llm_num_ctx: int = Field(
        default=4096,
        ge=512,
        le=131072,
        description=(
            "Context window. Must fit prompt + completion; Ollama's default (2048) "
            "would silently truncate a ~1k-token prompt plus a ~768-token answer."
        ),
    )

    # Prompt-bounding caps (consumed by the future PromptBuilder) so prompt
    # size — and therefore latency, cost, and parse reliability — stays
    # predictable regardless of how much news is supplied.
    llm_max_news_articles: int = Field(default=15, ge=1, le=100)
    llm_max_list_items: int = Field(default=8, ge=1, le=50)

    # Response-repair retry: how many corrective re-prompts the AI service may
    # issue when the model returns non-conforming JSON, before giving up.
    llm_max_repair_attempts: int = Field(default=1, ge=0, le=3)

    # --- Strategy Engine ---
    # Thresholds specific to strategy decision logic that are not already
    # covered by an existing setting above (RSI/pullback/breakout/volume
    # thresholds are deliberately *reused* from the blocks above rather than
    # duplicated here, so a strategy's read of "oversold" or "a pullback"
    # never drifts from the Technical Engine's own definition).
    strategy_level_proximity_pct: float = Field(
        default=0.02,
        gt=0.0,
        le=0.25,
        description="How close price must be to a mapped support/resistance level to count as 'near' it (Swing strategy).",
    )

    # --- Market Scanner ---
    # Centralized here, same pattern as the news_*/llm_* blocks: one
    # configuration surface, overridable via the TA_ env prefix (e.g.
    # TA_SCANNER_MAX_CONCURRENCY).
    scanner_max_concurrency: int = Field(
        default=20, ge=1, le=200,
        description="Max symbols analyzed concurrently in one scan; bounds load on the shared data provider.",
    )
    scanner_max_symbols_per_scan: int = Field(
        default=2000, ge=1, le=20000,
        description="Hard ceiling on symbols accepted by a single scan request, to protect the service from unbounded input.",
    )
    scanner_default_include_news: bool = Field(
        default=True, description="Whether a scan fetches News Agent data per symbol by default.",
    )
    scanner_news_concurrency: int = Field(
        default=5, ge=1, le=50,
        description="Max concurrent News Agent calls; kept lower than technical concurrency since news providers are more rate-limit sensitive.",
    )
    scanner_default_top_n: int = Field(default=10, ge=1, le=500)

    # Combined-score blend weights (0-1, need not sum to exactly 1 — they are
    # normalized by the Ranking Engine at combination time, the same
    # normalize-internally convention the confluence engine's weight_* block
    # above already uses).
    scanner_weight_technical: float = Field(default=0.40, ge=0.0, le=1.0)
    scanner_weight_news: float = Field(default=0.20, ge=0.0, le=1.0)
    scanner_weight_portfolio: float = Field(default=0.15, ge=0.0, le=1.0)
    scanner_weight_opportunity: float = Field(default=0.25, ge=0.0, le=1.0)

    # News scoring (deterministic coverage/recency heuristic — NOT sentiment;
    # the News Agent performs zero sentiment analysis and the Scanner does
    # not either. See scanner/news_scoring.py).
    scanner_news_recency_half_life_hours: float = Field(
        default=24.0, gt=0.0,
        description="Half-life used to decay an article's contribution to the news score by age.",
    )
    scanner_news_saturation_count: int = Field(
        default=8, ge=1,
        description="Recency-weighted article count that saturates the news coverage score at 100.",
    )

    # Caching (performance: avoid recomputation across nearby scanner calls;
    # never used by the frozen Technical/News/AI engines themselves).
    scanner_technical_cache_ttl_seconds: float = Field(
        default=60.0, ge=0.0,
        description="How long a per-(ticker,period,interval) TechnicalAnalysisResult is reused within the Scanner. 0 disables caching.",
    )
    scanner_result_cache_ttl_seconds: float = Field(
        default=30.0, ge=0.0,
        description="How long a full scan's opportunity list is reused by /scanner/top and /scanner/opportunities for identical parameters. 0 disables caching.",
    )

    # Optional AI enrichment of top-ranked opportunities (disabled by default
    # per the Scanner's design contract: LLM analysis must be opt-in).
    scanner_default_include_ai: bool = Field(default=False)
    scanner_ai_max_opportunities: int = Field(
        default=5, ge=0, le=50,
        description="Hard ceiling on how many top-ranked opportunities may be sent to the AI Analysis Service in one scan, regardless of what a caller requests.",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — avoids re-parsing env vars on every call."""
    return Settings()
