from __future__ import annotations

from config.settings import Settings, get_settings
from data.base import MarketDataProvider
from data.fallback_provider import FallbackDataProvider
from data.stooq_provider import StooqProvider
from data.yfinance_provider import YFinanceProvider
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_one(name: str, settings: Settings) -> MarketDataProvider | None:
    name = name.strip().lower()
    if name == "yfinance":
        return YFinanceProvider(
            timeout=settings.data_request_timeout,
            max_retries=settings.data_max_retries,
            retry_backoff=settings.data_retry_backoff,
            impersonate=settings.yfinance_impersonate,
        )
    if name == "stooq":
        return StooqProvider(timeout=settings.data_request_timeout)
    logger.warning("Unknown data source '%s' in configuration — skipping", name)
    return None


def create_data_provider(settings: Settings | None = None) -> MarketDataProvider:
    """Build the production data provider (a fallback chain) from settings.

    The chain is real sources only, in the configured order (default:
    yfinance -> stooq). If exactly one source is configured, that provider is
    returned directly; otherwise it is wrapped in a ``FallbackDataProvider``.
    Synthetic data is intentionally not selectable here.
    """
    settings = settings or get_settings()
    providers = [p for p in (_build_one(n, settings) for n in settings.data_sources) if p is not None]
    if not providers:
        # Never leave the app with no data source; default to a hardened yfinance.
        logger.warning("No valid data sources configured; defaulting to yfinance only")
        providers = [YFinanceProvider()]
    if len(providers) == 1:
        return providers[0]
    return FallbackDataProvider(providers)
