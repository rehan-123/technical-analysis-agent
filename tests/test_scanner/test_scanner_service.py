from __future__ import annotations

import pytest

from agent.news_agent import NewsAgent
from agent.technical_analysis_agent import TechnicalAnalysisAgent
from config.settings import Settings
from data.base import MarketDataProvider
from data.synthetic_provider import SyntheticDataProvider
from models.ai_analysis import AIAnalysisResult, Recommendation
from models.strategy import StrategyName
from portfolio.portfolio_manager import PortfolioManager
from scanner.exceptions import NoSymbolsProvidedError, ScanTooLargeError
from scanner.scanner_service import MarketScannerService
from services.news_service import NewsService
from tests.test_news.fakes import FakeNewsProvider, default_article_set
from utils.exceptions import DataFetchError


class _CountingProvider(MarketDataProvider):
    """Wraps another provider and counts calls, to verify the Scanner's
    technical-result cache actually prevents redundant fetches."""

    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner
        self.calls = 0

    async def get_ohlcv(self, ticker: str, period: str, interval: str):
        self.calls += 1
        return await self._inner.get_ohlcv(ticker, period, interval)


class _FlakyTechnicalAgent(TechnicalAnalysisAgent):
    """A real TechnicalAnalysisAgent that raises for one specific ticker —
    used to prove one symbol's failure never aborts the batch."""

    def __init__(self, *, fail_ticker: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fail_ticker = fail_ticker.upper()

    async def analyze(self, ticker: str, period: str | None = None, interval: str | None = None):
        if ticker.strip().upper() == self._fail_ticker:
            raise DataFetchError(f"simulated failure for {ticker}")
        return await super().analyze(ticker, period=period, interval=interval)


class _FakeAIAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def analyze(self, ticker, *, technical=None, news=None, additional_inputs=None, model=None):
        self.calls.append(ticker)
        return AIAnalysisResult(
            ticker=ticker,
            recommendation=Recommendation.BUY,
            confidence=77,
            investment_thesis="Solid setup with confirming technicals.",
        )


class _FailingAIAgent:
    async def analyze(self, *args, **kwargs):
        raise RuntimeError("LLM backend unavailable")


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def technical_agent(settings) -> TechnicalAnalysisAgent:
    provider = SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014)
    return TechnicalAnalysisAgent(settings=settings, data_provider=provider)


@pytest.fixture
def news_agent(settings) -> NewsAgent:
    provider = FakeNewsProvider(default_article_set())
    return NewsAgent(service=NewsService(provider=provider, settings=settings), settings=settings)


@pytest.fixture
def service(technical_agent, news_agent, settings) -> MarketScannerService:
    return MarketScannerService(technical_agent=technical_agent, news_agent=news_agent, settings=settings)


@pytest.mark.asyncio
async def test_scan_returns_ranked_opportunities(service):
    result = await service.scan(["AAA", "BBB", "CCC"])

    assert result.summary.requested == 3
    assert result.summary.scanned == 3
    assert result.summary.succeeded + result.summary.failed == 3
    assert result.summary.include_ai is False

    tickers_seen = {o.ticker for o in result.opportunities} | {f.ticker for f in result.failures}
    assert tickers_seen <= {"AAA", "BBB", "CCC"}

    scores = [o.combined_score for o in result.opportunities]
    assert scores == sorted(scores, reverse=True)
    assert [o.ranking for o in result.opportunities] == list(range(1, len(result.opportunities) + 1))


@pytest.mark.asyncio
async def test_scan_deduplicates_symbols(service):
    result = await service.scan(["AAA", "aaa", " AAA "])
    assert result.summary.requested == 1


@pytest.mark.asyncio
async def test_scan_empty_symbols_raises(service):
    with pytest.raises(NoSymbolsProvidedError):
        await service.scan([])


@pytest.mark.asyncio
async def test_scan_over_symbol_ceiling_raises(technical_agent, news_agent):
    tight_settings = Settings(scanner_max_symbols_per_scan=2)
    tight_service = MarketScannerService(
        technical_agent=technical_agent, news_agent=news_agent, settings=tight_settings
    )
    with pytest.raises(ScanTooLargeError):
        await tight_service.scan(["AAA", "BBB", "CCC"])


@pytest.mark.asyncio
async def test_one_symbol_failure_does_not_abort_the_batch(settings, news_agent):
    provider = SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014)
    flaky_agent = _FlakyTechnicalAgent(fail_ticker="BBB", settings=settings, data_provider=provider)
    service = MarketScannerService(technical_agent=flaky_agent, news_agent=news_agent, settings=settings)

    result = await service.scan(["AAA", "BBB", "CCC"])

    assert result.summary.scanned == 3
    assert result.summary.failed == 1
    failure_tickers = {f.ticker for f in result.failures}
    assert failure_tickers == {"BBB"}
    opportunity_tickers = {o.ticker for o in result.opportunities}
    assert "BBB" not in opportunity_tickers


@pytest.mark.asyncio
async def test_include_news_false_skips_news_fetch(service, news_agent):
    calls_before = len(news_agent._service._provider.calls)  # type: ignore[attr-defined]
    await service.scan(["AAA"], include_news=False)
    calls_after = len(news_agent._service._provider.calls)  # type: ignore[attr-defined]
    assert calls_after == calls_before


@pytest.mark.asyncio
async def test_portfolio_none_yields_neutral_portfolio_score(service):
    result = await service.scan(["AAA", "BBB", "CCC"], portfolio=None)
    for opp in result.opportunities:
        assert opp.portfolio_score == 50
        assert opp.position_size is None


@pytest.mark.asyncio
async def test_portfolio_with_cash_can_size_a_position(settings, news_agent):
    provider = SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014)
    agent = TechnicalAnalysisAgent(settings=settings, data_provider=provider)
    service = MarketScannerService(technical_agent=agent, news_agent=news_agent, settings=settings)
    portfolio = PortfolioManager.empty(name="test", cash=100_000.0)

    result = await service.scan(["AAA", "BBB", "CCC"], portfolio=portfolio)

    # At least one candidate should have received an actual position size,
    # since there is ample cash and no existing exposure to conflict with.
    assert any(o.position_size is not None and o.position_size.shares > 0 for o in result.opportunities)


@pytest.mark.asyncio
async def test_technical_results_are_cached_within_ttl(settings):
    inner = SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014)
    counting = _CountingProvider(inner)
    agent = TechnicalAnalysisAgent(settings=settings, data_provider=counting)
    service = MarketScannerService(technical_agent=agent, settings=settings)

    await service.scan(["AAA", "BBB"], include_news=False)
    calls_after_first = counting.calls
    assert calls_after_first == 2

    await service.scan(["AAA", "BBB"], include_news=False)
    # Same service instance -> same cache -> no additional provider calls.
    assert counting.calls == calls_after_first


@pytest.mark.asyncio
async def test_zero_ttl_disables_technical_caching():
    settings = Settings(scanner_technical_cache_ttl_seconds=0)
    inner = SyntheticDataProvider(seed=7, start_price=150.0, drift=0.0009, volatility=0.014)
    counting = _CountingProvider(inner)
    agent = TechnicalAnalysisAgent(settings=settings, data_provider=counting)
    service = MarketScannerService(technical_agent=agent, settings=settings)

    await service.scan(["AAA"], include_news=False)
    await service.scan(["AAA"], include_news=False)
    assert counting.calls == 2


@pytest.mark.asyncio
async def test_strategy_filter_only_returns_matching_strategy(service):
    result = await service.scan(["AAA", "BBB", "CCC"], strategy=StrategyName.TREND_FOLLOWING)
    for opp in result.opportunities:
        assert opp.strategy is StrategyName.TREND_FOLLOWING
    assert result.summary.strategy_filter is StrategyName.TREND_FOLLOWING


@pytest.mark.asyncio
async def test_enrich_with_ai_annotates_top_opportunities(service):
    result = await service.scan(["AAA", "BBB", "CCC"])
    assert result.opportunities, "fixture must produce at least one opportunity for this test to be meaningful"

    ai_agent = _FakeAIAgent()
    enriched = await service.enrich_with_ai(result, ai_agent=ai_agent)

    assert enriched.summary.include_ai is True
    assert enriched.summary.ai_analyzed == len(result.opportunities)
    assert len(ai_agent.calls) == len(result.opportunities)
    top = enriched.opportunities[0]
    assert any("AI thesis:" in r for r in top.reasoning)


@pytest.mark.asyncio
async def test_enrich_with_ai_respects_configured_cap(technical_agent, news_agent):
    capped_settings = Settings(scanner_ai_max_opportunities=1)
    service = MarketScannerService(technical_agent=technical_agent, news_agent=news_agent, settings=capped_settings)
    result = await service.scan(["AAA", "BBB", "CCC"])
    assert len(result.opportunities) >= 2, "need at least 2 opportunities to prove the cap binds"

    ai_agent = _FakeAIAgent()
    enriched = await service.enrich_with_ai(result, ai_agent=ai_agent)

    assert len(ai_agent.calls) == 1
    assert enriched.summary.ai_analyzed == 1


@pytest.mark.asyncio
async def test_enrich_with_ai_failure_degrades_gracefully(service):
    result = await service.scan(["AAA", "BBB", "CCC"])
    assert result.opportunities

    enriched = await service.enrich_with_ai(result, ai_agent=_FailingAIAgent())

    assert enriched.summary.ai_analyzed == 0
    assert enriched.summary.include_ai is True
    assert all("AI enrichment unavailable" in w for o in enriched.opportunities for w in o.warnings if "AI" in w)
    # The batch itself must not fail, and every opportunity must survive.
    assert len(enriched.opportunities) == len(result.opportunities)


@pytest.mark.asyncio
async def test_enrich_with_ai_on_empty_opportunities_is_a_noop(service):
    result = await service.scan(["AAA"], strategy=StrategyName.MEAN_REVERSION)
    if result.opportunities:
        pytest.skip("fixture happened to produce a mean-reversion opportunity; not the case under test")
    enriched = await service.enrich_with_ai(result, ai_agent=_FakeAIAgent())
    assert enriched.summary.ai_analyzed == 0
    assert enriched.opportunities == result.opportunities
