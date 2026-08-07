from __future__ import annotations

import math
from datetime import datetime, timezone

from config.settings import Settings, get_settings
from models.news import NewsAnalysisResult


def score_news(
    news: NewsAnalysisResult | None,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> tuple[int, list[str]]:
    """Deterministic news coverage/recency score in ``[0, 100]``.

    This is explicitly **not** a sentiment score: the News Agent performs
    zero sentiment analysis (see ``models.news.NewsAnalysisResult``'s own
    docstring), and the Scanner — which "never performs news analysis
    itself" — does not either. It measures how much *recent* coverage
    exists, nothing about whether that coverage is positive or negative.

    Each article contributes a weight that decays exponentially with age
    (half-life = ``scanner_news_recency_half_life_hours``); the summed
    weight is scaled so that ``scanner_news_saturation_count``
    recency-weighted articles saturates the score at 100. Absence of news —
    not fetched, or fetched but empty — scores a neutral 50 rather than 0:
    no coverage is not evidence of a bad opportunity, just missing
    information.

    Returns:
        ``(score, notes)`` — ``notes`` explain how the score was derived,
        folded by the caller into an ``Opportunity``'s ``reasoning``.
    """
    settings = settings or get_settings()

    if news is None:
        return 50, ["News was not fetched for this scan; news_score defaulted to neutral (50)."]
    if not news.articles:
        return 50, ["No recent news articles were found; news_score defaulted to neutral (50)."]

    now = now or datetime.now(timezone.utc)
    half_life_hours = max(settings.scanner_news_recency_half_life_hours, 1e-6)
    decay_lambda = math.log(2) / half_life_hours

    weight = 0.0
    for article in news.articles:
        age_hours = max(0.0, (now - article.published_at).total_seconds() / 3600.0)
        weight += math.exp(-decay_lambda * age_hours)

    saturation = max(settings.scanner_news_saturation_count, 1)
    score = int(max(0, min(100, round((weight / saturation) * 100.0))))
    notes = [
        f"{len(news.articles)} recent article(s) found; recency-weighted coverage "
        f"{weight:.2f} of {saturation} needed to saturate the news score."
    ]
    return score, notes
