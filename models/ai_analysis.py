from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from models.analysis_result import TechnicalAnalysisResult
from models.news import NewsAnalysisResult

# ---------------------------------------------------------------------------
# Enumerations
#
# The AI Analysis layer is a new bounded context, so it defines its own closed
# vocabularies as real ``str, Enum`` classes rather than mirroring the
# ``Literal`` aliases used by the Version 1.0 technical-analysis models. Enums
# give cleaner OpenAPI schemas, IDE autocompletion, and explicit membership,
# and subclassing ``str`` means they still serialize as plain JSON strings —
# so nothing is lost on the wire.
# ---------------------------------------------------------------------------


class Recommendation(str, Enum):
    """The AI's directional call — a closed, validated vocabulary.

    Deliberately an enum, not free text: a downstream Chief Decision Agent
    consumes this field programmatically, so the set of possible values must
    be fixed and comparable. ``NO_ACTION`` is distinct from ``HOLD``: HOLD is
    a position stance, NO_ACTION means the evidence did not support any call.
    """

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_ACTION = "NO_ACTION"


class NewsSentiment(str, Enum):
    """The model's characterization of the supplied news.

    IMPORTANT: this is the LLM's reading of the headlines it was given, NOT a
    computed sentiment score. The News Agent still performs zero sentiment
    analysis; this field exists only within the AI layer's interpretation.
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class TechnicalAlignment(str, Enum):
    """Whether the technical picture agrees with the overall thesis."""

    ALIGNED = "ALIGNED"
    DIVERGENT = "DIVERGENT"
    MIXED = "MIXED"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AIAnalysisRequest(BaseModel):
    """Input contract for the AI Analysis Agent.

    The AI agent is a pure *consumer*: it never fetches data or calls another
    agent. It receives already-built specialist results and synthesizes a
    thesis. This model carries those inputs.

    Extensibility (the reason it is shaped this way):
      * ``technical`` and ``news`` are the two known Version 2.0 inputs, kept
        as concrete, strongly-typed fields — and **optional**, so a
        news-only or technical-only thesis is representable without
        fabricating a missing input.
      * ``additional_inputs`` is the open extension point for future
        specialist outputs (Risk, Macro, Portfolio, Options). It is typed
        ``Mapping[str, BaseModel]`` rather than ``dict[str, Any]``
        deliberately: the ``BaseModel`` bound preserves validation and
        self-documentation, so future agents thread their *validated* results
        through here — e.g. ``{"risk": RiskAnalysisResult(...)}`` — with **no
        change to this class, the AI service, or the agent**. A result can
        later be promoted to a first-class typed field if it becomes core,
        without breaking callers using the map.

    Frozen for the same reasons as ``NewsRequest``: an immutable request has a
    stable identity, which is what makes it a safe cache key for the future
    ``AICache`` seam. (Hashability is not claimed here — the payload models are
    rich and not themselves hashable — but immutability guarantees the
    content cannot change after a key is derived from it.)
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., min_length=1, description="Ticker the thesis is about")
    technical: TechnicalAnalysisResult | None = Field(
        default=None, description="Technical Analysis Agent result, if available"
    )
    news: NewsAnalysisResult | None = Field(
        default=None, description="News Agent result, if available"
    )
    additional_inputs: Mapping[str, BaseModel] = Field(
        default_factory=dict,
        description="Future specialist-agent results, keyed by domain (e.g. 'risk', 'macro')",
    )
    model: str | None = Field(
        default=None, description="Optional per-request LLM model override"
    )

    @field_validator("additional_inputs", mode="before")
    @classmethod
    def _reject_non_basemodel_values(cls, value: object) -> object:
        """Enforce the ``Mapping[str, BaseModel]`` contract strictly.

        Pydantic v2's annotation alone is not sufficient here: because the
        abstract ``BaseModel`` base declares no fields, validating a raw
        ``dict`` against it succeeds (extra keys are ignored) and silently
        yields an empty model instead of raising. That would reopen exactly
        the weak-contract hole this typed mapping exists to close.

        Running in ``mode="before"`` — ahead of coercion — this guard rejects
        any value that is not already a ``BaseModel`` instance, so callers must
        pass validated specialist results (e.g. ``RiskAnalysisResult(...)``),
        never bare dicts. Keys are still required to be strings; that part the
        annotation enforces on its own.
        """
        if value is None:
            return value
        if not isinstance(value, Mapping):
            raise ValueError("additional_inputs must be a mapping")
        for key, item in value.items():
            if not isinstance(item, BaseModel):
                raise ValueError(
                    f"additional_inputs[{key!r}] must be a pydantic BaseModel instance, "
                    f"got {type(item).__name__}"
                )
        return value


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

#: Non-advice framing carried on every result. The platform emits analysis for
#: research, not personalized financial advice; this is asserted in the output
#: itself, not left implicit.
_DISCLAIMER = (
    "This is an AI-generated analytical thesis for research and informational "
    "purposes only. It is not personalized financial advice, and no "
    "recommendation herein should be acted upon without independent verification."
)


class AIAnalysisResult(BaseModel):
    """Structured output contract of the AI Analysis Agent.

    Maps the approved thesis shape onto typed fields. Closed-vocabulary fields
    are enums (validated on construction); genuinely open fields are
    ``list[str]``. Parallels the other agents' result shape (``agent``,
    ``ticker``, a timestamp, payload) so a Chief Decision Agent can consume
    every specialist uniformly.
    """

    agent: str = "ai_analysis_agent"
    ticker: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    recommendation: Recommendation
    confidence: int = Field(..., ge=0, le=100, description="0–100 self-assessed conviction")
    investment_thesis: str

    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    macro_considerations: list[str] = Field(default_factory=list)
    watch_items: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)

    news_sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    technical_alignment: TechnicalAlignment = TechnicalAlignment.MIXED

    # Observability — which backend produced this and any provider metadata
    # (latency, token counts). Populated by the AI service in a later phase.
    model_used: str = ""
    llm_metadata: dict = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def disclaimer(self) -> str:
        """Non-advice framing, emitted with every result. Computed so it is
        always present and can never be overridden to something misleading."""
        return _DISCLAIMER
