from __future__ import annotations


class AIAnalysisError(Exception):
    """Base class for orchestration-level errors raised by the AI analysis
    service.

    Kept in the service (AI) domain, mirroring how the News and LLM domains own
    their exceptions. It sits *above* — and never duplicates — the LLM
    transport exceptions (``LLMError`` and subclasses) or ``PromptBuildError``;
    those continue to represent their own concerns and are re-raised or mapped
    as-is. This hierarchy covers only the failures that are genuinely new at the
    orchestration layer: turning a raw model response into a validated result.
    """


class ResponseParseError(AIAnalysisError):
    """Raised when the raw LLM output cannot be parsed into a JSON object.

    Purely structural: the text was not JSON, or not a JSON object, even after
    tolerant extraction (fenced blocks, surrounding prose, whitespace). This is
    a *parser* failure and is never retried as a transport error.
    """


class InvalidAIResponse(AIAnalysisError):
    """Raised when parsed JSON does not satisfy the ``AIAnalysisResult``
    contract.

    The JSON was well-formed but semantically invalid — e.g. a confidence
    outside 0–100, a recommendation outside the enum, or a missing required
    field. The validation itself is performed by the existing
    ``AIAnalysisResult`` Pydantic model (not re-implemented here); this
    exception is the service-domain wrapper around that model's
    ``ValidationError`` so callers get a single, stable failure type.
    """
