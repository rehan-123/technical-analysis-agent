from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from models.ai_analysis import AIAnalysisResult
from services.ai_exceptions import InvalidAIResponse, ResponseParseError
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches a ```json ... ``` or ``` ... ``` fenced block and captures its body.
# Non-greedy so the first fenced block wins.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class ResponseParser:
    """Converts a raw LLM text response into a validated ``AIAnalysisResult``.

    Two clearly separated failure modes, so a caller can react appropriately:
      * structural (not JSON / not an object) -> ``ResponseParseError``
      * semantic (well-formed JSON that violates the model contract)
        -> ``InvalidAIResponse``

    Validation is delegated entirely to the ``AIAnalysisResult`` model — the
    confidence bounds, the recommendation enum, and required fields are already
    enforced there, so this parser never re-implements those rules. Its only
    jobs are to *extract* JSON robustly and to *route* the model's validation
    outcome into the service's exception vocabulary.
    """

    def parse(self, text: str, *, ticker: str, model_used: str = "") -> AIAnalysisResult:
        """Parse ``text`` into an ``AIAnalysisResult``.

        Args:
            text: The raw model output.
            ticker: The ticker under analysis, used only to backfill the field
                if the model omitted it (the service knows it authoritatively).
            model_used: The backend/model identifier, recorded on the result
                for observability. Never affects parsing.

        Raises:
            ResponseParseError: the text is not a JSON object even after
                tolerant extraction.
            InvalidAIResponse: the JSON is valid but violates the
                ``AIAnalysisResult`` contract.
        """
        payload = self._extract_json_object(text)

        # Backfill fields the service owns authoritatively, without overriding
        # anything the model actually returned.
        payload.setdefault("ticker", ticker)
        if model_used and not payload.get("model_used"):
            payload["model_used"] = model_used

        try:
            return AIAnalysisResult.model_validate(payload)
        except ValidationError as exc:
            # The model's own validation is the single source of truth for
            # confidence bounds / enum membership / required fields; we only
            # translate its failure into the service vocabulary.
            raise InvalidAIResponse(
                f"LLM response failed AIAnalysisResult validation: {exc.error_count()} error(s)"
            ) from exc

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Recover a JSON object from ``text``, tolerating common LLM wrapping.

        Handled, in order: a raw JSON object; a ```json fenced block; and a
        JSON object embedded in surrounding prose (first ``{`` … last ``}``).
        Anything that still is not a JSON *object* raises ``ResponseParseError``.
        """
        if not text or not text.strip():
            raise ResponseParseError("LLM response was empty")

        candidates: list[str] = []
        stripped = text.strip()
        candidates.append(stripped)

        fenced = _FENCE_RE.search(text)
        if fenced:
            candidates.append(fenced.group(1).strip())

        # First '{' to last '}' — recovers an object embedded in prose.
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first : last + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
            # Valid JSON but not an object (e.g. a list or scalar) — not usable.
            raise ResponseParseError(
                f"LLM response parsed to {type(parsed).__name__}, expected a JSON object"
            )

        raise ResponseParseError("LLM response did not contain a parseable JSON object")
