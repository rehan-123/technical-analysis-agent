from __future__ import annotations

import time

from config.settings import Settings, get_settings
from llm.base import LLMProvider
from llm.exceptions import LLMError
from llm.provider_factory import create_llm_provider
from models.ai_analysis import AIAnalysisRequest, AIAnalysisResult
from services.ai_exceptions import InvalidAIResponse, ResponseParseError
from services.prompt_builder import PromptBuilder
from services.response_parser import ResponseParser
from utils.logger import get_logger

logger = get_logger(__name__)


class AIAnalysisService:
    """Orchestrates the AI analysis pipeline — and only orchestrates.

    Coordinates already-built components in sequence:

        AIAnalysisRequest
            -> PromptBuilder.build            (prompt construction)
            -> LLMProvider.generate           (transport, incl. its own retries)
            -> ResponseParser.parse           (extraction + validation)
            -> AIAnalysisResult

    It contains no prompt formatting, no parsing, and no business logic; each
    of those lives in the component it delegates to. Every collaborator is
    injected (Dependency Injection), so the whole pipeline is exercisable with
    a fake provider and no network.

    Retry policy (deliberately non-duplicated):
      * Transport retries — timeouts, connection resets, 5xx — already live
        inside the provider (``OllamaProvider.generate``). The service does NOT
        wrap them; nesting would multiply attempts.
      * Response-repair retries — re-prompting when the model returns
        unparseable or invalid JSON — are the orchestration layer's own
        concern and are handled here, bounded by ``llm_max_repair_attempts``.
        Parser/validation errors are never treated as transport failures.
    """

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder | None = None,
        parser: ResponseParser | None = None,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        """All collaborators are injectable; sensible defaults are constructed
        when omitted.

        ``provider`` may be injected directly (tests pass a fake); when it is
        ``None`` the service builds one lazily via the factory on first use, so
        constructing the service never performs I/O or requires a configured
        backend.
        """
        self._settings = settings or get_settings()
        self._prompt_builder = prompt_builder or PromptBuilder(settings=self._settings)
        self._parser = parser or ResponseParser()
        self._provider = provider  # may be None -> lazily created via factory

    def _get_provider(self) -> LLMProvider:
        """Return the injected provider, or lazily create the configured one.

        Construction is deferred to first use (not ``__init__``) so a
        misconfigured backend surfaces as an ``LLMConfigurationError`` on a
        request — mappable to a clean 503 by the API layer — rather than
        breaking service construction. No provider name is hardcoded; the
        factory reads ``settings.llm_provider``.
        """
        if self._provider is None:
            self._provider = create_llm_provider(self._settings)
        return self._provider

    async def analyze(self, request: AIAnalysisRequest) -> AIAnalysisResult:
        """Run the full pipeline for ``request`` and return a validated result.

        Raises:
            PromptBuildError: no renderable content in the request (from the
                builder, re-raised unchanged).
            LLMError (or subclass): provider/transport failure, after the
                provider's own retries are exhausted.
            ResponseParseError / InvalidAIResponse: the model's output could
                not be turned into a valid ``AIAnalysisResult`` after the
                permitted repair attempts.
        """
        provider = self._get_provider()
        provider_name = type(provider).__name__
        started = time.perf_counter()

        # Prompt construction (PromptBuildError propagates unchanged).
        package = self._prompt_builder.build(request)
        model_override = request.model or package.model_hints.get("model")

        max_repairs = max(0, self._settings.llm_max_repair_attempts)
        last_parse_error: ResponseParseError | InvalidAIResponse | None = None

        # attempt 0 = initial call; up to max_repairs additional repair calls.
        for attempt in range(max_repairs + 1):
            user_prompt = package.user_prompt
            if attempt > 0 and last_parse_error is not None:
                user_prompt = self._repair_prompt(package.user_prompt, last_parse_error)

            try:
                response = await provider.generate(
                    user_prompt,
                    system=package.system_prompt,
                    model=model_override,
                )
            except LLMError as exc:
                # Transport failure — provider already retried internally. Do
                # not retry again here; log the reason (never the prompt) and
                # propagate.
                self._log_failure(request.ticker, provider_name, started, str(exc))
                raise

            try:
                result = self._parser.parse(
                    response.text, ticker=request.ticker, model_used=response.model
                )
            except (ResponseParseError, InvalidAIResponse) as exc:
                last_parse_error = exc
                if attempt < max_repairs:
                    logger.info(
                        "AI analysis for %s: response invalid on attempt %d/%d, repairing",
                        request.ticker, attempt + 1, max_repairs + 1,
                    )
                    continue
                self._log_failure(request.ticker, provider_name, started, f"parse/validation: {exc}")
                raise

            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "AI analysis for %s succeeded via %s in %.1fms (attempt %d/%d)",
                request.ticker, provider_name, elapsed_ms, attempt + 1, max_repairs + 1,
            )
            return result

        # Unreachable: the loop either returns or raises on the final attempt.
        assert last_parse_error is not None
        raise last_parse_error

    @staticmethod
    def _repair_prompt(
        original_user_prompt: str, error: ResponseParseError | InvalidAIResponse
    ) -> str:
        """Append a corrective instruction after an invalid response.

        Deterministic and content-free about the *data* — it only tells the
        model its previous output was invalid and to return valid JSON. It does
        not echo the model's bad output or any secret.
        """
        return (
            f"{original_user_prompt}\n\n"
            "Your previous response was not valid. "
            f"Reason: {error}. "
            "Respond again with a single valid JSON object matching the required "
            "schema exactly, and nothing else."
        )

    @staticmethod
    def _log_failure(ticker: str, provider: str, started: float, reason: str) -> None:
        """Log a failure with timing and reason — never prompt or response
        contents, keys, or confidential data."""
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning(
            "AI analysis for %s failed via %s after %.1fms: %s",
            ticker, provider, elapsed_ms, reason,
        )
