from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Final, Mapping

import httpx

from config.settings import Settings, get_settings
from llm.base import LLMProvider
from llm.exceptions import LLMConfigurationError
from llm.ollama_provider import OllamaProvider
from utils.logger import get_logger

logger = get_logger(__name__)

#: Signature every provider builder must satisfy. Taking the injected
#: ``Settings`` and optional shared ``httpx.AsyncClient`` means the factory
#: passes dependencies *through* to providers rather than letting them reach
#: for globals, preserving Dependency Injection end to end.
LLMProviderBuilder = Callable[[Settings, httpx.AsyncClient | None], LLMProvider]


def _build_ollama(settings: Settings, client: httpx.AsyncClient | None) -> LLMProvider:
    return OllamaProvider(settings=settings, client=client)


#: Fixed registry of known LLM backends, keyed by ``Settings.llm_provider``.
#: A registry rather than an if/elif chain so the factory is open for
#: extension and closed for modification: adding OpenAI/Anthropic/Gemini/Azure
#: means adding a builder and one entry here, never editing
#: ``create_llm_provider`` and never touching a single caller.
#:
#: Wrapped in ``MappingProxyType`` so it is immutable at runtime, not merely by
#: convention — the set of backends a deployment can use is fixed at import
#: time and auditable by reading this file.
#:
#: Values are *builders*, not instances — construction is deferred until a
#: provider is actually selected, so importing this module never constructs a
#: client or validates a backend nobody enabled.
_PROVIDER_REGISTRY: Final[Mapping[str, LLMProviderBuilder]] = MappingProxyType(
    {
        OllamaProvider.PROVIDER_NAME: _build_ollama,
    }
)


def available_llm_providers() -> tuple[str, ...]:
    """Return the names of every registered LLM provider, sorted.

    Useful for diagnostics, configuration validation, and error messages that
    tell an operator what they *could* have configured.
    """
    return tuple(sorted(_PROVIDER_REGISTRY))


def create_llm_provider(
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> LLMProvider:
    """Construct the configured LLM provider.

    The single place in the codebase that knows which concrete backend exists;
    every other module depends only on the ``LLMProvider`` abstraction. Reads
    ``Settings.llm_provider`` and returns an instance of that backend.

    This function only *constructs*. It performs no generation, prompt-building,
    validation, or retrying — those belong to the provider (transport) and to
    later phases.

    Args:
        settings: Injected configuration. Falls back to cached global settings.
        client: Optional shared HTTP client passed through to the provider, so
            callers (and tests) control transport and connection-pool lifetime.

    Returns:
        A ready-to-use ``LLMProvider``.

    Raises:
        LLMConfigurationError: if ``llm_provider`` is empty (the AI layer is
            disabled) or names a backend that is not registered.
    """
    settings = settings or get_settings()
    name = (settings.llm_provider or "").strip().lower()

    if not name:
        raise LLMConfigurationError(
            "No LLM provider configured: 'llm_provider' is empty. "
            f"Available providers: {', '.join(available_llm_providers())}"
        )

    builder = _PROVIDER_REGISTRY.get(name)
    if builder is None:
        raise LLMConfigurationError(
            f"Unknown LLM provider '{name}'. "
            f"Available providers: {', '.join(available_llm_providers())}"
        )

    provider = builder(settings, client)
    logger.info("LLM provider '%s' selected (%s)", name, type(provider).__name__)
    return provider
