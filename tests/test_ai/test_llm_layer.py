from __future__ import annotations

import httpx
import pytest

from config.settings import Settings
from llm.base import LLMProvider, LLMResponse
from llm.exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from llm.ollama_provider import OllamaProvider
from llm.provider_factory import (
    _PROVIDER_REGISTRY,
    available_llm_providers,
    create_llm_provider,
)


def _settings(**overrides) -> Settings:
    base = dict(
        llm_provider="ollama",
        llm_base_url="http://test-ollama:11434",
        llm_model="qwen2.5:7b",
        llm_max_retries=2,
        llm_retry_backoff=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def _client(handler) -> httpx.AsyncClient:
    """AsyncClient wired to an in-process transport — never touches a socket."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_body(content: str = '{"recommendation": "BUY"}', model: str = "qwen2.5:7b") -> dict:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 123456,
        "eval_count": 42,
    }


# --- Exception hierarchy ------------------------------------------------------


class TestExceptionHierarchy:
    def test_all_errors_share_a_common_base(self):
        for exc in (LLMConfigurationError, LLMProviderError, LLMRateLimitError, LLMResponseError):
            assert issubclass(exc, LLMError)

    def test_rate_limit_is_a_provider_error(self):
        """Handlers catching the parent must keep working for 429s."""
        assert issubclass(LLMRateLimitError, LLMProviderError)

    def test_provider_error_renders_context_into_message(self):
        exc = LLMProviderError("boom", provider="ollama", model="qwen2.5:7b", status_code=502, url="http://x/y")
        text = str(exc)
        assert "ollama" in text and "qwen2.5:7b" in text and "502" in text

    def test_provider_error_context_is_attribute_accessible(self):
        exc = LLMProviderError("boom", provider="ollama", status_code=500)
        assert exc.provider == "ollama"
        assert exc.status_code == 500
        assert exc.model is None

    def test_rate_limit_carries_retry_after(self):
        exc = LLMRateLimitError("limited", retry_after=30.0, provider="ollama")
        assert exc.retry_after == 30.0
        assert exc.status_code == 429
        assert "retry_after=30.0s" in str(exc)


# --- Provider interface -------------------------------------------------------


class TestProviderInterface:
    def test_ollama_implements_the_abstraction(self):
        assert issubclass(OllamaProvider, LLMProvider)

    def test_abstract_provider_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_llm_response_is_a_typed_model_with_expected_fields(self):
        resp = LLMResponse(text="hi", model="m", raw={"a": 1}, latency_ms=12.5)
        assert resp.text == "hi" and resp.model == "m"
        assert resp.raw == {"a": 1} and resp.latency_ms == 12.5

    def test_llm_response_defaults(self):
        resp = LLMResponse(text="hi", model="m")
        assert resp.raw == {} and resp.latency_ms == 0.0

    def test_interface_is_provider_agnostic(self):
        """The ABC must not leak Ollama-specific vocabulary."""
        import inspect

        import llm.base as base
        source = inspect.getsource(base).lower()
        for term in ("ollama", "openai", "anthropic", "gemini", "azure", "/api/chat"):
            assert term not in source, f"provider-specific term leaked into ABC: {term}"


# --- Ollama transport ---------------------------------------------------------


class TestOllamaSuccess:
    @pytest.mark.asyncio
    async def test_successful_generation_returns_llm_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_body(content="hello world"))

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            resp = await provider.generate("prompt")

        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello world"
        assert resp.model == "qwen2.5:7b"
        assert resp.latency_ms >= 0.0
        # native metadata is retained, message is not duplicated into raw
        assert resp.raw.get("eval_count") == 42
        assert "message" not in resp.raw

    @pytest.mark.asyncio
    async def test_request_targets_api_chat_with_stream_false(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            await provider.generate("prompt")

        assert seen["url"].endswith("/api/chat")
        assert seen["body"]["stream"] is False
        assert seen["body"]["format"] == "json"          # default JSON bias
        assert seen["body"]["options"]["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_system_prompt_and_user_prompt_are_placed_in_messages(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            await provider.generate("USER", system="SYSTEM")

        roles = [(m["role"], m["content"]) for m in seen["body"]["messages"]]
        assert roles == [("system", "SYSTEM"), ("user", "USER")]

    @pytest.mark.asyncio
    async def test_model_override_is_honoured(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body(model="qwen2.5:14b"))

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            resp = await provider.generate("prompt", model="qwen2.5:14b")

        assert seen["body"]["model"] == "qwen2.5:14b"
        assert resp.model == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_explicit_schema_is_sent_as_format(self):
        seen: dict = {}
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            await provider.generate("prompt", schema=schema)

        assert seen["body"]["format"] == schema


class TestOllamaErrorHandling:
    @pytest.mark.asyncio
    async def test_rate_limit_maps_to_rate_limit_error_and_is_not_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429, headers={"Retry-After": "15"}, json={})

        async with OllamaProvider(settings=_settings(llm_max_retries=3), client=_client(handler)) as provider:
            with pytest.raises(LLMRateLimitError) as exc_info:
                await provider.generate("prompt")

        assert calls["n"] == 1
        assert exc_info.value.retry_after == 15.0
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_client_error_is_not_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={})

        async with OllamaProvider(settings=_settings(llm_max_retries=3), client=_client(handler)) as provider:
            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate("prompt")

        assert calls["n"] == 1
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_server_error_is_retried_then_raises(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={})

        async with OllamaProvider(settings=_settings(llm_max_retries=3), client=_client(handler)) as provider:
            with pytest.raises(LLMProviderError):
                await provider.generate("prompt")

        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_transient_error_then_success_recovers(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={})
            return httpx.Response(200, json=_ok_body(content="recovered"))

        async with OllamaProvider(settings=_settings(llm_max_retries=3), client=_client(handler)) as provider:
            resp = await provider.generate("prompt")

        assert calls["n"] == 2
        assert resp.text == "recovered"

    @pytest.mark.asyncio
    async def test_timeout_is_reported_as_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        async with OllamaProvider(settings=_settings(llm_max_retries=1), client=_client(handler)) as provider:
            with pytest.raises(LLMProviderError, match="timed out"):
                await provider.generate("prompt")

    @pytest.mark.asyncio
    async def test_transport_error_is_retried_then_raises_with_context(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection reset", request=request)

        async with OllamaProvider(settings=_settings(llm_max_retries=2), client=_client(handler)) as provider:
            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate("prompt")

        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_invalid_json_envelope_raises_response_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>not json</html>")

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(LLMResponseError):
                await provider.generate("prompt")

    @pytest.mark.asyncio
    async def test_missing_message_content_raises_response_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "m", "done": True})  # no message

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as provider:
            with pytest.raises(LLMResponseError):
                await provider.generate("prompt")


class TestOllamaResourceManagement:
    @pytest.mark.asyncio
    async def test_injected_client_is_not_closed_by_the_provider(self):
        client = _client(lambda r: httpx.Response(200, json=_ok_body()))
        provider = OllamaProvider(settings=_settings(), client=client)
        await provider.generate("prompt")
        await provider.aclose()
        assert not client.is_closed
        await client.aclose()

    @pytest.mark.asyncio
    async def test_context_manager_closes_only_owned_clients(self):
        client = _client(lambda r: httpx.Response(200, json=_ok_body()))
        async with OllamaProvider(settings=_settings(), client=client) as provider:
            await provider.generate("prompt")
        assert not client.is_closed
        await client.aclose()

    def test_construction_performs_no_io_and_needs_no_event_loop(self):
        assert OllamaProvider(settings=_settings()) is not None


# --- Factory ------------------------------------------------------------------


class TestFactory:
    def test_returns_the_configured_provider(self):
        provider = create_llm_provider(_settings(llm_provider="ollama"))
        assert isinstance(provider, OllamaProvider)
        assert isinstance(provider, LLMProvider)  # callers bind to the abstraction

    def test_provider_name_is_case_and_whitespace_insensitive(self):
        assert isinstance(create_llm_provider(_settings(llm_provider="  Ollama ")), OllamaProvider)

    def test_empty_provider_raises_configuration_error(self):
        with pytest.raises(LLMConfigurationError, match="empty"):
            create_llm_provider(_settings(llm_provider=""))

    def test_unknown_provider_raises_configuration_error_listing_available(self):
        with pytest.raises(LLMConfigurationError, match="ollama"):
            create_llm_provider(_settings(llm_provider="gpt5"))

    def test_injected_client_is_passed_through(self):
        client = httpx.AsyncClient()
        provider = create_llm_provider(_settings(), client=client)
        assert provider._client is client  # noqa: SLF001 — verifying DI wiring

    def test_available_providers_is_sorted_and_includes_ollama(self):
        names = available_llm_providers()
        assert "ollama" in names
        assert list(names) == sorted(names)

    def test_registry_is_immutable_at_runtime(self):
        with pytest.raises(TypeError):
            _PROVIDER_REGISTRY["openai"] = lambda settings, client: None  # type: ignore[index]

    def test_registry_holds_builders_not_instances(self):
        assert all(callable(builder) for builder in _PROVIDER_REGISTRY.values())
