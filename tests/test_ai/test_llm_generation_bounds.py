from __future__ import annotations

import httpx
import pytest

from config.settings import Settings
from llm.exceptions import LLMProviderError
from llm.ollama_provider import OllamaProvider


def _settings(**overrides) -> Settings:
    base = dict(
        llm_provider="ollama",
        llm_base_url="http://test-ollama:11434",
        llm_model="qwen2.5:7b",
        llm_max_retries=3,
        llm_retry_backoff=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_body(**extra) -> dict:
    body = {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": '{"recommendation": "BUY"}'},
        "done": True,
    }
    body.update(extra)
    return body


class TestGenerationBounds:
    """The completion — not the prompt — is the dominant cost on CPU."""

    @pytest.mark.asyncio
    async def test_num_predict_and_num_ctx_are_sent(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=_ok_body())

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as p:
            await p.generate("prompt")

        opts = captured["options"]
        assert opts["num_predict"] == 768
        assert opts["num_ctx"] == 4096
        assert opts["temperature"] == 0.2
        assert captured["stream"] is False

    @pytest.mark.asyncio
    async def test_bounds_are_configurable(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=_ok_body())

        settings = _settings(llm_num_predict=256, llm_num_ctx=8192)
        async with OllamaProvider(settings=settings, client=_client(handler)) as p:
            await p.generate("prompt")

        assert captured["options"]["num_predict"] == 256
        assert captured["options"]["num_ctx"] == 8192

    def test_defaults_are_sized_for_cpu_inference(self):
        s = Settings()
        assert s.llm_request_timeout == 120.0        # 60s cannot fit a full thesis on CPU
        assert s.llm_num_predict == 768
        assert s.llm_num_ctx == 4096
        # context must accommodate prompt + completion
        assert s.llm_num_ctx > s.llm_num_predict


class TestTimeoutRetryPolicy:
    """A read timeout means the model was still generating; retrying re-runs
    the entire expensive generation, so it must NOT be retried."""

    @pytest.mark.asyncio
    async def test_read_timeout_is_not_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ReadTimeout("timed out", request=request)

        async with OllamaProvider(settings=_settings(llm_max_retries=3), client=_client(handler)) as p:
            with pytest.raises(LLMProviderError, match="not retried"):
                await p.generate("prompt")

        assert calls["n"] == 1, "read timeout must fail fast, not multiply wall-clock by max_retries"

    @pytest.mark.asyncio
    async def test_connect_timeout_is_still_retried(self):
        """The request never reached the model, so retrying is cheap."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectTimeout("no route", request=request)

        async with OllamaProvider(settings=_settings(llm_max_retries=3), client=_client(handler)) as p:
            with pytest.raises(LLMProviderError):
                await p.generate("prompt")

        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_read_timeout_error_names_the_timeout_value(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        settings = _settings(llm_request_timeout=42.0)
        async with OllamaProvider(settings=settings, client=_client(handler)) as p:
            with pytest.raises(LLMProviderError, match="42.0s"):
                await p.generate("prompt")


class TestUsageLogging:
    @pytest.mark.asyncio
    async def test_token_counts_and_throughput_are_logged(self, caplog):
        import logging

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_body(
                prompt_eval_count=1012, eval_count=640,
                eval_duration=64_000_000_000, total_duration=70_000_000_000,
            ))

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as p:
            with caplog.at_level(logging.INFO):
                await p.generate("prompt")

        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "prompt_tokens=1012" in msg
        assert "output_tokens=640" in msg
        assert "tok/s" in msg

    @pytest.mark.asyncio
    async def test_usage_log_never_contains_prompt_or_response_content(self, caplog):
        import logging

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_body(prompt_eval_count=10, eval_count=5))

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as p:
            with caplog.at_level(logging.DEBUG):
                await p.generate("SECRET_PROMPT_TEXT")

        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "SECRET_PROMPT_TEXT" not in msg
        assert "recommendation" not in msg

    @pytest.mark.asyncio
    async def test_missing_counters_do_not_break_logging(self):
        """Older Ollama builds may omit the counters entirely."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_body())

        async with OllamaProvider(settings=_settings(), client=_client(handler)) as p:
            result = await p.generate("prompt")
        assert result.text
