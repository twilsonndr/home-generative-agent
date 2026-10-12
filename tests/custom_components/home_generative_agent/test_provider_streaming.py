# ruff: noqa: S101
"""Verify that OpenAI and Anthropic providers are constructed with streaming enabled."""

from __future__ import annotations

import pathlib

_INIT = pathlib.Path("custom_components/home_generative_agent/__init__.py").read_text()


def _provider_block(marker: str, chars: int = 600) -> str:
    idx = _INIT.index(marker)
    return _INIT[idx : idx + chars]


def test_openai_provider_has_streaming_true() -> None:
    """ChatOpenAI construction includes streaming=True."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "streaming=True" in block


def test_openai_provider_has_stream_usage_true() -> None:
    """ChatOpenAI construction includes stream_usage=True to preserve usage_metadata."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "stream_usage=True" in block


def test_anthropic_provider_has_streaming_true() -> None:
    """ChatAnthropic construction includes streaming=True."""
    block = _provider_block("anthropic_chat = SharedClientChatAnthropic(")
    assert "streaming=True" in block


def test_openai_provider_http_client_preserved() -> None:
    """ChatOpenAI construction still wires the sync and async HTTP clients."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "http_client=openai_http_client" in block
    assert "http_async_client=http_async_client" in block


def test_anthropic_provider_primed_before_publish() -> None:
    """The SDK client is primed off the loop before the provider exists (#587, #618)."""
    start = _INIT.index("anthropic_chat = SharedClientChatAnthropic(")
    end = _INIT.index("except Exception:", start)
    block = _INIT[start:end]
    prime = block.index("await async_prime_async_client(hass, anthropic_chat)")
    publish = block.index("anthropic_provider = anthropic_chat.configurable_fields(")
    assert prime < publish


def test_openai_provider_defaults_to_chat_completions() -> None:
    """Cloud OpenAI stays on Chat Completions unless a model needs /responses."""
    block = _provider_block("openai_provider = ChatOpenAI(", chars=1600)
    assert "use_responses_api=False" in block
    assert 'use_responses_api=ConfigurableField(id="use_responses_api")' in block


def test_openai_compatible_provider_defaults_to_chat_completions() -> None:
    """OpenAI-compatible servers default to Chat Completions, choosable per model."""
    start = _INIT.index("openai_compatible_provider = ChatOpenAI(")
    end = _INIT.index("except Exception:", start)
    block = _INIT[start:end]
    assert "use_responses_api=False" in block
    assert 'use_responses_api=ConfigurableField(id="use_responses_api")' in block


def test_every_openai_model_config_applies_the_api_choice() -> None:
    """Chat, VLM and summary (both OpenAI providers) and fallbacks honor it."""
    calls = _INIT.count("_openai_api_config(") - _INIT.count("def _openai_api_config(")
    # chat x2, vlm x2, summarization x2, cloud fallback x1
    assert calls == 7
