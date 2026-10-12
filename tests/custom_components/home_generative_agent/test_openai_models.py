# ruff: noqa: S101
"""OpenAI model listing and Responses API request shaping."""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest
from langchain_core.runnables import ConfigurableField, RunnableBinding
from langchain_openai import ChatOpenAI

from custom_components.home_generative_agent.core import utils
from custom_components.home_generative_agent.core.fallback import (
    unsupported_sampling_param,
)
from custom_components.home_generative_agent.core.openai_models import (
    openai_supports_no_reasoning,
)
from custom_components.home_generative_agent.core.utils import (
    filter_openai_models,
    list_openai_models,
    merge_model_options,
    openai_api_preference,
    openai_uses_responses_api,
    resolve_openai_api,
)


class _FakeClient:
    def __init__(self, response: httpx.Response | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        self.calls.append((url, headers))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _patch_client(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response | Exception
) -> _FakeClient:
    client = _FakeClient(response)
    monkeypatch.setattr(utils, "get_async_client", lambda _hass: client)
    return client


@pytest.mark.asyncio
async def test_list_openai_models_reads_official_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI listing hits /v1/models with the key and returns the ids."""
    client = _patch_client(
        monkeypatch,
        httpx.Response(
            200, json={"data": [{"id": "gpt-6-luna"}, {"id": "gpt-transcribe"}, {}]}
        ),
    )
    assert await list_openai_models(None, "sk-test") == [  # type: ignore[arg-type]
        "gpt-6-luna",
        "gpt-transcribe",
    ]
    assert client.calls == [
        ("https://api.openai.com/v1/models", {"Authorization": "Bearer sk-test"})
    ]


@pytest.mark.asyncio
async def test_list_openai_models_compatible_base_url_keyless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyless compatible server gets /v1 appended and no auth header."""
    client = _patch_client(monkeypatch, httpx.Response(200, json={"data": []}))
    assert await list_openai_models(None, None, "http://box:8000") == []  # type: ignore[arg-type]
    assert client.calls == [("http://box:8000/v1/models", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": "nope"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=[{"id": "x"}]),
        httpx.ConnectError("down"),
    ],
)
async def test_list_openai_models_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch, response: Any
) -> None:
    """Any listing failure falls back to an empty list."""
    _patch_client(monkeypatch, response)
    assert await list_openai_models(None, "sk-test") == []  # type: ignore[arg-type]


def test_filter_openai_models_keeps_fine_tuned_models() -> None:
    """Fine-tuned ids are classified by their base model."""
    ids = [
        "ft:gpt-4o-mini-2024-07-18:acme::abc123",
        "ft:gpt-4o-mini-tts:acme::def456",
        "ft:davinci-002:acme::ghi789",
    ]
    assert filter_openai_models(ids, "chat") == [
        "ft:gpt-4o-mini-2024-07-18:acme::abc123"
    ]
    assert filter_openai_models(ids, "tts") == ["ft:gpt-4o-mini-tts:acme::def456"]


def test_openai_routing_classifies_fine_tuned_models() -> None:
    """Routing rules see through the ft: prefix."""
    assert openai_uses_responses_api("ft:gpt-6-luna:acme::x1")
    assert not openai_supports_no_reasoning("ft:gpt-6-astra:acme::x2")


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        ("gpt-6-luna", True),
        ("gpt-6-sol", True),
        ("gpt-5.1", True),
        ("gpt-6.1-sol", False),
        ("gpt-6.2-sol", False),
        ("gpt-6-astra", False),
        ("GPT-6-Astra", False),
    ],
)
def test_openai_supports_no_reasoning(model: str, *, supported: bool) -> None:
    """Only GPT-6 Astra and GPT-6.1+ Sol reject the "none" effort."""
    assert openai_supports_no_reasoning(model) is supported


def test_filter_openai_models_by_category() -> None:
    """Each category only keeps the model families that can serve it."""
    ids = [
        "gpt-6-luna",
        "gpt-6.1-sol",
        "o4-mini",
        "gpt-transcribe",
        "gpt-live-transcribe",
        "whisper-1",
        "gpt-4o-mini-tts",
        "gpt-image-2",
        "gpt-realtime-2",
        "text-embedding-3-small",
        "omni-moderation-latest",
    ]
    assert filter_openai_models(ids, "chat") == ["gpt-6-luna", "gpt-6.1-sol", "o4-mini"]
    assert filter_openai_models(ids, "vlm") == filter_openai_models(ids, "chat")
    assert filter_openai_models(ids, "stt") == [
        "gpt-live-transcribe",
        "gpt-transcribe",
        "whisper-1",
    ]
    assert filter_openai_models(ids, "tts") == ["gpt-4o-mini-tts"]
    assert filter_openai_models(ids, "embedding") == ["text-embedding-3-small"]


def test_merge_model_options_keeps_builtin_order_first() -> None:
    """Built-ins lead in their own order; listed extras follow, deduplicated."""
    assert merge_model_options(["b", "a"], ["a", "c", "b", "d"]) == [
        "b",
        "a",
        "c",
        "d",
    ]


def _bad_request(param: str | None, code: str | None, message: str) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {
        "message": message,
        "type": "invalid_request_error",
        "param": param,
        "code": code,
    }
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}",
        response=response,  # pyright: ignore[reportArgumentType]  # httpx vs openai 3's httpx2
        body=body,
    )


@pytest.mark.parametrize("param", [None, "reasoning_effort", "tools"])
def test_gpt6_tools_with_reasoning_effort_is_droppable(param: str | None) -> None:
    """The GPT-6 Chat Completions tools/reasoning 400 drops reasoning_effort."""
    err = _bad_request(
        param,
        None,
        "Function tools with reasoning_effort are not supported for gpt-6-luna "
        "in /v1/chat/completions. To use function tools, use /v1/responses or "
        "set reasoning_effort to 'none'.",
    )
    assert unsupported_sampling_param(err) == "reasoning_effort"


def test_responses_api_dotted_reasoning_param_is_droppable() -> None:
    """The Responses API reports the effort as reasoning.effort."""
    err = _bad_request(
        "reasoning.effort",
        "unsupported_parameter",
        "Unsupported parameter: 'reasoning.effort' is not supported with this model.",
    )
    assert unsupported_sampling_param(err) == "reasoning_effort"


def test_unrelated_not_supported_message_is_not_droppable() -> None:
    """Other 'not supported' 400s still surface instead of retrying."""
    err = _bad_request("messages", None, "Audio input is not supported.")
    assert unsupported_sampling_param(err) is None


def test_responses_api_payload_sends_reasoning_effort_with_tools() -> None:
    """Tools + reasoning effort go out as a Responses request, not chat/completions."""
    model = ChatOpenAI(
        api_key="sk-test",  # type: ignore[arg-type]
        model="gpt-6-luna",
        use_responses_api=True,
        reasoning_effort="low",
        max_completion_tokens=256,
    )

    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return city

    bound = model.bind_tools([get_weather])
    assert isinstance(bound, RunnableBinding)
    payload = model._get_request_payload(
        [("user", "weather in Paris?")], **bound.kwargs
    )
    assert payload["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in payload
    assert payload["max_output_tokens"] == 256
    assert "input" in payload
    assert "messages" not in payload
    assert payload["tools"][0]["name"] == "get_weather"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-6-luna", True),
        ("gpt-6.1-sol", True),
        ("GPT-6-Astra", True),
        ("gpt-5", False),
        ("gpt-4o", False),
        ("o4-mini", False),
        ("gpt-5.1-codex", False),
        (None, False),
    ],
)
def test_openai_uses_responses_api_only_for_gpt6(
    model: str | None, *, expected: bool
) -> None:
    """Only GPT-6-family models leave the tested Chat Completions path."""
    assert openai_uses_responses_api(model) is expected


def test_configurable_use_responses_api_routes_gpt6_tool_calls() -> None:
    """The per-model configurable flips a Chat Completions provider to /responses."""
    # Any: _prepare lives on RunnableConfigurableFields, which the return type
    # of configurable_fields() does not expose.
    base: Any = ChatOpenAI(
        api_key="sk-test",  # type: ignore[arg-type]
        use_responses_api=False,
    ).configurable_fields(
        model_name=ConfigurableField(id="model_name"),
        reasoning_effort=ConfigurableField(id="reasoning_effort"),
        use_responses_api=ConfigurableField(id="use_responses_api"),
    )

    def payload_for(model: str) -> dict[str, Any]:
        chat, _ = base._prepare(
            {
                "configurable": {
                    "model_name": model,
                    "reasoning_effort": "low",
                    "use_responses_api": openai_uses_responses_api(model),
                }
            }
        )
        assert isinstance(chat, ChatOpenAI)
        return chat._get_request_payload([("user", "hi")], tools=[_WEATHER_TOOL])

    gpt6 = payload_for("gpt-6-luna")
    assert gpt6["reasoning"] == {"effort": "low"}
    assert "input" in gpt6

    gpt5 = payload_for("gpt-5")
    assert gpt5["reasoning_effort"] == "low"
    assert "messages" in gpt5


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.parametrize(
    ("provider_type", "category", "model", "preference", "expected"),
    [
        ("openai", "chat", "gpt-6-luna", None, "responses"),
        ("openai", "chat", "gpt-5", None, "chat_completions"),
        ("openai", "vlm", "gpt-6-luna", None, "chat_completions"),
        ("openai", "summarization", "gpt-6-luna", "auto", "chat_completions"),
        ("openai_compatible", "chat", "gpt-6-luna", None, "chat_completions"),
        ("openai", "chat", "gpt-6-luna", "chat_completions", "chat_completions"),
        ("openai", "chat", "gpt-5", "responses", "responses"),
        ("openai_compatible", "vlm", "qwen3-vl", "responses", "responses"),
        ("openai", "chat", "gpt-5", "bogus", "chat_completions"),
        # Responses-only models (listed by /v1/models) leave Chat Completions
        # in every category; OpenAI-compatible servers are left alone.
        ("openai", "chat", "gpt-5.1-codex", None, "responses"),
        ("openai", "vlm", "gpt-5-pro", None, "responses"),
        ("openai", "summarization", "o3-pro-2025-06-10", None, "responses"),
        ("openai", "chat", "gpt-5.6-sol", None, "responses"),
        ("openai", "chat", "gpt-4o-mini", None, "chat_completions"),
        ("openai", "vlm", "gpt-5-pro", "chat_completions", "chat_completions"),
        ("openai_compatible", "chat", "gpt-5.1-codex", None, "chat_completions"),
    ],
)
def test_resolve_openai_api(
    provider_type: str,
    category: str,
    model: str,
    preference: str | None,
    expected: str,
) -> None:
    """An explicit choice wins; automatic picks Responses only where needed."""
    assert (
        resolve_openai_api(
            provider_type=provider_type,
            category=category,
            model=model,
            preference=preference,
        )
        == expected
    )


def test_openai_api_preference_per_model_wins() -> None:
    """A remembered per-model choice beats the flat one."""
    by_model = {"gpt-5": "responses", "gpt-4o": "auto"}
    assert openai_api_preference("chat_completions", by_model, "gpt-5") == "responses"
    # Invalid remembered values fall through to the flat choice.
    assert openai_api_preference("chat_completions", by_model, "gpt-4o") == (
        "chat_completions"
    )
    assert openai_api_preference(None, by_model, "gpt-6-luna") is None


def test_openai_api_preference_fallback_ignores_flat() -> None:
    """Fallback models only use their own remembered choice."""
    assert (
        openai_api_preference("responses", None, "gpt-5", default_to_flat=False) is None
    )
    assert (
        openai_api_preference(
            "chat_completions",
            {"gpt-5": "responses"},
            "gpt-5",
            default_to_flat=False,
        )
        == "responses"
    )


def test_responses_only_models_cover_langchain_routing() -> None:
    """Every name langchain-openai treats as Responses-only routes there here."""
    from langchain_openai.chat_models import base as lc_base  # noqa: PLC0415

    prefixes = lc_base._RESPONSES_API_ONLY_PREFIXES
    for name in (*prefixes, "gpt-5.1-codex", "codex-mini-latest"):
        assert lc_base._model_prefers_responses_api(name)
        assert (
            resolve_openai_api(provider_type="openai", category="vlm", model=name)
            == "responses"
        ), name
