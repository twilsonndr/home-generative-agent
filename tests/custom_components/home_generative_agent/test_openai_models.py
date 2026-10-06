# ruff: noqa: S101
"""OpenAI model listing and Responses API request shaping."""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI

from custom_components.home_generative_agent.core import utils
from custom_components.home_generative_agent.core.fallback import (
    unsupported_sampling_param,
)
from custom_components.home_generative_agent.core.utils import (
    filter_openai_models,
    list_openai_models,
    merge_model_options,
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
        f"Error code: 400 - {{'error': {body}}}", response=response, body=body
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
