"""OpenAI model-name rules shared by routing, reasoning and fallback code."""

from __future__ import annotations

import re
from typing import Any

_FINE_TUNE_PREFIX = "ft:"

# GPT-6 models without the "none" reasoning effort (GPT-6 Astra, GPT-6.1 Sol).
# On Chat Completions they cannot call function tools at any effort, so their
# tool calls need the Responses API. GPT-6 Sol and GPT-6 Luna accept "none",
# and with it can call tools on Chat Completions.
_NO_NONE_EFFORT_RE = re.compile(r"^gpt-6(?:\.\d+)?-astra|^gpt-6\.[1-9]\d*-sol")


def openai_base_model(model: Any) -> str:
    """
    Return the lowercased base model of an OpenAI model id.

    Fine-tuned ids look like ``ft:gpt-4o-mini-2024-07-18:org:suffix:id``; they
    serve whatever their base model serves, so they are classified by it.
    """
    name = str(model or "").lower()
    if name.startswith(_FINE_TUNE_PREFIX):
        return name.split(":", 2)[1]
    return name


def openai_supports_no_reasoning(model: Any) -> bool:
    """Return False for OpenAI models that reject reasoning_effort "none"."""
    return not _NO_NONE_EFFORT_RE.search(openai_base_model(model))


def openai_chat_tools_need_no_reasoning(model: Any) -> bool:
    """
    Return True when tools on Chat Completions need reasoning_effort "none".

    GPT-6 Sol and Luna call tools on Chat Completions only with "none"; their
    default effort is rejected alongside tools.
    """
    return openai_base_model(model).startswith(
        "gpt-6"
    ) and openai_supports_no_reasoning(model)
