# ruff: noqa: S101
"""Token usage tracking, cost estimates and the cost sensors."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from homeassistant.util import dt as dt_util
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.home_generative_agent.config_flow import _model_prices_error
from custom_components.home_generative_agent.const import CONF_MODEL_PRICES
from custom_components.home_generative_agent.core import cost_tracker as ct
from custom_components.home_generative_agent.core.cost_tracker import (
    META_CATEGORY,
    META_PROVIDER,
    ModelPrice,
    UsageCostTracker,
    UsageRecord,
    estimate_cost,
    normalize_model_name,
    parse_model_prices,
    price_for,
)
from custom_components.home_generative_agent.core.usage_cost_sensor import (
    build_usage_cost_sensors,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gpt-5-2025-08-07", "gpt-5"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("models/gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
        ("OpenAI/GPT-6-Luna", "gpt-6-luna"),
        (None, ""),
    ],
)
def test_normalize_model_name(raw: Any, expected: str) -> None:
    """Provider prefixes and date snapshots map to the base model."""
    assert normalize_model_name(raw) == expected


def test_price_lookup_order() -> None:
    """User override beats built-in; Ollama is free; unknown is None."""
    overrides = {"gpt-5": ModelPrice(9.0, 9.0)}
    assert price_for("gpt-5", "openai", overrides) == ModelPrice(9.0, 9.0)
    assert price_for("gpt-5-2025-08-07", "openai") == ct.BUILTIN_PRICES["gpt-5"]
    assert price_for("qwen3:8b", "ollama") == ModelPrice(0.0, 0.0, 0.0)
    # Sources disagreed on GPT-6 Luna, so it has no built-in price.
    assert price_for("gpt-6-luna", "openai") is None
    # A newer point release is not silently priced as its base model.
    assert price_for("gpt-5.4", "openai") is None
    # An OpenAI-compatible server bills its own rates, even for an OpenAI name.
    assert price_for("openai/gpt-5", "openai_compatible") is None
    assert price_for("gpt-5", "openai_compatible", overrides) == ModelPrice(9.0, 9.0)


def test_estimate_cost_with_cached_input() -> None:
    """Cached input tokens bill at the cached rate, the rest at input."""
    price = ModelPrice(input=1.0, output=10.0, cached_input=0.1)
    cost = estimate_cost(price, 1_000_000, 400_000, 100_000)
    assert cost == pytest.approx(0.6 + 0.04 + 1.0)
    # No cached rate: cache reads bill as input.
    assert estimate_cost(ModelPrice(1.0, 10.0), 1_000_000, 400_000, 0) == 1.0
    assert estimate_cost(None, 10, 0, 10) is None


@pytest.mark.parametrize(
    "text",
    [
        '{"gpt-6-luna": [0.1, 0.5, 0.01]}',
        '{"gpt-6-luna": [0.1, 0.5]}',
        '{"gpt-6-luna": {"input": 0.1, "output": 0.5, "cached_input": 0.01}}',
        "",
        None,
    ],
)
def test_parse_model_prices_valid(text: str | None) -> None:
    """Lists, objects and empty input are accepted."""
    prices = parse_model_prices(text)
    if text:
        assert prices["gpt-6-luna"].input == pytest.approx(0.1)
        assert prices["gpt-6-luna"].output == pytest.approx(0.5)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        '{"gpt-6-luna": [0.1]}',
        '{"gpt-6-luna": [-1, 0.5]}',
        '{"gpt-6-luna": {"input": "cheap", "output": 0.5}}',
        '{"gpt-6-luna": [true, 0.5]}',
        '{"gpt-6-luna": {"input": 0.1, "output": 0.5, "cached_input": -1}}',
        '{"gpt-6-luna": [NaN, 0.5]}',
        '{"gpt-6-luna": [0.1, Infinity]}',
    ],
)
def test_parse_model_prices_invalid(text: str) -> None:
    """Malformed tables raise ValueError and fail the options form."""
    with pytest.raises(ValueError):  # noqa: PT011
        parse_model_prices(text)
    assert _model_prices_error({CONF_MODEL_PRICES: text}) == "invalid_model_prices"


def test_options_accepts_valid_prices() -> None:
    """A valid table (or none) passes the options form."""
    assert _model_prices_error({CONF_MODEL_PRICES: '{"x": [1, 2]}'}) is None
    assert _model_prices_error({}) is None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


# With this price a call's input token count sets its cost: 1M tokens is $1.
TEST_PRICES = {"gpt-5": ModelPrice(1.0, 0.0)}


def _tracker(hass: HomeAssistant) -> UsageCostTracker:
    return UsageCostTracker(hass, TEST_PRICES)


def _record(ts: float, cost: float | None, model: str = "gpt-5", **kw: Any) -> Any:
    """Record a call costing ``cost`` at TEST_PRICES (None: an unpriced model)."""
    return UsageRecord(
        ts=ts,
        category=kw.get("category", "chat"),
        provider=kw.get("provider", "openai"),
        model="gpt-6-luna" if cost is None else model,
        api=kw.get("api", "responses"),
        input_tokens=round((cost or 0) * 1_000_000),
        cached_tokens=0,
        output_tokens=kw.get("output_tokens", 10),
    )


@pytest.mark.asyncio
async def test_summary_windows_and_projections(hass: HomeAssistant) -> None:
    """24h and 7d windows sum costs and project over the covered time."""
    tracker = _tracker(hass)
    now = time.time()
    tracker._started = now - 10 * ct.DAY_S
    tracker.async_record(_record(now - 3 * ct.DAY_S, 0.70))
    tracker.async_record(_record(now - 3600, 0.20))
    tracker.async_record(_record(now - 60, None))

    summary = tracker.summary(now)
    day, week = summary["day"], summary["week"]
    assert day["cost_usd"] == pytest.approx(0.20)
    assert day["calls"] == 2
    assert day["unpriced_calls"] == 1
    assert day["unpriced_models"] == ["gpt-6-luna"]
    assert day["projected_monthly_usd"] == pytest.approx(0.20 * ct.DAYS_PER_MONTH)
    assert week["cost_usd"] == pytest.approx(0.90)
    assert week["projected_monthly_usd"] == pytest.approx(0.90 / 7 * ct.DAYS_PER_MONTH)
    assert week["by_model"]["gpt-5 (responses, openai)"]["calls"] == 2
    assert week["by_model"]["gpt-6-luna (responses, openai)"]["cost_usd"] is None
    assert summary["last_call"]["model"] == "gpt-6-luna"


@pytest.mark.asyncio
async def test_projection_uses_time_actually_tracked(hass: HomeAssistant) -> None:
    """Two days of data project from two days, not seven."""
    tracker = _tracker(hass)
    now = time.time()
    tracker._started = now - 2 * ct.DAY_S
    tracker.async_record(_record(now - ct.DAY_S, 1.0))
    week = tracker.summary(now)["week"]
    assert week["covered_days"] == pytest.approx(2.0)
    assert week["projected_monthly_usd"] == pytest.approx(1.0 / 2 * ct.DAYS_PER_MONTH)


@pytest.mark.asyncio
async def test_old_records_are_pruned(hass: HomeAssistant) -> None:
    """Records older than seven days are dropped when new ones arrive."""
    tracker = _tracker(hass)
    now = time.time()
    tracker.async_record(_record(now - 8 * ct.DAY_S, 5.0))
    tracker.async_record(_record(now, 1.0))
    assert len(tracker._records) == 1


@pytest.mark.asyncio
async def test_records_persist_across_restarts(hass: HomeAssistant) -> None:
    """Flushed records load back; expired ones do not."""
    tracker = _tracker(hass)
    now = time.time()
    tracker.async_record(_record(now - 60, 0.5))
    await tracker.async_flush()

    reloaded = _tracker(hass)
    await reloaded.async_load()
    assert reloaded.summary(now)["day"]["cost_usd"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_record_threadsafe_from_executor(hass: HomeAssistant) -> None:
    """Sync model calls in executor threads are recorded on the loop."""
    tracker = _tracker(hass)
    worker = threading.Thread(
        target=tracker.record_threadsafe, args=(_record(time.time(), 0.1),)
    )
    worker.start()
    worker.join()
    await asyncio.sleep(0)
    await hass.async_block_till_done()
    assert len(tracker._records) == 1


# ---------------------------------------------------------------------------
# Callback, end to end through a LangChain chat model
# ---------------------------------------------------------------------------


def _model_returning(message: AIMessage, tracker: UsageCostTracker) -> Any:
    return FakeMessagesListChatModel(responses=[message]).with_config(
        callbacks=[tracker.callback],
        metadata={META_CATEGORY: "chat", META_PROVIDER: "openai"},
    )


@pytest.mark.asyncio
async def test_callback_records_responses_api_call(hass: HomeAssistant) -> None:
    """Usage, model, API and cost are taken from the provider's reply."""
    tracker = UsageCostTracker(hass)
    reply = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 2000,
            "output_tokens": 100,
            "total_tokens": 2100,
            "input_token_details": {"cache_read": 1000},
        },
        response_metadata={"model_name": "gpt-5-2025-08-07", "id": "resp_abc"},
    )
    await _model_returning(reply, tracker).ainvoke("hello")
    await hass.async_block_till_done()

    last = tracker.summary()["last_call"]
    assert last["model"] == "gpt-5"
    assert last["api"] == "responses"
    assert last["category"] == "chat"
    assert last["cached_tokens"] == 1000
    price = ct.BUILTIN_PRICES["gpt-5"]
    expected = estimate_cost(price, 2000, 1000, 100)
    assert last["cost"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_callback_records_chat_completions_sync_call(
    hass: HomeAssistant,
) -> None:
    """A sync call from an executor thread (Sentinel's path) is recorded too."""
    tracker = UsageCostTracker(hass)
    reply = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "gpt-6-luna", "id": "chatcmpl-1"},
    )
    model = _model_returning(reply, tracker)
    await hass.async_add_executor_job(model.invoke, "hello")
    await hass.async_block_till_done()

    last = tracker.summary()["last_call"]
    assert last["api"] == "chat_completions"
    assert last["cost"] is None  # no built-in price for gpt-6-luna


@pytest.mark.asyncio
async def test_callback_ignores_untagged_models(hass: HomeAssistant) -> None:
    """Models without the HGA metadata (e.g. tests, embeddings) are not counted."""
    tracker = UsageCostTracker(hass)
    reply = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    model = FakeMessagesListChatModel(responses=[reply]).with_config(
        callbacks=[tracker.callback]
    )
    await model.ainvoke("hello")
    await hass.async_block_till_done()
    assert tracker.summary()["last_call"] is None


@pytest.mark.asyncio
async def test_user_price_applies_to_recorded_calls(hass: HomeAssistant) -> None:
    """A price entered in the options prices an otherwise unknown model."""
    tracker = UsageCostTracker(
        hass, parse_model_prices('{"gpt-6-luna": [0.1, 0.5, 0.01]}')
    )
    reply = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        },
        response_metadata={"model_name": "gpt-6-luna", "id": "resp_1"},
    )
    await _model_returning(reply, tracker).ainvoke("hello")
    await hass.async_block_till_done()
    assert tracker.summary()["last_call"]["cost"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensors_expose_costs_and_breakdown(hass: HomeAssistant) -> None:
    """Each sensor reads its window; attributes carry the per-model breakdown."""
    tracker = _tracker(hass)
    now = time.time()
    tracker._started = now - 10 * ct.DAY_S
    tracker.async_record(_record(now - 60, 0.25))

    sensors = {
        s.entity_description.key: s
        for s in build_usage_cost_sensors(hass, tracker, "entry1")
    }
    assert set(sensors) == {
        "llm_cost_last_call",
        "llm_cost_24h",
        "llm_cost_7d",
        "llm_cost_monthly_projection_24h",
        "llm_cost_monthly_projection_7d",
    }
    assert sensors["llm_cost_last_call"].native_value == pytest.approx(0.25)
    assert sensors["llm_cost_24h"].native_value == pytest.approx(0.25)
    assert sensors["llm_cost_monthly_projection_24h"].native_value == pytest.approx(
        0.25 * ct.DAYS_PER_MONTH
    )
    attrs = sensors["llm_cost_7d"].extra_state_attributes
    assert attrs is not None
    assert attrs["by_model"]["gpt-5 (responses, openai)"]["input_tokens"] == 250_000
    assert sensors["llm_cost_24h"].native_unit_of_measurement == "USD"
    assert sensors["llm_cost_24h"].unique_id == "llm_cost_24h::entry1"


@pytest.mark.asyncio
async def test_streamed_chat_completions_reply_is_labeled(hass: HomeAssistant) -> None:
    """A streamed Chat Completions reply has no id but is still labeled."""
    tracker = UsageCostTracker(hass)
    reply = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "gpt-5-2025-08-07", "finish_reason": "stop"},
    )
    await _model_returning(reply, tracker).ainvoke("hello")
    await hass.async_block_till_done()
    assert tracker.summary()["last_call"]["api"] == "chat_completions"


@pytest.mark.asyncio
async def test_call_without_usage_counts_as_unpriced(hass: HomeAssistant) -> None:
    """A reply with no usage (no stream usage) is counted, not dropped or free."""
    tracker = UsageCostTracker(hass)
    reply = AIMessage(content="hi", response_metadata={"model_name": "gpt-5"})
    await _model_returning(reply, tracker).ainvoke("hello")
    await hass.async_block_till_done()
    day = tracker.summary()["day"]
    assert day["calls"] == 1
    assert day["unpriced_calls"] == 1


@pytest.mark.asyncio
async def test_closed_tracker_drops_late_calls(hass: HomeAssistant) -> None:
    """After unload, a call still in flight does not reach the old tracker."""
    tracker = _tracker(hass)
    await tracker.async_flush()
    tracker.async_record(_record(time.time(), 1.0))
    assert len(tracker._records) == 0


@pytest.mark.asyncio
async def test_record_cap_limits_projection_span(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the cap drops calls, projections use only the time still covered."""
    monkeypatch.setattr(ct, "MAX_RECORDS", 2)
    tracker = _tracker(hass)
    now = time.time()
    tracker._started = now - 10 * ct.DAY_S
    for age in (3, 2, 1):
        tracker.async_record(_record(now - age * ct.DAY_S, 1.0))
    week = tracker.summary(now)["week"]
    assert week["covered_days"] == pytest.approx(2.0)
    assert week["projected_monthly_usd"] == pytest.approx(2.0 / 2 * ct.DAYS_PER_MONTH)


@pytest.mark.asyncio
async def test_price_change_reprices_retained_calls(hass: HomeAssistant) -> None:
    """A corrected price applies to calls already recorded, after a reload too."""
    tracker = _tracker(hass)
    now = time.time()
    tracker.async_record(_record(now - 60, 0.5))
    await tracker.async_flush()

    repriced = UsageCostTracker(hass, {"gpt-5": ModelPrice(2.0, 0.0)})
    await repriced.async_load()
    summary = repriced.summary(now)
    assert summary["day"]["cost_usd"] == pytest.approx(1.0)
    assert summary["last_call"]["cost"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_load_ignores_legacy_and_bad_records(hass: HomeAssistant) -> None:
    """A stored cost from an earlier build is ignored; malformed rows are skipped."""
    now = time.time()
    legacy = {
        "ts": now - 60,
        "category": "chat",
        "provider": "openai",
        "model": "gpt-5",
        "api": "responses",
        "input_tokens": 1_000_000,
        "cached_tokens": 0,
        "output_tokens": 0,
        "cost": 99.0,
    }
    tracker = _tracker(hass)
    await tracker._store.async_save(
        {"started": now - ct.DAY_S, "records": [legacy, "junk", {"ts": now}]}
    )
    reloaded = _tracker(hass)
    await reloaded.async_load()
    assert len(reloaded._records) == 1
    assert reloaded.summary(now)["day"]["cost_usd"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_unreported_usage_is_unpriced_but_free_models_stay_free(
    hass: HomeAssistant,
) -> None:
    """No usage means unknown cost for a paid model, but zero for a free one."""
    tracker = _tracker(hass)
    paid = _record(time.time(), 0.0)
    paid.usage_reported = False
    free = _record(time.time(), 0.0, provider="ollama", model="llama3")
    free.usage_reported = False
    assert tracker.cost_of(paid) is None
    assert tracker.cost_of(free) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        {"started": "yesterday", "records": []},
        {"started": None, "records": 5},
        {"records": {"not": "a list"}},
    ],
)
async def test_load_survives_malformed_store(
    hass: HomeAssistant, stored: dict[str, Any]
) -> None:
    """A corrupted container never fails setup; it just loads nothing."""
    tracker = _tracker(hass)
    await tracker._store.async_save(stored)
    reloaded = _tracker(hass)
    await reloaded.async_load()
    assert len(reloaded._records) == 0
    assert reloaded.summary()["week"]["calls"] == 0


@pytest.mark.asyncio
async def test_load_skips_records_with_bad_field_types(hass: HomeAssistant) -> None:
    """Rows with wrongly typed fields are skipped so summary() cannot fail."""
    now = time.time()
    good = {
        "ts": now - 60,
        "category": "chat",
        "provider": "openai",
        "model": "gpt-5",
        "api": "responses",
        "input_tokens": 1_000_000,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    bad_rows = [
        {**good, "input_tokens": "1000"},
        {**good, "ts": "now"},
        {**good, "model": None},
        {**good, "cached_tokens": -1},
        {**good, "usage_reported": "yes"},
        {**good, "ts": float("nan")},
    ]
    tracker = _tracker(hass)
    await tracker._store.async_save({"started": now, "records": [good, *bad_rows]})
    reloaded = _tracker(hass)
    await reloaded.async_load()
    assert len(reloaded._records) == 1
    assert reloaded.summary(now)["day"]["cost_usd"] == pytest.approx(1.0)


def test_provider_scoped_price_wins_for_that_provider_only() -> None:
    """A price nested under a provider type leaves other providers alone."""
    overrides = parse_model_prices(
        '{"openai_compatible": {"gpt-5": [0.5, 2.0]}, "my-model": [1, 2]}'
    )
    assert price_for("gpt-5", "openai_compatible", overrides) == ModelPrice(0.5, 2.0)
    # First-party OpenAI keeps its built-in price.
    assert price_for("gpt-5", "openai", overrides) == ct.BUILTIN_PRICES["gpt-5"]
    # Plain entries still apply to every non-local provider.
    assert price_for("my-model", "openai_compatible", overrides) == ModelPrice(1, 2)


def test_plain_price_does_not_make_ollama_paid() -> None:
    """Ollama stays free unless priced under its own provider type."""
    plain = parse_model_prices('{"gpt-oss": [1, 2]}')
    assert price_for("gpt-oss", "ollama", plain) == ModelPrice(0.0, 0.0, 0.0)
    scoped = parse_model_prices('{"ollama": {"gpt-oss": [0.1, 0.2]}}')
    assert price_for("gpt-oss", "ollama", scoped) == ModelPrice(0.1, 0.2)


def test_provider_key_with_price_fields_is_a_model_entry() -> None:
    """A provider-named key holding a price object is read as a model price."""
    prices = parse_model_prices('{"openai": {"input": 1, "output": 2}}')
    assert prices == {"openai": ModelPrice(1.0, 2.0)}


def test_invalid_scoped_price_names_provider_and_model() -> None:
    """Errors inside a provider block say which entry is wrong."""
    with pytest.raises(ValueError, match="openai_compatible/gpt-5"):
        parse_model_prices('{"openai_compatible": {"gpt-5": [1]}}')


@pytest.mark.asyncio
async def test_periodic_tick_prunes_and_saves_expired_records(
    hass: HomeAssistant,
) -> None:
    """With no new calls, the refresh tick still expires old records."""
    tracker = _tracker(hass)
    now = time.time()
    tracker._records.append(_record(now - 8 * ct.DAY_S, 1.0))
    tracker._records.append(_record(now - 60, 0.5))
    with patch.object(tracker._store, "async_delay_save") as delay_save:
        unsub = tracker.async_start_refresh()
        async_fire_time_changed(hass, dt_util.utcnow() + ct.REFRESH_INTERVAL)
        await hass.async_block_till_done()
        unsub()
    assert len(tracker._records) == 1
    delay_save.assert_called_once()
    assert tracker.summary()["last_call"]["cost"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_periodic_tick_without_expired_records_does_not_save(
    hass: HomeAssistant,
) -> None:
    """A tick that removes nothing does not rewrite the store."""
    tracker = _tracker(hass)
    tracker._records.append(_record(time.time() - 60, 0.5))
    with patch.object(tracker._store, "async_delay_save") as delay_save:
        unsub = tracker.async_start_refresh()
        async_fire_time_changed(hass, dt_util.utcnow() + ct.REFRESH_INTERVAL)
        await hass.async_block_till_done()
        unsub()
    delay_save.assert_not_called()


def test_oversized_price_is_rejected_not_crashing() -> None:
    """A JSON integer too large for a float is an invalid price, not a crash."""
    huge = "1" + "0" * 400
    with pytest.raises(ValueError, match="non-negative"):
        parse_model_prices(f'{{"gpt-5": [{huge}, 1]}}')
    assert (
        _model_prices_error({CONF_MODEL_PRICES: f'{{"x": [{huge}, 1]}}'})
        == "invalid_model_prices"
    )


@pytest.mark.asyncio
async def test_load_skips_oversized_numbers(hass: HomeAssistant) -> None:
    """Huge stored integers are skipped instead of aborting the load."""
    huge = 10**400
    now = time.time()
    row = {
        "ts": huge,
        "category": "chat",
        "provider": "openai",
        "model": "gpt-5",
        "api": None,
        "input_tokens": 1,
        "cached_tokens": 0,
        "output_tokens": 1,
    }
    reloaded = _tracker(hass)
    # HA's own JSON layer cannot round-trip such integers; feed the loader
    # directly, as a hand-edited or foreign store could.
    with patch.object(
        reloaded._store,
        "async_load",
        return_value={"started": huge, "records": [row]},
    ):
        await reloaded.async_load()
    assert len(reloaded._records) == 0
    assert reloaded._started <= now + 1


@pytest.mark.asyncio
async def test_breakdown_separates_providers_of_the_same_model(
    hass: HomeAssistant,
) -> None:
    """An unpriced proxy call does not hide the priced OpenAI subtotal."""
    tracker = UsageCostTracker(hass, {"gpt-5": ModelPrice(1.0, 0.0)})
    now = time.time()
    tracker._started = now - ct.DAY_S
    tracker.async_record(_record(now - 60, 0.5))  # openai, priced
    tracker.async_record(_record(now - 30, 0.25, provider="openai_compatible"))
    by_model = tracker.summary(now)["day"]["by_model"]
    assert by_model["gpt-5 (responses, openai)"]["cost_usd"] == pytest.approx(0.5)
    # Plain entries price compatible servers too; scope one to change that.
    assert by_model["gpt-5 (responses, openai_compatible)"]["calls"] == 1


@pytest.mark.parametrize("bad", [10**400, -5, ct.MAX_TOKEN_COUNT + 1])
def test_implausible_token_counts_are_rejected(bad: int) -> None:
    """Stored rows and provider usage with absurd counts never reach the math."""
    row = {
        "ts": time.time(),
        "category": "chat",
        "provider": "openai",
        "model": "gpt-5",
        "api": None,
        "input_tokens": bad,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    assert ct._record_from_dict(row) is None
    assert ct._int(bad) == 0


@pytest.mark.asyncio
async def test_callback_ignores_absurd_reported_usage(hass: HomeAssistant) -> None:
    """A provider reporting an absurd count records zero, not a crash."""
    tracker = UsageCostTracker(hass)
    reply = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 10**400,
            "output_tokens": 5,
            "total_tokens": 10**400,
        },
        response_metadata={"model_name": "gpt-5", "id": "resp_1"},
    )
    await _model_returning(reply, tracker).ainvoke("hello")
    await hass.async_block_till_done()
    last = tracker.summary()["last_call"]
    assert last["input_tokens"] == 0
    assert last["output_tokens"] == 5
