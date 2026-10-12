"""
Token usage and cost estimates for every model call.

A LangChain callback (``UsageCostCallback``) is attached to each configured
chat, vision and summarization model. It reads the token usage the provider
reports at the end of each call and records it with the feature, provider,
model and OpenAI API that made the call. ``UsageCostTracker`` keeps seven days
of records in Home Assistant storage and summarizes them for the cost sensors:
the last call, the last 24 hours, the last 7 days, and monthly projections from
each window, broken down per model and API.

Costs are estimates. Prices come from a small built-in table (only models whose
published price sources agreed when it was checked, see PRICES_CHECKED) plus any
per-model prices the user enters in the options; a model with no known price is
counted as unpriced rather than given a guessed cost. Local Ollama models are
free.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import TYPE_CHECKING, Any, TypeGuard

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.storage import Store
from langchain_core.callbacks import BaseCallbackHandler

from ..const import (  # noqa: TID252
    DOMAIN,
    OPENAI_API_CHAT_COMPLETIONS,
    OPENAI_API_RESPONSES,
    SIGNAL_USAGE_COST_UPDATED,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from datetime import datetime
    from uuid import UUID

    from langchain_core.outputs import LLMResult

LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.usage_costs"
SAVE_DELAY_S = 30
RETENTION_S = 7 * 24 * 3600
DAY_S = 24 * 3600
WEEK_S = 7 * DAY_S
# Average month length used for the projections.
DAYS_PER_MONTH = 30.437
TOKENS_PER_PRICE_UNIT = 1_000_000
# Largest token count accepted for one call. Far above any real request, and
# small enough that cost arithmetic can never overflow a float.
MAX_TOKEN_COUNT = 10**12
# Highest accepted price, USD per 1M tokens. Far above any real model, and low
# enough that MAX_TOKEN_COUNT tokens on every kept record stays finite.
MAX_PRICE = 1_000_000.0
# Shortest span a projection scales from (see _window_summary).
MIN_PROJECTION_SPAN_S = 3600.0
# Per-call records kept at most; at a call a minute this is about two weeks.
MAX_RECORDS = 20_000
# The window sensors only change on a new call; this re-summarizes them so the
# 24-hour and 7-day sums also fall as old calls leave the windows.
REFRESH_INTERVAL = timedelta(minutes=5)

# Metadata keys set on each configured model's RunnableConfig.
META_CATEGORY = "hga_category"
META_PROVIDER = "hga_provider"

# Date the built-in prices were checked against the providers' price pages or,
# where those could not be reached, against several agreeing price trackers.
PRICES_CHECKED = "2026-10-12"


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per 1M tokens. cached_input None bills cache reads as input."""

    input: float
    output: float
    cached_input: float | None = None

    @property
    def is_paid(self) -> bool:
        """Return True when any token kind (cached input included) costs money."""
        return bool(self.input or self.output or self.cached_input)


# Only models whose published prices agreed across sources when checked.
# GPT-6 Luna / Sol are left out on purpose: sources disagreed. Users add any
# model (or correct one) in the integration options.
BUILTIN_PRICES: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-6-astra": ModelPrice(10.0, 50.0, 1.0),
    "gpt-5": ModelPrice(1.25, 10.0, 0.125),
    "gpt-5-mini": ModelPrice(0.25, 2.0, 0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.40, 0.005),
    # Anthropic (first-party API rates)
    "claude-opus-5-5": ModelPrice(4.0, 20.0, 0.20),
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, 0.10),
    "claude-opus-4-7": ModelPrice(5.0, 25.0, 0.50),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 0.30),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 0.10),
    # Google
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50, 0.03),
}

FREE_PROVIDER_TYPES = frozenset({"ollama"})
# Built-in prices are first-party list prices; an OpenAI-compatible server
# (a proxy, reseller or local server) bills its own rates, so it is priced only
# from the user's table.
UNPRICED_BY_DEFAULT_PROVIDER_TYPES = frozenset({"openai_compatible"})
OPENAI_PROVIDER_TYPES = frozenset({"openai", "openai_compatible"})
# Provider types a price table can scope entries to (see parse_model_prices).
PRICE_SCOPE_PROVIDER_TYPES = frozenset(
    {"openai", "openai_compatible", "ollama", "gemini", "anthropic"}
)
_PRICE_FIELDS = frozenset({"input", "output", "cached_input"})

_DATE_SUFFIX_RE = re.compile(r"(?:-\d{4}-\d{2}-\d{2}|-\d{8}|@\d{8})$")


def normalize_model_name(model: Any) -> str:
    """Lower-case a model id and drop provider prefixes and date snapshots."""
    name = str(model or "").strip().lower()
    for prefix in ("models/", "openai/", "anthropic/", "google/"):
        name = name.removeprefix(prefix)
    return _DATE_SUFFIX_RE.sub("", name)


def scoped_price_key(provider_type: str, model: Any) -> str:
    """Return the price-table key for a model priced for one provider type."""
    # A unit separator cannot appear in a normalized model name, so a scoped
    # key never collides with a model-only one.
    return f"{provider_type}\x1f{normalize_model_name(model)}"


def _is_provider_scope(key: str, raw: Any) -> bool:
    """Return True for a provider-type key whose value maps models to prices."""
    return (
        key in PRICE_SCOPE_PROVIDER_TYPES
        and isinstance(raw, dict)
        and not _PRICE_FIELDS.intersection(raw)
    )


def parse_model_prices(text: Any) -> dict[str, ModelPrice]:
    """
    Parse the user's price table: a JSON object of model -> price.

    Each price is either ``{"input": .., "output": .., "cached_input": ..}``
    or a list ``[input, output]`` / ``[input, output, cached_input]``, in USD
    per 1M tokens. A provider-type key (``"openai_compatible"``, ``"ollama"``,
    ...) holding its own model -> price object prices those models for that
    provider only, ahead of any model-only entry. Raises ValueError with a
    short reason when invalid.
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return {}
    data = json.loads(text) if isinstance(text, str) else text
    if not isinstance(data, dict):
        msg = "expected a JSON object of model -> price"
        raise ValueError(msg)  # noqa: TRY004 — surfaced as a form error
    prices: dict[str, ModelPrice] = {}
    for key, raw in data.items():
        if _is_provider_scope(str(key), raw):
            for model, scoped_raw in raw.items():
                label = f"{key}/{model}"
                prices[scoped_price_key(str(key), model)] = _parse_price(
                    label, scoped_raw
                )
            continue
        prices[normalize_model_name(key)] = _parse_price(str(key), raw)
    return prices


def _parse_price(model: str, raw: Any) -> ModelPrice:
    if isinstance(raw, dict):
        values = (raw.get("input"), raw.get("output"), raw.get("cached_input"))
    elif isinstance(raw, list) and len(raw) in (2, 3):
        values = (raw[0], raw[1], raw[2] if len(raw) == 3 else None)  # noqa: PLR2004
    else:
        msg = f"price for {model!r} must be an object or a 2-3 item list"
        raise ValueError(msg)
    input_price, output_price, cached_price = values
    for value in (input_price, output_price):
        if not _valid_price(value):
            msg = f"price for {model!r} needs non-negative input and output numbers"
            raise ValueError(msg)
    if cached_price is not None and not _valid_price(cached_price):
        msg = f"cached_input for {model!r} must be a non-negative number"
        raise ValueError(msg)
    return ModelPrice(
        float(input_price),  # type: ignore[arg-type]
        float(output_price),  # type: ignore[arg-type]
        None if cached_price is None else float(cached_price),
    )


def _valid_price(value: Any) -> bool:
    """Return True for a non-negative number no larger than MAX_PRICE."""
    return _is_number(value) and 0 <= value <= MAX_PRICE


def price_for(
    model: Any,
    provider_type: str | None,
    overrides: Mapping[str, ModelPrice] | None = None,
) -> ModelPrice | None:
    """
    Return the price for a model, or None when unknown.

    Order: the user's price for this provider type, then local providers are
    free, then the user's model-only price, then the built-in table (not used
    for OpenAI-compatible servers, which may bill anything).
    """
    name = normalize_model_name(model)
    overrides = overrides or {}
    if provider_type:
        scoped = overrides.get(scoped_price_key(provider_type, name))
        if scoped is not None:
            return scoped
    if provider_type in FREE_PROVIDER_TYPES:
        return ModelPrice(0.0, 0.0, 0.0)
    if name in overrides:
        return overrides[name]
    if provider_type in UNPRICED_BY_DEFAULT_PROVIDER_TYPES:
        return None
    return BUILTIN_PRICES.get(name)


def estimate_cost(
    price: ModelPrice | None,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> float | None:
    """Return the USD cost of one call, or None when the price is unknown."""
    if price is None:
        return None
    cached = max(0, min(cached_tokens, input_tokens))
    cached_rate = price.input if price.cached_input is None else price.cached_input
    return (
        (input_tokens - cached) * price.input
        + cached * cached_rate
        + output_tokens * price.output
    ) / TOKENS_PER_PRICE_UNIT


@dataclass(slots=True)
class UsageRecord:
    """
    One model call.

    Only raw usage is kept; the cost is worked out from the current prices
    whenever a summary is built, so a new or corrected price applies to every
    retained call, not just to calls made after the change.
    """

    ts: float
    category: str
    provider: str
    model: str
    api: str | None
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    # False when the provider reported no usage (e.g. a streamed reply from a
    # server that does not send stream usage).
    usage_reported: bool = True


_RECORD_FIELDS = tuple(f.name for f in fields(UsageRecord))


def _is_number(value: Any) -> TypeGuard[float]:
    """Return True for a finite int or float that fits a float (bool excluded)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        # JSON integers are unbounded; one too large for a float overflows.
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_count(value: Any) -> bool:
    """Return True for a plausible token count: an int in [0, MAX_TOKEN_COUNT]."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_TOKEN_COUNT
    )


def _record_from_dict(raw: Any) -> UsageRecord | None:
    """
    Rebuild a stored record, or None when it is malformed.

    Every field is type-checked so a corrupted store skips bad rows instead of
    failing setup or a later summary. Unknown keys (e.g. a cost saved by an
    earlier build) are ignored; costs are recomputed from the current prices.
    """
    if not isinstance(raw, dict):
        return None
    api = raw.get("api")
    usage_reported = raw.get("usage_reported", True)
    valid = (
        _is_number(raw.get("ts"))
        and all(
            isinstance(raw.get(key), str) for key in ("category", "provider", "model")
        )
        and (api is None or isinstance(api, str))
        and all(
            _is_count(raw.get(key))
            for key in ("input_tokens", "cached_tokens", "output_tokens")
        )
        and isinstance(usage_reported, bool)
    )
    if not valid:
        return None
    return UsageRecord(
        ts=float(raw["ts"]),
        category=raw["category"],
        provider=raw["provider"],
        model=raw["model"],
        api=api,
        input_tokens=raw["input_tokens"],
        cached_tokens=raw["cached_tokens"],
        output_tokens=raw["output_tokens"],
        usage_reported=usage_reported,
    )


def _record_to_dict(rec: UsageRecord) -> dict[str, Any]:
    """Shallow dict of a record (asdict deep-copies; all fields are scalars)."""
    return {name: getattr(rec, name) for name in _RECORD_FIELDS}


def _window_summary(
    records: Iterable[UsageRecord],
    cost_fn: Callable[[UsageRecord], float | None],
    *,
    since: float,
    span_s: float,
) -> dict[str, Any]:
    """Sum a window and project it to a month over the time actually covered."""
    calls = priced = 0
    cost = 0.0
    tokens_in = tokens_out = 0
    by_model: dict[str, dict[str, Any]] = {}
    unpriced_models: set[str] = set()
    for rec in records:
        if rec.ts < since:
            continue
        calls += 1
        tokens_in += rec.input_tokens
        tokens_out += rec.output_tokens
        # Provider is part of the key: the same model name can be priced for
        # one provider and unpriced (or priced differently) for another.
        detail = f"{rec.api}, {rec.provider}" if rec.api else rec.provider
        key = f"{rec.model} ({detail})" if detail else rec.model
        bucket = by_model.setdefault(
            key,
            {
                "provider": rec.provider,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += rec.input_tokens
        bucket["output_tokens"] += rec.output_tokens
        rec_cost = cost_fn(rec)
        if rec_cost is None:
            unpriced_models.add(rec.model)
            bucket["cost_usd"] = None
            continue
        priced += 1
        cost += rec_cost
        if bucket["cost_usd"] is not None:
            bucket["cost_usd"] += rec_cost
    # At least an hour: a few minutes of data scaled to a month is noise.
    days = max(span_s, MIN_PROJECTION_SPAN_S) / DAY_S
    return {
        "cost_usd": cost,
        "projected_monthly_usd": cost / days * DAYS_PER_MONTH,
        "calls": calls,
        "priced_calls": priced,
        "unpriced_calls": calls - priced,
        "unpriced_models": sorted(unpriced_models),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "average_cost_per_call_usd": cost / priced if priced else None,
        "covered_days": round(days, 2),
        "by_model": {
            key: {
                **val,
                "cost_usd": None
                if val["cost_usd"] is None
                else round(val["cost_usd"], 6),
            }
            for key, val in sorted(by_model.items())
        },
    }


class UsageCostTracker:
    """Keeps seven days of per-call usage and summarizes it for the sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        price_overrides: Mapping[str, ModelPrice] | None = None,
    ) -> None:
        """Initialize the tracker (call async_load before use)."""
        self.hass = hass
        self._overrides = dict(price_overrides or {})
        self._records: deque[UsageRecord] = deque(maxlen=MAX_RECORDS)
        self._started: float = time.time()
        # Timestamp of the newest call the record cap dropped, while it is
        # still inside the retention window: projections start after it.
        self._evicted_until: float | None = None
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, STORAGE_KEY, private=True
        )
        self._closed = False
        self.callback = UsageCostCallback(self)

    async def async_load(self) -> None:
        """Load stored records, dropping any older than the retention window."""
        try:
            data = await self._store.async_load()
        except Exception:  # noqa: BLE001 — a bad store must not block setup
            LOGGER.warning("Could not read stored usage costs; starting fresh.")
            data = None
        if not isinstance(data, dict):
            return
        started = data.get("started")
        if _is_number(started):
            self._started = float(started)
        evicted_until = data.get("evicted_until")
        if _is_number(evicted_until):
            self._evicted_until = float(evicted_until)
        records = data.get("records")
        if not isinstance(records, list):
            return
        cutoff = time.time() - RETENTION_S
        for raw in records:
            rec = _record_from_dict(raw)
            if rec is not None and rec.ts >= cutoff:
                self._append(rec)
        self._prune()

    def _data_to_save(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "evicted_until": self._evicted_until,
            "records": [_record_to_dict(rec) for rec in self._records],
        }

    def _append(self, record: UsageRecord) -> None:
        """Keep one record, noting the call the record cap drops, if any."""
        if self._records and len(self._records) == self._records.maxlen:
            dropped = self._records[0].ts
            self._evicted_until = max(self._evicted_until or dropped, dropped)
        self._records.append(record)

    async def async_flush(self) -> None:
        """
        Write pending records now and stop recording (on unload).

        Calls still in flight when the entry reloads finish on this tracker;
        once closed they are dropped instead of scheduling a delayed save that
        would overwrite the next tracker's file with this tracker's records.
        """
        self._closed = True
        try:
            await self._store.async_save(self._data_to_save())
        except Exception:
            LOGGER.warning("Could not save usage costs.", exc_info=True)

    @callback
    def async_start_refresh(self) -> Callable[[], None]:
        """Re-summarize the sensors periodically; returns the unsubscribe."""

        @callback
        def _tick(_now: datetime) -> None:
            # Expire old calls even when no new ones arrive, so the windows,
            # the last-call sensor and the stored file honor the retention.
            if self._prune() and not self._closed:
                self._store.async_delay_save(self._data_to_save, SAVE_DELAY_S)
            self._async_notify()

        return async_track_time_interval(self.hass, _tick, REFRESH_INTERVAL)

    @callback
    def _async_notify(self) -> None:
        """Send one summary to every cost sensor."""
        async_dispatcher_send(self.hass, SIGNAL_USAGE_COST_UPDATED, self.summary())

    def _prune(self) -> bool:
        """Drop records past the retention window; return True if any went."""
        cutoff = time.time() - RETENTION_S
        pruned = False
        while self._records and self._records[0].ts < cutoff:
            self._records.popleft()
            pruned = True
        if self._evicted_until is not None and self._evicted_until < cutoff:
            # The dropped calls are past retention anyway: no gap remains.
            self._evicted_until = None
            pruned = True
        return pruned

    def price_for(self, model: Any, provider_type: str | None) -> ModelPrice | None:
        """Return the price used for a model."""
        return price_for(model, provider_type, self._overrides)

    def cost_of(self, record: UsageRecord) -> float | None:
        """Return a call's cost at the current prices, or None if unpriced."""
        price = self.price_for(record.model, record.provider)
        if price is None or (not record.usage_reported and price.is_paid):
            # No usage for a paid model: count the call as unpriced rather
            # than as free.
            return None
        return estimate_cost(
            price, record.input_tokens, record.cached_tokens, record.output_tokens
        )

    @callback
    def async_record(self, record: UsageRecord) -> None:
        """Store one call; must run on the event loop."""
        if self._closed:
            return
        self._append(record)
        self._prune()
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY_S)
        self._async_notify()

    def record_threadsafe(self, record: UsageRecord) -> None:
        """Store one call from any thread (sync model calls run in executors)."""
        if threading.get_ident() == getattr(self.hass, "loop_thread_id", None):
            self.async_record(record)
        else:
            self.hass.loop.call_soon_threadsafe(self.async_record, record)

    def summary(self, now: float | None = None) -> dict[str, Any]:
        """Return the last call plus 24-hour and 7-day windows."""
        now = time.time() if now is None else now
        tracked_s = max(0.0, now - self._started)
        if self._evicted_until is not None:
            # The record cap dropped calls inside the windows: project only
            # over the time after the newest dropped call, which the kept
            # records fully cover.
            tracked_s = min(tracked_s, max(0.0, now - self._evicted_until))
        last = self._records[-1] if self._records else None
        return {
            "last_call": None
            if last is None
            else {**_record_to_dict(last), "cost": self.cost_of(last)},
            "day": _window_summary(
                self._records,
                self.cost_of,
                since=now - DAY_S,
                span_s=min(DAY_S, tracked_s),
            ),
            "week": _window_summary(
                self._records,
                self.cost_of,
                since=now - WEEK_S,
                span_s=min(WEEK_S, tracked_s),
            ),
            "prices_checked": PRICES_CHECKED,
        }


def _api_from_response(
    response_metadata: Mapping[str, Any], provider_type: str | None
) -> str | None:
    """
    Tell the OpenAI API apart by the response id it returned.

    Responses API replies always carry their ``resp_`` id, streamed or not.
    Streamed Chat Completions replies carry no id in the response metadata, so
    any other reply from an OpenAI provider is a Chat Completions call.
    """
    response_id = str(response_metadata.get("id") or "")
    if response_id.startswith("resp_"):
        return OPENAI_API_RESPONSES
    if response_id.startswith("chatcmpl-") or provider_type in OPENAI_PROVIDER_TYPES:
        return OPENAI_API_CHAT_COMPLETIONS
    return None


def _int(value: Any) -> int:
    """Return a reported token count, or 0 when missing or implausible."""
    return value if _is_count(value) else 0


class UsageCostCallback(BaseCallbackHandler):
    """Records token usage for each finished model call."""

    # Runs inline: cheap bookkeeping, and thread hops are handled by the
    # tracker (sync calls from executors end up in record_threadsafe).
    run_inline = True

    def __init__(self, tracker: UsageCostTracker) -> None:
        """Bind the callback to a tracker."""
        super().__init__()
        self._tracker = tracker
        self._runs: dict[UUID, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[Any]],  # noqa: ARG002
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember which feature and provider started this run."""
        meta = metadata or {}
        if META_CATEGORY not in meta:
            return
        params = kwargs.get("invocation_params") or {}
        with self._lock:
            self._runs[run_id] = {
                "category": str(meta.get(META_CATEGORY)),
                "provider": str(meta.get(META_PROVIDER) or ""),
                "model": params.get("model") or params.get("model_name"),
            }

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **_: Any) -> None:
        """Record the usage the provider reported for this run."""
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return
        messages = [
            message
            for generations in response.generations
            for generation in generations
            if (message := getattr(generation, "message", None)) is not None
        ]
        # Only the first reply with usage counts (n>1 replies share one bill).
        message = next(
            (m for m in messages if getattr(m, "usage_metadata", None)),
            messages[0] if messages else None,
        )
        usage = getattr(message, "usage_metadata", None) or {}
        meta = getattr(message, "response_metadata", None) or {}
        model = meta.get("model_name") or meta.get("model") or run["model"]
        details = usage.get("input_token_details") or {}
        counts = (
            usage.get("input_tokens"),
            details.get("cache_read"),
            usage.get("output_tokens"),
        )
        input_tokens, cached, output_tokens = (_int(c) for c in counts)
        # Input and output counts are always part of LangChain usage, while a
        # cache-read count is optional. A missing or implausible count makes
        # the call unpriced rather than cheaper than it was.
        usage_valid = (
            _is_count(counts[0])
            and _is_count(counts[2])
            and (counts[1] is None or _is_count(counts[1]))
        )
        self._tracker.record_threadsafe(
            UsageRecord(
                ts=time.time(),
                category=run["category"],
                provider=run["provider"],
                model=normalize_model_name(model) or "unknown",
                api=_api_from_response(meta, run["provider"]),
                input_tokens=input_tokens,
                cached_tokens=cached,
                output_tokens=output_tokens,
                usage_reported=bool(usage) and usage_valid,
            )
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:  # noqa: ARG002
        """Forget a failed run."""
        with self._lock:
            self._runs.pop(run_id, None)
