"""Sensors that show estimated model costs from the usage tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from ..const import SIGNAL_USAGE_COST_UPDATED  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Callable

    from .cost_tracker import UsageCostTracker

CURRENCY = "USD"


@dataclass(frozen=True, kw_only=True)
class UsageCostSensorDescription(SensorEntityDescription):
    """Describes one cost sensor: how to read its value and attributes."""

    value_fn: Callable[[dict[str, Any]], float | None]
    attrs_fn: Callable[[dict[str, Any]], dict[str, Any]]


def _last_call_attrs(summary: dict[str, Any]) -> dict[str, Any]:
    last = summary["last_call"] or {}
    return {
        "model": last.get("model"),
        "api": last.get("api"),
        "provider": last.get("provider"),
        "feature": last.get("category"),
        "input_tokens": last.get("input_tokens"),
        "cached_input_tokens": last.get("cached_tokens"),
        "output_tokens": last.get("output_tokens"),
        "average_cost_per_call_7d_usd": summary["week"]["average_cost_per_call_usd"],
        "prices_checked": summary["prices_checked"],
    }


def _window_attrs(window: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def attrs(summary: dict[str, Any]) -> dict[str, Any]:
        data = summary[window]
        return {
            "calls": data["calls"],
            "unpriced_calls": data["unpriced_calls"],
            "unpriced_models": data["unpriced_models"],
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "covered_days": data["covered_days"],
            "by_model": data["by_model"],
            "prices_checked": summary["prices_checked"],
        }

    return attrs


SENSORS: tuple[UsageCostSensorDescription, ...] = (
    UsageCostSensorDescription(
        key="llm_cost_last_call",
        translation_key="llm_cost_last_call",
        suggested_display_precision=6,
        value_fn=lambda s: (s["last_call"] or {}).get("cost"),
        attrs_fn=_last_call_attrs,
    ),
    UsageCostSensorDescription(
        key="llm_cost_24h",
        translation_key="llm_cost_24h",
        suggested_display_precision=4,
        value_fn=lambda s: s["day"]["cost_usd"],
        attrs_fn=_window_attrs("day"),
    ),
    UsageCostSensorDescription(
        key="llm_cost_7d",
        translation_key="llm_cost_7d",
        suggested_display_precision=4,
        value_fn=lambda s: s["week"]["cost_usd"],
        attrs_fn=_window_attrs("week"),
    ),
    UsageCostSensorDescription(
        key="llm_cost_monthly_projection_24h",
        translation_key="llm_cost_monthly_projection_24h",
        suggested_display_precision=2,
        value_fn=lambda s: s["day"]["projected_monthly_usd"],
        attrs_fn=_window_attrs("day"),
    ),
    UsageCostSensorDescription(
        key="llm_cost_monthly_projection_7d",
        translation_key="llm_cost_monthly_projection_7d",
        suggested_display_precision=2,
        value_fn=lambda s: s["week"]["projected_monthly_usd"],
        attrs_fn=_window_attrs("week"),
    ),
)


class UsageCostSensor(SensorEntity):
    """One estimated-cost sensor, refreshed after every recorded call."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_native_unit_of_measurement = CURRENCY
    _attr_icon = "mdi:cash"

    def __init__(
        self,
        hass: HomeAssistant,
        tracker: UsageCostTracker,
        entry_id: str,
        description: UsageCostSensorDescription,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self.entity_description = description
        self._description = description
        self._tracker = tracker
        self._attr_unique_id = f"{description.key}::{entry_id}"
        self._refresh()

    def _refresh(self) -> None:
        summary = self._tracker.summary()
        self._attr_native_value = self._description.value_fn(summary)
        self._attr_extra_state_attributes = self._description.attrs_fn(summary)

    async def async_added_to_hass(self) -> None:
        """Refresh on every recorded call."""

        @callback
        def _on_updated() -> None:
            self._refresh()
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(self.hass, SIGNAL_USAGE_COST_UPDATED, _on_updated)
        )


def build_usage_cost_sensors(
    hass: HomeAssistant, tracker: UsageCostTracker, entry_id: str
) -> list[UsageCostSensor]:
    """Return one sensor per description."""
    return [UsageCostSensor(hass, tracker, entry_id, desc) for desc in SENSORS]
