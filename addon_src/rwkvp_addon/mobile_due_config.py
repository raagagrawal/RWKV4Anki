"""Mobile Due settings, read from the add-on's config.json mapping.

All keys are optional; missing or invalid values fall back to safe defaults.
Mobile Due is off unless ``mobile_due_mode`` is set to ``shadow`` or ``live``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .mobile_due import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_PRIORITY_BUCKETS,
    GuardrailSettings,
    MobileDueSettings,
    PredictionTime,
    validate_prediction_time,
)
from .review_load_policy import DEFAULT_MINIMUM_RETENTION_STEP

MODE_KEY = "mobile_due_mode"
SEARCH_KEY = "mobile_due_search"
PREDICTION_TIME_KEY = "mobile_due_prediction_time"
RUN_AFTER_SYNC_KEY = "mobile_due_run_after_sync"
SYNC_AFTER_APPLY_KEY = "mobile_due_sync_after_apply"
AUTO_SYNC_HOURS_KEY = "mobile_due_auto_sync_hours"
LIMIT_KEY = "mobile_due_limit"
MINIMUM_KEY = "mobile_due_minimum"
RANDOM_PERCENT_KEY = "mobile_due_random_card_percent"
PRIORITY_BUCKETS_KEY = "mobile_due_priority_buckets"
HORIZON_KEY = "mobile_due_horizon_days"
GUARDRAILS_KEY = "mobile_due_guardrails"


class MobileDueMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class MobileDueConfig:
    mode: MobileDueMode = MobileDueMode.OFF
    search: str = ""
    prediction_time: str = PredictionTime.END_OF_DAY.value
    run_after_sync: bool = True
    sync_after_apply: bool = True
    auto_sync_hours: float = 0.0
    settings: MobileDueSettings = MobileDueSettings()
    guardrails_enabled: bool = True
    guardrails: GuardrailSettings = GuardrailSettings()

    @property
    def enabled(self) -> bool:
        return self.mode != MobileDueMode.OFF


DEFAULTS: dict[str, Any] = {
    MODE_KEY: MobileDueMode.OFF.value,
    SEARCH_KEY: "",
    PREDICTION_TIME_KEY: PredictionTime.END_OF_DAY.value,
    RUN_AFTER_SYNC_KEY: True,
    SYNC_AFTER_APPLY_KEY: True,
    AUTO_SYNC_HOURS_KEY: 0,
    LIMIT_KEY: 0,
    MINIMUM_KEY: 0,
    RANDOM_PERCENT_KEY: 0,
    PRIORITY_BUCKETS_KEY: DEFAULT_PRIORITY_BUCKETS,
    HORIZON_KEY: DEFAULT_HORIZON_DAYS,
    GUARDRAILS_KEY: True,
}


def mobile_due_config(
    config: Mapping[str, Any] | None,
    *,
    minimum_retention_extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP,
) -> MobileDueConfig:
    source = dict(config or {})
    mode = _mode(source.get(MODE_KEY))
    default_time = PredictionTime.END_OF_DAY.value
    try:
        prediction_time = validate_prediction_time(source.get(PREDICTION_TIME_KEY) or default_time)
    except ValueError:
        prediction_time = default_time
    settings = MobileDueSettings(
        limit=_int(source.get(LIMIT_KEY), 0, minimum=0),
        minimum=_int(source.get(MINIMUM_KEY), 0, minimum=0),
        minimum_retention_extra_quantum=float(minimum_retention_extra_quantum),
        random_card_percent=_float(source.get(RANDOM_PERCENT_KEY), 0.0, 0.0, 100.0),
        priority_buckets=_int(source.get(PRIORITY_BUCKETS_KEY), DEFAULT_PRIORITY_BUCKETS, minimum=1, maximum=60),
        horizon_days=_int(source.get(HORIZON_KEY), DEFAULT_HORIZON_DAYS, minimum=1, maximum=36500),
    )
    return MobileDueConfig(
        mode=mode,
        search=str(source.get(SEARCH_KEY) or "").strip(),
        prediction_time=prediction_time,
        run_after_sync=_bool(source.get(RUN_AFTER_SYNC_KEY), True),
        sync_after_apply=_bool(source.get(SYNC_AFTER_APPLY_KEY), True),
        auto_sync_hours=_float(source.get(AUTO_SYNC_HOURS_KEY), 0.0, 0.0, 24.0),
        settings=settings,
        guardrails_enabled=_bool(source.get(GUARDRAILS_KEY), True),
    )


def with_mode(config: Mapping[str, Any], mode: MobileDueMode) -> dict[str, Any]:
    updated = dict(config)
    updated[MODE_KEY] = mode.value
    return updated


def _mode(value: object) -> MobileDueMode:
    try:
        return MobileDueMode(str(value or "").strip().lower())
    except ValueError:
        return MobileDueMode.OFF


def _bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _int(value: object, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if minimum is not None and number < minimum:
        return default
    if maximum is not None and number > maximum:
        return maximum
    return number


def _float(value: object, default: float, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < minimum:
        return default
    return min(number, maximum)
