"""Mobile Due: make AnkiMobile/AnkiDroid show RWKV's picks as their Due queue.

Mobile clients cannot run add-ons, but they faithfully honour the ``due`` field
of synced review cards. This module plans a rewrite of that one field so that,
after a sync, the phone's Due queue contains exactly the review cards whose
RWKV Immediate prediction at a target time is below desired retention.

Cards RWKV does not pick are moved to the first future day on which RWKV
predicts they will fall below desired retention. If runs ever stop, every card
therefore still carries a sensible RWKV-projected date and the phone degrades
into an ordinary static scheduler instead of facing a backlog.

Only ``due`` is written. ``ivl``, ``type``, ``queue``, ``left`` and the FSRS
memory state are left untouched so the phone's own scheduler stays coherent.

This module is pure: no ``aqt`` or ``anki`` imports, so it can be unit tested
and reused by a headless runner.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .anki_api import DeckRetention
from .curve_reschedule import (
    CARD_TYPE_REVIEW,
    QUEUE_TYPE_REVIEW,
    CardScheduleInfo,
    anki_day_for_timestamp,
)
from .filtered_deck import (
    FilteredDeckCandidate,
    FilteredDeckSettings,
    select_candidates_with_minimum,
)
from .filtered_deck_sort import FilteredDeckOrder, FilteredDeckSortInfo
from .review_load_policy import DEFAULT_MINIMUM_RETENTION_STEP
from .review_rows import SECONDS_PER_DAY, ReviewData, prediction_row_for_card
from .review_rows import CardInfo as ReviewCardInfo

FALLBACK_DESIRED_RETENTION = 0.9
DEFAULT_HORIZON_DAYS = 365
DEFAULT_PRIORITY_BUCKETS = 7
END_OF_DAY_MARGIN_SECONDS = 60

Predictor = Callable[[list[dict[str, Any]]], Iterable[float]]


class PredictionTime(str, Enum):
    END_OF_DAY = "end_of_day"
    NOW = "now"


class MobileDueSkipReason(str, Enum):
    NOT_FOUND = "not_found"
    IN_FILTERED_DECK = "in_filtered_deck"
    NOT_REVIEW = "not_review"
    NO_HISTORY = "no_history"
    NO_PREDICTION = "no_prediction"


@dataclass(frozen=True)
class MobileDueSettings:
    """Selection and placement policy.

    ``limit`` caps how many cards are made due now (0 = no cap). ``minimum``,
    ``minimum_retention_extra_quantum`` and ``random_card_percent`` behave as in
    RWKV filtered decks and the Live Session. ``priority_buckets`` spreads the
    due-now cards over that many days of apparent overdueness (most urgent =
    most overdue) so a phone sorting by due date shows the lowest-R cards
    first; 1 disables it. ``horizon_days`` bounds how far ahead a not-due card
    may be projected.
    """

    limit: int = 0
    minimum: int = 0
    minimum_retention_extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP
    random_card_percent: float = 0.0
    priority_buckets: int = DEFAULT_PRIORITY_BUCKETS
    horizon_days: int = DEFAULT_HORIZON_DAYS

    def __post_init__(self) -> None:
        if int(self.limit) < 0:
            raise ValueError("Mobile Due limit must be 0 (no cap) or positive.")
        if int(self.minimum) < 0:
            raise ValueError("Mobile Due minimum must not be negative.")
        if int(self.priority_buckets) < 1:
            raise ValueError("Mobile Due priority buckets must be at least 1.")
        if int(self.horizon_days) < 1:
            raise ValueError("Mobile Due horizon must be at least 1 day.")


@dataclass(frozen=True)
class MobileDueChange:
    card_id: int
    old_due: int
    new_due: int
    old_mod: int
    old_interval: int
    due_now: bool
    predicted_retrievability: float
    desired_retention: float
    rank: int | None = None


@dataclass(frozen=True)
class MobileDuePlan:
    created_at: float
    target_timestamp: float
    today: int
    in_scope_count: int
    due_now_ids: tuple[int, ...]
    changes: tuple[MobileDueChange, ...]
    unchanged_count: int
    skipped: Mapping[MobileDueSkipReason, int] = field(default_factory=dict)
    expected_retrievability: float | None = None
    capped_by_limit: int = 0
    projected_to_horizon: int = 0

    @property
    def due_now_count(self) -> int:
        return len(self.due_now_ids)

    @property
    def changed_count(self) -> int:
        return len(self.changes)

    @property
    def pulled_in_count(self) -> int:
        """Cards made due now that the current schedule did not have due."""
        return sum(1 for c in self.changes if c.due_now and c.old_due > self.today)

    @property
    def pushed_out_count(self) -> int:
        """Cards currently due that RWKV moves to a future day."""
        return sum(1 for c in self.changes if not c.due_now and c.old_due <= self.today)


def resolve_target_timestamp(
    prediction_time: str,
    *,
    now: float,
    day_cutoff: int,
) -> float:
    """Return the moment RWKV predicts recall for.

    ``end_of_day``: just before the Anki day rolls over, so cards that will
    cross desired retention before the user's last study block of the day are
    already due. ``now``: the current moment. ``HH:MM``: that local clock time
    within the current Anki day; if it has already passed, ``now``.
    """

    value = str(prediction_time or PredictionTime.END_OF_DAY.value).strip().lower()
    end_of_day = float(day_cutoff) - END_OF_DAY_MARGIN_SECONDS
    if value == PredictionTime.END_OF_DAY.value:
        return max(float(now), end_of_day)
    if value == PredictionTime.NOW.value:
        return float(now)
    hours, minutes = _parse_clock(value)
    day_start = float(day_cutoff) - SECONDS_PER_DAY
    local = time.localtime(now)
    candidate = time.mktime(
        (
            local.tm_year,
            local.tm_mon,
            local.tm_mday,
            hours,
            minutes,
            0,
            0,
            0,
            -1,
        )
    )
    # The clock time may belong to the next calendar date when the Anki day
    # starts after midnight (e.g. 02:00 with a 04:00 rollover).
    for shift in (0.0, SECONDS_PER_DAY, -SECONDS_PER_DAY):
        shifted = candidate + shift
        if day_start <= shifted < float(day_cutoff):
            return max(float(now), shifted)
    return float(now)


def validate_prediction_time(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {PredictionTime.END_OF_DAY.value, PredictionTime.NOW.value}:
        return text
    _parse_clock(text)
    return text


def _parse_clock(value: str) -> tuple[int, int]:
    try:
        hours_text, minutes_text = value.split(":", 1)
        hours, minutes = int(hours_text), int(minutes_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            'Mobile Due prediction time must be "end_of_day", "now" or HH:MM.'
        ) from exc
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ValueError('Mobile Due prediction time must be "end_of_day", "now" or HH:MM.')
    return hours, minutes


def is_mobile_due_eligible(schedule: CardScheduleInfo) -> MobileDueSkipReason | None:
    if int(schedule.original_deck_id):
        return MobileDueSkipReason.IN_FILTERED_DECK
    if int(schedule.card_type) != CARD_TYPE_REVIEW or int(schedule.queue) != QUEUE_TYPE_REVIEW:
        return MobileDueSkipReason.NOT_REVIEW
    return None


def build_mobile_due_plan(
    *,
    card_ids: Iterable[int],
    schedule_infos: Mapping[int, CardScheduleInfo],
    review_data: ReviewData,
    retentions: Iterable[DeckRetention],
    max_interval_by_deck: Mapping[int, int | None],
    predictor: Predictor,
    settings: MobileDueSettings,
    today: int,
    day_cutoff: int,
    target_timestamp: float,
    created_at: float | None = None,
    progress=None,
) -> MobileDuePlan:
    skipped: dict[MobileDueSkipReason, int] = {}

    def skip(reason: MobileDueSkipReason) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    retention_by_deck = {int(r.deck_id): r for r in retentions}
    eligible: list[CardScheduleInfo] = []
    for card_id in dict.fromkeys(int(value) for value in card_ids):
        schedule = schedule_infos.get(card_id)
        if schedule is None:
            skip(MobileDueSkipReason.NOT_FOUND)
            continue
        reason = is_mobile_due_eligible(schedule)
        if reason is not None:
            skip(reason)
            continue
        if review_data.last_by_card.get(card_id) is None:
            skip(MobileDueSkipReason.NO_HISTORY)
            continue
        eligible.append(schedule)

    _progress(progress, 0, 1, "Predicting recall for Mobile Due")
    rows = [
        _prediction_row(review_data, schedule.card_id, target_timestamp)
        for schedule in eligible
    ]
    predictions = list(predictor(rows)) if rows else []
    if len(predictions) != len(rows):
        raise ValueError("RWKV Mobile Due prediction count did not match candidates.")

    candidates: list[FilteredDeckCandidate] = []
    schedule_by_card: dict[int, CardScheduleInfo] = {}
    for schedule, row, prediction in zip(eligible, rows, predictions, strict=True):
        value = float(prediction)
        if not math.isfinite(value):
            skip(MobileDueSkipReason.NO_PREDICTION)
            continue
        retention = retention_by_deck.get(int(schedule.source_deck_id or 0))
        desired = float(retention.desired_retention) if retention else FALLBACK_DESIRED_RETENTION
        same_day = (
            float(retention.same_day_desired_retention)
            if retention is not None and retention.same_day_desired_retention is not None
            else desired
        )
        candidates.append(
            FilteredDeckCandidate(
                card_id=schedule.card_id,
                deck_id=schedule.source_deck_id,
                desired_retention=desired,
                same_day_desired_retention=same_day,
                predicted_retrievability=value,
                elapsed_days=float(row.get("elapsed_days", -1)),
                sort_info=FilteredDeckSortInfo(
                    card_id=schedule.card_id,
                    modified_secs=schedule.modified_secs,
                ),
            )
        )
        schedule_by_card[schedule.card_id] = schedule

    limit = int(settings.limit) or len(candidates)
    selection_settings = FilteredDeckSettings(
        limit=max(1, limit),
        order_index=int(FilteredDeckOrder.RETRIEVABILITY_ASCENDING),
        minimum=min(int(settings.minimum), max(1, limit)),
        minimum_retention_extra_quantum=settings.minimum_retention_extra_quantum,
        random_card_percent=settings.random_card_percent,
    )
    if candidates:
        below, selected, _extra = select_candidates_with_minimum(
            candidates,
            settings=selection_settings,
        )
    else:
        below, selected = [], []
    selected_ids = [candidate.card_id for candidate in selected]
    selected_set = set(selected_ids)
    below_set = {candidate.card_id for candidate in below}

    new_due: dict[int, int] = {}
    rank_by_card: dict[int, int] = {}
    buckets = max(1, int(settings.priority_buckets))
    count = len(selected_ids)
    for rank, card_id in enumerate(selected_ids):
        bucket = (rank * buckets) // count if count else 0
        new_due[card_id] = int(today) - (buckets - 1 - bucket)
        rank_by_card[card_id] = rank

    # Below desired retention but cut by the limit: earliest possible later day.
    capped = [c for c in candidates if c.card_id in below_set and c.card_id not in selected_set]
    for candidate in capped:
        new_due[candidate.card_id] = int(today) + 1

    remaining = [
        c for c in candidates if c.card_id not in selected_set and c.card_id not in below_set
    ]
    projected, at_horizon = _project_crossing_days(
        remaining,
        review_data=review_data,
        schedule_by_card=schedule_by_card,
        max_interval_by_deck=max_interval_by_deck,
        predictor=predictor,
        today=int(today),
        day_cutoff=int(day_cutoff),
        target_timestamp=float(target_timestamp),
        horizon_days=int(settings.horizon_days),
        progress=progress,
    )
    new_due.update(projected)

    changes: list[MobileDueChange] = []
    unchanged = 0
    for candidate in candidates:
        schedule = schedule_by_card[candidate.card_id]
        target_due = new_due[candidate.card_id]
        if int(schedule.due) == target_due:
            unchanged += 1
            continue
        changes.append(
            MobileDueChange(
                card_id=candidate.card_id,
                old_due=int(schedule.due),
                new_due=int(target_due),
                old_mod=int(schedule.modified_secs),
                old_interval=int(schedule.interval),
                due_now=candidate.card_id in selected_set,
                predicted_retrievability=float(candidate.predicted_retrievability),
                desired_retention=float(candidate.active_desired_retention),
                rank=rank_by_card.get(candidate.card_id),
            )
        )

    expected = (
        sum(candidate.predicted_retrievability for candidate in selected) / len(selected)
        if selected
        else None
    )
    _progress(progress, 1, 1, "Planned Mobile Due")
    return MobileDuePlan(
        created_at=float(time.time() if created_at is None else created_at),
        target_timestamp=float(target_timestamp),
        today=int(today),
        in_scope_count=len(candidates),
        due_now_ids=tuple(selected_ids),
        changes=tuple(changes),
        unchanged_count=unchanged,
        skipped=dict(skipped),
        expected_retrievability=expected,
        capped_by_limit=len(capped),
        projected_to_horizon=at_horizon,
    )


def _project_crossing_days(
    candidates: Sequence[FilteredDeckCandidate],
    *,
    review_data: ReviewData,
    schedule_by_card: Mapping[int, CardScheduleInfo],
    max_interval_by_deck: Mapping[int, int | None],
    predictor: Predictor,
    today: int,
    day_cutoff: int,
    target_timestamp: float,
    horizon_days: int,
    progress=None,
) -> tuple[dict[int, int], int]:
    """Find, per card, the first day offset d >= 1 with R(T + d days) < DR.

    Every card is known to be at or above DR at offset 0. Offsets are probed
    exponentially (1, 2, 4, ...) up to each card's cap, then bisected. Each
    probe round is one predictor call with at most one row per card.
    """

    if not candidates:
        return {}, 0
    lo: dict[int, int] = {}
    hi: dict[int, int] = {}
    cap: dict[int, int] = {}
    threshold: dict[int, float] = {}
    for candidate in candidates:
        card_id = candidate.card_id
        lo[card_id] = 0
        threshold[card_id] = float(candidate.desired_retention)
        cap[card_id] = _offset_cap(
            schedule_by_card[card_id],
            review_data=review_data,
            max_interval_by_deck=max_interval_by_deck,
            today=today,
            day_cutoff=day_cutoff,
            horizon_days=horizon_days,
        )

    def run_round(probes: dict[int, int], label: str) -> None:
        if not probes:
            return
        ids = list(probes)
        rows = [
            _prediction_row(
                review_data,
                card_id,
                target_timestamp + probes[card_id] * SECONDS_PER_DAY,
            )
            for card_id in ids
        ]
        values = list(predictor(rows))
        if len(values) != len(rows):
            raise ValueError("RWKV Mobile Due projection count did not match candidates.")
        for card_id, value in zip(ids, values, strict=True):
            offset = probes[card_id]
            prediction = float(value)
            # A missing prediction counts as "below": the card comes back early
            # rather than being pushed out to the horizon.
            if not math.isfinite(prediction) or prediction < threshold[card_id]:
                hi[card_id] = offset
            else:
                lo[card_id] = offset

    step = 1
    rounds = 0
    while True:
        probes = {
            card_id: min(step, cap[card_id])
            for card_id in lo
            if card_id not in hi and lo[card_id] < cap[card_id]
        }
        if not probes:
            break
        rounds += 1
        _progress(progress, rounds, rounds + 1, f"Projecting Mobile Due dates (+{step} days)")
        run_round(probes, "expand")
        step *= 2

    while True:
        probes = {
            card_id: (lo[card_id] + hi[card_id]) // 2
            for card_id in hi
            if hi[card_id] - lo[card_id] > 1
        }
        if not probes:
            break
        rounds += 1
        _progress(progress, rounds, rounds + 1, "Refining Mobile Due dates")
        run_round(probes, "bisect")

    result: dict[int, int] = {}
    at_horizon = 0
    for card_id in lo:
        if card_id in hi:
            offset = hi[card_id]
        else:
            offset = cap[card_id]
            at_horizon += 1
        result[card_id] = int(today) + max(1, int(offset))
    return result, at_horizon


def _offset_cap(
    schedule: CardScheduleInfo,
    *,
    review_data: ReviewData,
    max_interval_by_deck: Mapping[int, int | None],
    today: int,
    day_cutoff: int,
    horizon_days: int,
) -> int:
    offset_cap = max(1, int(horizon_days))
    max_interval = max_interval_by_deck.get(int(schedule.source_deck_id or 0))
    last_review = review_data.last_by_card.get(int(schedule.card_id))
    if max_interval is not None and last_review is not None:
        last_day = anki_day_for_timestamp(
            last_review.timestamp_seconds,
            today=today,
            day_cutoff=day_cutoff,
        )
        offset_cap = min(offset_cap, int(last_day) + int(max_interval) - int(today))
    return max(1, offset_cap)


def _prediction_row(
    review_data: ReviewData,
    card_id: int,
    target_timestamp: float,
) -> dict[str, Any]:
    card = review_data.cards.get(int(card_id)) or ReviewCardInfo(int(card_id), None, None, None)
    return prediction_row_for_card(
        card,
        review_data.last_by_card.get(int(card_id)),
        target_timestamp_seconds=float(target_timestamp),
        next_day_at=review_data.next_day_at,
        day_offset_origin=review_data.day_offset_origin,
    )


def _progress(progress, current: int, total: int, text: str) -> None:
    if progress is None:
        return
    update = getattr(progress, "update", None)
    if callable(update):
        update(current, total, text)


# --- Guardrails --------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailSettings:
    spike_multiplier: float = 3.0
    spike_minimum_cards: int = 50
    max_due_now_fraction: float = 0.5
    max_due_now_fraction_min_scope: int = 100
    history_runs: int = 7


def guardrail_violations(
    plan: MobileDuePlan,
    *,
    recent_due_now_counts: Sequence[int],
    settings: GuardrailSettings | None = None,
) -> list[str]:
    """Return human-readable reasons this plan should not be auto-applied."""

    resolved = settings or GuardrailSettings()
    violations: list[str] = []
    history = [int(value) for value in recent_due_now_counts][-int(resolved.history_runs):]
    if len(history) >= 3:
        baseline = sorted(history)[len(history) // 2]
        limit = max(int(resolved.spike_minimum_cards), baseline * float(resolved.spike_multiplier))
        if plan.due_now_count > limit:
            violations.append(
                f"{plan.due_now_count} cards due now is more than "
                f"{resolved.spike_multiplier:g}x the recent median ({baseline})."
            )
    if plan.in_scope_count >= int(resolved.max_due_now_fraction_min_scope):
        fraction = plan.due_now_count / max(1, plan.in_scope_count)
        if fraction > float(resolved.max_due_now_fraction):
            violations.append(
                f"{fraction:.0%} of review cards would be due now "
                f"(limit {resolved.max_due_now_fraction:.0%})."
            )
    return violations


def plan_summary_lines(plan: MobileDuePlan) -> list[str]:
    lines = [
        f"Review cards considered: {plan.in_scope_count}",
        f"Due now (RWKV below desired retention): {plan.due_now_count}",
        f"  newly pulled in: {plan.pulled_in_count}",
        f"Currently due but moved later: {plan.pushed_out_count}",
        f"Due dates changed: {plan.changed_count} (unchanged: {plan.unchanged_count})",
    ]
    if plan.expected_retrievability is not None:
        lines.append(f"Mean predicted recall of due-now cards: {plan.expected_retrievability:.3f}")
    if plan.capped_by_limit:
        lines.append(f"Below desired retention but over the card limit: {plan.capped_by_limit}")
    if plan.projected_to_horizon:
        lines.append(f"Projected to the horizon / max interval: {plan.projected_to_horizon}")
    if plan.skipped:
        skipped = ", ".join(
            f"{reason.value.replace('_', ' ')} {count}"
            for reason, count in sorted(plan.skipped.items(), key=lambda item: item[0].value)
        )
        lines.append(f"Left untouched: {skipped}")
    return lines


def plan_with_changes(plan: MobileDuePlan, changes: Iterable[MobileDueChange]) -> MobileDuePlan:
    return replace(plan, changes=tuple(changes))
