from __future__ import annotations

import math
import time

import pytest

from rwkvp_addon.anki_api import DeckRetention
from rwkvp_addon.curve_reschedule import CardScheduleInfo
from rwkvp_addon.mobile_due import (
    GuardrailSettings,
    MobileDueChange,
    MobileDuePlan,
    lock_currently_due,
    MobileDueSettings,
    MobileDueSkipReason,
    build_mobile_due_plan,
    guardrail_violations,
    resolve_target_timestamp,
    validate_prediction_time,
)
from rwkvp_addon.review_rows import SECONDS_PER_DAY, CardInfo, LastReviewInfo, ReviewData

TODAY = 1000
# Anki day rolls over at DAY_CUTOFF; "today" spans [DAY_CUTOFF - 1 day, DAY_CUTOFF).
DAY_CUTOFF = 1_800_000_000
NOW = DAY_CUTOFF - 10 * 3600  # 14h into the Anki day
DECK = 1
DR = 0.9


def day_start(day: int) -> float:
    return DAY_CUTOFF - SECONDS_PER_DAY + (day - TODAY) * SECONDS_PER_DAY


class FakeRWKV:
    """R = 0.9 ** (elapsed_days / S): S days of stability puts R at 0.9 after S days."""

    def __init__(self, stability_by_card: dict[int, float], nan_cards=()):
        self.stability = stability_by_card
        self.nan_cards = set(nan_cards)
        self.calls: list[list[int]] = []

    def __call__(self, rows):
        ids = [int(row["card_id"]) for row in rows]
        self.calls.append(ids)
        out = []
        for row in rows:
            card_id = int(row["card_id"])
            if card_id in self.nan_cards:
                out.append(float("nan"))
                continue
            elapsed = float(row["elapsed_seconds"]) / SECONDS_PER_DAY
            out.append(0.9 ** (elapsed / self.stability[card_id]))
        return out

    def r_at(self, card_id: int, last_ts: float, ts: float) -> float:
        return 0.9 ** (((ts - last_ts) / SECONDS_PER_DAY) / self.stability[card_id])


def make_world(cards: dict[int, dict]):
    """cards: id -> {last_day, stability, due, type=2, queue=2, odid=0, ivl, history=True}"""
    last_by_card = {}
    card_infos = {}
    schedules = {}
    for cid, spec in cards.items():
        last_ts = day_start(spec["last_day"]) + 12 * 3600
        if spec.get("history", True):
            last_by_card[cid] = LastReviewInfo(
                review_id=int(last_ts * 1000),
                # Today is raw day offset -1 relative to next_day_at (see review_rows).
                day_offset=spec["last_day"] - TODAY - 1,
                timestamp_seconds=last_ts,
                interval=spec.get("ivl", 10),
                lapse_count=0,
            )
        card_infos[cid] = CardInfo(cid, cid * 10, DECK, 1)
        schedules[cid] = CardScheduleInfo(
            card_id=cid,
            source_deck_id=DECK,
            card_type=spec.get("type", 2),
            queue=spec.get("queue", 2),
            due=spec["due"],
            interval=spec.get("ivl", 10),
            original_deck_id=spec.get("odid", 0),
            modified_secs=111,
        )
    review_data = ReviewData(
        rows=[],
        revlogs=[],
        cards=card_infos,
        decks={},
        last_by_card=last_by_card,
        next_day_at=DAY_CUTOFF,
        day_offset_origin=0,
    )
    return review_data, schedules


def plan_for(cards, *, settings=None, max_interval=None, nan_cards=(), target=None):
    review_data, schedules = make_world(cards)
    fake = FakeRWKV({cid: spec.get("stability", 10.0) for cid, spec in cards.items()}, nan_cards)
    plan = build_mobile_due_plan(
        card_ids=list(cards),
        schedule_infos=schedules,
        review_data=review_data,
        retentions=[DeckRetention(DECK, "Deck", DR, 1)],
        max_interval_by_deck={DECK: max_interval},
        predictor=fake,
        settings=settings or MobileDueSettings(priority_buckets=1),
        today=TODAY,
        day_cutoff=DAY_CUTOFF,
        target_timestamp=target if target is not None else DAY_CUTOFF - 60,
        created_at=NOW,
    )
    return plan, fake, review_data


def expected_crossing_offset(fake, cid, last_ts, target, cap):
    for d in range(1, cap + 1):
        if fake.r_at(cid, last_ts, target + d * SECONDS_PER_DAY) < DR:
            return d
    return cap


def test_due_now_is_exactly_the_below_dr_set_and_others_get_crossing_day():
    cards = {
        1: {"last_day": TODAY - 20, "stability": 10.0, "due": TODAY + 5},  # R<DR: pull in
        2: {"last_day": TODAY - 2, "stability": 10.0, "due": TODAY},  # R>DR: push out
        3: {"last_day": TODAY - 1, "stability": 3.0, "due": TODAY + 30},  # crosses soon
        4: {"last_day": TODAY - 5, "stability": 50.0, "due": TODAY - 1},  # far future
    }
    target = DAY_CUTOFF - 60
    plan, fake, review_data = plan_for(cards, target=target)
    assert set(plan.due_now_ids) == {1}
    new_due = {c.card_id: c.new_due for c in plan.changes}
    assert new_due[1] == TODAY
    for cid in (2, 3, 4):
        last_ts = review_data.last_by_card[cid].timestamp_seconds
        offset = expected_crossing_offset(fake, cid, last_ts, target, 365)
        assert new_due[cid] == TODAY + offset
        assert new_due[cid] > TODAY
    assert plan.pulled_in_count == 1
    assert plan.pushed_out_count == 2  # cards 2 and 4 were due, now later


def test_projection_matches_brute_force_for_many_stabilities():
    cards = {
        cid: {"last_day": TODAY - (cid % 9), "stability": 0.5 + cid * 1.7, "due": TODAY}
        for cid in range(1, 80)
    }
    target = DAY_CUTOFF - 60
    plan, fake, review_data = plan_for(cards, target=target)
    by_card = {c.card_id: c for c in plan.changes}
    for cid in cards:
        if cid in plan.due_now_ids:
            continue
        last_ts = review_data.last_by_card[cid].timestamp_seconds
        expected = TODAY + expected_crossing_offset(fake, cid, last_ts, target, 365)
        got = by_card[cid].new_due if cid in by_card else cards[cid]["due"]
        assert got == expected, cid


def test_predictor_never_sees_duplicate_cards_in_one_call():
    cards = {cid: {"last_day": TODAY - 1, "stability": 2.0 + cid, "due": TODAY} for cid in range(1, 40)}
    _plan, fake, _ = plan_for(cards)
    assert fake.calls
    for call in fake.calls:
        assert len(call) == len(set(call))
    # Exponential + bisection keeps probe rounds logarithmic in the horizon.
    assert len(fake.calls) <= 1 + 2 * (math.ceil(math.log2(365)) + 1)


def test_priority_buckets_put_lowest_recall_most_overdue():
    cards = {
        1: {"last_day": TODAY - 100, "stability": 5.0, "due": TODAY},  # lowest R
        2: {"last_day": TODAY - 40, "stability": 5.0, "due": TODAY},
        3: {"last_day": TODAY - 15, "stability": 5.0, "due": TODAY},  # highest R (still < DR)
    }
    plan, _fake, _ = plan_for(cards, settings=MobileDueSettings(priority_buckets=3))
    assert plan.due_now_ids == (1, 2, 3)
    due = {c.card_id: c.new_due for c in plan.changes}
    due.update({cid: cards[cid]["due"] for cid in cards if cid not in due})
    assert due[1] == TODAY - 2
    assert due[2] == TODAY - 1
    assert due[3] == TODAY
    assert all(d <= TODAY for d in due.values())


def test_single_bucket_makes_all_due_today_and_unchanged_cards_are_not_rewritten():
    cards = {
        1: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY},
        2: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY - 3},
    }
    plan, _fake, _ = plan_for(cards, settings=MobileDueSettings(priority_buckets=1))
    assert plan.unchanged_count == 1
    assert [(c.card_id, c.new_due) for c in plan.changes] == [(2, TODAY)]


def test_untouched_states():
    cards = {
        1: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY, "odid": 99},  # filtered deck
        2: {"last_day": TODAY - 30, "stability": 5.0, "due": 123, "type": 1, "queue": 1},  # learning
        3: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY, "type": 3, "queue": 3},  # relearn
        4: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY, "queue": -1},  # suspended
        5: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY, "queue": -2},  # buried
        6: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY, "history": False},
        7: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY},  # NaN prediction
        8: {"last_day": TODAY - 30, "stability": 5.0, "due": TODAY + 3},  # the one real card
    }
    plan, _fake, _ = plan_for(cards, nan_cards={7})
    assert {c.card_id for c in plan.changes} == {8}
    assert plan.skipped == {
        MobileDueSkipReason.IN_FILTERED_DECK: 1,
        MobileDueSkipReason.NOT_REVIEW: 4,
        MobileDueSkipReason.NO_HISTORY: 1,
        MobileDueSkipReason.NO_PREDICTION: 1,
    }


def test_limit_caps_due_now_and_moves_the_rest_to_tomorrow():
    cards = {cid: {"last_day": TODAY - 10 * cid, "stability": 2.0, "due": TODAY + 9} for cid in range(1, 6)}
    plan, _fake, _ = plan_for(cards, settings=MobileDueSettings(limit=2, priority_buckets=1))
    assert plan.due_now_count == 2
    assert set(plan.due_now_ids) == {5, 4}  # lowest recall first
    assert plan.capped_by_limit == 3
    due = {c.card_id: c.new_due for c in plan.changes}
    for cid in (1, 2, 3):
        assert due[cid] == TODAY + 1


def test_max_interval_and_horizon_cap_projection():
    cards = {1: {"last_day": TODAY - 1, "stability": 10_000.0, "due": TODAY}}
    plan, _fake, _ = plan_for(cards, max_interval=30)
    assert plan.changes[0].new_due == TODAY - 1 + 30
    assert plan.projected_to_horizon == 1
    plan, _fake, _ = plan_for(cards, settings=MobileDueSettings(priority_buckets=1, horizon_days=10))
    assert plan.changes[0].new_due == TODAY + 10


def test_later_target_time_selects_superset():
    cards = {cid: {"last_day": TODAY - cid, "stability": 9.0, "due": TODAY} for cid in range(1, 30)}
    early, _f, _ = plan_for(cards, target=NOW)
    late, _f, _ = plan_for(cards, target=DAY_CUTOFF - 60)
    assert set(early.due_now_ids) <= set(late.due_now_ids)


def test_resolve_target_timestamp():
    assert resolve_target_timestamp("end_of_day", now=NOW, day_cutoff=DAY_CUTOFF) == DAY_CUTOFF - 60
    assert resolve_target_timestamp("now", now=NOW, day_cutoff=DAY_CUTOFF) == NOW
    # A clock time later in the current Anki day is honoured; a past one falls back to now.
    lt = time.localtime(NOW + 3600)
    later = f"{lt.tm_hour:02d}:{lt.tm_min:02d}"
    resolved = resolve_target_timestamp(later, now=NOW, day_cutoff=DAY_CUTOFF)
    assert NOW < resolved <= NOW + 3600
    lt = time.localtime(NOW - 3600)
    earlier = f"{lt.tm_hour:02d}:{lt.tm_min:02d}"
    assert resolve_target_timestamp(earlier, now=NOW, day_cutoff=DAY_CUTOFF) == NOW
    assert validate_prediction_time(" End_Of_Day ") == "end_of_day"
    with pytest.raises(ValueError):
        validate_prediction_time("25:00")
    with pytest.raises(ValueError):
        validate_prediction_time("evening")


def test_guardrails():
    cards = {cid: {"last_day": TODAY - 50, "stability": 2.0, "due": TODAY} for cid in range(1, 201)}
    plan, _f, _ = plan_for(cards)
    assert plan.due_now_count == 200
    violations = guardrail_violations(plan, recent_due_now_counts=[20, 25, 30])
    assert any("recent median" in v for v in violations)
    assert any("of review cards" in v for v in violations)
    assert guardrail_violations(plan, recent_due_now_counts=[190, 200, 210], settings=GuardrailSettings(max_due_now_fraction=1.0)) == []


def test_settings_validation():
    with pytest.raises(ValueError):
        MobileDueSettings(priority_buckets=0)
    with pytest.raises(ValueError):
        MobileDueSettings(limit=-1)


def test_minimum_widening_never_sends_above_dr_cards_to_tomorrow():
    """Regression: widened-but-unselected cards must be projected, not capped."""
    cards = {1: {"last_day": TODAY - 40, "stability": 5.0, "due": TODAY},
             2: {"last_day": TODAY - 35, "stability": 5.0, "due": TODAY}}
    for cid in range(3, 33):  # just above DR at T
        cards[cid] = {"last_day": TODAY - 1, "stability": 8.0 + cid * 0.1, "due": TODAY + 3}
    target = DAY_CUTOFF - 60
    plan, fake, review_data = plan_for(
        cards, settings=MobileDueSettings(minimum=5, priority_buckets=1), target=target
    )
    assert plan.due_now_count == 5  # 2 below DR + 3 widened picks
    assert plan.capped_by_limit == 0
    by_card = {c.card_id: c.new_due for c in plan.changes}
    for cid in cards:
        if cid in plan.due_now_ids:
            continue
        last_ts = review_data.last_by_card[cid].timestamp_seconds
        expected = TODAY + expected_crossing_offset(fake, cid, last_ts, target, 365)
        assert by_card.get(cid, cards[cid]["due"]) == expected


def test_no_crossing_within_horizon_never_pulls_a_card_earlier():
    cards = {1: {"last_day": TODAY - 1, "stability": 10_000.0, "due": TODAY + 200}}
    plan, _f, _ = plan_for(cards, settings=MobileDueSettings(priority_buckets=1, horizon_days=30))
    assert plan.changes == ()  # stays at +200 rather than moving to +30
    # ...but the deck's max interval still wins over a later current schedule.
    plan, _f, _ = plan_for(cards, max_interval=50, settings=MobileDueSettings(priority_buckets=1, horizon_days=30))
    assert plan.changes[0].new_due == TODAY - 1 + 50


def test_lock_currently_due_drops_every_change_to_due_cards():
    def change(cid, old, new):
        return MobileDueChange(cid, old, new, 0, 10, new <= TODAY, 0.8, 0.9)

    plan = MobileDuePlan(
        created_at=0, target_timestamp=0, today=TODAY, in_scope_count=4,
        due_now_ids=(1, 2), unchanged_count=0,
        changes=(
            change(1, TODAY, TODAY - 3),   # already due, re-bucketed: locked
            change(2, TODAY + 4, TODAY),   # pulled in from the future: allowed
            change(3, TODAY - 1, TODAY + 6),  # due, pushed out: locked
            change(4, TODAY + 9, TODAY + 2),  # future to future: allowed
        ),
    )
    locked = lock_currently_due(plan)
    assert [c.card_id for c in locked.changes] == [2, 4]
    assert locked.locked_due_count == 2
    assert locked.pushed_out_count == 0
    assert all(c.old_due > TODAY for c in locked.changes)


def test_saturation_guardrail_uses_collection_history_and_ignores_max_interval_caps():
    cards = {cid: {"last_day": TODAY - 1, "stability": 10_000.0, "due": TODAY} for cid in range(1, 151)}
    plan, _f, _ = plan_for(cards)
    assert plan.due_now_count == 0 and plan.beyond_horizon_count == 150
    # No history: only an extreme fraction (> 90%) trips it.
    assert any("never drop below" in v for v in guardrail_violations(plan, recent_due_now_counts=[]))
    # A mature collection that is usually ~95% beyond the horizon is normal for it.
    mature = dict(cards)
    for cid in range(151, 159):
        mature[cid] = {"last_day": TODAY - 1, "stability": 4.0, "due": TODAY + 1}
    mature_plan, _f, _ = plan_for(mature)
    assert 0.9 < mature_plan.beyond_horizon_count / mature_plan.in_scope_count < 0.99
    assert not any(
        "never drop below" in v
        for v in guardrail_violations(mature_plan, recent_due_now_counts=[], recent_beyond_horizon_fractions=[0.93, 0.95, 0.94])
    )
    # ...but a jump from its usual 20% is not.
    assert any(
        "never drop below" in v
        for v in guardrail_violations(plan, recent_due_now_counts=[], recent_beyond_horizon_fractions=[0.2, 0.2, 0.25])
    )
    # Cards stopped by a short deck max interval are not "beyond the horizon".
    capped, _f, _ = plan_for(cards, max_interval=30)
    assert capped.projected_to_horizon == 150 and capped.beyond_horizon_count == 0
    assert guardrail_violations(capped, recent_due_now_counts=[]) == []


def test_finishing_reviews_does_not_trip_guardrails():
    """Regression: a small due-now count after the user clears the queue is normal."""
    cards = {cid: {"last_day": TODAY - 1, "stability": 3.0 + cid, "due": TODAY + 2} for cid in range(1, 151)}
    plan, _f, _ = plan_for(cards)
    assert guardrail_violations(plan, recent_due_now_counts=[200, 210, 190, 205]) == []
