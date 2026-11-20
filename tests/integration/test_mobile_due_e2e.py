"""End to end: real Anki 26.05 collection + real RWKV-SRS predictions -> Mobile Due.

Checks the claims Mobile Due makes to the user against the actual model:
due-now cards really are below desired retention at the target time, other
cards land on RWKV's first below-DR day, only ``due`` changes, the result is
what Anki itself reports as due today, and restore puts dates back.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rwkv_harness  # noqa: E402

_REASON = rwkv_harness.unavailable_reason()
if _REASON is None:
    rwkv_harness.prepare_environment()  # must precede any rwkvp_addon import
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")

DAY = 86400


def _card_rows(col):
    return {
        int(row[0]): tuple(row[1:])
        for row in col.db.all(
            "select id, type, queue, due, ivl, left, odid, odue, factor, reps, lapses from cards"
        )
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("mobile_due_e2e")
    col = rwkv_harness.make_collection(root / "col", n_notes=80, days=120, seed=11)
    try:
        yield col, root
    finally:
        col.close()


def test_mobile_due_end_to_end(world):
    from rwkvp_addon.mobile_due import (
        MobileDueSettings,
        build_mobile_due_plan,
        plan_summary_lines,
        resolve_target_timestamp,
    )
    from rwkvp_addon.mobile_due_collection import (
        apply_mobile_due_changes,
        collect_mobile_due_inputs,
        restore_baseline_due_dates,
    )
    from rwkvp_addon.mobile_due_ledger import RUN_KIND_LIVE, MobileDueLedger

    col, root = world
    inputs = collect_mobile_due_inputs(col)
    assert len(inputs.card_ids) >= 20
    before_rows = _card_rows(col)
    target = resolve_target_timestamp("end_of_day", now=time.time(), day_cutoff=inputs.day_cutoff)
    assert inputs.day_cutoff - DAY <= target < inputs.day_cutoff

    with rwkv_harness.open_rwkv_predictor(col, inputs.card_ids, root / "store") as handle:
        started = time.perf_counter()
        plan = build_mobile_due_plan(
            card_ids=inputs.card_ids,
            schedule_infos=inputs.schedule_infos,
            review_data=handle.review_data,
            retentions=inputs.retentions,
            max_interval_by_deck=inputs.max_interval_by_deck,
            predictor=handle.predictor,
            settings=MobileDueSettings(priority_buckets=3),
            today=inputs.today,
            day_cutoff=inputs.day_cutoff,
            target_timestamp=target,
        )
        plan_seconds = time.perf_counter() - started
        print("\n".join(plan_summary_lines(plan)), f"\nplanned in {plan_seconds:.2f}s")

        dr_by_deck = {r.deck_id: r.desired_retention for r in inputs.retentions}

        def dr(card_id):
            return dr_by_deck[inputs.schedule_infos[card_id].source_deck_id]

        # 1. Due-now cards are exactly those below DR at T (no limit/minimum set).
        ids = list(plan.due_now_ids)
        considered = [cid for cid in inputs.card_ids if cid in handle.review_data.last_by_card]
        r_at_t = dict(zip(considered, handle.predict_at(considered, target)))
        below = {cid for cid, r in r_at_t.items() if r < dr(cid)}
        assert set(ids) == below
        # Ranked lowest recall first.
        ranked = [r_at_t[cid] for cid in ids]
        assert ranked == sorted(ranked)

        # 2. Every other changed card lands on RWKV's first below-DR day (within its cap).
        later = [c for c in plan.changes if not c.due_now]
        checked = 0
        for change in later:
            offset = change.new_due - inputs.today
            assert offset >= 1
            r_on_day = handle.predict_at([change.card_id], target + offset * DAY)[0]
            if offset < rwkv_harness_cap(inputs, handle, change.card_id):
                assert r_on_day < dr(change.card_id), change
            if offset > 1:
                r_before = handle.predict_at([change.card_id], target + (offset - 1) * DAY)[0]
                assert r_before >= dr(change.card_id), change
            checked += 1
        assert checked > 0

    # 3. Apply: only due (and a pinned last_review_time) changes, as one undo step.
    result = apply_mobile_due_changes(col, list(plan.changes))
    assert result.conflicts == ()
    assert len(result.applied) == plan.changed_count
    after_rows = _card_rows(col)
    changed = {c.card_id for c in result.applied}
    for cid, row in before_rows.items():
        if cid in changed:
            # type, queue, ivl, left, odid, odue, factor, reps, lapses unchanged; due changed.
            assert row[:2] == after_rows[cid][:2] and row[3:] == after_rows[cid][3:], cid
        else:
            assert row == after_rows[cid], cid

    # 4. Anki itself now reports exactly the RWKV set as due review cards.
    in_scope_due = set(col.find_cards("is:due is:review -is:learn")) & set(inputs.card_ids)
    assert in_scope_due == set(plan.due_now_ids)

    # 5. Ledger + restore returns every unreviewed card to its pre-RWKV date.
    ledger = MobileDueLedger(root / "ledger")
    ledger.record_changes(
        run_id="e2e", applied=result.applied, last_review_secs=result.last_review_secs, now=time.time() - 1
    )
    ledger.record_run(kind=RUN_KIND_LIVE, plan=plan, trigger="e2e", run_id="e2e", applied_count=len(result.applied))
    restored = restore_baseline_due_dates(col, ledger.baseline())
    assert restored.restored == len(changed)
    assert _card_rows(col) == before_rows


def rwkv_harness_cap(inputs, handle, card_id):
    """Mirror of the planner's cap: max interval from the last review day, and the horizon."""
    from rwkvp_addon.curve_reschedule import anki_day_for_timestamp

    cap = 365
    info = inputs.schedule_infos[card_id]
    max_ivl = inputs.max_interval_by_deck.get(info.source_deck_id)
    last = handle.review_data.last_by_card[card_id]
    if max_ivl is not None:
        last_day = anki_day_for_timestamp(last.timestamp_seconds, today=inputs.today, day_cutoff=inputs.day_cutoff)
        cap = min(cap, last_day + max_ivl - inputs.today)
    return max(1, cap)
