"""Mobile Due writes against a real Anki 26.05 collection (no GUI, no RWKV)."""

from __future__ import annotations

import time

import pytest

anki = pytest.importorskip("anki")
from anki.collection import Collection  # noqa: E402

from rwkvp_addon.mobile_due import MobileDueChange  # noqa: E402
from rwkvp_addon.mobile_due_collection import (  # noqa: E402
    UNDO_LABEL,
    PlanOutdatedError,
    apply_mobile_due_changes,
    collect_mobile_due_inputs,
    restore_baseline_due_dates,
)
from rwkvp_addon.mobile_due_ledger import (  # noqa: E402
    RUN_KIND_LIVE,
    RUN_KIND_SHADOW,
    MobileDueLedger,
)


@pytest.fixture()
def col(tmp_path):
    collection = Collection(str(tmp_path / "collection.anki2"))
    try:
        yield collection
    finally:
        collection.close()


def add_cards(col, count, deck_name="Default"):
    deck_id = col.decks.id(deck_name)
    model = col.models.by_name("Basic")
    ids = []
    for index in range(count):
        note = col.new_note(model)
        note["Front"] = f"{deck_name} q{index}"
        note["Back"] = f"a{index}"
        col.add_note(note, deck_id)
        ids.extend(note.card_ids())
    return ids


_REVLOG_SEQ = iter(range(10**6))


def make_review(col, card_ids, *, due_offset=0, ivl=10, reviewed_days_ago=10):
    today = col.sched.today
    for card_id in card_ids:
        review_ms = int((time.time() - reviewed_days_ago * 86400) * 1000) + next(_REVLOG_SEQ)
        col.db.execute(
            "insert into revlog (id,cid,usn,ease,ivl,lastIvl,factor,time,type) values (?,?,?,?,?,?,?,?,?)",
            review_ms, card_id, -1, 3, ivl, 0, 0, 4000, 1,
        )
        col.db.execute(
            "update cards set type=2, queue=2, ivl=?, due=?, mod=?, usn=-1 where id=?",
            ivl,
            today + due_offset,
            int(time.time()) - 100,
            card_id,
        )


def change_for(col, card_id, new_due):
    card = col.get_card(card_id)
    return MobileDueChange(
        card_id=card_id,
        old_due=card.due,
        new_due=new_due,
        old_mod=card.mod,
        old_interval=card.ivl,
        due_now=new_due <= col.sched.today,
        predicted_retrievability=0.8,
        desired_retention=0.9,
    )


def test_collect_inputs_scope(col):
    review = add_cards(col, 4)
    make_review(col, review)
    new_cards = add_cards(col, 1)
    suspended = add_cards(col, 1)
    make_review(col, suspended)
    col.sched.suspend_cards(suspended)
    buried = add_cards(col, 1)
    make_review(col, buried)
    col.sched.bury_cards(buried)
    learning = add_cards(col, 1)
    col.db.execute("update cards set type=1, queue=1, due=? where id=?", int(time.time()), learning[0])
    other_deck = add_cards(col, 2, deck_name="Other")
    make_review(col, other_deck)

    inputs = collect_mobile_due_inputs(col)
    assert set(inputs.card_ids) == set(review) | set(other_deck)
    assert not set(inputs.card_ids) & set(new_cards + suspended + buried + learning)
    assert inputs.today == col.sched.today
    assert inputs.day_cutoff == col.sched.day_cutoff
    assert {r.deck_id for r in inputs.retentions} == {col.decks.id("Default"), col.decks.id("Other")}
    assert all(0 < r.desired_retention <= 1 for r in inputs.retentions)

    narrowed = collect_mobile_due_inputs(col, extra_search='deck:Other')
    assert set(narrowed.card_ids) == set(other_deck)


def test_collect_excludes_filtered_deck_cards(col):
    cards = add_cards(col, 3)
    make_review(col, cards, due_offset=-1)
    filtered_id = col.decks.new_filtered("RWKV test filter")
    deck = col.decks.get(filtered_id)
    deck["terms"] = [[f"cid:{cards[0]}", 10, 0]]
    col.decks.save(deck)
    col.sched.rebuild_filtered_deck(filtered_id)
    assert col.get_card(cards[0]).odid
    inputs = collect_mobile_due_inputs(col)
    assert cards[0] not in inputs.card_ids
    assert set(inputs.card_ids) == set(cards[1:])


def test_apply_writes_only_due_as_one_undo_step(col):
    cards = add_cards(col, 3)
    make_review(col, cards, due_offset=5, ivl=12)
    today = col.sched.today
    before = {cid: col.get_card(cid) for cid in cards}
    changes = [change_for(col, cards[0], today), change_for(col, cards[1], today - 3), change_for(col, cards[2], today + 40)]
    result = apply_mobile_due_changes(col, changes)
    assert [c.card_id for c in result.applied] == cards
    assert result.conflicts == ()
    for cid, expected in zip(cards, (today, today - 3, today + 40)):
        card = col.get_card(cid)
        assert card.due == expected
        assert (card.type, card.queue, card.ivl, card.left, card.odid) == (
            before[cid].type, before[cid].queue, before[cid].ivl, before[cid].left, before[cid].odid,
        )
        assert card.usn == -1  # will be sent on next sync
        # FSRS's elapsed-days input is pinned to the real last review.
        last_ms = col.db.scalar("select max(id) from revlog where cid=?", cid)
        assert card.last_review_time == last_ms // 1000
    # Due-now cards are what the scheduler would show as due today.
    due_today = set(col.find_cards("is:due"))
    assert {cards[0], cards[1]} <= due_today
    assert cards[2] not in due_today

    assert col.undo_status().undo == UNDO_LABEL
    col.undo()
    for cid in cards:
        assert col.get_card(cid).due == before[cid].due


def test_apply_skips_cards_changed_since_planning(col):
    cards = add_cards(col, 2)
    make_review(col, cards, due_offset=3)
    today = col.sched.today
    changes = [change_for(col, cid, today) for cid in cards]
    # Card 0 is touched (e.g. answered or synced from the phone) after planning.
    card = col.get_card(cards[0])
    card.due = today + 9
    time.sleep(1.1)
    col.update_card(card)
    result = apply_mobile_due_changes(col, changes)
    assert result.conflicts == (cards[0],)
    assert [c.card_id for c in result.applied] == [cards[1]]
    assert col.get_card(cards[0]).due == today + 9
    assert col.get_card(cards[1]).due == today


def test_restore_puts_back_unreviewed_cards_only(col, tmp_path):
    cards = add_cards(col, 3)
    make_review(col, cards, due_offset=4)
    today = col.sched.today
    original = {cid: col.get_card(cid).due for cid in cards}
    ledger = MobileDueLedger(tmp_path / "mobile_due")
    changes = [change_for(col, cid, today) for cid in cards]
    run_id = ledger.new_run_id()
    result = apply_mobile_due_changes(
        col,
        changes,
        before_write=lambda applied, reviews: ledger.record_changes(
            run_id=run_id, applied=applied, last_review_secs=reviews, now=time.time() - 5
        ),
    )
    ledger.record_run(kind=RUN_KIND_LIVE, plan=None, trigger="test", run_id=run_id, applied_count=len(result.applied))

    # Card 2 is reviewed afterwards: it must keep its new schedule.
    reviewed_at_ms = int(time.time() * 1000)
    col.db.execute(
        "insert into revlog (id,cid,usn,ease,ivl,lastIvl,factor,time,type) values (?,?,?,?,?,?,?,?,?)",
        reviewed_at_ms, cards[2], -1, 3, 15, 10, 2500, 5000, 1,
    )
    col.db.execute("update cards set due=? where id=?", today + 15, cards[2])

    restore = restore_baseline_due_dates(col, ledger.baseline())
    assert (restore.restored, restore.reviewed_since, restore.not_review, restore.missing) == (2, 1, 0, 0)
    assert col.get_card(cards[0]).due == original[cards[0]]
    assert col.get_card(cards[1]).due == original[cards[1]]
    assert col.get_card(cards[2]).due == today + 15


def test_ledger_roundtrip(tmp_path):
    ledger = MobileDueLedger(tmp_path / "md")
    assert not ledger.has_live_run()
    first = MobileDueChange(1, 100, 90, 5, 10, True, 0.7, 0.9, 0)
    again = MobileDueChange(1, 90, 95, 6, 10, False, 0.92, 0.9, None)
    ledger.record_run(kind=RUN_KIND_SHADOW, plan=None, trigger="t")
    ledger.record_run(kind=RUN_KIND_LIVE, plan=None, trigger="t", applied_count=0)
    assert not ledger.has_live_run()  # a live run that changed nothing does not count
    ledger.record_changes(run_id="r1", applied=[first], last_review_secs={1: 500}, now=1000.0)
    ledger.record_run(kind=RUN_KIND_LIVE, plan=None, trigger="t", run_id="r1", applied_count=1, now=1000.0)
    assert ledger.has_live_run()
    # Moved again without a review in between: the pre-RWKV value is kept.
    ledger.record_changes(run_id="r2", applied=[again], last_review_secs={1: 900}, now=2000.0)
    assert ledger.baseline()[1].original_due == 100
    assert ledger.baseline()[1].first_changed_at == 1000.0
    # Reviewed after RWKV moved it, then moved again: the post-review schedule becomes the baseline.
    third = MobileDueChange(1, 140, 120, 7, 30, False, 0.95, 0.9, None)
    ledger.record_changes(run_id="r3", applied=[third], last_review_secs={1: 2500}, now=3000.0)
    assert ledger.baseline()[1].original_due == 140
    assert ledger.baseline()[1].first_changed_at == 3000.0
    # A torn trailing line (crash mid-write) is ignored.
    with ledger.runs_path.open("a") as handle:
        handle.write('{"kind": "li')
    assert len(ledger.runs()) == 3
    changes_files = list(ledger.changes_dir.glob("*.jsonl"))
    assert changes_files and all(len(p.stem) == 10 and p.stem[4] == "-" for p in changes_files)
    archived = ledger.archive_baseline(now=4000.0)
    assert archived is not None and archived.exists()
    assert ledger.baseline() == {}


def test_restore_after_review_and_second_move_returns_scheduler_due(col, tmp_path):
    """Regression: scheduler +20 -> RWKV +3 -> reviewed (scheduler +15) -> RWKV +8 -> restore = +15."""
    cards = add_cards(col, 1)
    make_review(col, cards, due_offset=20)
    today = col.sched.today
    ledger = MobileDueLedger(tmp_path / "md")

    def apply_with_ledger(new_due, now):
        return apply_mobile_due_changes(
            col,
            [change_for(col, cards[0], new_due)],
            before_write=lambda applied, reviews: ledger.record_changes(
                run_id=ledger.new_run_id(), applied=applied, last_review_secs=reviews, now=now
            ),
        )

    apply_with_ledger(today + 3, time.time() - 3600)
    # The phone answers it (synced in): new revlog, scheduler puts it at +15.
    col.db.execute(
        "insert into revlog (id,cid,usn,ease,ivl,lastIvl,factor,time,type) values (?,?,?,?,?,?,?,?,?)",
        int((time.time() - 1800) * 1000), cards[0], -1, 3, 15, 10, 0, 3000, 1,
    )
    col.db.execute("update cards set due=?, mod=? where id=?", today + 15, int(time.time()) - 1800, cards[0])
    apply_with_ledger(today + 8, time.time() - 60)
    assert col.get_card(cards[0]).due == today + 8
    restore = restore_baseline_due_dates(col, ledger.baseline())
    assert restore.restored == 1
    assert col.get_card(cards[0]).due == today + 15


def test_ledger_failure_aborts_apply_with_nothing_written(col):
    cards = add_cards(col, 2)
    make_review(col, cards, due_offset=4)
    today = col.sched.today
    before = {cid: col.get_card(cid).due for cid in cards}

    def broken_ledger(_applied, _reviews):
        raise OSError("disk full")

    with pytest.raises(OSError):
        apply_mobile_due_changes(col, [change_for(col, cid, today) for cid in cards], before_write=broken_ledger)
    assert {cid: col.get_card(cid).due for cid in cards} == before
    assert col.undo_status().undo != UNDO_LABEL


def test_apply_refuses_a_plan_from_a_previous_day(col):
    cards = add_cards(col, 1)
    make_review(col, cards, due_offset=4)
    with pytest.raises(PlanOutdatedError):
        apply_mobile_due_changes(col, [change_for(col, cards[0], col.sched.today)], plan_today=col.sched.today - 1)


def test_all_conflicts_leave_no_empty_undo_step(col):
    cards = add_cards(col, 1)
    make_review(col, cards, due_offset=4)
    change = change_for(col, cards[0], col.sched.today)
    stale = MobileDueChange(**{**change.__dict__, "old_mod": change.old_mod - 50})
    result = apply_mobile_due_changes(col, [stale])
    assert result.applied == () and result.conflicts == (cards[0],)
    assert col.undo_status().undo != UNDO_LABEL
