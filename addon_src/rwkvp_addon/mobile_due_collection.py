"""Collection-side reads and writes for Mobile Due.

Uses only the ``anki`` collection API (no ``aqt``), so the same functions serve
the desktop add-on and any headless runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .anki_api import (
    DeckRetention,
    card_schedule_info_for_ids,
    deck_maximum_review_interval,
    deck_retention,
    find_cards,
    is_filtered_deck,
)
from .curve_reschedule import CARD_TYPE_REVIEW, QUEUE_TYPE_REVIEW, CardScheduleInfo
from .mobile_due import MobileDueChange
from .mobile_due_ledger import BaselineEntry
from .search_filters import required_search_with_filter

# Review-queue cards only. Learning/relearning steps, new, suspended, buried
# and filtered-deck cards are left to the phone's own scheduler.
MOBILE_DUE_BASE_SEARCH = "is:review -is:learn -is:suspended -is:buried -deck:filtered"
UNDO_LABEL = "RWKV Mobile Due"
RESTORE_UNDO_LABEL = "RWKV Mobile Due: Restore"
APPLY_CHUNK_SIZE = 500


@dataclass(frozen=True)
class MobileDueInputs:
    search: str
    card_ids: tuple[int, ...]
    schedule_infos: Mapping[int, CardScheduleInfo]
    retentions: tuple[DeckRetention, ...]
    max_interval_by_deck: Mapping[int, int | None]
    today: int
    day_cutoff: int


@dataclass(frozen=True)
class ApplyResult:
    applied: tuple[MobileDueChange, ...]
    conflicts: tuple[int, ...]
    changes: object | None = None


@dataclass(frozen=True)
class RestoreResult:
    restored: int
    reviewed_since: int
    not_review: int
    missing: int
    changes: object | None = None


def mobile_due_search(extra_search: str | None = None) -> str:
    return required_search_with_filter(MOBILE_DUE_BASE_SEARCH, extra_search or None)


def collect_mobile_due_inputs(col, *, extra_search: str | None = None) -> MobileDueInputs:
    search = mobile_due_search(extra_search)
    card_ids = tuple(find_cards(col, search))
    schedule_infos = card_schedule_info_for_ids(col, card_ids)
    deck_ids = sorted(
        {
            int(info.source_deck_id)
            for info in schedule_infos.values()
            if info.source_deck_id is not None
        }
    )
    retentions: list[DeckRetention] = []
    max_intervals: dict[int, int | None] = {}
    for deck_id in deck_ids:
        if is_filtered_deck(col, deck_id):
            continue
        retentions.append(deck_retention(col, deck_id))
        max_intervals[deck_id] = deck_maximum_review_interval(col, deck_id)
    return MobileDueInputs(
        search=search,
        card_ids=card_ids,
        schedule_infos=schedule_infos,
        retentions=tuple(retentions),
        max_interval_by_deck=max_intervals,
        today=int(col.sched.today),
        day_cutoff=int(col.sched.day_cutoff),
    )


def apply_mobile_due_changes(col, changes: Sequence[MobileDueChange]) -> ApplyResult:
    """Write planned ``due`` values as a single undoable step.

    A card is skipped (a conflict) if anything about it changed since it was
    planned: answered on the desktop, synced in from the phone, moved, or
    rescheduled. The next run plans it afresh.
    """

    if not changes:
        return ApplyResult(applied=(), conflicts=(), changes=None)
    undo_target = col.add_custom_undo_entry(UNDO_LABEL)
    applied: list[MobileDueChange] = []
    conflicts: list[int] = []
    for start in range(0, len(changes), APPLY_CHUNK_SIZE):
        chunk = changes[start : start + APPLY_CHUNK_SIZE]
        last_review_secs = _latest_graded_review_secs_by_card(
            col, (change.card_id for change in chunk)
        )
        cards = []
        for change in chunk:
            try:
                card = col.get_card(change.card_id)
            except Exception:
                conflicts.append(change.card_id)
                continue
            if not _still_as_planned(card, change):
                conflicts.append(change.card_id)
                continue
            if not _pin_last_review_time(card, last_review_secs.get(change.card_id)):
                conflicts.append(change.card_id)
                continue
            card.due = int(change.new_due)
            cards.append(card)
            applied.append(change)
        if cards:
            col.update_cards(cards)
    op_changes = col.merge_undo_entries(undo_target)
    return ApplyResult(applied=tuple(applied), conflicts=tuple(conflicts), changes=op_changes)


def _pin_last_review_time(card, revlog_secs: int | None) -> bool:
    """Make sure moving ``due`` cannot change FSRS's elapsed-days input.

    When ``last_review_time`` is unset, Anki derives the last review day from
    ``due - ivl``, so moving ``due`` would corrupt the elapsed time FSRS uses
    when the card is next answered (on any device). Fill it from the revlog,
    exactly as Anki itself records it on every review. Returns False when it
    cannot be established, in which case the card must not be moved.
    """

    if not hasattr(card, "last_review_time"):
        return False
    current = getattr(card, "last_review_time", None)
    if current:
        return True
    if not revlog_secs:
        return False
    card.last_review_time = int(revlog_secs)
    return True


def _latest_graded_review_secs_by_card(col, card_ids: Iterable[int]) -> dict[int, int]:
    ids = sorted({int(card_id) for card_id in card_ids})
    if not ids:
        return {}
    rows = col.db.all(
        "SELECT cid, MAX(id) FROM revlog "
        f"WHERE cid IN ({','.join(map(str, ids))}) AND ease BETWEEN 1 AND 4 "
        "GROUP BY cid"
    )
    return {int(cid): int(review_id) // 1000 for cid, review_id in rows if review_id}


def _still_as_planned(card, change: MobileDueChange) -> bool:
    return (
        int(card.mod) == int(change.old_mod)
        and int(card.due) == int(change.old_due)
        and int(card.type) == CARD_TYPE_REVIEW
        and int(card.queue) == QUEUE_TYPE_REVIEW
        and int(getattr(card, "odid", 0) or 0) == 0
    )


def restore_baseline_due_dates(
    col,
    baseline: Mapping[int, BaselineEntry],
) -> RestoreResult:
    """Put back the pre-RWKV ``due`` of every card not reviewed since RWKV moved it.

    Cards reviewed since then already carry a fresh schedule from whichever
    app answered them, so they are left alone.
    """

    if not baseline:
        return RestoreResult(0, 0, 0, 0, None)
    latest_review = _latest_review_ms_by_card(col, baseline.keys())
    existing = set(_existing_card_ids(col, baseline.keys()))
    undo_target = col.add_custom_undo_entry(RESTORE_UNDO_LABEL)
    restored = reviewed = not_review = missing = 0
    entries = list(baseline.values())
    for start in range(0, len(entries), APPLY_CHUNK_SIZE):
        cards = []
        for entry in entries[start : start + APPLY_CHUNK_SIZE]:
            if entry.card_id not in existing:
                missing += 1
                continue
            if latest_review.get(entry.card_id, 0) > int(entry.first_changed_at * 1000):
                reviewed += 1
                continue
            card = col.get_card(entry.card_id)
            if (
                int(card.type) != CARD_TYPE_REVIEW
                or int(card.queue) != QUEUE_TYPE_REVIEW
                or int(getattr(card, "odid", 0) or 0)
            ):
                not_review += 1
                continue
            if int(card.due) != int(entry.original_due):
                card.due = int(entry.original_due)
                cards.append(card)
            restored += 1
        if cards:
            col.update_cards(cards)
    op_changes = col.merge_undo_entries(undo_target)
    return RestoreResult(restored, reviewed, not_review, missing, op_changes)


def _latest_review_ms_by_card(col, card_ids: Iterable[int]) -> dict[int, int]:
    ids = sorted({int(card_id) for card_id in card_ids})
    latest: dict[int, int] = {}
    for start in range(0, len(ids), APPLY_CHUNK_SIZE):
        chunk = ids[start : start + APPLY_CHUNK_SIZE]
        rows = col.db.all(
            f"SELECT cid, MAX(id) FROM revlog WHERE cid IN ({','.join(map(str, chunk))}) GROUP BY cid"
        )
        for card_id, review_id in rows:
            latest[int(card_id)] = int(review_id or 0)
    return latest


def _existing_card_ids(col, card_ids: Iterable[int]) -> list[int]:
    ids = sorted({int(card_id) for card_id in card_ids})
    found: list[int] = []
    for start in range(0, len(ids), APPLY_CHUNK_SIZE):
        chunk = ids[start : start + APPLY_CHUNK_SIZE]
        found.extend(
            int(row[0])
            for row in col.db.all(f"SELECT id FROM cards WHERE id IN ({','.join(map(str, chunk))})")
        )
    return found
