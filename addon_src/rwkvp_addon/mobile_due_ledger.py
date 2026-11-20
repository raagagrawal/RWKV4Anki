"""Append-only audit log for Mobile Due runs.

Layout under ``<profile store>/mobile_due/``::

    runs.jsonl                one summary line per run (shadow, live, aborted, restore)
    changes/YYYY-MM-DD.jsonl  one line per rewritten card, by local run date
    baseline.jsonl            the first pre-RWKV ``due`` ever seen per card

``baseline.jsonl`` is what "Restore pre-RWKV due dates" reads. A card is added
the first time Mobile Due changes it, and again whenever it is moved after
having been reviewed since, so the latest line per card is always the schedule
the phone or desktop scheduler last produced before RWKV took it over.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mobile_due import MobileDueChange, MobileDuePlan

RUN_KIND_SHADOW = "shadow"
RUN_KIND_LIVE = "live"
RUN_KIND_ABORTED = "aborted"
RUN_KIND_RESTORE = "restore"


@dataclass(frozen=True)
class BaselineEntry:
    card_id: int
    original_due: int
    first_changed_at: float


class MobileDueLedger:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def for_store_root(cls, store_root: Path) -> MobileDueLedger:
        return cls(Path(store_root) / "mobile_due")

    @property
    def runs_path(self) -> Path:
        return self.root / "runs.jsonl"

    @property
    def changes_dir(self) -> Path:
        return self.root / "changes"

    @property
    def baseline_path(self) -> Path:
        return self.root / "baseline.jsonl"

    @property
    def pushed_out_path(self) -> Path:
        return self.root / "pushed_out.jsonl"

    def ensure(self) -> None:
        self.changes_dir.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:12]

    def record_changes(
        self,
        *,
        run_id: str,
        applied: Iterable[MobileDueChange],
        last_review_secs: Mapping[int, int] | None = None,
        today: int | None = None,
        now: float | None = None,
    ) -> None:
        """Durably log rewritten cards. Call BEFORE writing them to the collection.

        Raises on any I/O error so the caller can abort the apply: a change
        must never reach the collection without a way to restore it.
        """

        applied_list = list(applied)
        if not applied_list:
            return
        self.ensure()
        timestamp = float(time.time() if now is None else now)
        self._update_baseline(applied_list, last_review_secs or {}, timestamp)
        if today is not None:
            # Track push-outs (due -> later). A later change to a tracked card
            # clears its mark. Only those rows are written, so the file stays small.
            tracked = self.pushed_out_unreviewed_candidates()
            rows = []
            for change in applied_list:
                pushed = change.old_due <= int(today) < change.new_due
                if pushed or change.card_id in tracked:
                    rows.append(
                        {"card_id": change.card_id, "timestamp": timestamp, "pushed_out": pushed}
                    )
            _append_jsonl(self.pushed_out_path, rows)
        day = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        _append_jsonl(
            self.changes_dir / f"{day}.jsonl",
            (
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "card_id": change.card_id,
                    "old_due": change.old_due,
                    "new_due": change.new_due,
                    "old_mod": change.old_mod,
                    "old_ivl": change.old_interval,
                    "due_now": change.due_now,
                    "r": round(change.predicted_retrievability, 6),
                    "dr": change.desired_retention,
                    "rank": change.rank,
                }
                for change in applied_list
            ),
        )

    def record_run(
        self,
        *,
        kind: str,
        plan: MobileDuePlan | None,
        trigger: str,
        run_id: str | None = None,
        applied_count: int = 0,
        conflicts: int = 0,
        violations: Iterable[str] = (),
        note: str = "",
        extra: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> str:
        """Append one run summary line."""

        self.ensure()
        resolved_id = run_id or self.new_run_id()
        timestamp = float(time.time() if now is None else now)
        summary: dict[str, Any] = {
            "run_id": resolved_id,
            "timestamp": timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp)),
            "kind": kind,
            "trigger": trigger,
            "applied": int(applied_count),
            "conflicts": int(conflicts),
            "violations": list(violations),
            "note": note,
        }
        if plan is not None:
            summary.update(
                {
                    "today": plan.today,
                    "target_timestamp": plan.target_timestamp,
                    "in_scope": plan.in_scope_count,
                    "due_now": plan.due_now_count,
                    "planned_changes": plan.changed_count,
                    "pulled_in": plan.pulled_in_count,
                    "pushed_out": plan.pushed_out_count,
                    "unchanged": plan.unchanged_count,
                    "locked_due": plan.locked_due_count,
                    "capped_by_limit": plan.capped_by_limit,
                    "projected_to_horizon": plan.projected_to_horizon,
                    "beyond_horizon": plan.beyond_horizon_count,
                    "beyond_horizon_fraction": (
                        plan.beyond_horizon_count / plan.in_scope_count
                        if plan.in_scope_count
                        else 0.0
                    ),
                    "expected_retrievability": plan.expected_retrievability,
                    "skipped": {reason.value: count for reason, count in plan.skipped.items()},
                }
            )
        if extra:
            summary.update(dict(extra))
        _append_jsonl(self.runs_path, [summary])
        return resolved_id

    def _update_baseline(
        self,
        changes: Iterable[MobileDueChange],
        last_review_secs: Mapping[int, int],
        timestamp: float,
    ) -> None:
        """Remember the schedule each card had before RWKV (last) took it over.

        A card enters the baseline the first time Mobile Due moves it. If it is
        reviewed afterwards, its next move records the post-review schedule
        instead: that is the scheduler's own answer and what restore should put
        back.
        """

        known = self.baseline()
        fresh = []
        for change in changes:
            entry = known.get(change.card_id)
            reviewed = int(last_review_secs.get(change.card_id, 0) or 0)
            if entry is None or reviewed > entry.first_changed_at:
                fresh.append(change)
        if not fresh:
            return
        _append_jsonl(
            self.baseline_path,
            (
                {
                    "card_id": change.card_id,
                    "original_due": change.old_due,
                    "first_changed_at": timestamp,
                }
                for change in fresh
            ),
        )

    def archive_baseline(self, now: float | None = None) -> Path | None:
        if not self.baseline_path.exists():
            return None
        stamp = time.strftime("%Y-%m-%dT%H%M%S", time.localtime(time.time() if now is None else now))
        target = self.root / f"baseline_restored_{stamp}.jsonl"
        os.replace(self.baseline_path, target)
        return target

    # -- reading ---------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        return list(_read_jsonl(self.runs_path))

    def recent_values(self, key: str, limit: int = 7) -> list[float]:
        """Recent planned values, held-back runs included (see guardrails)."""
        values = [
            float(run[key])
            for run in self.runs()
            if run.get("kind") in {RUN_KIND_LIVE, RUN_KIND_SHADOW, RUN_KIND_ABORTED}
            and run.get(key) is not None
        ]
        return values[-int(limit):]

    def recent_due_now_counts(self, limit: int = 7) -> list[int]:
        return [int(value) for value in self.recent_values("due_now", limit)]

    def pushed_out_unreviewed_candidates(self) -> dict[int, float]:
        """Cards whose latest change moved them from due to a later day.

        Maps card id -> time of that change. The phone may still show such a
        card as due (and answer it) until it syncs, so automatic runs must not
        rewrite it again before a newer review has been seen.
        """

        latest: dict[int, tuple[float, bool]] = {}
        for row in _read_jsonl(self.pushed_out_path):
            try:
                latest[int(row["card_id"])] = (float(row["timestamp"]), bool(row["pushed_out"]))
            except (KeyError, TypeError, ValueError):
                continue
        return {card_id: ts for card_id, (ts, pushed) in latest.items() if pushed}

    def has_live_run(self) -> bool:
        """True once a live run has actually rewritten at least one card."""
        return any(
            run.get("kind") == RUN_KIND_LIVE and int(run.get("applied") or 0) > 0
            for run in self.runs()
        )

    def last_run(self) -> dict[str, Any] | None:
        runs = self.runs()
        return runs[-1] if runs else None

    def baseline(self) -> dict[int, BaselineEntry]:
        entries: dict[int, BaselineEntry] = {}
        for row in _read_jsonl(self.baseline_path):
            try:
                card_id = int(row["card_id"])
                entry = BaselineEntry(
                    card_id=card_id,
                    original_due=int(row["original_due"]),
                    first_changed_at=float(row["first_changed_at"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            entries[card_id] = entry  # the latest line for a card wins
        return entries


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    if not payload:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a crash must not block later runs.
                continue
            if isinstance(value, dict):
                yield value
