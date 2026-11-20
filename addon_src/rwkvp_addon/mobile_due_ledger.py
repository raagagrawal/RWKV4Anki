"""Append-only audit log for Mobile Due runs.

Layout under ``<profile store>/mobile_due/``::

    runs.jsonl                one summary line per run (shadow, live, aborted, restore)
    changes/YYYY-MM-DD.jsonl  one line per rewritten card, by local run date
    baseline.jsonl            the first pre-RWKV ``due`` ever seen per card

``baseline.jsonl`` is what "Restore pre-RWKV due dates" reads. A card is only
added the first time Mobile Due changes it, so its value is always the
schedule Anki/FSRS had produced before RWKV touched it.
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

    def ensure(self) -> None:
        self.changes_dir.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------

    def record_run(
        self,
        *,
        kind: str,
        plan: MobileDuePlan | None,
        trigger: str,
        applied: Iterable[MobileDueChange] = (),
        conflicts: int = 0,
        violations: Iterable[str] = (),
        note: str = "",
        extra: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> str:
        self.ensure()
        run_id = uuid.uuid4().hex[:12]
        timestamp = float(time.time() if now is None else now)
        applied_list = list(applied)
        summary: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp)),
            "kind": kind,
            "trigger": trigger,
            "applied": len(applied_list),
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
                    "capped_by_limit": plan.capped_by_limit,
                    "projected_to_horizon": plan.projected_to_horizon,
                    "expected_retrievability": plan.expected_retrievability,
                    "skipped": {reason.value: count for reason, count in plan.skipped.items()},
                }
            )
        if extra:
            summary.update(dict(extra))
        if applied_list:
            self._append_baseline(applied_list, timestamp)
            day = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            _append_jsonl(
                self.changes_dir / f"{day}.jsonl",
                (
                    {
                        "run_id": run_id,
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
        _append_jsonl(self.runs_path, [summary])
        return run_id

    def _append_baseline(self, changes: Iterable[MobileDueChange], timestamp: float) -> None:
        known = set(self.baseline())
        fresh = [change for change in changes if change.card_id not in known]
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

    def recent_due_now_counts(self, limit: int = 7) -> list[int]:
        counts = [
            int(run["due_now"])
            for run in self.runs()
            if run.get("kind") in {RUN_KIND_LIVE, RUN_KIND_SHADOW} and "due_now" in run
        ]
        return counts[-int(limit):]

    def has_live_run(self) -> bool:
        return any(run.get("kind") == RUN_KIND_LIVE for run in self.runs())

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
            entries.setdefault(card_id, entry)
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
