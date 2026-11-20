"""Desktop wiring for Mobile Due.

Flow of one run (``_start_run``)::

    collect (collection thread) -> plan (background, RWKV) -> decide (main)
        -> apply as one undoable CollectionOp -> ledger -> sync push

Triggers: the RWKV -> Mobile Due menu, Anki's own sync finishing (when
``mobile_due_run_after_sync`` is on), and an optional timer that syncs every
``mobile_due_auto_sync_hours`` while Anki is open. Our own post-apply sync is
flagged so it does not trigger another run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from aqt import gui_hooks, mw
from aqt.qt import QAction, QActionGroup, QMenu, QTimer
from aqt.utils import openFolder, qconnect, tooltip

from ..addon_config import addon_config_for_mw, minimum_review_widening_extra, write_addon_config_for_mw
from ..anki_api import is_fsrs_enabled
from ..dataset_export import load_review_data_for_checkpoint, open_checkpoint_runtime_from_load
from ..filtered_deck_predictors import filtered_deck_predictor_for_mode
from ..mobile_due import (
    MobileDuePlan,
    build_mobile_due_plan,
    guardrail_violations,
    plan_summary_lines,
    resolve_target_timestamp,
)
from ..mobile_due_collection import (
    ApplyResult,
    collect_mobile_due_inputs,
    apply_mobile_due_changes,
    restore_baseline_due_dates,
)
from ..mobile_due_config import MobileDueConfig, MobileDueMode, mobile_due_config, with_mode
from ..mobile_due_ledger import (
    RUN_KIND_ABORTED,
    RUN_KIND_LIVE,
    RUN_KIND_RESTORE,
    RUN_KIND_SHADOW,
    MobileDueLedger,
)
from ..progress import TimedProgressReporter
from ..review_rows import checkpoint_scope_cards_for_card_ids
from ..runtime import manager_for_mw, store_for_mw
from ..rwkv_modes import RetrievabilityMode
from .background_stages import run_background_stages
from .checkpoint_failure import handle_checkpoint_failure, require_checkpoint_for_use
from .common import ProgressStage, run_with_progress_stages, show_quiet_info
from .web_message import ask_web_confirmation, show_web_warning

TITLE = "RWKV Mobile Due"
TRIGGER_MENU_PREVIEW = "menu_preview"
TRIGGER_MENU_APPLY = "menu_apply"
TRIGGER_SYNC = "after_sync"

_menu: QMenu | None = None
_mode_actions: dict[MobileDueMode, QAction] = {}
_timer: QTimer | None = None
_running = False
_closing = False
_skip_next_sync_hook = False
_hooks_installed = False


class NothingToPlan(Exception):
    pass


@dataclass(frozen=True)
class _Collected:
    inputs: object
    review_load: object
    target_timestamp: float


# --- installation --------------------------------------------------------------


def install_mobile_due_menu(parent_menu: QMenu) -> None:
    global _menu, _hooks_installed
    if _menu is not None:
        return
    _menu = QMenu("Mobile Due", parent_menu)
    parent_menu.addMenu(_menu)

    preview = QAction("Preview (no changes)...", _menu)
    qconnect(preview.triggered, lambda _c=False: _start_run(TRIGGER_MENU_PREVIEW, interactive=True))
    _menu.addAction(preview)

    apply_now = QAction("Apply Now...", _menu)
    qconnect(apply_now.triggered, lambda _c=False: _start_run(TRIGGER_MENU_APPLY, interactive=True))
    _menu.addAction(apply_now)

    _menu.addSeparator()
    group = QActionGroup(_menu)
    group.setExclusive(True)
    labels = {
        MobileDueMode.OFF: "Mode: Off",
        MobileDueMode.SHADOW: "Mode: Shadow (plan and log after each sync, change nothing)",
        MobileDueMode.LIVE: "Mode: Live (rewrite due dates after each sync)",
    }
    for mode, label in labels.items():
        action = QAction(label, _menu)
        action.setCheckable(True)
        group.addAction(action)
        qconnect(action.triggered, lambda _c=False, m=mode: _set_mode(m))
        _menu.addAction(action)
        _mode_actions[mode] = action

    settings = QAction("Settings...", _menu)
    qconnect(settings.triggered, lambda _c=False: _show_settings())
    _menu.addAction(settings)

    _menu.addSeparator()
    restore = QAction("Restore Pre-RWKV Due Dates...", _menu)
    qconnect(restore.triggered, lambda _c=False: _confirm_restore())
    _menu.addAction(restore)

    open_log = QAction("Open Mobile Due Log Folder", _menu)
    qconnect(open_log.triggered, lambda _c=False: _open_log_folder())
    _menu.addAction(open_log)

    qconnect(_menu.aboutToShow, _refresh_mode_actions)

    if not _hooks_installed:
        gui_hooks.sync_did_finish.append(_on_sync_did_finish)
        gui_hooks.profile_did_open.append(_on_profile_did_open)
        gui_hooks.profile_will_close.append(_on_profile_will_close)
        _hooks_installed = True


def _config() -> MobileDueConfig:
    config = addon_config_for_mw(mw)
    return mobile_due_config(
        config,
        minimum_retention_extra_quantum=minimum_review_widening_extra(config),
    )


def _ledger() -> MobileDueLedger:
    return MobileDueLedger.for_store_root(store_for_mw(mw).root)


def _show_settings() -> None:
    from .mobile_due_settings import MobileDueSettingsDialog

    def saved() -> None:
        _refresh_mode_actions()
        _reschedule_timer()

    MobileDueSettingsDialog(mw, live_allowed=_ledger().has_live_run(), on_saved=saved).exec()


def _refresh_mode_actions() -> None:
    mode = _config().mode
    for candidate, action in _mode_actions.items():
        action.setChecked(candidate == mode)


def _set_mode(mode: MobileDueMode) -> None:
    def write() -> None:
        write_addon_config_for_mw(mw, with_mode(addon_config_for_mw(mw), mode))
        _refresh_mode_actions()
        _reschedule_timer()
        tooltip(f"{TITLE}: mode set to {mode.value}.", parent=mw)

    if mode == MobileDueMode.LIVE and not _ledger().has_live_run():
        _refresh_mode_actions()
        show_web_warning(
            "Run RWKV -> Mobile Due -> Apply Now once and review its summary before "
            "turning on Live mode. Live mode will then rewrite due dates after every sync.",
            title=TITLE,
            parent=mw,
        )
        return
    write()


# --- hooks and timer ------------------------------------------------------------


def _on_profile_did_open() -> None:
    global _closing
    _closing = False
    _reschedule_timer()


def _on_profile_will_close() -> None:
    global _closing, _timer
    _closing = True
    if _timer is not None:
        _timer.stop()


def _on_sync_did_finish() -> None:
    global _skip_next_sync_hook
    if _skip_next_sync_hook:
        _skip_next_sync_hook = False
        return
    if _closing or mw.col is None:
        return
    config = _config()
    if not config.enabled or not config.run_after_sync:
        return
    # Let Anki finish refreshing its main window after the sync first.
    mw.progress.single_shot(500, lambda: _start_run(TRIGGER_SYNC, interactive=False), False)


def _reschedule_timer() -> None:
    global _timer
    hours = _config().auto_sync_hours if _config().enabled else 0.0
    if _timer is None:
        _timer = QTimer(mw)
        qconnect(_timer.timeout, _on_timer)
    _timer.stop()
    if hours > 0:
        _timer.start(int(hours * 3600 * 1000))


def _on_timer() -> None:
    if _closing or mw.col is None or _running:
        return
    # Never interrupt reviewing or an open dialog on the desktop.
    if getattr(mw, "state", None) not in {"deckBrowser", "overview"}:
        return
    if not _can_sync():
        return
    mw._sync_collection_and_media(mw._refresh_after_sync)  # our hook then runs


def _can_sync() -> bool:
    try:
        return bool(mw.pm.sync_auth()) and not mw.media_syncer.is_syncing()
    except Exception:
        return False


def _push_sync() -> None:
    global _skip_next_sync_hook
    if _closing or not _can_sync():
        return
    _skip_next_sync_hook = True
    try:
        mw._sync_collection_and_media(mw._refresh_after_sync)
    except Exception:
        _skip_next_sync_hook = False
        raise


# --- running ----------------------------------------------------------------------


def _start_run(trigger: str, *, interactive: bool) -> None:
    global _running
    if _running:
        if interactive:
            tooltip(f"{TITLE} is already running.", parent=mw)
        return
    if mw.col is None:
        return
    config = _config()
    manager = manager_for_mw(mw)
    if interactive:
        if not is_fsrs_enabled(mw.col):
            show_web_warning("RWKV requires FSRS to be enabled.", title=TITLE, parent=mw)
            return
        if not require_checkpoint_for_use(mw, manager=manager):
            return
    elif (
        not manager.has_checkpoint
        or manager.status() == "legacy"
        or not is_fsrs_enabled(mw.col)
    ):
        return

    store = store_for_mw(mw)

    def collect_op(col, progress, _previous):
        inputs = collect_mobile_due_inputs(col, extra_search=config.search)
        if not inputs.card_ids:
            raise NothingToPlan("No review cards matched Mobile Due's scope.")
        review_load = load_review_data_for_checkpoint(
            col, store, manager, progress, allow_incremental=True
        )
        target = resolve_target_timestamp(
            config.prediction_time, now=time.time(), day_cutoff=inputs.day_cutoff
        )
        return _Collected(inputs, review_load, target)

    def plan_op(_col, progress, previous: _Collected):
        inputs = previous.inputs
        review_data = previous.review_load.review_data
        readiness, runtime = open_checkpoint_runtime_from_load(
            manager,
            previous.review_load,
            progress,
            scope_cards=checkpoint_scope_cards_for_card_ids(inputs.card_ids, review_data),
        )
        try:
            predictor = filtered_deck_predictor_for_mode(
                RetrievabilityMode.IMMEDIATE, runtime, progress
            )
            return build_mobile_due_plan(
                card_ids=inputs.card_ids,
                schedule_infos=inputs.schedule_infos,
                review_data=readiness.review_data,
                retentions=inputs.retentions,
                max_interval_by_deck=inputs.max_interval_by_deck,
                predictor=predictor,
                settings=config.settings,
                today=inputs.today,
                day_cutoff=inputs.day_cutoff,
                target_timestamp=previous.target_timestamp,
                progress=progress,
            )
        finally:
            runtime.close()

    def done() -> None:
        global _running
        _running = False

    def success(plan: MobileDuePlan) -> None:
        done()
        _decide(plan, trigger=trigger, interactive=interactive, config=config)

    def failure(exc: Exception) -> None:
        done()
        if isinstance(exc, NothingToPlan):
            if interactive:
                show_quiet_info(str(exc), title=TITLE, parent=mw)
            return
        if not interactive:
            _safe_record(kind=RUN_KIND_ABORTED, plan=None, trigger=trigger, note=f"error: {exc}")
            tooltip(f"{TITLE} skipped: {exc}", parent=mw, period=6000)
            return
        handle_checkpoint_failure(
            exc,
            lambda: _start_run(trigger, interactive=interactive),
            parent=mw,
        )

    def cancelled() -> None:
        done()

    _running = True
    stages = [
        ProgressStage(collect_op, uses_collection=True),
        ProgressStage(plan_op, uses_collection=False),
    ]
    if interactive:
        run_with_progress_stages(
            parent=mw,
            title=TITLE,
            label="Planning Mobile Due",
            stages=stages,
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )
    else:
        run_background_stages(
            stages=stages,
            reporter=TimedProgressReporter(),
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )


def _decide(plan: MobileDuePlan, *, trigger: str, interactive: bool, config: MobileDueConfig) -> None:
    ledger = _ledger()
    violations = (
        guardrail_violations(
            plan,
            recent_due_now_counts=ledger.recent_due_now_counts(config.guardrails.history_runs),
            settings=config.guardrails,
        )
        if config.guardrails_enabled
        else []
    )
    summary = "\n".join(plan_summary_lines(plan))
    target = time.strftime("%a %H:%M", time.localtime(plan.target_timestamp))
    header = f"Predicted for {target}."

    if trigger == TRIGGER_MENU_PREVIEW:
        _safe_record(kind=RUN_KIND_SHADOW, plan=plan, trigger=trigger, violations=violations)
        warning = ("\n\nGuardrail warnings:\n- " + "\n- ".join(violations)) if violations else ""
        show_quiet_info(f"{header}\n\n{summary}{warning}", title=TITLE, parent=mw)
        return

    if interactive:
        if not plan.changes:
            _safe_record(kind=RUN_KIND_LIVE, plan=plan, trigger=trigger)
            show_quiet_info(f"{header}\n\n{summary}\n\nNothing to change.", title=TITLE, parent=mw)
            return
        warning = (
            "\n\nGuardrail warnings (you can still apply):\n- " + "\n- ".join(violations)
            if violations
            else ""
        )
        ask_web_confirmation(
            parent=mw,
            title=TITLE,
            message=(
                f"{header}\n\n{summary}{warning}\n\n"
                "Rewrite these due dates? Only the due field of review cards changes; "
                "this is one undoable step and every old value is logged."
            ),
            confirm_label="Apply Due Dates",
            destructive=True,
            on_result=lambda ok: _apply(plan, trigger=trigger, config=config) if ok else None,
        )
        return

    # Automatic run after a sync.
    if config.mode == MobileDueMode.SHADOW:
        _safe_record(kind=RUN_KIND_SHADOW, plan=plan, trigger=trigger, violations=violations)
        tooltip(
            f"{TITLE} (shadow): {plan.due_now_count} due now, "
            f"{plan.changed_count} dates would change.",
            parent=mw,
        )
        return
    if config.mode != MobileDueMode.LIVE:
        return
    if not ledger.has_live_run():
        _safe_record(
            kind=RUN_KIND_ABORTED,
            plan=plan,
            trigger=trigger,
            note="first live run must be applied manually",
        )
        return
    if violations:
        _safe_record(kind=RUN_KIND_ABORTED, plan=plan, trigger=trigger, violations=violations)
        tooltip(
            f"{TITLE} held back: " + " ".join(violations) + " Use Apply Now to override.",
            parent=mw,
            period=10000,
        )
        return
    if not plan.changes:
        _safe_record(kind=RUN_KIND_LIVE, plan=plan, trigger=trigger)
        return
    _apply(plan, trigger=trigger, config=config)


def _apply(plan: MobileDuePlan, *, trigger: str, config: MobileDueConfig) -> None:
    from aqt.operations import CollectionOp

    def success(result: ApplyResult) -> None:
        _safe_record(
            kind=RUN_KIND_LIVE,
            plan=plan,
            trigger=trigger,
            applied=result.applied,
            conflicts=len(result.conflicts),
        )
        skipped = f", {len(result.conflicts)} changed meanwhile and skipped" if result.conflicts else ""
        tooltip(
            f"{TITLE}: {plan.due_now_count} due now; {len(result.applied)} due dates updated{skipped}.",
            parent=mw,
            period=6000,
        )
        if config.sync_after_apply:
            mw.progress.single_shot(300, _push_sync, False)

    def failure(exc: Exception) -> None:
        _safe_record(kind=RUN_KIND_ABORTED, plan=plan, trigger=trigger, note=f"apply error: {exc}")
        show_web_warning(str(exc), title=TITLE, parent=mw)

    CollectionOp(mw, lambda col: apply_mobile_due_changes(col, list(plan.changes))).success(
        success
    ).failure(failure).run_in_background()


def _safe_record(**kwargs) -> None:
    try:
        _ledger().record_run(**kwargs)
    except Exception as exc:  # the log must never break studying
        print(f"{TITLE}: could not write ledger: {exc}")


# --- restore and log --------------------------------------------------------------


def _confirm_restore() -> None:
    if mw.col is None:
        return
    baseline = _ledger().baseline()
    if not baseline:
        show_quiet_info("Mobile Due has not changed any due dates yet.", title=TITLE, parent=mw)
        return
    ask_web_confirmation(
        parent=mw,
        title=TITLE,
        message=(
            f"Mobile Due has moved {len(baseline)} cards since it was enabled.\n\n"
            "Restore the original due date of every one of them that has not been "
            "reviewed since? Reviewed cards keep their new schedule. Mobile Due will "
            "be switched off so the next sync does not undo this."
        ),
        confirm_label="Restore Due Dates",
        destructive=True,
        on_result=lambda ok: _restore(baseline) if ok else None,
    )


def _restore(baseline) -> None:
    from aqt.operations import CollectionOp

    write_addon_config_for_mw(mw, with_mode(addon_config_for_mw(mw), MobileDueMode.OFF))
    _reschedule_timer()

    def success(result) -> None:
        ledger = _ledger()
        _safe_record(
            kind=RUN_KIND_RESTORE,
            plan=None,
            trigger="menu_restore",
            note=(
                f"restored {result.restored}, reviewed since {result.reviewed_since}, "
                f"no longer review {result.not_review}, missing {result.missing}"
            ),
        )
        ledger.archive_baseline()
        show_quiet_info(
            f"Restored {result.restored} cards.\n"
            f"Kept {result.reviewed_since} reviewed since, "
            f"{result.not_review} no longer in review, {result.missing} deleted.\n\n"
            "Mobile Due is now off. Sync to send this to your phone.",
            title=TITLE,
            parent=mw,
        )

    def op(col):
        result = restore_baseline_due_dates(col, baseline)
        if result.changes is None:
            from anki.collection import OpChanges

            return _RestoreWithChanges(result, OpChanges())
        return _RestoreWithChanges(result, result.changes)

    CollectionOp(mw, op).success(lambda wrapped: success(wrapped.result)).failure(
        lambda exc: show_web_warning(str(exc), title=TITLE, parent=mw)
    ).run_in_background()


@dataclass(frozen=True)
class _RestoreWithChanges:
    result: object
    changes: object


def _open_log_folder() -> None:
    ledger = _ledger()
    ledger.ensure()
    openFolder(str(ledger.root))
