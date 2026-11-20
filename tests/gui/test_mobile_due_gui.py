"""Offscreen Qt smoke tests for the Mobile Due menu, settings dialog and run decisions."""

from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
aqt = pytest.importorskip("aqt")
from aqt.qt import QApplication, QMenu  # noqa: E402

from rwkvp_addon.mobile_due import MobileDuePlan  # noqa: E402
from rwkvp_addon.mobile_due_ledger import MobileDueLedger  # noqa: E402


class FakeAddonManager:
    def __init__(self, config):
        self.config = dict(config)

    def addonFromModule(self, _name):
        return "rwkv4anki"

    def getConfig(self, _addon):
        return dict(self.config)

    def addonMeta(self, _addon):
        return {"config": dict(self.config)}

    def writeConfig(self, _addon, config):
        self.config = dict(config)


class FakeProgress:
    def __init__(self):
        self.shots = []

    def single_shot(self, ms, func, requires_collection=True):
        self.shots.append(func)


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from anki.collection import Collection
    from rwkvp_addon.addon_config import default_addon_config

    col = Collection(str(tmp_path / "c.anki2"))
    fake_mw = types.SimpleNamespace(
        col=col,
        addonManager=FakeAddonManager(default_addon_config()),
        progress=FakeProgress(),
        state="deckBrowser",
        pm=types.SimpleNamespace(sync_auth=lambda: object()),
        media_syncer=types.SimpleNamespace(is_syncing=lambda: False),
        synced=[],
    )
    fake_mw._sync_collection_and_media = lambda after: fake_mw.synced.append(after)
    fake_mw._refresh_after_sync = lambda: None
    monkeypatch.setattr(aqt, "mw", fake_mw, raising=False)
    sys.modules.pop("rwkvp_addon.gui.mobile_due", None)
    sys.modules.pop("rwkvp_addon.gui.mobile_due_settings", None)
    import rwkvp_addon.gui.mobile_due as module

    monkeypatch.setattr(module, "mw", fake_mw)
    ledger = MobileDueLedger(tmp_path / "ledger")
    monkeypatch.setattr(module, "_ledger", lambda: ledger)
    monkeypatch.setattr(module, "tooltip", lambda *a, **k: fake_mw.__dict__.setdefault("tips", []).append(a[0]))
    module._menu = None
    module._hooks_installed = True  # do not touch real gui_hooks lists
    module._closing = False
    module._skip_next_sync_hook = False
    module._running = False
    yield types.SimpleNamespace(app=app, mw=fake_mw, module=module, ledger=ledger)
    col.close()


def plan(due_now=3, changes=(), in_scope=10):
    return MobileDuePlan(
        created_at=0.0,
        target_timestamp=0.0,
        today=100,
        in_scope_count=in_scope,
        due_now_ids=tuple(range(due_now)),
        changes=tuple(changes),
        unchanged_count=0,
    )


def test_menu_installs_with_expected_actions(gui):
    parent = QMenu()
    gui.module.install_mobile_due_menu(parent)
    labels = [a.text() for a in gui.module._menu.actions() if a.text()]
    assert labels[:2] == ["Preview (no changes)...", "Apply Now..."]
    assert "Settings..." in labels
    assert "Restore Pre-RWKV Due Dates..." in labels
    gui.module._refresh_mode_actions()
    from rwkvp_addon.mobile_due_config import MobileDueMode

    assert gui.module._mode_actions[MobileDueMode.OFF].isChecked()


def test_settings_dialog_round_trip(gui):
    from rwkvp_addon.gui.mobile_due_settings import MobileDueSettingsDialog
    from rwkvp_addon.mobile_due_config import mobile_due_config

    saved = []
    dialog = MobileDueSettingsDialog(None, live_allowed=False, on_saved=lambda: saved.append(1))
    dialog.mode.setCurrentIndex(dialog.mode.findData("shadow"))
    dialog.search.setText("deck:Default")
    dialog.prediction.setCurrentIndex(dialog.prediction.findData("specific"))
    from aqt.qt import QTime

    dialog.clock.setTime(QTime(21, 30))
    dialog.auto_hours.setValue(3)
    dialog.buckets.setValue(5)
    dialog._save()
    assert saved == [1]
    config = mobile_due_config(gui.mw.addonManager.config)
    assert config.mode.value == "shadow"
    assert config.search == "deck:Default"
    assert config.prediction_time == "21:30"
    assert config.auto_sync_hours == 3
    assert config.settings.priority_buckets == 5
    # Unrelated add-on keys survive.
    assert gui.mw.addonManager.config["model_generation"] == "v1"


def test_settings_dialog_blocks_live_before_first_manual_apply_and_bad_search(gui, monkeypatch):
    from rwkvp_addon.gui import mobile_due_settings as settings_module

    warnings = []
    monkeypatch.setattr(settings_module, "showWarning", lambda msg, **k: warnings.append(msg))
    dialog = settings_module.MobileDueSettingsDialog(None, live_allowed=False, on_saved=lambda: None)
    dialog.mode.setCurrentIndex(dialog.mode.findData("live"))
    dialog._save()
    assert warnings and "Apply Now" in warnings[-1]
    dialog.mode.setCurrentIndex(dialog.mode.findData("off"))
    dialog.search.setText('deck:"unclosed')
    dialog._save()
    assert "not valid" in warnings[-1]


def _set_mode(gui, mode):
    gui.mw.addonManager.config["mobile_due_mode"] = mode


def _bump_sync_stamp(gui):
    gui.mw.col.db.execute("update col set ls = ls + 1000")


def _sync(gui, *, exchanged=True):
    gui.module._on_sync_will_start()
    if exchanged:
        _bump_sync_stamp(gui)
    gui.module._on_sync_did_finish()


def test_sync_hook_runs_only_after_a_sync_that_exchanged_data(gui):
    _set_mode(gui, "off")
    _sync(gui)
    assert gui.mw.progress.shots == []
    _set_mode(gui, "shadow")
    _sync(gui, exchanged=False)  # failed, cancelled or no-change sync
    assert gui.mw.progress.shots == []
    _sync(gui)
    assert len(gui.mw.progress.shots) == 1
    gui.module._skip_next_sync_hook = True  # our own push
    _sync(gui)
    assert len(gui.mw.progress.shots) == 1
    assert gui.module._skip_next_sync_hook is False
    gui.module._closing = True
    _sync(gui)
    assert len(gui.mw.progress.shots) == 1


def _decide(gui, p, *, trigger="after_sync", interactive=False):
    gui.module._running = True
    gui.module._decide(p, trigger=trigger, interactive=interactive, config=gui.module._config())


def test_auto_shadow_logs_without_applying(gui, monkeypatch):
    applied = []
    monkeypatch.setattr(gui.module, "_apply", lambda *a, **k: applied.append(a))
    _set_mode(gui, "shadow")
    _decide(gui, plan())
    assert applied == []
    assert gui.ledger.runs()[-1]["kind"] == "shadow"
    assert gui.module._running is False


def test_auto_live_requires_manual_apply_respects_guardrails_and_locks_due_cards(gui, monkeypatch):
    from rwkvp_addon.mobile_due import MobileDueChange

    applied = []
    monkeypatch.setattr(gui.module, "_apply", lambda p, **k: applied.append((p, k)))
    _set_mode(gui, "live")
    pull_in = MobileDueChange(1, 105, 100, 0, 10, True, 0.8, 0.9, 0)
    push_out = MobileDueChange(2, 99, 120, 0, 10, False, 0.95, 0.9, None)
    _decide(gui, plan(changes=[pull_in]))
    assert applied == []
    assert gui.ledger.runs()[-1]["kind"] == "aborted"

    gui.ledger.record_run(kind="live", plan=plan(), trigger="menu_apply", applied_count=3)
    for count in (10, 12, 11):
        gui.ledger.record_run(kind="live", plan=plan(due_now=count), trigger="after_sync", applied_count=1)
    _decide(gui, plan(due_now=400, changes=[pull_in], in_scope=2000))
    assert applied == []
    assert gui.ledger.runs()[-1]["kind"] == "aborted"
    assert gui.ledger.runs()[-1]["violations"]

    _decide(gui, plan(due_now=12, changes=[pull_in, push_out], in_scope=2000))
    assert len(applied) == 1
    sent, kwargs = applied[0]
    assert [c.card_id for c in sent.changes] == [1]  # the already-due card is never touched
    assert sent.locked_due_count == 1
    assert kwargs["interactive"] is False


def test_auto_run_abandons_when_mode_changed_while_planning(gui, monkeypatch):
    from rwkvp_addon.mobile_due import MobileDueChange

    applied = []
    monkeypatch.setattr(gui.module, "_apply", lambda p, **k: applied.append(p))
    _set_mode(gui, "live")
    gui.ledger.record_run(kind="live", plan=plan(), trigger="menu_apply", applied_count=3)
    started = gui.module._config()
    _set_mode(gui, "off")  # e.g. Restore switched it off mid-run
    gui.module._running = True
    gui.module._decide(
        plan(changes=[MobileDueChange(1, 105, 100, 0, 10, True, 0.8, 0.9, 0)]),
        trigger="after_sync",
        interactive=False,
        config=started,
    )
    assert applied == []
    assert gui.module._running is False


def test_interactive_apply_lets_user_choose_to_move_due_cards(gui, monkeypatch):
    from rwkvp_addon.mobile_due import MobileDueChange

    applied, prompts = [], []

    def fake_confirm(**kwargs):
        prompts.append(kwargs)

    monkeypatch.setattr(gui.module, "ask_web_confirmation", fake_confirm)
    monkeypatch.setattr(gui.module, "_apply", lambda p, **k: applied.append(p))
    changes = [MobileDueChange(1, 105, 100, 0, 10, True, 0.8, 0.9, 0), MobileDueChange(2, 99, 120, 0, 10, False, 0.95, 0.9, None)]
    _decide(gui, plan(changes=changes), trigger="menu_apply", interactive=True)
    prompt = prompts[-1]
    assert prompt["checkbox_checked"] is True  # first run: full replacement offered by default
    prompt["on_checkbox_changed"](False)
    prompt["on_result"](True)
    assert [c.card_id for c in applied[-1].changes] == [1]
    _decide(gui, plan(changes=changes), trigger="menu_apply", interactive=True)
    prompts[-1]["on_result"](True)
    assert [c.card_id for c in applied[-1].changes] == [1, 2]
    _decide(gui, plan(changes=changes), trigger="menu_apply", interactive=True)
    prompts[-1]["on_result"](False)
    assert gui.module._running is False


def test_settings_and_restore_refused_while_running(gui, monkeypatch):
    shown = []
    monkeypatch.setattr(gui.module, "ask_web_confirmation", lambda **k: shown.append(k))
    gui.module._running = True
    gui.module._confirm_restore()
    gui.module._show_settings()
    assert shown == []
    assert any("running" in tip for tip in gui.mw.tips)


def test_push_sync_flags_own_sync(gui):
    gui.module._push_sync()
    assert len(gui.mw.synced) == 1
    assert gui.module._skip_next_sync_hook is True


def test_timer_does_not_sync_while_reviewing(gui):
    _set_mode(gui, "live")
    gui.mw.state = "review"
    gui.module._on_timer()
    assert gui.mw.synced == []
    gui.mw.state = "deckBrowser"
    gui.module._on_timer()
    assert len(gui.mw.synced) == 1
