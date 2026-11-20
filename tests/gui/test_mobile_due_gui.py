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


def test_sync_hook_schedules_run_only_when_enabled_and_skips_own_sync(gui):
    _set_mode(gui, "off")
    gui.module._on_sync_did_finish()
    assert gui.mw.progress.shots == []
    _set_mode(gui, "shadow")
    gui.module._on_sync_did_finish()
    assert len(gui.mw.progress.shots) == 1
    gui.module._skip_next_sync_hook = True
    gui.module._on_sync_did_finish()
    assert len(gui.mw.progress.shots) == 1
    assert gui.module._skip_next_sync_hook is False
    gui.module._closing = True
    gui.module._on_sync_did_finish()
    assert len(gui.mw.progress.shots) == 1


def test_auto_shadow_logs_without_applying(gui, monkeypatch):
    applied = []
    monkeypatch.setattr(gui.module, "_apply", lambda *a, **k: applied.append(a))
    _set_mode(gui, "shadow")
    gui.module._decide(plan(), trigger="after_sync", interactive=False, config=gui.module._config())
    assert applied == []
    assert gui.ledger.runs()[-1]["kind"] == "shadow"


def test_auto_live_requires_a_prior_manual_apply_and_respects_guardrails(gui, monkeypatch):
    from rwkvp_addon.mobile_due import MobileDueChange

    applied = []
    monkeypatch.setattr(gui.module, "_apply", lambda p, **k: applied.append(p))
    _set_mode(gui, "live")
    change = MobileDueChange(1, 105, 100, 0, 10, True, 0.8, 0.9, 0)
    gui.module._decide(plan(changes=[change]), trigger="after_sync", interactive=False, config=gui.module._config())
    assert applied == []
    assert gui.ledger.runs()[-1]["kind"] == "aborted"

    gui.ledger.record_run(kind="live", plan=plan(), trigger="menu_apply")
    for count in (10, 12, 11):
        gui.ledger.record_run(kind="live", plan=plan(due_now=count), trigger="after_sync")
    spike = plan(due_now=400, changes=[change], in_scope=2000)
    gui.module._decide(spike, trigger="after_sync", interactive=False, config=gui.module._config())
    assert applied == []
    assert gui.ledger.runs()[-1]["kind"] == "aborted"
    assert gui.ledger.runs()[-1]["violations"]

    normal = plan(due_now=12, changes=[change], in_scope=2000)
    gui.module._decide(normal, trigger="after_sync", interactive=False, config=gui.module._config())
    assert applied == [normal]


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
