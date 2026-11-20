"""A small native settings dialog for Mobile Due.

The main RWKV Settings window only renders bool/int/choice options, while
Mobile Due needs a free-text Anki search and a clock time, so it gets its own
dialog. Values are merged onto the latest persisted add-on config.
"""

from __future__ import annotations

from collections.abc import Callable

from aqt import mw
from aqt.qt import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTime,
    QTimeEdit,
    QVBoxLayout,
)
from aqt.utils import qconnect, showWarning

from ..addon_config import addon_config_for_mw, write_addon_config_for_mw
from ..mobile_due import PredictionTime
from ..mobile_due_collection import mobile_due_search
from ..mobile_due_config import (
    AUTO_SYNC_HOURS_KEY,
    GUARDRAILS_KEY,
    HORIZON_KEY,
    LIMIT_KEY,
    MODE_KEY,
    PREDICTION_TIME_KEY,
    PRIORITY_BUCKETS_KEY,
    RUN_AFTER_SYNC_KEY,
    SEARCH_KEY,
    SYNC_AFTER_APPLY_KEY,
    MobileDueMode,
    mobile_due_config,
)

_SPECIFIC_TIME = "specific"


class MobileDueSettingsDialog(QDialog):
    def __init__(self, parent, *, live_allowed: bool, on_saved: Callable[[], None]) -> None:
        super().__init__(parent)
        self.setWindowTitle("RWKV Mobile Due Settings")
        self._live_allowed = live_allowed
        self._on_saved = on_saved
        config = mobile_due_config(addon_config_for_mw(mw))

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Mobile Due rewrites the due date of review cards so that AnkiMobile and "
            "AnkiDroid show RWKV's picks as their Due queue after a sync."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)

        self.mode = QComboBox()
        for mode, label in (
            (MobileDueMode.OFF, "Off"),
            (MobileDueMode.SHADOW, "Shadow: plan and log after each sync, change nothing"),
            (MobileDueMode.LIVE, "Live: rewrite due dates after each sync"),
        ):
            self.mode.addItem(label, mode.value)
        self.mode.setCurrentIndex(max(0, self.mode.findData(config.mode.value)))
        form.addRow("Mode", self.mode)

        self.search = QLineEdit(config.search)
        self.search.setPlaceholderText("All review cards. Example: deck:Medicine -tag:leech")
        self.search.setToolTip(
            "Optional Anki search that narrows which review cards Mobile Due manages. "
            "Learning, new, suspended, buried and filtered-deck cards are never touched."
        )
        form.addRow("Cards in scope", self.search)

        self.prediction = QComboBox()
        self.prediction.addItem("End of the Anki day (recommended)", PredictionTime.END_OF_DAY.value)
        self.prediction.addItem("Time of the run", PredictionTime.NOW.value)
        self.prediction.addItem("A specific time today", _SPECIFIC_TIME)
        self.clock = QTimeEdit()
        self.clock.setDisplayFormat("HH:mm")
        if config.prediction_time in {PredictionTime.END_OF_DAY.value, PredictionTime.NOW.value}:
            self.prediction.setCurrentIndex(self.prediction.findData(config.prediction_time))
            self.clock.setTime(QTime(20, 0))
        else:
            self.prediction.setCurrentIndex(self.prediction.findData(_SPECIFIC_TIME))
            hours, minutes = (int(part) for part in config.prediction_time.split(":"))
            self.clock.setTime(QTime(hours, minutes))
        self.prediction.setToolTip(
            "RWKV predicts recall for this moment. A later time makes cards that will "
            "drop below desired retention before you study due already."
        )
        qconnect(self.prediction.currentIndexChanged, lambda _i: self._sync_clock())
        form.addRow("Predict recall for", self.prediction)
        form.addRow("Time", self.clock)
        self._sync_clock()

        self.run_after_sync = QCheckBox("Run after every sync")
        self.run_after_sync.setChecked(config.run_after_sync)
        form.addRow("", self.run_after_sync)
        self.sync_after_apply = QCheckBox("Sync again after applying, to push to the phone")
        self.sync_after_apply.setChecked(config.sync_after_apply)
        form.addRow("", self.sync_after_apply)

        self.auto_hours = QDoubleSpinBox()
        self.auto_hours.setRange(0.0, 24.0)
        self.auto_hours.setSingleStep(0.5)
        self.auto_hours.setDecimals(1)
        self.auto_hours.setSuffix(" h")
        self.auto_hours.setValue(config.auto_sync_hours)
        self.auto_hours.setToolTip(
            "While Anki is open on the deck list, sync (and therefore refresh Mobile Due) "
            "this often. 0 turns it off. Never interrupts a review session."
        )
        form.addRow("Auto-sync every", self.auto_hours)

        self.limit = QSpinBox()
        self.limit.setRange(0, 100000)
        self.limit.setValue(config.settings.limit)
        self.limit.setSpecialValueText("No limit")
        form.addRow("Max cards due now", self.limit)

        self.buckets = QSpinBox()
        self.buckets.setRange(1, 60)
        self.buckets.setValue(config.settings.priority_buckets)
        self.buckets.setSuffix(" days")
        self.buckets.setToolTip(
            "Spreads due-now cards over this many days of apparent overdueness, lowest "
            "predicted recall most overdue, so a phone sorting by due date shows the "
            "most-forgotten cards first. 1 makes every due-now card due today."
        )
        form.addRow("Priority spread", self.buckets)

        self.horizon = QSpinBox()
        self.horizon.setRange(1, 36500)
        self.horizon.setValue(config.settings.horizon_days)
        self.horizon.setSuffix(" days")
        self.horizon.setToolTip(
            "Furthest ahead a not-yet-due card is projected. The deck's maximum interval "
            "also applies."
        )
        form.addRow("Projection horizon", self.horizon)

        self.guardrails = QCheckBox("Hold back automatic runs that look abnormal")
        self.guardrails.setChecked(config.guardrails_enabled)
        form.addRow("", self.guardrails)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        qconnect(buttons.accepted, self._save)
        qconnect(buttons.rejected, self.reject)
        layout.addWidget(buttons)

    def _sync_clock(self) -> None:
        self.clock.setEnabled(self.prediction.currentData() == _SPECIFIC_TIME)

    def _save(self) -> None:
        mode = str(self.mode.currentData())
        if mode == MobileDueMode.LIVE.value and not self._live_allowed:
            showWarning(
                "Run RWKV -> Mobile Due -> Apply Now once and review its summary "
                "before turning on Live mode.",
                parent=self,
            )
            return
        search = self.search.text().strip()
        if search and mw.col is not None:
            try:
                mw.col.find_cards(mobile_due_search(search), order=False)
            except Exception as exc:
                showWarning(f"That search is not valid:\n\n{exc}", parent=self)
                return
        prediction = str(self.prediction.currentData())
        if prediction == _SPECIFIC_TIME:
            prediction = self.clock.time().toString("HH:mm")
        config = dict(addon_config_for_mw(mw))
        config.update(
            {
                MODE_KEY: mode,
                SEARCH_KEY: search,
                PREDICTION_TIME_KEY: prediction,
                RUN_AFTER_SYNC_KEY: self.run_after_sync.isChecked(),
                SYNC_AFTER_APPLY_KEY: self.sync_after_apply.isChecked(),
                AUTO_SYNC_HOURS_KEY: float(self.auto_hours.value()),
                LIMIT_KEY: int(self.limit.value()),
                PRIORITY_BUCKETS_KEY: int(self.buckets.value()),
                HORIZON_KEY: int(self.horizon.value()),
                GUARDRAILS_KEY: self.guardrails.isChecked(),
            }
        )
        write_addon_config_for_mw(mw, config)
        self._on_saved()
        self.accept()
