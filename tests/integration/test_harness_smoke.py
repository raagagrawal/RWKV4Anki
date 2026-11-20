"""Smoke test for the headless RWKV harness (no aqt)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rwkv_harness  # noqa: E402

# Other test modules (the Qt GUI tests) may import aqt in the same session.
_AQT_PRELOADED = "aqt" in sys.modules

_REASON = rwkv_harness.unavailable_reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")


@pytest.fixture
def small_collection(tmp_path):
    col = rwkv_harness.make_collection(tmp_path / "col", n_notes=24, days=40, seed=7)
    try:
        yield col
    finally:
        col.close()


def test_collection_has_realistic_state_mix(small_collection):
    summary = small_collection.harness_summary
    after = summary["after_reopen"]
    # Tiny decks may have nothing due exactly today; everything else must exist.
    assert set(summary["missing_states"]) <= {"review_due_today", "review_due_past"}
    assert after["review_due_past"] + after["review_due_today"] > 0
    assert after["fsrs"] is True
    assert {"Parent", "Parent::Child"} <= set(after["decks"])
    assert after["today"] == 40
    assert after["revlog"] > after["cards"]
    assert after["with_memory_state"] > 0
    # Revlog ids (ms) are spread over the simulated period, not one instant.
    first, last = small_collection.db.first("select min(id), max(id) from revlog")
    assert (last - first) / 86_400_000 > 30


def test_predictions_finite_in_unit_interval(small_collection, tmp_path):
    col = small_collection
    card_ids = list(col.find_cards("is:review"))
    assert card_ids
    with rwkv_harness.open_rwkv_predictor(col, card_ids, tmp_path / "store") as handle:
        review_data = handle.review_data
        assert review_data.next_day_at == col.sched.day_cutoff
        assert len(review_data.rows) > 0
        target = review_data.next_day_at - 1
        rows = handle.rows_for(card_ids, target)
        predictions = list(handle.predictor(rows))
        assert len(predictions) == len(card_ids)
        for value in predictions:
            assert math.isfinite(value)
            assert 0.0 < value < 1.0
        # Same card at two timestamps in one call == two separate calls.
        pair = [rows[0], handle.rows_for(card_ids[:1], target + 16 * 86400)[0]]
        together = list(handle.predictor(pair))
        separate = [list(handle.predictor([row]))[0] for row in pair]
        assert together == pytest.approx(separate, abs=1e-6)
    assert handle.runtime.closed
    # Headless: nothing pulled in the GUI layer, and the Rust backend was used
    # from the harness's _vendor layout.
    if not _AQT_PRELOADED:
        assert "aqt" not in sys.modules
    import rwkv_srs
    from rwkvp_addon.vendor_bootstrap import rwkv_backend_name

    assert Path(rwkv_srs.__file__).parent == rwkv_harness.VENDOR_DIR / "rwkv_srs"
    assert "rust" in rwkv_backend_name().lower()
