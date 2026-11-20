"""Headless integration harness for RWKV4Anki (no aqt / Qt).

Two public entry points:

* :func:`make_collection` builds a fresh Anki collection with FSRS enabled, a
  ``Parent`` / ``Parent::Child`` deck tree (two presets with different desired
  retention) and a realistic review history. The history is produced by really
  answering cards through Anki's v3 scheduler while a *fake clock* walks
  forward day by day. The clock is libfaketime (``pip install libfaketime``,
  which ships a prebuilt ``libfaketime.so.1``) preloaded into a child Python
  process, so Anki's Rust backend, revlog ids, card/note ids, FSRS memory state
  and ``last_review_time`` are all mutually consistent. The final simulated day
  is the real "today", so the collection is valid when reopened under the real
  clock.

* :func:`open_rwkv_predictor` mirrors the filtered-deck dialog's
  ``_run_build`` (collect_op -> plan_op) for ``RetrievabilityMode.IMMEDIATE``
  and yields the review data, the predictor callable and the checkpoint
  manager. The runtime lease is closed (and the manager unloaded) on exit.

:func:`prepare_environment` must run before any ``rwkvp_addon`` import because
``rwkvp_addon.constants.VENDOR_ROOT`` is computed at import time; both entry
points call it for you. It sets::

    RWKV4ANKI_DEV_VENDOR_ROOT=<this dir>/_vendor
    RWKV_SRS_BACKEND=rust

and prepends ``<repo>/addon_src`` to ``sys.path``. Model files are looked up in
``$RWKV4ANKI_TEST_MODELS_DIR`` (default ``/home/claude/dl``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
ADDON_SRC = REPO_ROOT / "addon_src"
VENDOR_DIR = HARNESS_DIR / "_vendor"
MODELS_DIR_ENV_VAR = "RWKV4ANKI_TEST_MODELS_DIR"
DEFAULT_MODELS_DIR = Path("/home/claude/dl")
VENDOR_ENV_VAR = "RWKV4ANKI_DEV_VENDOR_ROOT"  # mirrors rwkvp_addon.constants
BACKEND_ENV_VAR = "RWKV_SRS_BACKEND"  # mirrors rwkvp_addon.rwkv_backend

MODEL_FILES = {
    "v1": ("RWKV_trained_on_101_4999.safetensors", "RWKV_trained_on_5000_10000.safetensors"),
    "v5": ("v5o-a.safetensors",),
}

_SIM_FLAG_ENV_VAR = "RWKV4ANKI_HARNESS_SIMULATING"


# --------------------------------------------------------------------------
# Environment / availability
# --------------------------------------------------------------------------


def models_dir() -> Path:
    return Path(os.environ.get(MODELS_DIR_ENV_VAR) or DEFAULT_MODELS_DIR)


def _rwkv_srs_package_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("rwkv_srs")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve().parent
    # A previously prepared _vendor copy may already be on sys.path.
    return path


def libfaketime_library() -> Path | None:
    """Locate libfaketime.so.1 from the ``libfaketime`` pip package, or the system."""

    override = os.environ.get("RWKV4ANKI_LIBFAKETIME")
    if override and Path(override).is_file():
        return Path(override)
    try:
        spec = importlib.util.find_spec("libfaketime")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin is not None:
        candidate = (
            Path(spec.origin).parent / "vendor" / "libfaketime" / "src" / "libfaketime.so.1"
        )
        if candidate.is_file():
            return candidate
    for system in (
        "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1",
        "/usr/lib/faketime/libfaketime.so.1",
    ):
        if Path(system).is_file():
            return Path(system)
    return None


def unavailable_reason(model_generation: str = "v1") -> str | None:
    """Return why the harness cannot run here, or ``None`` when it can."""

    if sys.platform != "linux":
        return "harness fake clock (libfaketime LD_PRELOAD) is only wired up for Linux"
    try:
        import anki  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"anki is not importable: {exc}"
    if _rwkv_srs_package_dir() is None:
        return "rwkv_srs is not installed"
    folder = "v5" if model_generation in {"v5", "v5o"} else "v1"
    missing = [name for name in MODEL_FILES[folder] if not (models_dir() / name).is_file()]
    if missing:
        return f"model files missing from {models_dir()}: {', '.join(missing)}"
    if libfaketime_library() is None:
        return "libfaketime is unavailable (pip install libfaketime)"
    return None


def prepare_environment() -> Path:
    """Build ``_vendor`` (rwkv_srs package + pretrained models) and set env vars.

    Layout mirrors the packaged add-on (scripts/package_addon.py)::

        _vendor/rwkv_srs/{__init__.py,_native.abi3.so,...}   -> symlinks to the venv
        _vendor/rwkv_srs/pretrained/v1/<model>.safetensors   -> symlinks to models dir
        _vendor/rwkv_srs/pretrained/v5/v5o-a.safetensors
    """

    already = sys.modules.get("rwkvp_addon.constants")
    if already is not None and Path(already.VENDOR_ROOT) != VENDOR_DIR.resolve():
        raise RuntimeError(
            "rwkvp_addon.constants was imported before prepare_environment(); "
            f"VENDOR_ROOT={already.VENDOR_ROOT}"
        )
    package_src = _rwkv_srs_package_dir()
    if package_src is None:
        raise RuntimeError("rwkv_srs is not installed in this interpreter")
    vendor_pkg = VENDOR_DIR / "rwkv_srs"
    if package_src == vendor_pkg.resolve():
        package_src = None  # already pointing at our own vendor dir
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = VENDOR_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Generated by tests/integration/rwkv_harness.py\n*\n")
    vendor_pkg.mkdir(exist_ok=True)
    if package_src is not None:
        for entry in package_src.iterdir():
            if entry.name in {"__pycache__", "pretrained"}:
                continue
            _ensure_symlink(vendor_pkg / entry.name, entry)
    for folder, names in MODEL_FILES.items():
        target_dir = vendor_pkg / "pretrained" / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = models_dir() / name
            if source.is_file():
                _ensure_symlink(target_dir / name, source)
    os.environ[VENDOR_ENV_VAR] = str(VENDOR_DIR)
    os.environ[BACKEND_ENV_VAR] = "rust"
    if str(ADDON_SRC) not in sys.path:
        sys.path.insert(0, str(ADDON_SRC))
    return VENDOR_DIR


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        return
    link.symlink_to(target)


# --------------------------------------------------------------------------
# Collection construction
# --------------------------------------------------------------------------


def make_collection(
    tmp_dir: str | os.PathLike[str],
    *,
    n_notes: int = 60,
    days: int = 120,
    seed: int = 0,
    end_epoch: float | None = None,
    name: str = "collection.anki2",
):
    """Create a simulated FSRS collection and return it open (real clock).

    ``end_epoch`` is the moment the simulation finishes (must not be in the
    future); the last study session starts 45 minutes before it. The default,
    :func:`default_end_epoch`, is 06:00 local today (or "now" before then), so
    the same arguments rebuild a byte-identical history all day long: Anki's
    fuzz is seeded from card ids, which are fake-clock timestamps. The returned
    collection has ``harness_summary`` (dict) attached describing the mix.
    """

    from anki.collection import Collection

    if n_notes < 12:
        raise ValueError("n_notes must be at least 12 to produce every card state")
    if days < 10:
        raise ValueError("days must be at least 10")
    lib = libfaketime_library()
    if lib is None:
        raise RuntimeError("libfaketime is unavailable (pip install libfaketime)")
    path = Path(tmp_dir).resolve() / name
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    end = float(default_end_epoch() if end_epoch is None else end_epoch)
    if end > time.time() + 1:
        raise ValueError("end_epoch must not be in the future")
    env = dict(os.environ)
    env.update(
        {
            "LD_PRELOAD": f"{lib}:{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else str(lib),
            "FAKETIME": "+0",
            "FAKETIME_NO_CACHE": "1",
            "DONT_FAKE_MONOTONIC": "1",
            _SIM_FLAG_ENV_VAR: "1",
        }
    )
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--simulate",
        str(path),
        "--n-notes",
        str(n_notes),
        "--days",
        str(days),
        "--seed",
        str(seed),
        "--end-epoch",
        repr(end),
    ]
    completed = subprocess.run(
        command, env=env, cwd=str(path.parent), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "collection simulation failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    if time.time() < summary["last_fake_time"]:
        time.sleep(max(0.0, summary["last_fake_time"] - time.time()) + 0.01)
    col = Collection(str(path))
    summary["after_reopen"] = collection_summary(col)
    col.harness_summary = summary  # type: ignore[attr-defined]
    return col


def default_end_epoch(now: float | None = None) -> float:
    """06:00 local time today when that has passed (and is after the default
    04:00 rollover), else ``now``. Deterministic within a day."""

    now = time.time() if now is None else float(now)
    local = time.localtime(now)
    six_am = time.mktime(
        (local.tm_year, local.tm_mon, local.tm_mday, 6, 0, 0, 0, 0, -1)
    )
    return six_am if six_am <= now else now


def collection_summary(col) -> dict[str, Any]:
    today = int(col.sched.today)
    cutoff = int(col.sched.day_cutoff)
    q = col.db.scalar
    return {
        "today": today,
        "day_cutoff": cutoff,
        "crt": int(col.crt),
        "fsrs": bool(col.get_config("fsrs", False)),
        "notes": q("select count() from notes"),
        "cards": q("select count() from cards"),
        "revlog": q("select count() from revlog"),
        "new": q("select count() from cards where type=0 and queue=0"),
        "learning": q("select count() from cards where type=1 and queue in (1,3)"),
        "relearning": q("select count() from cards where type=3 and queue in (1,3)"),
        "review_due_past": q("select count() from cards where queue=2 and due<?", today),
        "review_due_today": q("select count() from cards where queue=2 and due=?", today),
        "review_due_future": q("select count() from cards where queue=2 and due>?", today),
        "suspended": q("select count() from cards where queue=-1"),
        "buried": q("select count() from cards where queue in (-2,-3)"),
        "with_memory_state": q(
            "select count() from cards where type!=0 and json_extract(data,'$.s') is not null"
        ),
        "decks": sorted(d.name for d in col.decks.all_names_and_ids()),
    }


class _FakeClock:
    """Drive libfaketime's absolute (frozen) clock via the FAKETIME env var."""

    def __init__(self, start: float) -> None:
        self.t = 0.0
        self._correction = 0.0
        self.set(start)

    def set(self, epoch: float) -> None:
        self.t = float(epoch)
        for _ in range(3):
            os.environ["FAKETIME"] = self._format(self.t + self._correction)
            error = time.time() - self.t
            if abs(error) < 0.002:
                return
            # DST transitions make local-time strings ambiguous; compensate.
            self._correction -= error
        raise RuntimeError(f"fake clock mismatch: wanted {self.t}, got {time.time()}")

    def advance(self, seconds: float) -> None:
        self.set(self.t + seconds)

    @staticmethod
    def _format(epoch: float) -> str:
        whole = math.floor(epoch)
        micros = int(round((epoch - whole) * 1_000_000))
        if micros >= 1_000_000:
            whole, micros = whole + 1, 0
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(whole)) + f".{micros:06d}"


@dataclass
class _Latent:
    stability: float
    last_answer: float | None = None
    reviews: int = 0


@dataclass
class _Simulation:
    col: Any
    clock: _FakeClock
    rng: random.Random
    latent: dict[int, _Latent] = field(default_factory=dict)
    answers: int = 0

    def latent_for(self, card_id: int) -> _Latent:
        state = self.latent.get(card_id)
        if state is None:
            state = _Latent(stability=self.rng.lognormvariate(math.log(2.0), 0.7))
            self.latent[card_id] = state
        return state

    def choose_rating(self, card, forced: int | None = None) -> int:
        from anki.scheduler_pb2 import CardAnswer

        rng = self.rng
        state = self.latent_for(card.id)
        if forced is not None:
            rating = forced
        elif card.type == 0:
            rating = _weighted(rng, ((CardAnswer.AGAIN, 0.25), (CardAnswer.HARD, 0.05),
                                     (CardAnswer.GOOD, 0.6), (CardAnswer.EASY, 0.1)))
        elif card.type in (1, 3):
            rating = _weighted(rng, ((CardAnswer.AGAIN, 0.12), (CardAnswer.HARD, 0.05),
                                     (CardAnswer.GOOD, 0.83)))
        else:
            elapsed = 0.0 if state.last_answer is None else (self.clock.t - state.last_answer) / 86400
            recall = (1.0 + elapsed / (9.0 * state.stability)) ** -1.0
            if rng.random() < recall:
                rating = _weighted(rng, ((CardAnswer.HARD, 0.1), (CardAnswer.GOOD, 0.8),
                                         (CardAnswer.EASY, 0.1)))
            else:
                rating = CardAnswer.AGAIN
        if card.type == 2:
            if rating == CardAnswer.AGAIN:
                state.stability = max(0.2, state.stability * rng.uniform(0.15, 0.35))
            else:
                growth = rng.uniform(1.6, 3.2)
                growth *= {CardAnswer.HARD: 0.75, CardAnswer.EASY: 1.3}.get(rating, 1.0)
                state.stability *= growth
        elif card.type == 0 and rating == CardAnswer.AGAIN:
            state.stability *= 0.8
        return rating

    def answer(self, card, states, *, forced: int | None = None) -> int:
        """Answer one card through the real v3 scheduler at the fake time."""

        col = self.col
        card.start_timer()
        self.clock.advance(min(60.0, self.rng.lognormvariate(math.log(6.0), 0.6)))
        rating = self.choose_rating(card, forced)
        answer = col.sched.build_answer(card=card, states=states, rating=rating)
        col.sched.answer_card(answer)
        state = self.latent_for(card.id)
        state.last_answer = self.clock.t
        state.reviews += 1
        self.answers += 1
        self.clock.advance(self.rng.uniform(0.5, 3.0))
        return rating

    def answer_card_id(self, card_id: int, *, forced: int | None = None) -> int:
        card = self.col.get_card(card_id)
        states = self.col._backend.get_scheduling_states(card_id)
        return self.answer(card, states, forced=forced)

    def study_queue(self, max_answers: int = 5000) -> int:
        from anki.cards import Card

        done = 0
        while done < max_answers:
            queued = self.col.sched.get_queued_cards(fetch_limit=1)
            if not queued.cards:
                break
            top = queued.cards[0]
            self.answer(Card(self.col, backend_card=top.card), top.states)
            done += 1
        return done


def _weighted(rng: random.Random, choices: Sequence[tuple[int, float]]) -> int:
    roll = rng.random() * sum(weight for _, weight in choices)
    for value, weight in choices:
        roll -= weight
        if roll <= 0:
            return value
    return choices[-1][0]


def _simulate(path: Path, *, n_notes: int, days: int, seed: int, end_epoch: float) -> dict:
    if os.environ.get(_SIM_FLAG_ENV_VAR) != "1" or "FAKETIME" not in os.environ:
        raise RuntimeError("_simulate must run in the libfaketime child process")
    from anki.collection import Collection

    rng = random.Random(seed)
    session_lead = 45 * 60  # every session starts 45 minutes "before now" o'clock

    def session_start(day: int) -> float:
        return end_epoch - (days - day) * 86400 - session_lead

    clock = _FakeClock(session_start(0) - 600)
    col = Collection(str(path))
    sim = _Simulation(col=col, clock=clock, rng=rng)

    col.set_config("fsrs", True)
    parent_conf = col.decks.get_config(1)
    parent_conf["name"] = "Parent preset"
    parent_conf["desiredRetention"] = 0.90
    parent_conf["new"]["perDay"] = 9999
    parent_conf["rev"]["perDay"] = 9999
    col.decks.update_config(parent_conf)
    child_conf = col.decks.add_config_returning_id("Child preset", clone_from=parent_conf)
    child_conf_dict = col.decks.get_config(child_conf)
    child_conf_dict["desiredRetention"] = 0.85
    col.decks.update_config(child_conf_dict)
    parent_id = col.decks.id("Parent")
    child_id = col.decks.id("Parent::Child")
    child = col.decks.get(child_id)
    child["conf"] = child_conf
    col.decks.save(child)
    col.decks.select(parent_id)
    notetype = col.models.by_name("Basic")

    n_final_new = max(3, n_notes // 15)
    n_final_learning = 2
    n_regular = n_notes - n_final_new - n_final_learning
    intro_last_day = max(1, int(days * 0.7))
    intro_days = sorted(
        [0] * min(5, n_regular)
        + [rng.randint(0, intro_last_day) for _ in range(n_regular - min(5, n_regular))]
    )
    skipped = {d for d in range(1, days - 4) if rng.random() < 0.08}
    skipped |= {days - 3, days - 2, days - 1}  # leave overdue cards for "today"
    note_counter = 0

    def add_notes(count: int) -> list[int]:
        nonlocal note_counter
        card_ids = []
        for _ in range(count):
            note = col.new_note(notetype)
            note["Front"] = f"Question {note_counter}"
            note["Back"] = f"Answer {note_counter}"
            deck = child_id if note_counter % 3 == 2 else parent_id
            col.add_note(note, deck)
            card_ids.extend(note.card_ids())
            note_counter += 1
            clock.advance(rng.uniform(5, 20))
        return card_ids

    pending_intro = 0
    sessions = 0
    for day in range(days):
        pending_intro += sum(1 for d in intro_days if d == day)
        if day in skipped:
            continue
        clock.set(session_start(day))
        if pending_intro:
            add_notes(pending_intro)
            pending_intro = 0
        sim.study_queue()
        sessions += 1

    # Final day ("today"): partial session, so cards remain in every state.
    from anki.scheduler_pb2 import CardAnswer

    clock.set(session_start(days))
    if pending_intro:
        add_notes(pending_intro)
        pending_intro = 0
    learning_ids = add_notes(n_final_learning)
    for card_id in learning_ids:
        sim.answer_card_id(card_id, forced=CardAnswer.GOOD)  # step 1 -> step 2
    today = int(col.sched.today)
    due_reviews = [
        int(cid)
        for cid in col.db.list(
            "select id from cards where queue=2 and due<=? order by due, id", today
        )
    ]
    due_by_card = dict(col.db.all("select id, due from cards where queue=2 and due<=?", today))
    # Leave at least one overdue and one due-today card unanswered; answer about
    # half of the rest (most overdue first), failing two so they relearn.
    keep = set()
    for wanted in (lambda due: due < today, lambda due: due == today):
        candidates = [cid for cid in due_reviews if wanted(due_by_card[cid])]
        if candidates:
            keep.add(candidates[-1])
    to_answer = [cid for cid in due_reviews if cid not in keep][::2]
    n_relearn = 2
    forced_again = set(to_answer[:n_relearn])
    if len(forced_again) < n_relearn:
        # Small decks: fail cards that are not yet due (explicit early review).
        early = col.db.list(
            "select id from cards where queue=2 and due>? order by due, id limit ?",
            today,
            n_relearn - len(forced_again),
        )
        forced_again.update(int(cid) for cid in early)
        to_answer.extend(int(cid) for cid in early)
    for card_id in to_answer:
        sim.answer_card_id(card_id, forced=CardAnswer.AGAIN if card_id in forced_again else None)
    sessions += 1
    add_notes(n_final_new)
    future = [
        int(cid)
        for cid in col.db.list(
            "select id from cards where queue=2 and due>? order by id", today
        )
    ]
    if len(future) < 2:
        raise RuntimeError("simulation produced too few future review cards")
    col.sched.suspend_cards([future[0]])
    col.sched.bury_cards([future[1]], manual=True)
    summary = collection_summary(col)
    summary["missing_states"] = [
        key
        for key in (
            "new", "learning", "relearning", "review_due_past", "review_due_today",
            "review_due_future", "suspended", "buried",
        )
        if not summary[key]
    ]
    summary.update(
        {
            "sessions": sessions,
            "answers": sim.answers,
            "skipped_days": sorted(skipped),
            "end_epoch": end_epoch,
            "last_fake_time": clock.t,
            "suspended_card_id": future[0],
            "buried_card_id": future[1],
            "today_at_sim": today,
        }
    )
    col.close()
    return summary


# --------------------------------------------------------------------------
# RWKV predictor (mirrors gui/filtered_deck_dialog.py::_run_build)
# --------------------------------------------------------------------------


def make_progress(*, verbose: bool = False):
    """Return the add-on's TimedProgressReporter (``update``/``check_cancelled``/
    ``cancelled``); optionally print each new stage label."""

    prepare_environment()
    from rwkvp_addon.progress import TimedProgressReporter

    last = {"label": None}

    def callback(current, total, label, _eta):
        if verbose and label != last["label"]:
            print(f"  [progress] {label} ({current}/{total})", flush=True)
            last["label"] = label

    return TimedProgressReporter(callback)


@dataclass
class RWKVPredictorHandle:
    review_data: Any  # rwkvp_addon.review_rows.ReviewData
    predictor: Any  # Callable[[list[dict]], list[float]]
    manager: Any  # RWKVCheckpointManager
    runtime: Any  # ScopedRuntimeLease
    readiness: Any  # CheckpointReadiness
    review_load: Any  # ReviewDataLoad
    progress: Any
    timings: dict[str, float]

    def rows_for(self, card_ids: Iterable[int], target_timestamp_seconds: float) -> list[dict]:
        from rwkvp_addon.review_rows import prediction_rows_for_card_ids

        return prediction_rows_for_card_ids(
            card_ids, self.review_data, target_timestamp_seconds=target_timestamp_seconds
        )

    def predict_at(self, card_ids: Sequence[int], target_timestamp_seconds: float) -> list[float]:
        return list(self.predictor(self.rows_for(card_ids, target_timestamp_seconds)))


def make_manager(
    store_dir: str | os.PathLike[str],
    *,
    model_id: str | None = None,
    model_generation: str | None = None,
    predict_mode: str = "fast",
    profile_name: str = "harness",
    config_overrides: dict[str, Any] | None = None,
):
    """Construct the manager exactly like runtime.manager_for_mw, minus ``mw``."""

    prepare_environment()
    from rwkvp_addon.addon_config import (
        checkpoint_save_interval,
        configured_model_generation,
        configured_model_id,
        default_addon_config,
        enabled_prediction_cache_specs,
        exclude_deleted_card_revlogs,
        gpu_prediction_math,
        predict_many_batch_size,
        predict_many_mode,
        process_many_mode,
        state_precision,
    )
    from rwkvp_addon.checkpoint_manager import RWKVCheckpointManager
    from rwkvp_addon.profile_store import ProfileStore
    from rwkvp_addon.review_type_normalization import (
        filtered_review_normalization_policy_for_store,
    )
    from rwkvp_addon.rwkv_backend import configured_rwkv_backend
    from rwkvp_addon.rwkv_performance_modes import PREDICT_MANY_ORACLE_MODE

    config = default_addon_config()  # addon_src/config.json
    config["predict_many_mode"] = predict_mode  # no GPU here: CPU "fast"
    config["process_many_mode"] = "fast"
    if model_id is not None:
        config["model"] = model_id
    if model_generation is not None:
        config["model_generation"] = model_generation
    config.update(config_overrides or {})

    root = Path(store_dir).resolve()
    store = ProfileStore(root=root, profile_name=profile_name)
    store.ensure()
    mode = predict_many_mode(config)
    manager = RWKVCheckpointManager(
        store,
        model_id=configured_model_id(config),
        model_generation=configured_model_generation(config),
        state_precision=state_precision(config),
        gpu_prediction_math=gpu_prediction_math(config),
        prediction_cache_specs=enabled_prediction_cache_specs(config),
        predict_many_mode=mode,
        predict_many_batch_size=predict_many_batch_size(
            config, mode if configured_rwkv_backend() == "rust" else PREDICT_MANY_ORACLE_MODE
        ),
        process_many_mode=process_many_mode(config),
        checkpoint_save_interval=checkpoint_save_interval(config),
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs(config),
        filtered_review_normalization_policy=filtered_review_normalization_policy_for_store(
            store
        ),
    )
    return manager


def ensure_checkpoint(col, manager, progress) -> dict[str, Any] | None:
    """Build the initial checkpoint like gui/menu.py's checkpoint workflow.

    The filtered-deck path raises ``MissingCheckpointError`` without one; in the
    GUI the user runs "Initialize checkpoint" (load_review_data_for_checkpoint
    with ``force_export=True`` -> initialize_or_update_checkpoint_from_load with
    ``rebuild=True, force_save=True``). Returns None if one already existed.
    """

    prepare_environment()
    from rwkvp_addon.dataset_export import (
        initialize_or_update_checkpoint_from_load,
        load_review_data_for_checkpoint,
    )

    if manager.has_checkpoint:
        return None
    started = time.perf_counter()
    review_load = load_review_data_for_checkpoint(
        col, manager.store, manager, progress, force_export=True
    )
    readiness = initialize_or_update_checkpoint_from_load(
        manager, review_load, progress, rebuild=True, force_save=True
    )
    manager.wait_for_pending_save()
    result = readiness.checkpoint_result
    return {
        "seconds": time.perf_counter() - started,
        "status": getattr(result, "status", None),
        "processed_review_count": getattr(result, "processed_review_count", None),
    }


@contextmanager
def open_rwkv_predictor(
    col,
    card_ids: Iterable[int],
    store_dir: str | os.PathLike[str],
    *,
    manager=None,
    progress=None,
    verbose: bool = False,
    initialize_checkpoint: bool = True,
    unload_on_exit: bool = True,
    **manager_kwargs: Any,
) -> Iterator[RWKVPredictorHandle]:
    """Open an immediate-mode RWKV predictor for ``card_ids``.

    Same sequence as the filtered-deck dialog: load review data
    (``allow_incremental=True``) -> ``open_checkpoint_runtime_from_load`` with
    ``checkpoint_scope_cards_for_card_ids`` -> IMMEDIATE predictor. A missing
    checkpoint is first built via :func:`ensure_checkpoint` (disable with
    ``initialize_checkpoint=False``). Pass an existing ``manager`` to reuse it
    (and ``unload_on_exit=False`` to keep it loaded for the next call).

    The manager owns a non-daemon worker thread: always let this context exit
    (or call ``manager.unload()``), otherwise interpreter shutdown can hang in
    ``RWKVCheckpointManager.__del__``.
    """

    prepare_environment()
    from rwkvp_addon.dataset_export import (
        load_review_data_for_checkpoint,
        open_checkpoint_runtime_from_load,
    )
    from rwkvp_addon.filtered_deck_predictors import filtered_deck_predictor_for_mode
    from rwkvp_addon.review_rows import checkpoint_scope_cards_for_card_ids
    from rwkvp_addon.rwkv_modes import RetrievabilityMode

    ids = [int(card_id) for card_id in card_ids]
    if manager is None:
        manager = make_manager(store_dir, **manager_kwargs)
    elif manager_kwargs:
        raise TypeError("manager_kwargs are only used when manager is None")
    if progress is None:
        progress = make_progress(verbose=verbose)
    timings: dict[str, float] = {}
    runtime = None
    try:
        if initialize_checkpoint:
            built = ensure_checkpoint(col, manager, progress)
            if built is not None:
                timings["initialize_checkpoint"] = built["seconds"]
        started = time.perf_counter()
        review_load = load_review_data_for_checkpoint(
            col, manager.store, manager, progress, allow_incremental=True
        )
        timings["load_review_data"] = time.perf_counter() - started
        started = time.perf_counter()
        readiness, runtime = open_checkpoint_runtime_from_load(
            manager,
            review_load,
            progress,
            scope_cards=checkpoint_scope_cards_for_card_ids(ids, review_load.review_data),
        )
        timings["open_runtime"] = time.perf_counter() - started
        predictor = filtered_deck_predictor_for_mode(
            RetrievabilityMode.IMMEDIATE, runtime, progress
        )
        yield RWKVPredictorHandle(
            review_data=readiness.review_data,
            predictor=predictor,
            manager=manager,
            runtime=runtime,
            readiness=readiness,
            review_load=review_load,
            progress=progress,
            timings=timings,
        )
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            if unload_on_exit:
                manager.unload()


# --------------------------------------------------------------------------
# CLI (the simulation child process; also a quick manual entry point)
# --------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulate", type=Path, help="(internal) build a collection here")
    parser.add_argument("--n-notes", type=int, default=60)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end-epoch", type=float, default=None)
    args = parser.parse_args(argv)
    if args.simulate is None:
        parser.error("--simulate is required")
    summary = _simulate(
        args.simulate,
        n_notes=args.n_notes,
        days=args.days,
        seed=args.seed,
        end_epoch=args.end_epoch if args.end_epoch is not None else time.time(),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
