#!/usr/bin/env python3
"""Check whether RWKV immediate predictions decay monotonically with horizon.

Builds a simulated collection (rwkv_harness.make_collection), opens the
filtered-deck IMMEDIATE predictor for every review card (``is:review``: review
and relearning cards, including the suspended/buried ones) and predicts
retrievability at end-of-today and +1/+2/+4/+8/+16/+32/+64 days, one predictor
call per horizon. Also checks whether mixing horizons (the same card_id at
several target timestamps) inside one predictor call changes the results.

Usage::

    /home/claude/venv/bin/python tests/integration/check_future_monotonicity.py \
        [--n-notes 60] [--days 120] [--seed 0] [--json out.json] [--keep]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rwkv_harness  # noqa: E402

HORIZON_DAYS = (0, 1, 2, 4, 8, 16, 32, 64)  # 0 == end of today
STRICT_TOL = 0.0
LOOSE_TOL = 1e-4


def _label(days: int) -> str:
    return "end_of_today" if days == 0 else f"+{days}d"


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "min": min(values),
        "p10": _quantile(values, 0.1),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": _quantile(values, 0.9),
        "max": max(values),
    }


def _timed(predictor, rows):
    started = time.perf_counter()
    values = list(predictor(rows))
    return values, time.perf_counter() - started


def run(n_notes: int, days: int, seed: int, workdir: Path, model_id: str | None = None) -> dict:
    report: dict = {
        "params": {"n_notes": n_notes, "days": days, "seed": seed, "model_id": model_id}
    }
    started = time.perf_counter()
    col = rwkv_harness.make_collection(workdir, n_notes=n_notes, days=days, seed=seed)
    report["collection_build_seconds"] = time.perf_counter() - started
    report["collection"] = col.harness_summary["after_reopen"]
    try:
        card_ids = [int(cid) for cid in col.find_cards("is:review")]
        type_by_card = dict(
            col.db.all(f"select id, type from cards where id in ({','.join(map(str, card_ids))})")
        )
        queue_by_card = dict(
            col.db.all(f"select id, queue from cards where id in ({','.join(map(str, card_ids))})")
        )
        extra = {} if model_id is None else {"model_id": model_id}
        with rwkv_harness.open_rwkv_predictor(
            col, card_ids, workdir / "store", **extra
        ) as handle:
            report["model_id"] = handle.manager.model_id
            report["open_timings_seconds"] = dict(handle.timings)
            review_data = handle.review_data
            eod = review_data.next_day_at - 1
            report["day_handling"] = {
                "col.sched.today": int(col.sched.today),
                "col.sched.day_cutoff": int(col.sched.day_cutoff),
                "review_data.next_day_at": int(review_data.next_day_at),
                "review_data.day_offset_origin": int(review_data.day_offset_origin),
                "processable_rows": len(review_data.rows),
                "revlog_rows": len(review_data.revlogs),
            }
            targets = {d: eod + d * 86400 for d in HORIZON_DAYS}
            rows_by_horizon = {d: handle.rows_for(card_ids, t) for d, t in targets.items()}
            sample_rows = rows_by_horizon[0][:1] + rows_by_horizon[1][:1]
            report["day_handling"]["sample_rows"] = sample_rows

            # Warm-up so the first timed batch is not penalised by lazy init.
            handle.predictor(rows_by_horizon[0][:1])

            preds: dict[int, list[float]] = {}
            batch_seconds: dict[str, float] = {}
            for d in HORIZON_DAYS:
                preds[d], batch_seconds[_label(d)] = _timed(
                    handle.predictor, rows_by_horizon[d]
                )
            report["batch_seconds"] = batch_seconds
            report["cards"] = len(card_ids)

            # Determinism: repeat the first horizon after all others.
            repeat, _ = _timed(handle.predictor, rows_by_horizon[0])
            report["repeat_max_abs_diff"] = max(
                abs(a - b) for a, b in zip(repeat, preds[0], strict=True)
            )

            # --- Mixed-horizon batches -------------------------------------
            mixed = {}
            # (a) every card twice, interleaved (+1d, +16d) in ONE call.
            pair_rows = []
            for i in range(len(card_ids)):
                pair_rows.append(rows_by_horizon[1][i])
                pair_rows.append(rows_by_horizon[16][i])
            pair_vals, pair_secs = _timed(handle.predictor, pair_rows)
            diffs = [abs(pair_vals[2 * i] - preds[1][i]) for i in range(len(card_ids))]
            diffs += [abs(pair_vals[2 * i + 1] - preds[16][i]) for i in range(len(card_ids))]
            mixed["interleaved_+1d_+16d_max_abs_diff"] = max(diffs)
            mixed["interleaved_seconds"] = pair_secs
            # (b) all horizons x all cards in one call (horizon-major).
            all_rows = [row for d in HORIZON_DAYS for row in rows_by_horizon[d]]
            all_vals, all_secs = _timed(handle.predictor, all_rows)
            expected = [v for d in HORIZON_DAYS for v in preds[d]]
            mixed["all_horizons_one_call_max_abs_diff"] = max(
                abs(a - b) for a, b in zip(all_vals, expected, strict=True)
            )
            mixed["all_horizons_one_call_rows"] = len(all_rows)
            mixed["all_horizons_one_call_seconds"] = all_secs
            # (c) single card, two timestamps in one call vs two calls.
            single = []
            for i in range(min(10, len(card_ids))):
                together = list(handle.predictor([rows_by_horizon[1][i], rows_by_horizon[16][i]]))
                alone_1 = list(handle.predictor([rows_by_horizon[1][i]]))[0]
                alone_16 = list(handle.predictor([rows_by_horizon[16][i]]))[0]
                single.append(max(abs(together[0] - alone_1), abs(together[1] - alone_16)))
            mixed["single_card_pair_vs_separate_max_abs_diff"] = max(single)
            # (d) control: batch-of-one vs full batch (batch-composition effects).
            control = [
                abs(list(handle.predictor([rows_by_horizon[4][i]]))[0] - preds[4][i])
                for i in range(min(10, len(card_ids)))
            ]
            mixed["batch_of_one_vs_full_batch_max_abs_diff"] = max(control)
            report["mixed_horizon"] = mixed
    finally:
        col.close()

    # --- Monotonicity -----------------------------------------------------
    n = len(card_ids)
    increases = []
    strict_ok = loose_ok = 0
    examples = []
    for i, card_id in enumerate(card_ids):
        series = [preds[d][i] for d in HORIZON_DAYS]
        steps = [b - a for a, b in zip(series, series[1:])]
        worst = max(steps)
        increases.append(max(0.0, worst))
        strict_ok += worst <= STRICT_TOL
        loose_ok += worst <= LOOSE_TOL
        if worst > LOOSE_TOL:
            examples.append(
                {
                    "card_id": card_id,
                    "type": type_by_card.get(card_id),
                    "queue": queue_by_card.get(card_id),
                    "elapsed_days_at_eod": rows_by_horizon[0][i]["elapsed_days"],
                    "max_increase": worst,
                    "at_step": f"{_label(HORIZON_DAYS[steps.index(worst)])}->"
                    f"{_label(HORIZON_DAYS[steps.index(worst) + 1])}",
                    "series": [round(v, 5) for v in series],
                }
            )
    examples.sort(key=lambda e: -e["max_increase"])
    finite = all(math.isfinite(v) and 0.0 < v < 1.0 for d in HORIZON_DAYS for v in preds[d])
    report["all_predictions_finite_in_0_1"] = finite
    report["monotonicity"] = {
        "cards": n,
        "fraction_non_increasing_strict": strict_ok / n,
        f"fraction_non_increasing_tol_{LOOSE_TOL:g}": loose_ok / n,
        "max_increase": max(increases),
        "cards_with_any_increase": sum(1 for v in increases if v > 0),
        "increase_summary_over_violators": (
            _summary([v for v in increases if v > 0]) if any(v > 0 for v in increases) else None
        ),
        "worst_examples": examples[:8],
    }
    report["prediction_distribution"] = {
        _label(d): _summary(preds[d]) for d in HORIZON_DAYS
    }
    report["mean_drop_per_step"] = {
        f"{_label(a)}->{_label(b)}": statistics.fmean(
            preds[a][i] - preds[b][i] for i in range(n)
        )
        for a, b in zip(HORIZON_DAYS, HORIZON_DAYS[1:])
    }
    report["total_seconds"] = time.perf_counter() - started
    return report


def _print_report(report: dict) -> None:
    c = report["collection"]
    print(f"Collection: {c['cards']} cards, {c['revlog']} revlog, today={c['today']}, "
          f"new={c['new']} learn={c['learning']} relearn={c['relearning']} "
          f"due past/today/future={c['review_due_past']}/{c['review_due_today']}/"
          f"{c['review_due_future']} suspended={c['suspended']} buried={c['buried']} "
          f"(built in {report['collection_build_seconds']:.1f}s)")
    print(f"Model: {report['model_id']}")
    print(f"Predictor open: {json.dumps({k: round(v, 3) for k, v in report['open_timings_seconds'].items()})}")
    print(f"Day handling: {json.dumps({k: v for k, v in report['day_handling'].items() if k != 'sample_rows'})}")
    for row in report["day_handling"]["sample_rows"]:
        print(f"  sample row: {row}")
    print("\nPrediction distribution per horizon (one predictor call each):")
    print(f"  {'horizon':>13} {'min':>7} {'p10':>7} {'median':>7} {'mean':>7} {'p90':>7} {'max':>7} {'secs':>8}")
    for label, s in report["prediction_distribution"].items():
        secs = report["batch_seconds"][label]
        print(f"  {label:>13} {s['min']:7.4f} {s['p10']:7.4f} {s['median']:7.4f} {s['mean']:7.4f} "
              f"{s['p90']:7.4f} {s['max']:7.4f} {secs * 1000:6.1f}ms")
    m = report["monotonicity"]
    print(f"\nMonotonicity over {m['cards']} review cards:")
    for key, value in m.items():
        if key in {"worst_examples", "cards"}:
            continue
        print(f"  {key}: {value}")
    for example in m["worst_examples"]:
        print(f"  example: {example}")
    print(f"\nMean drop per step: {json.dumps({k: round(v, 4) for k, v in report['mean_drop_per_step'].items()})}")
    print(f"Repeat-call max abs diff: {report['repeat_max_abs_diff']:.3g}")
    print("Mixed-horizon batches:")
    for key, value in report["mixed_horizon"].items():
        print(f"  {key}: {value:.3g}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\nAll predictions finite in (0,1): {report['all_predictions_finite_in_0_1']}")
    print(f"Total {report['total_seconds']:.1f}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-notes", type=int, default=60)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default=None, help="v1 model id (default: config.json)")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--keep", action="store_true", help="keep the temp directory")
    args = parser.parse_args(argv)
    reason = rwkv_harness.unavailable_reason()
    if reason:
        print(f"cannot run: {reason}", file=sys.stderr)
        return 2
    workdir = Path(tempfile.mkdtemp(prefix="rwkv-monotonicity-"))
    try:
        report = run(args.n_notes, args.days, args.seed, workdir, args.model_id)
    finally:
        if args.keep:
            print(f"kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    _print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
