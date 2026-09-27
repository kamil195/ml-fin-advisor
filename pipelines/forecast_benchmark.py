"""
Offline benchmark: which user-specific daily method should serve? (STEP 13M)

This script answers one question with measurements instead of assumptions: for a
daily-spend forecast over a fixed horizon, does a weekday-aware user profile
beat simpler user-specific baselines — and does any of them beat a
**non-personalized** global average (the thing personalization is supposed to
improve on)?

Data source: **synthetic** user histories generated here with a fixed seed
(``random.Random``). No real user data, no model artifacts, no network. The
generator covers the shapes this product must survive:

* ``weekday_seasonal`` — weekend-heavy spend (where a weekday profile should win)
* ``flat``            — constant level + noise (where a simple mean is hard to beat)
* ``trend``           — steady upward drift (where a recency-weighted level wins)
* ``spiky``           — few large purchases among zero-spend days
* ``sparse``          — spend on ~2 days per week only

Validation is **expanding-window walk-forward**: every fold trains on days
strictly before its test window. There is no random split (see
``src/models/forecaster/personal_forecast.py``).

Honest reporting rules followed here:

* All four personalized methods AND the non-personalized global reference are
  reported side by side, sorted by WAPE. The script never forces a "win".
* MAPE is shown only over days with a meaningful denominator; WAPE/MAE/RMSE
  carry the comparison because zero-spend days are common.
* The legacy global artifact (``models/serving/forecast_results.json``, MAPE
  ≈ 6.10%) is **not technically comparable**: it aggregates per-category weekly
  totals over a different synthetic dataset/period. Its number must never be
  quoted as this pipeline's personalized accuracy — the script states that.
* Results are offline/synthetic only and are not product claims.

Run:  python pipelines/forecast_benchmark.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.forecaster.personal_forecast import (  # noqa: E402
    METHOD_EWMA,
    METHOD_PROPHET,
    METHOD_RECENT_MEAN,
    METHOD_WEEKDAY_PROFILE,
    DailySeries,
    predict_daily,
    predict_daily_prophet,
)

# ── Benchmark configuration (deterministic) ─────────────────────────────────

SEED = 13013
USERS_PER_STYLE = 6
HISTORY_DAYS = 180
HORIZON_DAYS = 30
MIN_TRAIN_DAYS = 60
#: Prophet is slow, so it is evaluated on the first N users of EVERY style — a
#: balanced sample, never one easy style only — and the sample size is reported
#: alongside its result so the comparison cannot be over-read.
PROPHET_SAMPLE_PER_STYLE = 1

STYLES = ("weekday_seasonal", "flat", "trend", "spiky", "sparse")

GLOBAL_REFERENCE = "global_flat_mean"  # non-personalized, reference only


def synth_series(seed: int, style: str) -> DailySeries:
    """Deterministic synthetic daily-spend series for one style."""
    rng = random.Random(seed)
    start = date(2026, 1, 5)  # a Monday, so weekday effects are stable
    base = 1800.0
    spend: list[float] = []

    for i in range(HISTORY_DAYS):
        day = start + timedelta(days=i)
        weekday = day.weekday()
        weekend = weekday >= 5

        if style == "weekday_seasonal":
            level = base * (1.55 if weekend else 1.0)
            value = level * rng.uniform(0.85, 1.15)
        elif style == "flat":
            value = base * rng.uniform(0.9, 1.1)
        elif style == "trend":
            level = base * (1.0 + 0.004 * i) * (1.15 if weekend else 1.0)
            value = level * rng.uniform(0.9, 1.1)
        elif style == "spiky":
            value = 0.0 if rng.random() < 0.75 else base * rng.uniform(2.0, 6.0)
        elif style == "sparse":
            value = 0.0 if weekday not in (1, 4) else base * rng.uniform(1.6, 2.4)
        else:  # pragma: no cover - guarded by STYLES
            raise ValueError(f"unknown style: {style}")

        spend.append(round(value, 2))

    return DailySeries(
        dates=[start + timedelta(days=i) for i in range(HISTORY_DAYS)],
        spend=spend,
        transaction_count=sum(1 for v in spend if v > 0),
        distinct_spend_days=sum(1 for v in spend if v > 0),
    )


def global_reference_pairs(series: DailySeries) -> list[tuple[float, float]]:
    """(actual, predicted) pairs for the NON-personalized global mean baseline.

    The mean is computed from the training portion only, so the reference is
    evaluated under exactly the same leakage-safe folds as the personalized
    methods (it is included to show what personalization must beat).
    """
    pairs: list[tuple[float, float]] = []
    for t0 in range(MIN_TRAIN_DAYS, series.history_days - HORIZON_DAYS + 1, HORIZON_DAYS):
        train = series.spend[:t0]
        actual = series.spend[t0 : t0 + HORIZON_DAYS]
        level = fmean(train) if train else 0.0
        pairs.extend((a, level) for a in actual)
    return pairs


def pairs_to_metrics(method: str, pairs: list[tuple[float, float]]) -> dict[str, object]:
    """MAE/RMSE/WAPE/bias/MAPE for a set of (actual, predicted) pairs."""
    errors = [p - a for a, p in pairs]
    n = len(pairs)
    sum_actual = sum(a for a, _ in pairs)
    mape_terms = [abs(p - a) / abs(a) for a, p in pairs if abs(a) >= 1.0]
    wape = (sum(abs(e) for e in errors) / sum_actual) if sum_actual > 0 else None
    return {
        "method": method,
        "n_points": n,
        "mae": round(fmean([abs(e) for e in errors]), 2) if n else 0.0,
        "rmse": round(math.sqrt(fmean([e * e for e in errors])), 2) if n else 0.0,
        "wape_pct": None if wape is None else round(wape * 100, 2),
        "mape_pct": round(fmean(mape_terms) * 100, 2) if mape_terms else None,
        "bias": round(fmean(errors), 2) if n else 0.0,
    }


def fold_pairs(
    series: DailySeries, method: str
) -> tuple[list[tuple[float, float]], list[float], int]:
    """One method's (actual, predicted) pairs over the shared walk-forward folds.

    Folds mirror ``personal_forecast.walk_forward_evaluate`` exactly (same
    start, same non-overlapping step); a unit test asserts the two agree, so the
    benchmark cannot silently drift from the evaluated implementation.
    """
    pairs: list[tuple[float, float]] = []
    window_wapes: list[float] = []
    n_windows = 0
    for t0 in range(MIN_TRAIN_DAYS, series.history_days - HORIZON_DAYS + 1, HORIZON_DAYS):
        train = DailySeries(
            dates=series.dates[:t0],
            spend=series.spend[:t0],
            transaction_count=series.transaction_count,
            distinct_spend_days=series.distinct_spend_days,
        )
        actual = series.spend[t0 : t0 + HORIZON_DAYS]
        if method == METHOD_PROPHET:
            predicted = predict_daily_prophet(train, HORIZON_DAYS)
            if predicted is None or len(predicted) < len(actual):
                continue
        else:
            _, predicted = predict_daily(train, len(actual), method, start=series.dates[t0])
        fold = list(zip(actual, predicted))
        pairs.extend(fold)
        n_windows += 1
        fold_actual = sum(a for a, _ in fold)
        if fold_actual > 0:
            window_wapes.append(sum(abs(p - a) for a, p in fold) / fold_actual)
    return pairs, window_wapes, n_windows


def _pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile as a percentage (reporting dispersion only)."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return round(ordered[idx] * 100, 2)


def main() -> int:
    """Run the benchmark and print an honest comparison table."""
    personalized = (METHOD_RECENT_MEAN, METHOD_WEEKDAY_PROFILE, METHOD_EWMA)
    tracked = personalized + (METHOD_PROPHET, GLOBAL_REFERENCE)

    pooled: dict[str, list[tuple[float, float]]] = {m: [] for m in tracked}
    user_wapes: dict[str, list[float]] = {m: [] for m in tracked}
    windows: dict[str, int] = {m: 0 for m in tracked}

    prophet_users = 0
    for style_index, style in enumerate(STYLES):
        for user_index in range(USERS_PER_STYLE):
            series = synth_series(SEED + style_index * 1000 + user_index, style)

            for method in personalized:
                pairs, wapes, n_windows = fold_pairs(series, method)
                pooled[method].extend(pairs)
                windows[method] += n_windows
                if wapes:
                    user_wapes[method].append(fmean(wapes))

            pooled[GLOBAL_REFERENCE].extend(global_reference_pairs(series))
            windows[GLOBAL_REFERENCE] += max(
                1, (HISTORY_DAYS - MIN_TRAIN_DAYS) // HORIZON_DAYS
            )

            # Prophet is expensive: evaluated on the first N users of EVERY
            # style (balanced sample), with the sample size reported.
            if user_index < PROPHET_SAMPLE_PER_STYLE:
                pairs, wapes, n_windows = fold_pairs(series, METHOD_PROPHET)
                pooled[METHOD_PROPHET].extend(pairs)
                windows[METHOD_PROPHET] += n_windows
                if wapes:
                    user_wapes[METHOD_PROPHET].append(fmean(wapes))
                prophet_users += 1

    rows: list[dict[str, object]] = []
    for method in tracked:
        row = pairs_to_metrics(method, pooled[method])
        wape_values = user_wapes[method]
        row["users_evaluated"] = len(wape_values) or USERS_PER_STYLE * len(STYLES)
        row["windows"] = windows[method]
        row["user_wape_median_pct"] = _pct(wape_values, 0.5)
        row["user_wape_p90_pct"] = _pct(wape_values, 0.9)
        rows.append(row)

    rows.sort(
        key=lambda r: (r["wape_pct"] if r["wape_pct"] is not None else float("inf"))
    )
    _report(rows, personalized, prophet_users)
    return 0


def _fmt(value: object) -> str:
    """Render a metric for the table ('-' when undefined, e.g. no denominator)."""
    return "-" if value is None else str(value)


def _report(
    rows: list[dict[str, object]], personalized: tuple[str, ...], prophet_users: int
) -> None:
    """Print the comparison table, the verdict, and the limitations."""
    print("=" * 92)
    print("STEP 13M — user-specific daily forecast benchmark (OFFLINE / SYNTHETIC)")
    print("=" * 92)
    print(
        f"synthetic users: {USERS_PER_STYLE * len(STYLES)}  |  history: {HISTORY_DAYS}d"
        f"  |  horizon: {HORIZON_DAYS}d  |  min train: {MIN_TRAIN_DAYS}d  |  seed: {SEED}"
    )
    print(
        f"prophet evaluated on {prophet_users} of {USERS_PER_STYLE * len(STYLES)} users "
        "(first user of every style — balanced sample, bounded runtime)"
    )
    print()

    header = (
        f"{'method':<20}{'users':>7}{'windows':>9}{'MAE':>10}{'RMSE':>10}"
        f"{'WAPE%':>8}{'MAPE%':>8}{'bias':>10}{'medWAPE%':>11}{'p90WAPE%':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['method']):<20}{int(row['users_evaluated']):>7}"
            f"{int(row['windows']):>9}"
            f"{row['mae']:>10}{row['rmse']:>10}"
            f"{_fmt(row['wape_pct']):>8}{_fmt(row['mape_pct']):>8}{row['bias']:>10}"
            f"{_fmt(row['user_wape_median_pct']):>11}{_fmt(row['user_wape_p90_pct']):>10}"
        )

    best = next(
        (r for r in rows if r["method"] in personalized and r["wape_pct"] is not None),
        None,
    )
    reference = next((r for r in rows if r["method"] == GLOBAL_REFERENCE), None)
    print()
    if best is not None:
        print(f"selected personalized method (lowest pooled WAPE): {best['method']}")
    if best is not None and reference is not None and reference["wape_pct"] is not None:
        delta = float(reference["wape_pct"]) - float(best["wape_pct"])
        verdict = "better than" if delta > 0 else "NOT better than"
        print(
            f"personalization check: {best['method']} is {verdict} the "
            f"non-personalized global mean ({best['wape_pct']}% vs {reference['wape_pct']}% "
            f"WAPE, delta {round(delta, 2)}pp)"
        )
    print()
    print("LIMITATIONS (must be preserved in any write-up):")
    print(" * synthetic data only — offline numbers, never product/customer accuracy claims")
    print(" * the legacy artifact models/serving/forecast_results.json (MAPE ~6.10%) is NOT")
    print("   comparable (different dataset; per-category WEEKLY totals over a demo period) and")
    print("   must never be quoted as this pipeline's personalized accuracy")
    print(" * MAPE is undefined for zero-spend days, so it is reported only where |actual| >= 1;")
    print("   WAPE/MAE/RMSE carry the comparison")
    print(" * expanding-window walk-forward folds only — no random split anywhere")
    print()
    print("=== JSON ===")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
