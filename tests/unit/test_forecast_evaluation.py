"""STEP 13M — forecast evaluation methodology tests.

Pins the properties that make the offline numbers trustworthy:

* time-based (expanding-window) validation only — no random split anywhere;
* a fold may never see the days it predicts (leakage check);
* metrics survive zero-spend days (WAPE/MAE/RMSE) and MAPE uses a safe
  denominator rule;
* the benchmark's folds match the evaluated implementation's folds, so the
  reported numbers cannot silently drift from the shipped code.

Everything here is synthetic and offline: no network, no Postgres, no product
accuracy claim.
"""

from __future__ import annotations

import importlib.util
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.models.forecaster import personal_forecast as pf

REPO = Path(__file__).resolve().parents[2]
DAY = timedelta(days=1)
BASE = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)  # a Monday


def _rows(days: int, amount: float = -100.0, *, every: int = 1) -> list[tuple]:
    """Rows with a spend every ``every`` days (0 spend on the other days)."""
    return [
        (BASE + i * DAY, amount)
        for i in range(days)
        if i % every == 0
    ]


def _series(days: int, amount: float = -100.0) -> pf.DailySeries:
    return pf.build_daily_series(_rows(days, amount))


def _load_benchmark():
    """Import pipelines/forecast_benchmark.py by path (it is a script)."""
    path = REPO / "pipelines" / "forecast_benchmark.py"
    spec = importlib.util.spec_from_file_location("pw_forecast_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── series construction ──────────────────────────────────────────────────────


def test_series_uses_spend_only_and_zero_fills():
    rows = [
        (BASE, -100.0),          # spend
        (BASE + DAY, 500.0),     # income -> excluded, not netted
        (BASE + 2 * DAY, -50.0),  # spend
        (BASE + 2 * DAY, 20.0),   # refund -> excluded, not netted
    ]
    series = pf.build_daily_series(rows)
    assert series.spend == [100.0, 0.0, 50.0]   # gap zero-filled
    assert series.transaction_count == 2        # only the two debits
    assert series.distinct_spend_days == 2
    assert series.history_days == 3


def test_series_end_date_prevents_future_leakage():
    """Truncation is what keeps a fold from seeing its own test window."""
    rows = [(BASE + i * DAY, -10.0) for i in range(10)]
    series = pf.build_daily_series(rows, end_date=(BASE + 4 * DAY).date())
    assert series.history_days == 5
    assert series.last_date == (BASE + 4 * DAY).date()


def test_quality_tier_thresholds_are_explicit():
    assert pf.quality_tier(_series(40, -100.0)) == pf.STATUS_PERSONALIZED
    assert pf.quality_tier(_series(20, -50.0)) == pf.STATUS_LIMITED
    assert pf.quality_tier(_series(5, -50.0)) == pf.STATUS_INSUFFICIENT
    assert pf.quality_tier(pf.build_daily_series([])) == pf.STATUS_INSUFFICIENT
    # Many transactions but too short a span is still not "personalized".
    dense_short = pf.build_daily_series([(BASE + timedelta(hours=h), -5.0) for h in range(0, 240, 2)])
    assert pf.quality_tier(dense_short) == pf.STATUS_INSUFFICIENT


# ── leakage safety ───────────────────────────────────────────────────────────


def test_evaluation_uses_no_random_split():
    """(17)(18) the implementation contains no random/shuffled splitting."""
    src = inspect.getsource(pf)
    for forbidden in ("import random", "np.random", "numpy.random",
                      "train_test_split", "shuffle(", ".shuffle"):
        assert forbidden not in src, f"random split construct present: {forbidden}"
    assert "walk_forward_evaluate" in src
    assert "expanding" in src  # the method is documented as expanding-window


def test_benchmark_folds_are_not_random():
    """The benchmark must not shuffle either (it only seeds synthetic data)."""
    src = (REPO / "pipelines" / "forecast_benchmark.py").read_text(encoding="utf-8")
    for forbidden in ("train_test_split", "shuffle(", ".shuffle", "sample("):
        assert forbidden not in src, f"random construct present: {forbidden}"


def test_fold_never_trains_on_its_test_window():
    """A fold's prediction equals one computed from strictly-prior days only."""
    series = _series(60, -100.0)
    horizon = 20
    train = pf.DailySeries(
        dates=series.dates[:40],
        spend=series.spend[:40],
        transaction_count=series.transaction_count,
        distinct_spend_days=series.distinct_spend_days,
    )
    _, manual = pf.predict_daily(
        train, horizon, pf.METHOD_RECENT_MEAN, start=series.dates[40]
    )

    metrics = pf.walk_forward_evaluate(
        series, horizon, (pf.METHOD_RECENT_MEAN,), min_train_days=40
    )[0]
    assert metrics.n_windows == 1
    assert metrics.n_points == horizon
    expected_wape = sum(
        abs(p - a) for a, p in zip(series.spend[40:60], manual)
    ) / sum(series.spend[40:60])
    assert metrics.wape == pytest.approx(expected_wape)


# ── metrics on zero-spend days ───────────────────────────────────────────────


def test_metrics_are_finite_with_zero_spend_days():
    """(19) zero-spend days must not break the metrics."""
    series = pf.build_daily_series(_rows(120, -100.0, every=4))  # 75% zero days
    assert series.spend.count(0.0) > 80
    metrics = pf.walk_forward_evaluate(
        series, 30, (pf.METHOD_WEEKDAY_PROFILE,), min_train_days=60
    )[0]
    assert metrics.n_windows > 0
    assert metrics.mae >= 0 and metrics.rmse >= 0
    assert metrics.wape is not None and metrics.wape >= 0
    assert metrics.mape is None or metrics.mape >= 0


def test_wape_is_undefined_when_there_is_no_actual_spend():
    """An all-zero actual window yields no WAPE rather than a divide-by-zero."""
    empty = pf.DailySeries(dates=[], spend=[], transaction_count=0, distinct_spend_days=0)
    metrics = pf._metrics_for(pf.METHOD_RECENT_MEAN, [(0.0, 5.0), (0.0, 0.0)], [])
    assert metrics.wape is None
    assert metrics.mae == pytest.approx(2.5)
    assert metrics.mape is None
    assert empty.history_days == 0


def test_mape_uses_a_safe_denominator():
    """MAPE ignores days below MAPE_MIN_ACTUAL instead of exploding."""
    pairs = [(0.0, 100.0), (100.0, 90.0)]  # first day would be a huge error
    metrics = pf._metrics_for(pf.METHOD_RECENT_MEAN, pairs, [])
    assert metrics.mape == pytest.approx(0.10)  # only the 100.0 day counts
    assert metrics.bias == pytest.approx((100.0 - 0.0 + 90.0 - 100.0) / 2)


# ── implementation / benchmark agreement ─────────────────────────────────────


def test_benchmark_folds_match_shipped_evaluation():
    """The benchmark measures exactly what the shipped code computes."""
    bench = _load_benchmark()
    series = bench.synth_series(bench.SEED, "weekday_seasonal")
    _, window_wapes, n_windows = bench.fold_pairs(series, pf.METHOD_RECENT_MEAN)
    metrics = pf.walk_forward_evaluate(
        series,
        bench.HORIZON_DAYS,
        (pf.METHOD_RECENT_MEAN,),
        min_train_days=bench.MIN_TRAIN_DAYS,
    )[0]
    assert metrics.n_windows == n_windows > 0
    assert metrics.per_window_wape == pytest.approx(window_wapes)


def test_benchmark_synthetic_data_is_deterministic():
    bench = _load_benchmark()
    first = bench.synth_series(13013, "trend")
    again = bench.synth_series(13013, "trend")
    other = bench.synth_series(13014, "trend")
    assert first.spend == again.spend
    assert first.spend != other.spend


# ── forecast shape / sanity ──────────────────────────────────────────────────


def test_interval_is_ordered_and_non_negative():
    result = pf.predict(_series(60, -100.0), 30)
    assert result["personalization_status"] == pf.STATUS_PERSONALIZED
    for point in result["daily"]:
        assert 0.0 <= point["p10"] <= point["p50"] <= point["p90"]
    assert result["total"]["p10"] <= result["total"]["p50"] <= result["total"]["p90"]
    assert result["expected_spend"] == result["total"]["p50"]
    assert len(result["daily"]) == 30
    assert result["history_days_used"] == 60
    assert result["engine_version"] == pf.FORECAST_ENGINE_VERSION


def test_weekday_profile_follows_the_users_own_weekday_pattern():
    """The selected method reflects THIS user's weekday shape, not a global one."""
    rows: list[tuple] = []
    for i in range(84):  # 12 weeks
        weekend = (BASE + i * DAY).weekday() >= 5
        rows.append((BASE + i * DAY, -300.0 if weekend else -10.0))
    weekend_heavy = pf.build_daily_series(rows)

    levels = pf.weekday_profile_levels(weekend_heavy)
    assert min(levels[5], levels[6]) > max(levels[0], levels[1], levels[2])
    # An inverted user yields the opposite profile from the same code path.
    inverted = pf.build_daily_series(
        [(ts, -10.0 if ts.weekday() >= 5 else -300.0) for ts, _ in rows]
    )
    inverted_levels = pf.weekday_profile_levels(inverted)
    assert min(inverted_levels[0], inverted_levels[1], inverted_levels[2]) > max(
        inverted_levels[5], inverted_levels[6]
    )
    assert pf.PRIMARY_METHOD == pf.METHOD_WEEKDAY_PROFILE