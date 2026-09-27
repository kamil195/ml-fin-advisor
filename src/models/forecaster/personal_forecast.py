"""
User-specific daily spend forecasting (STEP 13).

Produces a forecast from ONE authenticated user's own persisted transaction
history. Nothing global is ever substituted for a user's data, and no other
user's rows can influence an output.

Design constraints (deliberate):

* **Deterministic.** Pure functions over an explicit daily series; no
  randomness, no wall-clock reads inside the model, no LLM.
* **Leakage-safe evaluation.** ``walk_forward_evaluate`` uses expanding-window
  time-series validation (train strictly before test); there is no random
  split anywhere in this module.
* **Honest about sparse data.** ``quality_tier`` classifies the available
  history; the caller reports the tier verbatim instead of upgrading a thin
  series to "personalized".
* **Explainable.** Each forecast reports the method, the engine version, the
  history actually used, and a residual-based 80% interval — no fabricated
  precision score.

Methods implemented (all user-specific):

``recent_mean``      flat projection of the recent level (last 28 days)
``weekday_profile``  per-weekday mean over the recent 8 weeks (weekday-aware)
``ewma``             exponentially weighted level (7-day half-life), flat
``prophet``          weekly Prophet (optional; only if importable)

Which method serves production is decided by ``pipelines/forecast_benchmark.py``
(see ``PRIMARY_METHOD``/``FALLBACK_METHOD`` below) — the choice is measured, not
assumed. Both options are user-specific, so a fallback never leaks a global
series into a personalized response.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from statistics import fmean

logger = logging.getLogger(__name__)

#: Engine identifier: returned with every forecast, used in cache keys.
FORECAST_ENGINE_VERSION = "user-v1"

#: 80% interval multiplier (matches the Prophet wrapper's interval_width=0.80).
Z80 = 1.2816

#: Label for results produced from a user's own persisted history.
STATUS_PERSONALIZED = "personalized"
STATUS_LIMITED = "limited_history"
STATUS_INSUFFICIENT = "insufficient_history"
#: Label for the legacy global/static artifact (never mixed with the above).
STATUS_NOT_PERSONALIZED = "not_personalized"

FALLBACK_NONE = "none"
FALLBACK_USER_BASELINE = "user_recent_baseline"
#: Set when the requested method could not be used and the user-specific
#: primary method was substituted — reported so a caller is never misled.
FALLBACK_PRIMARY_SUBSTITUTED = "primary_substituted"
FALLBACK_GLOBAL_REFERENCE = "global_reference"

# ── Minimum-history thresholds (evidence-based, see benchmark report) ────────
# Tier A requires four complete weeks so weekday effects are estimable and a
# level shift is detectable; Tier B keeps a short window meaningful but flags
# the reduced reliability. Below Tier B no personalized number is produced.
MIN_DAYS_PERSONALIZED = 28
MIN_TXNS_PERSONALIZED = 20
MIN_SPEND_DAYS_PERSONALIZED = 12
MIN_DAYS_LIMITED = 14
MIN_TXNS_LIMITED = 8

#: Recent windows used by the level methods.
RECENT_LEVEL_DAYS = 28
WEEKDAY_PROFILE_DAYS = 56  # 8 complete weeks
EWMA_HALF_LIFE_DAYS = 7.0
#: Horizon bounds shared with the API layer.
MIN_HORIZON_DAYS = 7
MAX_HORIZON_DAYS = 90
DEFAULT_HORIZON_DAYS = 30


@dataclass(frozen=True)
class DailySeries:
    """A user's own daily spend series (zero-filled, chronological).

    ``dates``/``spend`` are aligned and include every calendar day between the
    first and last observed spend, with ``0.0`` on days without spend. Zero
    days are real observations (no spending happened) and are required for
    correct weekday profiles and metrics.
    """

    dates: list[date]
    spend: list[float]
    transaction_count: int
    distinct_spend_days: int

    @property
    def history_days(self) -> int:
        return len(self.dates)

    @property
    def last_date(self) -> date | None:
        return self.dates[-1] if self.dates else None

    @property
    def total_spend(self) -> float:
        return round(sum(self.spend), 2)


def build_daily_series(
    rows: list[tuple[datetime, float]],
    *,
    end_date: date | None = None,
) -> DailySeries:
    """Aggregate persisted transactions into a zero-filled daily spend series.

    Transformations (all deliberate — no silent financial assumptions):

    * Only **spend** is modelled: rows with ``amount >= 0`` (income, credits,
      refunds) are excluded rather than netted, matching the existing
      forecaster semantics (``ProphetModel._prepare_series`` and
      ``/consumer/forecast/live`` both filter to ``amount < 0``).
    * Spend is taken as ``abs(amount)``, grouped per UTC calendar day.
    * Days between the first and last observation are **zero-filled**, so a
      no-spend day is evidence of no spending rather than a gap.
    * Duplicate transactions are already collapsed at ingest (Step 12 unique
      dedupe index); same-day rows simply add up.
    * ``end_date`` (if given) truncates the series — used by the walk-forward
      evaluation to prevent any leakage of the test window into training.
    """
    by_day: dict[date, float] = {}
    used = 0
    for occurred_at, amount in rows:
        if amount is None or amount >= 0:
            continue  # spend only; credits/refunds/income are not netted
        day = occurred_at.date() if isinstance(occurred_at, datetime) else occurred_at
        if end_date is not None and day > end_date:
            continue
        by_day[day] = by_day.get(day, 0.0) + abs(float(amount))
        used += 1

    if not by_day:
        return DailySeries(dates=[], spend=[], transaction_count=0, distinct_spend_days=0)

    first, last = min(by_day), max(by_day)
    span = (last - first).days + 1
    dates = [first + timedelta(days=i) for i in range(span)]
    spend = [round(by_day.get(d, 0.0), 2) for d in dates]
    return DailySeries(
        dates=dates,
        spend=spend,
        transaction_count=used,
        distinct_spend_days=sum(1 for v in spend if v > 0),
    )


def quality_tier(series: DailySeries) -> str:
    """Classify available history into an explicit, honest tier.

    Returns one of ``STATUS_PERSONALIZED`` (Tier A), ``STATUS_LIMITED``
    (Tier B) or ``STATUS_INSUFFICIENT`` (Tier C). A Tier C user receives **no**
    numeric forecast — never a global series relabelled as their own.
    """
    if (
        series.history_days >= MIN_DAYS_PERSONALIZED
        and series.transaction_count >= MIN_TXNS_PERSONALIZED
        and series.distinct_spend_days >= MIN_SPEND_DAYS_PERSONALIZED
    ):
        return STATUS_PERSONALIZED
    if (
        series.history_days >= MIN_DAYS_LIMITED
        and series.transaction_count >= MIN_TXNS_LIMITED
        and series.distinct_spend_days >= 2
    ):
        return STATUS_LIMITED
    return STATUS_INSUFFICIENT


def insufficient_history_detail(series: DailySeries) -> dict[str, object]:
    """What the caller needs before a personalized forecast is possible."""
    return {
        "required": {
            "history_days": MIN_DAYS_LIMITED,
            "transactions": MIN_TXNS_LIMITED,
            "distinct_spend_days": 2,
        },
        "for_full_confidence": {
            "history_days": MIN_DAYS_PERSONALIZED,
            "transactions": MIN_TXNS_PERSONALIZED,
            "distinct_spend_days": MIN_SPEND_DAYS_PERSONALIZED,
        },
        "observed": {
            "history_days": series.history_days,
            "transactions": series.transaction_count,
            "distinct_spend_days": series.distinct_spend_days,
        },
    }


# ── Level estimators (all derived from ONE user's series only) ───────────────

METHOD_RECENT_MEAN = "recent_mean"
METHOD_WEEKDAY_PROFILE = "weekday_profile"
METHOD_EWMA = "ewma"
METHOD_PROPHET = "prophet"

#: Methods selectable for a personalized serving response.
METHODS = (METHOD_RECENT_MEAN, METHOD_WEEKDAY_PROFILE, METHOD_EWMA, METHOD_PROPHET)

#: Serving method, chosen by measurement — see the model docstring and
#: pipelines/forecast_benchmark.py. Measured on 30 synthetic users (180d history,
#: 30d horizon, expanding-window walk-forward, 2026-09 re-run):
#:
#:   method            WAPE%   MAE     medWAPE%   notes
#:   weekday_profile   37.66   724.13  9.47       <- selected
#:   prophet           41.91   767.85  18.26      5-user balanced sample only
#:   recent_mean       53.55  1029.73  19.01
#:   ewma              53.64  1031.47  18.94
#:   global_flat_mean  56.00  1076.77  -          NON-personalized reference
#:
#: Prophet did NOT win once sampled across all styles, and it is far slower at
#: request time, so it is not used for serving. Both PRIMARY and FALLBACK are
#: user-specific baselines. Offline/synthetic numbers — never product claims.
PRIMARY_METHOD = METHOD_WEEKDAY_PROFILE
FALLBACK_METHOD = METHOD_RECENT_MEAN


def recent_mean_level(series: DailySeries) -> float:
    """Mean daily spend over the most recent ``RECENT_LEVEL_DAYS`` days."""
    window = series.spend[-RECENT_LEVEL_DAYS:]
    return fmean(window) if window else 0.0


def weekday_profile_levels(series: DailySeries) -> list[float]:
    """Mean daily spend per weekday (Mon..Sun) over the recent 8 weeks."""
    cutoff = max(0, series.history_days - WEEKDAY_PROFILE_DAYS)
    buckets: list[list[float]] = [[] for _ in range(7)]
    for i in range(cutoff, series.history_days):
        buckets[series.dates[i].weekday()].append(series.spend[i])
    overall = fmean(series.spend[cutoff:]) if series.history_days > cutoff else 0.0
    # A weekday never observed falls back to the window's overall level, so the
    # profile stays user-specific and never borrows from another user.
    return [fmean(b) if b else overall for b in buckets]


def ewma_level(series: DailySeries, half_life_days: float = EWMA_HALF_LIFE_DAYS) -> float:
    """Exponentially weighted level of daily spend (recency-weighted mean)."""
    if not series.spend:
        return 0.0
    alpha = 1.0 - 0.5 ** (1.0 / half_life_days)
    level = series.spend[0]
    for value in series.spend[1:]:
        level = alpha * value + (1.0 - alpha) * level
    return level


def _levels_for(series: DailySeries, method: str) -> list[float]:
    """Seven per-weekday levels for ``method`` (index 0 = Monday)."""
    if method == METHOD_WEEKDAY_PROFILE:
        return weekday_profile_levels(series)
    if method == METHOD_RECENT_MEAN:
        value = recent_mean_level(series)
    elif method == METHOD_EWMA:
        value = ewma_level(series)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported daily method: {method}")
    return [value] * 7


def predict_daily(
    series: DailySeries,
    horizon_days: int,
    method: str,
    *,
    start: date | None = None,
) -> tuple[list[date], list[float]]:
    """Point forecasts for the next ``horizon_days`` days (user-specific only)."""
    if series.history_days == 0:
        raise ValueError("cannot forecast an empty series")
    levels = _levels_for(series, method)
    first = start if start is not None else series.last_date + timedelta(days=1)  # type: ignore[operator]
    dates = [first + timedelta(days=i) for i in range(horizon_days)]
    return dates, [round(max(levels[d.weekday()], 0.0), 2) for d in dates]


def _residual_scale(series: DailySeries, method: str, min_train: int = 7) -> float:
    """Dispersion of one-step errors, measured without leakage.

    For every history index ``i >= min_train`` the level is re-estimated from
    the strictly-prior days only, then compared with the observed value. The
    returned sigma drives the reported interval — it is a measured dispersion,
    not an assumed accuracy figure.
    """
    residuals: list[float] = []
    for i in range(min(min_train, series.history_days), series.history_days):
        past = DailySeries(
            dates=series.dates[:i],
            spend=series.spend[:i],
            transaction_count=series.transaction_count,
            distinct_spend_days=series.distinct_spend_days,
        )
        _, predicted = predict_daily(past, 1, method, start=series.dates[i])
        residuals.append(series.spend[i] - predicted[0])
    if len(residuals) < 2:
        return 0.0
    mean_r = fmean(residuals)
    return math.sqrt(fmean([(r - mean_r) ** 2 for r in residuals]))


def weekday_shares(series: DailySeries) -> list[float]:
    """Share of weekly spend per weekday (user's own history; sums to 1)."""
    totals = [0.0] * 7
    for i, value in enumerate(series.spend):
        totals[series.dates[i].weekday()] += value
    grand = sum(totals)
    if grand <= 0:
        return [1.0 / 7.0] * 7
    return [t / grand for t in totals]


def predict_daily_prophet(series: DailySeries, horizon_days: int) -> list[float] | None:
    """Prophet forecast at daily resolution, or ``None`` when unavailable.

    Prophet (as used elsewhere in this repository) models **weekly** totals. To
    compare and serve at daily resolution each projected week is distributed
    across its days using the *user's own* weekday shares — no global profile
    and no other user's data is involved. Returns ``None`` when Prophet is not
    importable so callers can fall back explicitly.
    """
    try:
        from prophet import Prophet  # lazy: keeps serving import-light
    except ImportError:  # pragma: no cover - depends on environment
        return None

    import pandas as pd

    weeks: dict[date, float] = {}
    for i, value in enumerate(series.spend):
        week_start = series.dates[i] - timedelta(days=series.dates[i].weekday())
        weeks[week_start] = weeks.get(week_start, 0.0) + value
    if len(weeks) < 4:
        return None

    frame = pd.DataFrame(
        {"ds": sorted(weeks), "y": [weeks[k] for k in sorted(weeks)]}
    )
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, interval_width=0.80)
    model.fit(frame)
    n_weeks = max(1, math.ceil(horizon_days / 7))
    future = model.make_future_dataframe(periods=n_weeks, freq="W")
    weekly = model.predict(future).tail(n_weeks)

    shares = weekday_shares(series)
    start = series.last_date + timedelta(days=1)  # type: ignore[operator]
    points: list[float] = []
    for _, row in weekly.iterrows():
        total = max(float(row["yhat"]), 0.0)
        for offset in range(7):
            if len(points) >= horizon_days:
                break
            shares_idx = (start + timedelta(days=offset)).weekday()
            points.append(round(total * shares[shares_idx], 2))
    return points[:horizon_days]


def predict(
    series: DailySeries,
    horizon_days: int,
    method: str | None = None,
    *,
    allow_prophet: bool = False,
) -> dict[str, object]:
    """Build a complete, user-specific forecast result.

    ``method=None`` uses ``PRIMARY_METHOD``; when that method cannot be computed
    the user-specific ``FALLBACK_METHOD`` is used and ``fallback_status`` says
    so. The reported interval is derived from measured one-step residual
    dispersion (``Z80``, 80% like the Prophet wrapper) and the **total** bounds
    are the sum of the daily bounds — a deliberately conservative assumption of
    correlated daily errors rather than a fabricated statistical claim.
    """
    chosen = method or PRIMARY_METHOD
    fallback = FALLBACK_NONE
    if chosen == METHOD_PROPHET and not allow_prophet:
        # Prophet is never served from the request path (cost/instability); say
        # so explicitly instead of silently returning a different method.
        chosen = PRIMARY_METHOD
        fallback = FALLBACK_PRIMARY_SUBSTITUTED
    if chosen == METHOD_WEEKDAY_PROFILE and series.history_days < 14:
        chosen, fallback = FALLBACK_METHOD, FALLBACK_USER_BASELINE

    if chosen == METHOD_PROPHET:
        points = predict_daily_prophet(series, horizon_days)
        if points is None or len(points) < horizon_days:
            chosen, fallback = FALLBACK_METHOD, FALLBACK_USER_BASELINE
            dates, points = predict_daily(series, horizon_days, chosen)
        else:
            start = series.last_date + timedelta(days=1)  # type: ignore[operator]
            dates = [start + timedelta(days=i) for i in range(horizon_days)]
    else:
        dates, points = predict_daily(series, horizon_days, chosen)

    sigma = _residual_scale(series, chosen if chosen != METHOD_PROPHET else METHOD_WEEKDAY_PROFILE)
    lower = [round(max(p - Z80 * sigma, 0.0), 2) for p in points]
    upper = [round(p + Z80 * sigma, 2) for p in points]

    total_p50 = round(sum(points), 2)
    total_p10 = round(sum(lower), 2)
    total_p90 = round(sum(upper), 2)
    tier = quality_tier(series)

    def _iso(d: date) -> str:
        return d.isoformat()

    return {
        "engine_version": FORECAST_ENGINE_VERSION,
        "method": chosen,
        "personalization_status": tier,
        "fallback_status": fallback,
        "interval_width": 0.80,
        "forecast_start": _iso(dates[0]) if dates else None,
        "forecast_end": _iso(dates[-1]) if dates else None,
        "horizon_days": horizon_days,
        "expected_spend": total_p50,
        "total": {"p10": total_p10, "p50": total_p50, "p90": total_p90},
        "daily": [
            {"date": _iso(d), "p50": p, "p10": lo, "p90": hi}
            for d, p, lo, hi in zip(dates, points, lower, upper)
        ],
        "history_days_used": series.history_days,
        "transaction_count_used": series.transaction_count,
        "distinct_spend_days_used": series.distinct_spend_days,
        "residual_sigma": round(sigma, 2),
    }


# ── Leakage-safe evaluation (STEP 13G) ──────────────────────────────────────
# Expanding-window (walk-forward) validation only. There is deliberately no
# random split anywhere: order matters in a time series, and shuffling days
# would let future information reach a model that is supposed to forecast it.

#: Minimum absolute actual used as a MAPE denominator. Zero-spend days make
#: plain MAPE undefined or explosive, so MAPE is reported only over points with
#: a meaningful denominator, and WAPE/MAE/RMSE carry the comparison.
MAPE_MIN_ACTUAL = 1.0


@dataclass
class ForecastMetrics:
    """Metrics for one method over one or more leakage-safe windows."""

    method: str
    n_windows: int
    n_points: int
    mae: float
    rmse: float
    wape: float | None
    mape: float | None
    bias: float
    per_window_wape: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "n_windows": self.n_windows,
            "n_points": self.n_points,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "wape_pct": None if self.wape is None else round(self.wape * 100, 4),
            "mape_pct": None if self.mape is None else round(self.mape * 100, 4),
            "bias": round(self.bias, 4),
        }


def _metrics_for(method: str, pairs: list[tuple[float, float]], windows: list[float]) -> ForecastMetrics:
    """Aggregate (actual, predicted) pairs into honest error metrics."""
    errors = [p - a for a, p in pairs]
    n = len(pairs)
    mae = fmean([abs(e) for e in errors]) if n else 0.0
    rmse = math.sqrt(fmean([e * e for e in errors])) if n else 0.0
    bias = fmean(errors) if n else 0.0
    sum_actual = sum(a for a, _ in pairs)
    wape = (sum(abs(e) for e in errors) / sum_actual) if sum_actual > 0 else None
    denominators = [
        abs(a) for a, _ in pairs if abs(a) >= MAPE_MIN_ACTUAL
    ]
    mape_terms = [
        abs(p - a) / abs(a) for a, p in pairs if abs(a) >= MAPE_MIN_ACTUAL
    ]
    mape = fmean(mape_terms) if denominators and mape_terms else None
    return ForecastMetrics(
        method=method,
        n_windows=len(windows),
        n_points=n,
        mae=mae,
        rmse=rmse,
        wape=wape,
        mape=mape,
        bias=bias,
        per_window_wape=windows,
    )


def walk_forward_evaluate(
    series: DailySeries,
    horizon_days: int,
    methods: tuple[str, ...] = METHODS,
    *,
    min_train_days: int = 21,
    step_days: int | None = None,
) -> list[ForecastMetrics]:
    """Evaluate methods on ``series`` using expanding-window validation.

    Folds are contiguous, non-overlapping test windows: each fold trains on
    every day strictly before the window and predicts exactly the window. ``step_days``
    defaults to ``horizon_days`` (non-overlapping windows), which keeps the
    sampled points independent of one another.
    """
    step = step_days or horizon_days
    results: list[ForecastMetrics] = []

    for method in methods:
        pairs: list[tuple[float, float]] = []
        window_wapes: list[float] = []
        for t0 in range(min_train_days, series.history_days - horizon_days + 1, step):
            train = DailySeries(
                dates=series.dates[:t0],
                spend=series.spend[:t0],
                transaction_count=series.transaction_count,
                distinct_spend_days=series.distinct_spend_days,
            )
            actual = series.spend[t0 : t0 + horizon_days]
            if method == METHOD_PROPHET:
                predicted = predict_daily_prophet(train, horizon_days)
                if predicted is None or len(predicted) < len(actual):
                    continue
            else:
                _, predicted = predict_daily(
                    train, len(actual), method, start=series.dates[t0]
                )
            fold_pairs = list(zip(actual, predicted))
            pairs.extend(fold_pairs)
            fold_actual = sum(a for a, _ in fold_pairs)
            if fold_actual > 0:
                window_wapes.append(
                    sum(abs(p - a) for a, p in fold_pairs) / fold_actual
                )
        results.append(_metrics_for(method, pairs, window_wapes))

    return results


def summarize(metrics: list[ForecastMetrics]) -> list[dict[str, object]]:
    """Compact, report-ready view of a set of metrics (sorted by WAPE)."""
    ranked = sorted(
        metrics,
        key=lambda m: (m.wape if m.wape is not None else float("inf")),
    )
    return [m.as_dict() for m in ranked]
