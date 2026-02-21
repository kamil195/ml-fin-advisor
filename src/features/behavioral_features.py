"""
Behavioral feature extraction — glue between the Behavior Modeling Layer
(``src.models.behavior``) and the feature pipeline.

Implements SPEC §5.2.4:
  - spending_regime
  - impulse_score
  - habit_strength
  - income_cycle_phase
  - lifestyle_drift_30d
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_spending_regime(
    df: pd.DataFrame,
    baseline_window: int = 90,
    comparison_window: int = 30,
) -> pd.Series:
    """
    Assign a spending regime label per row based on recent vs. baseline
    spend per user × category.

    Labels: ``normal``, ``elevated``, ``reduced``, ``irregular``.

    This is a simplified heuristic; the full Bayesian changepoint detector
    lives in ``src.models.behavior.regime_detector``.
    """
    regimes = pd.Series("normal", index=df.index)

    if "category_l2" not in df.columns or "amount" not in df.columns:
        return regimes

    sorted_df = df.sort_values(["user_id", "timestamp"])

    for (uid, cat), grp in sorted_df.groupby(["user_id", "category_l2"]):
        if len(grp) < 4:
            continue
        ts = pd.to_datetime(grp["timestamp"])
        amounts = grp["amount"].abs()
        latest = ts.max()

        baseline_mask = ts >= (latest - pd.Timedelta(days=baseline_window))
        comparison_mask = ts >= (latest - pd.Timedelta(days=comparison_window))

        baseline_mean = amounts[baseline_mask].mean()
        comparison_mean = amounts[comparison_mask].mean()
        baseline_std = amounts[baseline_mask].std(ddof=0)

        if baseline_std == 0 or np.isnan(baseline_std):
            continue

        z = (comparison_mean - baseline_mean) / baseline_std
        idx = grp.index

        if z > 2.0:
            regimes.loc[idx] = "elevated"
        elif z < -2.0:
            regimes.loc[idx] = "reduced"
        elif amounts[comparison_mask].std(ddof=0) > 2 * baseline_std:
            regimes.loc[idx] = "irregular"

    return regimes


def compute_impulse_score(df: pd.DataFrame) -> pd.Series:
    """
    Estimate impulse probability per transaction (SPEC §9.2.2).

    Uses a weighted heuristic across six signals. The full logistic
    regression model lives in ``src.models.behavior.impulse_scorer``.
    """
    from src.utils.constants import DISCRETIONARY_CATEGORIES

    scores = pd.Series(0.0, index=df.index)
    sorted_df = df.sort_values(["user_id", "timestamp"])

    for uid, grp in sorted_df.groupby("user_id"):
        idx = grp.index
        ts = pd.to_datetime(grp["timestamp"])
        amounts = grp["amount"].abs()
        median_amount = amounts.median()

        # Signal 1: unusual hour (weight 0.15)
        hour = ts.dt.hour
        user_median_hour = hour.median()
        hour_deviation = (hour - user_median_hour).abs() / 12.0
        s1 = hour_deviation.clip(upper=1.0) * 0.15

        # Signal 2: amount above median (weight 0.20)
        s2 = ((amounts / median_amount.clip(lower=1.0)) - 1.0).clip(0, 3) / 3.0 * 0.20

        # Signal 3: discretionary category (weight 0.15)
        s3 = pd.Series(0.0, index=idx)
        if "category_l2" in grp.columns:
            s3 = grp["category_l2"].map(
                lambda c: 0.15 if c in DISCRETIONARY_CATEGORIES else 0.0
            )

        # Signal 4: novel merchant (weight 0.15)
        merchant_counts = grp["merchant_name"].map(
            grp["merchant_name"].value_counts()
        )
        s4 = (merchant_counts == 1).astype(float) * 0.15

        # Signal 5: short gap since last same merchant (weight 0.20)
        s5 = pd.Series(0.0, index=idx)
        for merchant, mgrp in grp.groupby("merchant_name"):
            if len(mgrp) < 2:
                continue
            mts = pd.to_datetime(mgrp["timestamp"])
            gaps = mts.diff().dt.total_seconds() / 3600  # hours
            short_gap = (gaps < 48).fillna(False).astype(float) * 0.20
            s5.loc[mgrp.index] = short_gap

        # Signal 6: close to payday (weight 0.15)
        dom = ts.dt.day
        near_payday = ((dom <= 3) | ((dom >= 14) & (dom <= 17))).astype(float) * 0.15

        total = s1 + s2.values + s3.values + s4.values + s5.values + near_payday.values
        scores.loc[idx] = total.clip(0.0, 1.0).values

    return scores


def compute_habit_strength(df: pd.DataFrame) -> pd.Series:
    """
    Compute habit strength index per transaction (SPEC §9.2.3).

    H(c, u) = α · Recurrence + β · Consistency + γ · Duration
    """
    alpha, beta, gamma = 0.40, 0.35, 0.25
    duration_cap_months = 12

    habit = pd.Series(0.0, index=df.index)
    sorted_df = df.sort_values(["user_id", "timestamp"])

    for (uid, cat), grp in sorted_df.groupby(["user_id", "category_l2"]):
        if len(grp) < 2:
            continue
        idx = grp.index
        ts = pd.to_datetime(grp["timestamp"])

        # Recurrence: frequency relative to expected (1 txn/week as baseline)
        span_days = (ts.max() - ts.min()).days or 1
        expected_weekly = span_days / 7.0
        actual = len(grp)
        recurrence = min(actual / max(expected_weekly, 1.0), 1.0)

        # Consistency: 1 − CV of inter-purchase intervals
        if len(grp) >= 3:
            gaps = ts.diff().dt.total_seconds().dropna()
            cv = gaps.std() / gaps.mean() if gaps.mean() > 0 else 1.0
            consistency = max(1.0 - cv, 0.0)
        else:
            consistency = 0.0

        # Duration: months since first purchase, capped
        duration_months = min(span_days / 30.0, duration_cap_months) / duration_cap_months

        h = alpha * recurrence + beta * consistency + gamma * duration_months
        habit.loc[idx] = round(min(h, 1.0), 4)

    return habit


def compute_income_cycle_phase(
    df: pd.DataFrame,
    pay_period_days: int = 15,
) -> pd.Series:
    """
    Normalised position within the detected pay cycle (0.0 – 1.0).

    Uses a simple heuristic assuming bi-monthly pay (1st & 15th).
    The full detector lives in ``src.models.behavior.income_cycle``.
    """
    ts = pd.to_datetime(df["timestamp"])
    dom = ts.dt.day

    # Distance from nearest payday (1st or 15th)
    dist_from_1 = (dom - 1) % pay_period_days
    dist_from_15 = (dom - 15) % pay_period_days
    days_since = np.minimum(dist_from_1, dist_from_15)
    return (days_since / pay_period_days).clip(0.0, 1.0)


def compute_lifestyle_drift(
    df: pd.DataFrame,
    baseline_window: int = 90,
    comparison_window: int = 30,
) -> pd.Series:
    """
    Percentage change in median category spend vs. 90-day baseline (SPEC §9.2.5).
    """
    drift = pd.Series(0.0, index=df.index)
    sorted_df = df.sort_values(["user_id", "timestamp"])

    for (uid, cat), grp in sorted_df.groupby(["user_id", "category_l2"]):
        if len(grp) < 4:
            continue
        idx = grp.index
        ts = pd.to_datetime(grp["timestamp"])
        amounts = grp["amount"].abs()
        latest = ts.max()

        baseline_mask = ts >= (latest - pd.Timedelta(days=baseline_window))
        comparison_mask = ts >= (latest - pd.Timedelta(days=comparison_window))

        baseline_median = amounts[baseline_mask].median()
        comparison_median = amounts[comparison_mask].median()

        if baseline_median > 0:
            pct_change = (comparison_median - baseline_median) / baseline_median
            drift.loc[idx] = round(pct_change, 4)

    return drift


# ── Convenience: extract all behavioral features ──────────────────────────────


def extract_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all behavioral features and return as a DataFrame.
    """
    features = pd.DataFrame(index=df.index)
    features["spending_regime"] = compute_spending_regime(df)
    features["impulse_score"] = compute_impulse_score(df)
    features["habit_strength"] = compute_habit_strength(df)
    features["income_cycle_phase"] = compute_income_cycle_phase(df)
    features["lifestyle_drift_30d"] = compute_lifestyle_drift(df)
    return features
