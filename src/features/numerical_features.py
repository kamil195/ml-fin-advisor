"""
Numerical feature extraction for transactions.

Implements SPEC §5.2.2:
  - log_amount, amount_zscore_user, amount_pct_of_income
  - rolling_spend_7d, rolling_spend_30d
  - txn_count_24h
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Individual transforms ──────────────────────────────────────────────────────


def log_amount(amount: pd.Series) -> pd.Series:
    """``log1p(abs(amount))`` — compresses the heavy tail."""
    return np.log1p(amount.abs())


def amount_zscore_user(df: pd.DataFrame) -> pd.Series:
    """
    Z-score of transaction amount within each user's historical distribution.
    """

    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    return df.groupby("user_id")["amount"].transform(_zscore)


def amount_pct_of_income(
    amount: pd.Series,
    estimated_monthly_income: pd.Series | float,
) -> pd.Series:
    """``abs(amount) / estimated_monthly_income``."""
    income = (
        estimated_monthly_income
        if isinstance(estimated_monthly_income, pd.Series)
        else pd.Series(estimated_monthly_income, index=amount.index)
    )
    return amount.abs() / income.clip(lower=1.0)  # avoid div-by-zero


def rolling_spend(
    df: pd.DataFrame,
    window_days: int = 7,
) -> pd.Series:
    """
    Sum of debits (negative amounts) in a trailing window per user.
    DataFrame **must** be sorted by ``(user_id, timestamp)``.
    """
    debits = df["amount"].clip(upper=0).abs()
    result = pd.Series(0.0, index=df.index)

    for uid, grp in df.groupby("user_id"):
        idx = grp.index
        ts = grp["timestamp"]
        vals = debits.loc[idx]
        # pandas rolling with a time-based window needs a DatetimeIndex
        tmp = pd.Series(vals.values, index=pd.DatetimeIndex(ts))
        rolling = tmp.rolling(f"{window_days}D", min_periods=1).sum()
        result.loc[idx] = rolling.values

    return result


def calculate_velocity(
    df: pd.DataFrame,
    window_minutes: int = 10,
    time_column: str = "timestamp",
) -> pd.Series:
    """
    Per-user transaction velocity: number of *preceding* transactions within
    a trailing time window.

    Velocity is the classic fraud signal — a burst of transactions inside a
    short window is a red flag regardless of individual amounts. This is used
    both as a batch feature and by the serving layer's fraud assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``user_id`` and the time column.
    window_minutes : int
        Trailing window width. Defaults to 10 minutes.
    time_column : str
        Column holding transaction timestamps (naive or tz-aware).

    Returns
    -------
    pd.Series
        Integer count of prior transactions within the window, aligned to
        ``df.index``. A transaction with no history gets 0 (the current row
        is excluded, so a single transaction is never flagged).
    """
    work = df.sort_values(["user_id", time_column])
    result = pd.Series(0, index=df.index, dtype=int)

    for uid, grp in work.groupby("user_id", sort=False):
        idx = grp.index
        ts = pd.to_datetime(grp[time_column], utc=True)
        # rolling count of txns in the trailing window (includes the current
        # row), then subtract 1 so the current row is excluded (prior activity
        # within the window only)
        ones = pd.Series(1, index=pd.DatetimeIndex(ts))
        counts = ones.rolling(f"{window_minutes}min", min_periods=1).sum()
        prior = (counts - 1).clip(lower=0)
        result.loc[idx] = prior.astype(int).values

    return result.reindex(df.index).fillna(0).astype(int)


def txn_count_24h(df: pd.DataFrame) -> pd.Series:
    """
    Number of transactions in the preceding 24 hours for each user.
    DataFrame **must** be sorted by ``(user_id, timestamp)``.
    """
    result = pd.Series(0, index=df.index, dtype=int)

    for uid, grp in df.groupby("user_id"):
        idx = grp.index
        ts = grp["timestamp"]
        ones = pd.Series(1, index=pd.DatetimeIndex(ts))
        counts = ones.rolling("24h", min_periods=1).sum()
        result.loc[idx] = counts.values.astype(int)

    return result


# ── Convenience: extract all numerical features ───────────────────────────────


def extract_numerical_features(
    df: pd.DataFrame,
    estimated_monthly_income: pd.Series | float = 5_000.0,
) -> pd.DataFrame:
    """
    Compute all numerical features and return as a DataFrame aligned to
    the input index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``user_id``, ``timestamp``, ``amount``.
        Should be sorted by ``(user_id, timestamp)`` for rolling features.
    estimated_monthly_income : pd.Series | float
        Per-user monthly income estimate. A scalar applies the same value
        to every row.
    """
    sorted_df = df.sort_values(["user_id", "timestamp"])

    features = pd.DataFrame(index=sorted_df.index)
    features["log_amount"] = log_amount(sorted_df["amount"])
    features["amount_zscore_user"] = amount_zscore_user(sorted_df)
    features["amount_pct_of_income"] = amount_pct_of_income(
        sorted_df["amount"], estimated_monthly_income
    )
    features["rolling_spend_7d"] = rolling_spend(sorted_df, window_days=7)
    features["rolling_spend_30d"] = rolling_spend(sorted_df, window_days=30)
    features["txn_count_24h"] = txn_count_24h(sorted_df)

    # Re-align to original index ordering
    return features.loc[df.index]
