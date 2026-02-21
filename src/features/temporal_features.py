"""
Temporal feature extraction for transactions.

Implements SPEC §5.2.3:
  - Cyclical sin/cos encodings for hour, day-of-week, day-of-month
  - Boolean flags: is_weekend, is_holiday
  - Month phase (early / mid / late)
  - days_since_payday (requires income-cycle information)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Cyclical encoding ──────────────────────────────────────────────────────────


def cyclical_encode(values: pd.Series, period: int) -> pd.DataFrame:
    """
    Encode a numeric series as sin/cos pair with the given period.

    Returns a DataFrame with columns ``{name}_sin`` and ``{name}_cos``.
    """
    name = values.name or "value"
    radians = 2 * np.pi * values / period
    return pd.DataFrame(
        {
            f"{name}_sin": np.sin(radians),
            f"{name}_cos": np.cos(radians),
        },
        index=values.index,
    )


# ── Boolean flags ──────────────────────────────────────────────────────────────


def is_weekend(ts: pd.Series) -> pd.Series:
    """True for Saturday (5) and Sunday (6)."""
    return ts.dt.dayofweek.isin([5, 6]).astype(int)


def is_holiday(
    ts: pd.Series,
    country: str = "US",
) -> pd.Series:
    """
    True if the transaction date falls on a public holiday.

    Uses the ``holidays`` library when available; falls back to a
    hard-coded set of major US holidays.
    """
    try:
        import holidays as holidays_lib

        cal = holidays_lib.country_holidays(country)
        return ts.dt.date.map(lambda d: int(d in cal))
    except ImportError:
        # Fallback: US federal holidays (approximate)
        logger.debug("holidays library not installed — using fallback set.")
        return pd.Series(0, index=ts.index, dtype=int)


# ── Month phase ────────────────────────────────────────────────────────────────


def month_phase(day_of_month: pd.Series) -> pd.Series:
    """
    Categorise day of month into ``early`` (1–10), ``mid`` (11–20),
    ``late`` (21–31).
    """
    return pd.cut(
        day_of_month,
        bins=[0, 10, 20, 31],
        labels=["early", "mid", "late"],
        right=True,
        include_lowest=True,
    )


# ── Days since payday ─────────────────────────────────────────────────────────


def days_since_payday(
    timestamps: pd.Series,
    pay_dates: pd.Series | None = None,
    default_paydays: list[int] | None = None,
) -> pd.Series:
    """
    Compute the number of days since the most recent payday.

    Parameters
    ----------
    timestamps : pd.Series[datetime]
        Transaction timestamps.
    pay_dates : pd.Series[datetime] | None
        Actual detected income deposit dates per user.  If ``None``,
        falls back to ``default_paydays``.
    default_paydays : list[int] | None
        Day-of-month assumptions when no detected pay cycle is available
        (default: 1st and 15th).
    """
    if default_paydays is None:
        default_paydays = [1, 15]

    if pay_dates is not None and not pay_dates.empty:
        # Actual pay dates available — compute difference
        result = pd.Series(np.nan, index=timestamps.index)
        for i, ts in timestamps.items():
            past = pay_dates[pay_dates <= ts]
            if past.empty:
                result.at[i] = 0
            else:
                result.at[i] = (ts - past.iloc[-1]).days
        return result.astype(int)

    # Heuristic fallback: use default_paydays within the same month
    def _days_since(ts: pd.Timestamp) -> int:
        dom = ts.day
        past_paydays = [d for d in default_paydays if d <= dom]
        if past_paydays:
            return dom - max(past_paydays)
        # Wrap to previous month's last payday
        return dom + (31 - max(default_paydays))

    return timestamps.map(_days_since).astype(int)


# ── Convenience: extract all temporal features ─────────────────────────────────


def extract_temporal_features(
    df: pd.DataFrame,
    *,
    country: str = "US",
    pay_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute all temporal features and return as a DataFrame aligned to
    the input index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``timestamp`` column (datetime64).
    country : str
        ISO country code for holiday detection.
    pay_dates : pd.Series | None
        Optional detected payday timestamps.
    """
    ts = pd.to_datetime(df["timestamp"])

    parts: list[pd.DataFrame | pd.Series] = []

    # Cyclical encodings
    hour = ts.dt.hour.rename("hour_of_day")
    parts.append(cyclical_encode(hour, period=24))

    dow = ts.dt.dayofweek.rename("day_of_week")
    parts.append(cyclical_encode(dow, period=7))

    dom = ts.dt.day.rename("day_of_month")
    parts.append(cyclical_encode(dom, period=31))

    # Boolean flags
    parts.append(is_weekend(ts).rename("is_weekend"))
    parts.append(is_holiday(ts, country=country).rename("is_holiday"))

    # Month phase (one-hot encoded for numerical models)
    phase = month_phase(dom)
    phase_dummies = pd.get_dummies(phase, prefix="month_phase", dtype=int)
    parts.append(phase_dummies)

    # Days since payday
    parts.append(
        days_since_payday(ts, pay_dates=pay_dates).rename("days_since_payday")
    )

    return pd.concat(parts, axis=1).set_index(df.index)
