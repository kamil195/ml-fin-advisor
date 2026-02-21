"""
Income Cycle Detection & Alignment (SPEC §9.2.4).

Detects income deposits via amount clustering + recurrence,
then computes a normalised position within the detected pay cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IncomeCycleResult:
    """Detected income cycle parameters for a single user."""

    user_id: str
    pay_frequency_days: float
    detected_pay_dates: list[str] = field(default_factory=list)
    confidence: float = 0.0
    spender_type: str = "even"  # front-loaded | even | end-loaded


class IncomeCycleDetector:
    """
    Detect recurring income deposits and compute pay-cycle alignment.

    Parameters
    ----------
    min_income_amount : float
        Minimum credit amount to consider as potential income.
    min_recurrences : int
        Minimum number of similar-amount credits to detect a cycle.
    amount_tolerance_pct : float
        Percentage tolerance for clustering similar income amounts.
    """

    def __init__(
        self,
        min_income_amount: float = 500.0,
        min_recurrences: int = 2,
        amount_tolerance_pct: float = 0.10,
    ) -> None:
        self.min_income_amount = min_income_amount
        self.min_recurrences = min_recurrences
        self.amount_tolerance_pct = amount_tolerance_pct

    def _detect_income_deposits(
        self,
        df: pd.DataFrame,
        user_id: str,
    ) -> pd.DataFrame:
        """Identify likely income deposits for a user."""
        mask = (df["user_id"] == user_id) & (df["amount"] > self.min_income_amount)
        credits = df.loc[mask].sort_values("timestamp").copy()

        if len(credits) < self.min_recurrences:
            return pd.DataFrame()

        # Cluster by amount (within tolerance)
        credits["amount_bin"] = pd.cut(
            credits["amount"],
            bins=max(1, len(credits) // 2),
            labels=False,
        )

        # Find the most frequent amount cluster
        best_cluster = credits["amount_bin"].mode()
        if len(best_cluster) == 0:
            return pd.DataFrame()

        income = credits[credits["amount_bin"] == best_cluster.iloc[0]]

        if len(income) < self.min_recurrences:
            # Fall back: use all large credits
            income = credits

        return income

    def _estimate_pay_frequency(self, pay_dates: pd.Series) -> float:
        """Estimate the inter-pay-period in days."""
        ts = pd.to_datetime(pay_dates)
        if len(ts) < 2:
            return 30.0  # default to monthly

        gaps = ts.diff().dt.days.dropna()
        median_gap = float(gaps.median())

        # Round to nearest common frequency
        common_frequencies = [7, 14, 15, 30, 31]
        closest = min(common_frequencies, key=lambda f: abs(f - median_gap))
        return float(closest)

    def _classify_spender_type(
        self,
        df: pd.DataFrame,
        user_id: str,
        pay_freq: float,
    ) -> str:
        """
        Classify whether a user is front-loaded, even, or end-loaded
        relative to their pay cycle.
        """
        mask = (df["user_id"] == user_id) & (df["amount"] < 0)
        debits = df.loc[mask].copy()

        if len(debits) < 5:
            return "even"

        ts = pd.to_datetime(debits["timestamp"])
        dom = ts.dt.day

        # Split month into thirds
        early = (dom <= 10).sum()
        mid = ((dom > 10) & (dom <= 20)).sum()
        late = (dom > 20).sum()
        total = early + mid + late

        if total == 0:
            return "even"

        early_pct = early / total
        late_pct = late / total

        if early_pct > 0.45:
            return "front-loaded"
        elif late_pct > 0.45:
            return "end-loaded"
        return "even"

    def detect(
        self,
        df: pd.DataFrame,
        user_id: str,
    ) -> IncomeCycleResult:
        """
        Detect income cycle for a single user.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``user_id``, ``timestamp``, ``amount``.
        """
        income = self._detect_income_deposits(df, user_id)

        if income.empty:
            return IncomeCycleResult(
                user_id=user_id,
                pay_frequency_days=30.0,
                confidence=0.0,
                spender_type="even",
            )

        pay_dates = pd.to_datetime(income["timestamp"])
        pay_freq = self._estimate_pay_frequency(pay_dates)
        spender_type = self._classify_spender_type(df, user_id, pay_freq)

        # Confidence based on regularity
        if len(income) >= 3:
            gaps = pay_dates.diff().dt.days.dropna()
            cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 1.0
            confidence = max(1.0 - cv, 0.2)
        else:
            confidence = 0.3

        return IncomeCycleResult(
            user_id=user_id,
            pay_frequency_days=pay_freq,
            detected_pay_dates=[str(d.date()) for d in pay_dates],
            confidence=round(confidence, 3),
            spender_type=spender_type,
        )

    def compute_cycle_phase(
        self,
        df: pd.DataFrame,
        user_id: str | None = None,
    ) -> pd.Series:
        """
        Compute normalised cycle position (0.0–1.0) for all transactions.

        If ``user_id`` is supplied, only process that user.
        """
        phase = pd.Series(0.0, index=df.index)

        user_ids = [user_id] if user_id else df["user_id"].unique()

        for uid in user_ids:
            cycle = self.detect(df, uid)
            mask = df["user_id"] == uid
            subset = df.loc[mask]
            ts = pd.to_datetime(subset["timestamp"])

            if cycle.detected_pay_dates:
                pay_ts = pd.to_datetime(cycle.detected_pay_dates)
                # For each transaction, find days since most recent pay date
                for idx, t in ts.items():
                    past = pay_ts[pay_ts <= t]
                    if len(past) > 0:
                        days_since = (t - past.max()).days
                    else:
                        days_since = t.day  # fallback
                    phase.at[idx] = min(
                        days_since / max(cycle.pay_frequency_days, 1.0), 1.0
                    )
            else:
                # Fallback: simple day-of-month heuristic
                dom = ts.dt.day
                phase.loc[mask] = (dom / 31.0).clip(0.0, 1.0).values

        return phase

    def detect_all(
        self,
        df: pd.DataFrame,
    ) -> list[IncomeCycleResult]:
        """Detect income cycles for all users."""
        results = []
        for uid in df["user_id"].unique():
            results.append(self.detect(df, uid))
        logger.info("Detected income cycles for %d users.", len(results))
        return results
