"""
Impulse spending scorer (SPEC §9.2.2).

Estimates the probability that a given transaction is impulsive (unplanned)
using a logistic-regression model over six interpretable signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.constants import DISCRETIONARY_CATEGORIES

logger = logging.getLogger(__name__)


# Signal weights from SPEC §9.2.2
SIGNAL_WEIGHTS = {
    "repeat_merchant_short_gap": 0.20,
    "unusual_hour": 0.15,
    "amount_above_median": 0.20,
    "near_payday": 0.15,
    "discretionary_category": 0.15,
    "novel_merchant": 0.15,
}


@dataclass
class ImpulseSignals:
    """Raw signal values for a single transaction."""

    repeat_merchant_short_gap: float = 0.0
    unusual_hour: float = 0.0
    amount_above_median: float = 0.0
    near_payday: float = 0.0
    discretionary_category: float = 0.0
    novel_merchant: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.repeat_merchant_short_gap,
            self.unusual_hour,
            self.amount_above_median,
            self.near_payday,
            self.discretionary_category,
            self.novel_merchant,
        ])


class ImpulseScorer:
    """
    Logistic-regression model for impulse-purchase detection.

    Uses six interpretable signals specified in SPEC §9.2.2.
    Supports both a heuristic (weighted-sum) mode and a trained model.

    Parameters
    ----------
    threshold : float
        Classification threshold for the ``is_impulse`` label.
    use_model : bool
        If True, use trained logistic regression; otherwise weighted sum.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_model: bool = False,
    ) -> None:
        self.threshold = threshold
        self.use_model = use_model
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None

    # ── Signal extraction ──────────────────────────────────────────────────

    def _extract_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the six impulse signals for every row."""
        sorted_df = df.sort_values(["user_id", "timestamp"]).copy()
        ts = pd.to_datetime(sorted_df["timestamp"])
        signals = pd.DataFrame(index=sorted_df.index, dtype=float)

        # 1) Repeat merchant short gap (< 48h)
        signals["repeat_merchant_short_gap"] = 0.0
        for (uid, merchant), grp in sorted_df.groupby(
            ["user_id", "merchant_name"]
        ):
            if len(grp) < 2:
                continue
            mts = pd.to_datetime(grp["timestamp"])
            gaps_hours = mts.diff().dt.total_seconds() / 3600
            signals.loc[grp.index, "repeat_merchant_short_gap"] = (
                (gaps_hours < 48).fillna(False).astype(float)
            )

        # 2) Unusual hour
        hour = ts.dt.hour
        user_median_hour = sorted_df.groupby("user_id")["timestamp"].transform(
            lambda s: pd.to_datetime(s).dt.hour.median()
        )
        deviation = (hour - user_median_hour).abs() / 12.0
        signals["unusual_hour"] = deviation.clip(upper=1.0)

        # 3) Amount above user median for that category
        amounts = sorted_df["amount"].abs()
        if "category_l2" in sorted_df.columns:
            group_cols = ["user_id", "category_l2"]
        else:
            group_cols = ["user_id"]
        median_amt = sorted_df.groupby(group_cols)["amount"].transform(
            lambda s: s.abs().median()
        )
        ratio = (amounts / median_amt.clip(lower=1.0)) - 1.0
        signals["amount_above_median"] = ratio.clip(0, 3) / 3.0

        # 4) Near payday (within 3 days of 1st or 15th)
        dom = ts.dt.day
        signals["near_payday"] = (
            (dom <= 3) | ((dom >= 14) & (dom <= 17))
        ).astype(float)

        # 5) Discretionary category
        signals["discretionary_category"] = 0.0
        if "category_l2" in sorted_df.columns:
            signals["discretionary_category"] = sorted_df["category_l2"].map(
                lambda c: 1.0 if c in DISCRETIONARY_CATEGORIES else 0.0
            )

        # 6) Novel merchant (first visit)
        merchant_counts = sorted_df.groupby("user_id")["merchant_name"].transform(
            "count"
        )
        first_visit = sorted_df.groupby(["user_id", "merchant_name"]).cumcount() == 0
        signals["novel_merchant"] = first_visit.astype(float)

        return signals.loc[df.index]

    # ── Scoring ────────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute impulse scores for all transactions.

        Returns a DataFrame with columns ``impulse_score`` and ``is_impulse``.
        """
        signals = self._extract_signals(df)

        if self.use_model and self._model is not None:
            X = self._scaler.transform(signals.values)
            probs = self._model.predict_proba(X)[:, 1]
        else:
            # Weighted-sum heuristic
            weights = np.array(list(SIGNAL_WEIGHTS.values()))
            probs = (signals.values * weights).sum(axis=1)

        result = pd.DataFrame(index=df.index)
        result["impulse_score"] = np.clip(probs, 0.0, 1.0)
        result["is_impulse"] = result["impulse_score"] >= self.threshold
        return result

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> ImpulseScorer:
        """
        Train the logistic-regression model on labelled data.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data.
        labels : pd.Series
            Binary labels (1 = impulse, 0 = planned).
        """
        signals = self._extract_signals(df)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(signals.values)

        self._model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        self._model.fit(X, labels.values)
        self.use_model = True

        logger.info(
            "ImpulseScorer fitted — coefficients: %s",
            dict(zip(SIGNAL_WEIGHTS.keys(), self._model.coef_[0].round(3))),
        )
        return self
