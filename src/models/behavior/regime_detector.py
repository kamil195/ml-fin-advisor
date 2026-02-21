"""
Bayesian Online Changepoint Detection (BOCPD) for spending regime shifts.

Implements SPEC §9.2.1:
  - States: normal, elevated, reduced, irregular
  - Hazard function: constant λ = 1/60
  - Observation model: Gaussian with Normal-Inverse-Gamma conjugate prior
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeLabel = Literal["normal", "elevated", "reduced", "irregular"]


@dataclass
class RegimeResult:
    """Result of regime detection for a single user × category."""

    user_id: str
    category: str
    current_regime: RegimeLabel
    regime_probability: float
    change_date: str | None = None
    history: list[dict] = field(default_factory=list)


class BayesianChangepointDetector:
    """
    Online Bayesian Changepoint Detection (BOCPD) for detecting regime
    shifts in per-category weekly spend time-series.

    Parameters
    ----------
    hazard_lambda : float
        Inverse of expected regime duration in days.  Default ``1/60``
        means we expect regimes to last ~60 days.
    min_regime_length_days : int
        Minimum number of days before a new regime can be declared.
    """

    def __init__(
        self,
        hazard_lambda: float = 1.0 / 60.0,
        min_regime_length_days: int = 14,
    ) -> None:
        self.hazard_lambda = hazard_lambda
        self.min_regime_length_days = min_regime_length_days

    def _hazard(self, r: int) -> float:
        """Constant hazard function: P(run length = r) = λ."""
        return self.hazard_lambda

    def _detect_changepoints(
        self,
        series: np.ndarray,
    ) -> list[int]:
        """
        Run BOCPD on a 1-D series and return changepoint indices.

        Uses a simplified version with conjugate Normal-Inverse-Gamma
        updates.
        """
        n = len(series)
        if n < 3:
            return []

        # Prior parameters (Normal-Inverse-Gamma)
        mu0 = float(series.mean())
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = float(series.var() + 1e-6)

        # Run-length posterior: R[t, r] = P(r_t = r | x_{1:t})
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0

        # Sufficient statistics per run length
        mu = np.full(n + 1, mu0)
        kappa = np.full(n + 1, kappa0)
        alpha = np.full(n + 1, alpha0)
        beta_arr = np.full(n + 1, beta0)

        changepoints: list[int] = []

        for t in range(1, n + 1):
            x = series[t - 1]

            # Predictive probability under each run length
            pred_var = beta_arr[:t] * (kappa[:t] + 1) / (alpha[:t] * kappa[:t])
            pred_var = np.maximum(pred_var, 1e-10)
            pred_prob = (
                1.0
                / np.sqrt(2 * np.pi * pred_var)
                * np.exp(-0.5 * (x - mu[:t]) ** 2 / pred_var)
            )

            # Growth probabilities
            growth = R[t - 1, :t] * pred_prob * (1 - self.hazard_lambda)

            # Changepoint probability
            cp = np.sum(R[t - 1, :t] * pred_prob * self.hazard_lambda)

            # Update run-length distribution
            R[t, 1 : t + 1] = growth
            R[t, 0] = cp

            # Normalise
            evidence = R[t, : t + 1].sum()
            if evidence > 0:
                R[t, : t + 1] /= evidence

            # Detect changepoint: run-length 0 has high posterior mass
            if R[t, 0] > 0.5 and t > self.min_regime_length_days // 7:
                changepoints.append(t - 1)

            # Update sufficient statistics
            mu_new = (kappa[:t] * mu[:t] + x) / (kappa[:t] + 1)
            kappa_new = kappa[:t] + 1
            alpha_new = alpha[:t] + 0.5
            beta_new = (
                beta_arr[:t]
                + 0.5 * kappa[:t] * (x - mu[:t]) ** 2 / (kappa[:t] + 1)
            )

            mu[1 : t + 1] = mu_new
            kappa[1 : t + 1] = kappa_new
            alpha[1 : t + 1] = alpha_new
            beta_arr[1 : t + 1] = beta_new

            # Reset for new run
            mu[0] = mu0
            kappa[0] = kappa0
            alpha[0] = alpha0
            beta_arr[0] = beta0

        return changepoints

    def _classify_regime(
        self,
        current_segment: np.ndarray,
        baseline: np.ndarray,
    ) -> RegimeLabel:
        """Classify regime based on current vs. baseline statistics."""
        if len(current_segment) < 2 or len(baseline) < 2:
            return "normal"

        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std(ddof=0))
        current_mean = float(current_segment.mean())
        current_std = float(current_segment.std(ddof=0))

        if baseline_std < 1e-6:
            return "normal"

        z = (current_mean - baseline_mean) / baseline_std

        if current_std > 2 * baseline_std:
            return "irregular"
        elif z > 2.0:
            return "elevated"
        elif z < -2.0:
            return "reduced"
        return "normal"

    def detect(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
    ) -> RegimeResult:
        """
        Detect spending regime for a single user × category.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with ``user_id``, ``category_l2``,
            ``timestamp``, ``amount``.
        user_id : str
            Target user.
        category : str
            Target L2 category.
        """
        mask = (df["user_id"] == user_id)
        if "category_l2" in df.columns:
            mask = mask & (df["category_l2"] == category)

        subset = df.loc[mask].sort_values("timestamp").copy()

        if len(subset) < 4:
            return RegimeResult(
                user_id=user_id,
                category=category,
                current_regime="normal",
                regime_probability=1.0,
            )

        # Aggregate to weekly spend
        subset["week"] = pd.to_datetime(subset["timestamp"]).dt.to_period("W")
        weekly = subset.groupby("week")["amount"].apply(lambda s: s.abs().sum())
        series = weekly.values.astype(float)

        changepoints = self._detect_changepoints(series)

        # Classify current regime
        if changepoints:
            last_cp = changepoints[-1]
            current = series[last_cp:]
            baseline = series[:last_cp]
            change_date = str(weekly.index[last_cp])
        else:
            current = series
            baseline = series
            change_date = None

        regime = self._classify_regime(current, baseline)

        return RegimeResult(
            user_id=user_id,
            category=category,
            current_regime=regime,
            regime_probability=0.85,  # placeholder confidence
            change_date=change_date,
        )

    def detect_all(
        self,
        df: pd.DataFrame,
    ) -> list[RegimeResult]:
        """Run regime detection for all user × category combinations."""
        results: list[RegimeResult] = []

        if "category_l2" not in df.columns:
            logger.warning("category_l2 not in DataFrame — skipping regime detection.")
            return results

        for (uid, cat) in df.groupby(["user_id", "category_l2"]).groups:
            result = self.detect(df, uid, cat)
            results.append(result)

        logger.info("Detected regimes for %d user×category pairs.", len(results))
        return results
