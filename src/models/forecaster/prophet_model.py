"""
Prophet baseline wrapper for expense forecasting (SPEC §7.3).

Selected for users with < 6 months history and stable spending patterns.
Provides strong seasonality decomposition and robustness to missing data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProphetForecast:
    """Forecast output from Prophet."""

    category: str
    dates: list[str]
    p10: list[float]
    p50: list[float]
    p90: list[float]


class ProphetModel:
    """
    Prophet wrapper for per-category weekly expense forecasting.

    Parameters
    ----------
    yearly_seasonality : bool
        Enable yearly seasonality component.
    weekly_seasonality : bool
        Enable weekly seasonality component.
    changepoint_prior_scale : float
        Controls trend flexibility (higher = more flexible).
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._models: dict[str, object] = {}

    def _prepare_series(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
    ) -> pd.DataFrame:
        """Aggregate transactions to weekly spend for Prophet format."""
        mask = (df["user_id"] == user_id) & (df["amount"] < 0)
        if category != "all":
            mask = mask & (df.get("category_l2", pd.Series()) == category)

        subset = df.loc[mask].copy()
        if subset.empty:
            return pd.DataFrame(columns=["ds", "y"])

        subset["ds"] = pd.to_datetime(subset["timestamp"]).dt.to_period("W").dt.start_time
        weekly = subset.groupby("ds")["amount"].apply(lambda s: s.abs().sum()).reset_index()
        weekly.columns = ["ds", "y"]
        return weekly.sort_values("ds")

    def fit(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str = "all",
    ) -> ProphetModel:
        """Fit Prophet on weekly aggregated spend."""
        series = self._prepare_series(df, user_id, category)

        if len(series) < 4:
            logger.warning(
                "Insufficient data for Prophet (user=%s, cat=%s): %d weeks",
                user_id, category, len(series),
            )
            return self

        try:
            from prophet import Prophet

            model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                interval_width=0.80,
            )
            model.fit(series)
            self._models[f"{user_id}_{category}"] = model
            logger.info("Prophet fitted for user=%s, cat=%s", user_id, category)
        except ImportError:
            logger.warning("prophet not installed — using naive forecast fallback.")
            self._models[f"{user_id}_{category}"] = {
                "mean": float(series["y"].mean()),
                "std": float(series["y"].std(ddof=0)),
            }

        return self

    def predict(
        self,
        user_id: str,
        category: str = "all",
        horizon_weeks: int = 4,
    ) -> ProphetForecast:
        """Generate probabilistic forecast."""
        key = f"{user_id}_{category}"
        model = self._models.get(key)

        if model is None:
            return ProphetForecast(
                category=category,
                dates=[],
                p10=[],
                p50=[],
                p90=[],
            )

        if isinstance(model, dict):
            # Naive fallback
            mu, sigma = model["mean"], model["std"]
            dates = pd.date_range(
                start=pd.Timestamp.now().normalize(),
                periods=horizon_weeks,
                freq="W",
            )
            return ProphetForecast(
                category=category,
                dates=[str(d.date()) for d in dates],
                p10=[round(max(mu - 1.28 * sigma, 0), 2)] * horizon_weeks,
                p50=[round(mu, 2)] * horizon_weeks,
                p90=[round(mu + 1.28 * sigma, 2)] * horizon_weeks,
            )

        # Real Prophet
        future = model.make_future_dataframe(periods=horizon_weeks, freq="W")
        forecast = model.predict(future)
        fc = forecast.tail(horizon_weeks)

        return ProphetForecast(
            category=category,
            dates=[str(d.date()) for d in fc["ds"]],
            p10=[round(max(v, 0), 2) for v in fc["yhat_lower"]],
            p50=[round(max(v, 0), 2) for v in fc["yhat"]],
            p90=[round(max(v, 0), 2) for v in fc["yhat_upper"]],
        )
