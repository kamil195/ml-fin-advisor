"""
N-BEATS wrapper for expense forecasting (SPEC §7.3, Appendix A.4).

Selected for users with 6–18 months history and moderate variability.
High accuracy on univariate series without feature engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NBeatsForecast:
    """Forecast output from N-BEATS."""

    category: str
    dates: list[str]
    p10: list[float]
    p50: list[float]
    p90: list[float]


class NBeatsModel:
    """
    N-BEATS wrapper for per-category weekly expense forecasting.

    Uses the NeuralForecast library when available; falls back to a
    simple exponential-smoothing baseline.

    Parameters
    ----------
    stack_types : list[str]
        N-BEATS stack types.
    blocks_per_stack : int
        Number of blocks per stack.
    hidden_size : int
        Width of fully-connected layers.
    lookback_multiple : int
        Lookback = multiple × horizon.
    horizon : int
        Forecast horizon in weeks.
    """

    def __init__(
        self,
        stack_types: list[str] | None = None,
        blocks_per_stack: int = 3,
        hidden_size: int = 256,
        lookback_multiple: int = 5,
        horizon: int = 4,
    ) -> None:
        self.stack_types = stack_types or ["trend", "seasonality", "generic"]
        self.blocks_per_stack = blocks_per_stack
        self.hidden_size = hidden_size
        self.lookback_multiple = lookback_multiple
        self.horizon = horizon
        self._model: object | None = None
        self._fitted_data: dict[str, pd.DataFrame] = {}

    def _prepare_series(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
    ) -> pd.DataFrame:
        """Aggregate to NeuralForecast format: unique_id, ds, y."""
        mask = (df["user_id"] == user_id) & (df["amount"] < 0)
        if category != "all":
            mask = mask & (df.get("category_l2", pd.Series()) == category)

        subset = df.loc[mask].copy()
        if subset.empty:
            return pd.DataFrame()

        subset["ds"] = pd.to_datetime(subset["timestamp"]).dt.to_period("W").dt.start_time
        weekly = subset.groupby("ds")["amount"].apply(lambda s: s.abs().sum()).reset_index()
        weekly.columns = ["ds", "y"]
        weekly = weekly.sort_values("ds")
        weekly["unique_id"] = f"{user_id}_{category}"
        return weekly[["unique_id", "ds", "y"]]

    def fit(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str = "all",
    ) -> NBeatsModel:
        """Train N-BEATS on weekly aggregated spend."""
        series = self._prepare_series(df, user_id, category)
        key = f"{user_id}_{category}"

        if len(series) < self.lookback_multiple * self.horizon:
            logger.warning(
                "Insufficient data for N-BEATS (user=%s, cat=%s): %d weeks",
                user_id, category, len(series),
            )
            # Store for naive fallback
            if not series.empty:
                self._fitted_data[key] = series
            return self

        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATS

            model = NBEATS(
                h=self.horizon,
                input_size=self.lookback_multiple * self.horizon,
                stack_types=self.stack_types,
                n_blocks=[self.blocks_per_stack] * len(self.stack_types),
                mlp_units=[[self.hidden_size, self.hidden_size]] * len(self.stack_types),
                max_steps=200,
                random_seed=42,
            )
            nf = NeuralForecast(models=[model], freq="W")
            nf.fit(df=series)
            self._model = nf
            self._fitted_data[key] = series
            logger.info("N-BEATS fitted for user=%s, cat=%s", user_id, category)
        except ImportError:
            logger.warning("neuralforecast not installed — using ETS fallback.")
            self._fitted_data[key] = series

        return self

    def predict(
        self,
        user_id: str,
        category: str = "all",
    ) -> NBeatsForecast:
        """Generate forecast."""
        key = f"{user_id}_{category}"
        series = self._fitted_data.get(key)

        if series is None or series.empty:
            return NBeatsForecast(
                category=category, dates=[], p10=[], p50=[], p90=[]
            )

        if self._model is not None:
            try:
                fc = self._model.predict()
                dates = [str(d.date()) for d in fc["ds"]]
                p50 = [round(max(v, 0), 2) for v in fc.iloc[:, -1]]
                # Approximate quantiles
                std_est = np.std(series["y"].values[-8:]) if len(series) >= 8 else 1.0
                p10 = [round(max(v - 1.28 * std_est, 0), 2) for v in p50]
                p90 = [round(v + 1.28 * std_est, 2) for v in p50]
                return NBeatsForecast(
                    category=category, dates=dates, p10=p10, p50=p50, p90=p90
                )
            except Exception as e:
                logger.warning("N-BEATS prediction failed: %s", e)

        # Exponential-smoothing fallback
        values = series["y"].values
        alpha = 0.3
        level = values[-1]
        fitted_vals = [level]
        for v in values[-6:]:
            level = alpha * v + (1 - alpha) * level
            fitted_vals.append(level)

        mu = level
        std = float(np.std(values[-8:])) if len(values) >= 8 else float(np.std(values))

        dates = pd.date_range(
            start=series["ds"].max() + pd.Timedelta(weeks=1),
            periods=self.horizon,
            freq="W",
        )

        return NBeatsForecast(
            category=category,
            dates=[str(d.date()) for d in dates],
            p10=[round(max(mu - 1.28 * std, 0), 2)] * self.horizon,
            p50=[round(mu, 2)] * self.horizon,
            p90=[round(mu + 1.28 * std, 2)] * self.horizon,
        )
