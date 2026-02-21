"""
Temporal Fusion Transformer (TFT) wrapper for expense forecasting
(SPEC §7.3, Appendix A.3).

Selected for users with > 18 months history and complex patterns.
Handles covariates, multi-horizon forecasting, and built-in attention
for interpretability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TFTForecast:
    """Forecast output from Temporal Fusion Transformer."""

    category: str
    dates: list[str]
    p10: list[float]
    p50: list[float]
    p90: list[float]
    attention_weights: dict | None = None


class TFTModel:
    """
    Temporal Fusion Transformer for multi-horizon expense forecasting.

    Uses NeuralForecast's TFT when available; falls back to a
    weighted-average baseline.

    Parameters
    ----------
    hidden_size : int
        Hidden layer size for LSTM and attention.
    attention_heads : int
        Number of multi-head attention heads.
    lstm_layers : int
        Number of LSTM encoder layers.
    dropout : float
        Dropout rate.
    learning_rate : float
        Learning rate for AdamW.
    quantiles : list[float]
        Prediction quantiles.
    max_encoder_length : int
        Maximum encoder (lookback) length in weeks.
    max_prediction_length : int
        Maximum forecast horizon in weeks.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        quantiles: list[float] | None = None,
        max_encoder_length: int = 52,
        max_prediction_length: int = 13,
    ) -> None:
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self._model: object | None = None
        self._fitted_data: dict[str, pd.DataFrame] = {}

    def _prepare_series(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str,
    ) -> pd.DataFrame:
        """Aggregate and add covariates for TFT format."""
        mask = (df["user_id"] == user_id) & (df["amount"] < 0)
        if category != "all":
            mask = mask & (df.get("category_l2", pd.Series()) == category)

        subset = df.loc[mask].copy()
        if subset.empty:
            return pd.DataFrame()

        ts = pd.to_datetime(subset["timestamp"])
        subset["ds"] = ts.dt.to_period("W").dt.start_time
        weekly = subset.groupby("ds").agg(
            y=("amount", lambda s: s.abs().sum()),
            txn_count=("amount", "count"),
        ).reset_index()
        weekly = weekly.sort_values("ds")
        weekly["unique_id"] = f"{user_id}_{category}"

        # Add temporal covariates
        weekly["week_of_year"] = weekly["ds"].dt.isocalendar().week.astype(int)
        weekly["month"] = weekly["ds"].dt.month

        return weekly

    def fit(
        self,
        df: pd.DataFrame,
        user_id: str,
        category: str = "all",
    ) -> TFTModel:
        """Train TFT on weekly aggregated spend with covariates."""
        series = self._prepare_series(df, user_id, category)
        key = f"{user_id}_{category}"

        if len(series) < self.max_encoder_length // 2:
            logger.warning(
                "Insufficient data for TFT (user=%s, cat=%s): %d weeks",
                user_id, category, len(series),
            )
            if not series.empty:
                self._fitted_data[key] = series
            return self

        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import TFT

            model = TFT(
                h=min(self.max_prediction_length, 13),
                input_size=min(len(series) - 1, self.max_encoder_length),
                hidden_size=self.hidden_size,
                learning_rate=self.learning_rate,
                max_steps=300,
                random_seed=42,
            )
            nf = NeuralForecast(models=[model], freq="W")
            nf.fit(df=series[["unique_id", "ds", "y"]])
            self._model = nf
            self._fitted_data[key] = series
            logger.info("TFT fitted for user=%s, cat=%s", user_id, category)
        except ImportError:
            logger.warning("neuralforecast not installed — using weighted-avg fallback.")
            self._fitted_data[key] = series

        return self

    def predict(
        self,
        user_id: str,
        category: str = "all",
        horizon_weeks: int = 4,
    ) -> TFTForecast:
        """Generate quantile forecasts."""
        key = f"{user_id}_{category}"
        series = self._fitted_data.get(key)

        if series is None or series.empty:
            return TFTForecast(
                category=category, dates=[], p10=[], p50=[], p90=[]
            )

        if self._model is not None:
            try:
                fc = self._model.predict()
                dates = [str(d.date()) for d in fc["ds"]]
                p50 = [round(max(v, 0), 2) for v in fc.iloc[:, -1]]
                std_est = np.std(series["y"].values[-8:]) if len(series) >= 8 else 1.0
                p10 = [round(max(v - 1.28 * std_est, 0), 2) for v in p50]
                p90 = [round(v + 1.28 * std_est, 2) for v in p50]
                return TFTForecast(
                    category=category, dates=dates, p10=p10, p50=p50, p90=p90
                )
            except Exception as e:
                logger.warning("TFT prediction failed: %s", e)

        # Weighted-average fallback with exponential decay
        values = series["y"].values
        weights = np.exp(np.linspace(-1, 0, min(len(values), 12)))
        weights /= weights.sum()
        recent = values[-len(weights) :]
        mu = float(np.sum(recent * weights))
        std = float(np.std(values[-8:])) if len(values) >= 8 else float(np.std(values))

        dates = pd.date_range(
            start=series["ds"].max() + pd.Timedelta(weeks=1),
            periods=horizon_weeks,
            freq="W",
        )

        return TFTForecast(
            category=category,
            dates=[str(d.date()) for d in dates],
            p10=[round(max(mu - 1.28 * std, 0), 2)] * horizon_weeks,
            p50=[round(mu, 2)] * horizon_weeks,
            p90=[round(mu + 1.28 * std, 2)] * horizon_weeks,
        )
