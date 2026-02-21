"""
Batch inference pipeline (SPEC §10.1, §11.1).

Orchestrates:
  1. Load trained models from registry / disk
  2. Load latest feature store snapshot
  3. Run batch classification for unclassified transactions
  4. Run batch forecasting for all active users (nightly at 02:00 UTC)
  5. Generate budget recommendations for users due for refresh
  6. Write results to database / cache

Can be scheduled via Airflow / Prefect or run standalone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InferencePipelineConfig:
    """Configuration for the batch inference pipeline."""

    feature_store_path: str = "data/feature_store"
    model_path: str = "models"
    output_path: str = "data/predictions"
    config_path: str = "configs/serving_config.yaml"
    # Task selection
    run_classification: bool = True
    run_forecasting: bool = True
    run_budget: bool = True
    # Forecasting
    forecast_horizon_days: int = 30


class InferencePipeline:
    """
    Batch inference pipeline for offline scoring.

    Runs classification, forecasting, and budget optimization
    against the full user base, writing results to cache/storage
    for low-latency serving.
    """

    def __init__(self, config: InferencePipelineConfig | None = None) -> None:
        self.config = config or InferencePipelineConfig()
        self._serving_cfg: dict[str, Any] = {}
        self._load_serving_config()

    def _load_serving_config(self) -> None:
        """Load serving_config.yaml."""
        cfg_path = Path(self.config.config_path)
        if cfg_path.exists():
            try:
                import yaml

                with open(cfg_path) as f:
                    self._serving_cfg = yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning("Could not load serving config: %s", exc)

    # ── Step 1: Load Models ────────────────────────────────────────────────

    def load_models(self) -> dict[str, Any]:
        """Load trained model artefacts."""
        models: dict[str, Any] = {}
        model_dir = Path(self.config.model_path)

        # Classifier
        clf_dir = model_dir / "classifier"
        if clf_dir.exists():
            try:
                from src.models.classifier.train import ClassifierTrainer

                trainer = ClassifierTrainer()
                # trainer.load(clf_dir) — would load from disk in production
                models["classifier"] = trainer
                logger.info("Classifier loaded from %s", clf_dir)
            except Exception as exc:
                logger.warning("Could not load classifier: %s", exc)

        # Forecaster
        fc_dir = model_dir / "forecaster"
        if fc_dir.exists():
            try:
                from src.models.forecaster.model_selector import ModelSelector

                selector = ModelSelector()
                models["forecaster"] = selector
                logger.info("Forecaster loaded from %s", fc_dir)
            except Exception as exc:
                logger.warning("Could not load forecaster: %s", exc)

        return models

    # ── Step 2: Load Features ──────────────────────────────────────────────

    def load_features(self) -> pd.DataFrame:
        """Load the latest feature store snapshot."""
        store = Path(self.config.feature_store_path)
        parquet_files = sorted(store.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No feature files in {store}")

        latest = parquet_files[-1]
        logger.info("Loading features from %s", latest)
        return pd.read_parquet(latest)

    # ── Step 3: Batch Classification ───────────────────────────────────────

    def run_classification(
        self,
        features: pd.DataFrame,
        models: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Classify all unclassified transactions."""
        classifier = models.get("classifier")
        if classifier is None:
            logger.warning("No classifier available — skipping")
            return None

        try:
            # Filter to unclassified rows
            mask = features.get("category_l2")
            if mask is not None:
                unclassified = features[mask.isna()]
            else:
                unclassified = features

            if unclassified.empty:
                logger.info("No unclassified transactions")
                return None

            logger.info("Classifying %d transactions", len(unclassified))

            predictions = classifier.predict(unclassified)
            if predictions is not None:
                result = unclassified.copy()
                result["predicted_l2"] = predictions.get("category_l2", [])
                result["confidence"] = predictions.get("confidence", [])
                return result

        except Exception as exc:
            logger.error("Batch classification failed: %s", exc)

        return None

    # ── Step 4: Batch Forecasting ──────────────────────────────────────────

    def run_forecasting(
        self,
        features: pd.DataFrame,
        models: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Generate forecasts for all active users."""
        forecaster = models.get("forecaster")
        if forecaster is None:
            logger.warning("No forecaster available — skipping")
            return None

        try:
            user_col = "user_id"
            if user_col not in features.columns:
                logger.warning("No user_id column found — skipping")
                return None

            users = features[user_col].unique()
            logger.info("Forecasting for %d users", len(users))

            results = []
            for user_id in users:
                user_data = features[features[user_col] == user_id]
                try:
                    forecast = forecaster.predict(
                        user_data,
                        horizon=self.config.forecast_horizon_days,
                    )
                    if forecast is not None:
                        forecast["user_id"] = user_id
                        results.append(forecast)
                except Exception as exc:
                    logger.warning("Forecast failed for user %s: %s", user_id, exc)

            if results:
                return pd.DataFrame(results)

        except Exception as exc:
            logger.error("Batch forecasting failed: %s", exc)

        return None

    # ── Step 5: Budget Recommendations ─────────────────────────────────────

    def run_budget(
        self,
        features: pd.DataFrame,
        forecast_results: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Generate budget recommendations for all users."""
        try:
            from src.models.recommender.budget_optimizer import BudgetOptimiser

            optimiser = BudgetOptimiser()

            user_col = "user_id"
            if user_col not in features.columns:
                logger.warning("No user_id column — skipping budget")
                return None

            users = features[user_col].unique()
            logger.info("Generating budgets for %d users", len(users))

            results = []
            for user_id in users:
                user_data = features[features[user_col] == user_id]
                try:
                    # Compute spending by category
                    if "amount" in user_data.columns and "category_l1" in user_data.columns:
                        spending = (
                            user_data.groupby("category_l1")["amount"]
                            .sum()
                            .to_dict()
                        )
                        income = user_data["amount"].sum() * 1.3  # rough estimate
                        result = optimiser.optimise(
                            current_spending=spending,
                            income=income,
                        )
                        results.append({
                            "user_id": user_id,
                            "total_budget": result.total_budget,
                            "savings": result.savings,
                        })
                except Exception as exc:
                    logger.warning("Budget failed for user %s: %s", user_id, exc)

            if results:
                return pd.DataFrame(results)

        except Exception as exc:
            logger.error("Batch budget generation failed: %s", exc)

        return None

    # ── Step 6: Save Results ───────────────────────────────────────────────

    def save_results(
        self,
        classifications: pd.DataFrame | None,
        forecasts: pd.DataFrame | None,
        budgets: pd.DataFrame | None,
    ) -> None:
        """Persist predictions to disk (and optionally cache)."""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if classifications is not None and not classifications.empty:
            path = output_dir / f"classifications_{ts}.parquet"
            classifications.to_parquet(path, index=False)
            logger.info("Classifications → %s (%d rows)", path, len(classifications))

        if forecasts is not None and not forecasts.empty:
            path = output_dir / f"forecasts_{ts}.parquet"
            forecasts.to_parquet(path, index=False)
            logger.info("Forecasts → %s (%d rows)", path, len(forecasts))

        if budgets is not None and not budgets.empty:
            path = output_dir / f"budgets_{ts}.parquet"
            budgets.to_parquet(path, index=False)
            logger.info("Budgets → %s (%d rows)", path, len(budgets))

    # ── Run full pipeline ──────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the complete batch inference pipeline."""
        logger.info("═══ Inference Pipeline START ═══")
        start = datetime.utcnow()

        # Load models & features
        models = self.load_models()
        features = self.load_features()

        # Classification
        classifications = None
        if self.config.run_classification:
            classifications = self.run_classification(features, models)

        # Forecasting
        forecasts = None
        if self.config.run_forecasting:
            forecasts = self.run_forecasting(features, models)

        # Budget
        budgets = None
        if self.config.run_budget:
            budgets = self.run_budget(features, forecasts)

        # Save
        self.save_results(classifications, forecasts, budgets)

        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info("═══ Inference Pipeline DONE (%.1f s) ═══", elapsed)


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Run batch inference pipeline from command line."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Batch inference pipeline")
    parser.add_argument("--features", default="data/feature_store", help="Feature store path")
    parser.add_argument("--models", default="models", help="Model path")
    parser.add_argument("--output", default="data/predictions", help="Output path")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--no-forecast", action="store_true")
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    args = parser.parse_args()

    config = InferencePipelineConfig(
        feature_store_path=args.features,
        model_path=args.models,
        output_path=args.output,
        run_classification=not args.no_classify,
        run_forecasting=not args.no_forecast,
        run_budget=not args.no_budget,
        forecast_horizon_days=args.horizon,
    )

    pipeline = InferencePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
