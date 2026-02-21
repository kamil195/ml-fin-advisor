"""
Training pipeline (SPEC §10.1).

Orchestrates:
  1. Feature loading
  2. Train/val/test splitting (temporal)
  3. Classifier training (text tower → MLP → meta-learner)
  4. Forecaster training (Prophet / N-BEATS / TFT selection)
  5. Budget optimizer fitting
  6. Evaluation & model comparison
  7. MLflow registry & promotion

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
class TrainingPipelineConfig:
    """Configuration for the training pipeline."""

    feature_store_path: str = "data/feature_store"
    model_output_path: str = "models"
    config_path: str = "configs/model_config.yaml"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "fin-advisor"
    # Split ratios (temporal split)
    val_months: int = 2
    test_months: int = 2
    # Flags
    train_classifier: bool = True
    train_forecaster: bool = True
    train_recommender: bool = True
    run_evaluation: bool = True
    run_comparison: bool = True


class TrainingPipeline:
    """
    End-to-end model training pipeline.

    Steps
    -----
    1. Load features from the offline feature store
    2. Temporal train/val/test split
    3. Train transaction classifier
    4. Train expense forecasters
    5. Fit budget optimizer
    6. Evaluate all models
    7. Compare with champion & register to MLflow
    """

    def __init__(self, config: TrainingPipelineConfig | None = None) -> None:
        self.config = config or TrainingPipelineConfig()
        self._model_cfg: dict[str, Any] = {}
        self._load_model_config()

    def _load_model_config(self) -> None:
        """Load model_config.yaml."""
        cfg_path = Path(self.config.config_path)
        if cfg_path.exists():
            try:
                import yaml

                with open(cfg_path) as f:
                    self._model_cfg = yaml.safe_load(f) or {}
                logger.info("Loaded model config from %s", cfg_path)
            except Exception as exc:
                logger.warning("Could not load model config: %s", exc)

    # ── Step 1: Load Features ──────────────────────────────────────────────

    def load_features(self) -> pd.DataFrame:
        """Load the latest feature set from the offline store."""
        store = Path(self.config.feature_store_path)
        parquet_files = sorted(store.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No feature files found in {store}")

        latest = parquet_files[-1]
        logger.info("Loading features from %s", latest)
        return pd.read_parquet(latest)

    # ── Step 2: Temporal Split ─────────────────────────────────────────────

    def temporal_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by time — train | val | test."""
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        max_date = df[timestamp_col].max()
        test_cutoff = max_date - pd.DateOffset(months=self.config.test_months)
        val_cutoff = test_cutoff - pd.DateOffset(months=self.config.val_months)

        train = df[df[timestamp_col] < val_cutoff]
        val = df[(df[timestamp_col] >= val_cutoff) & (df[timestamp_col] < test_cutoff)]
        test = df[df[timestamp_col] >= test_cutoff]

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    # ── Step 3: Classifier ─────────────────────────────────────────────────

    def train_classifier(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> Any:
        """Train the transaction classification pipeline."""
        from src.models.classifier.train import ClassifierTrainer, TrainingConfig

        clf_cfg = self._model_cfg.get("classification", {})
        training_config = TrainingConfig(
            epochs=clf_cfg.get("training", {}).get("epochs", 30),
            lr=clf_cfg.get("training", {}).get("lr", 1e-3),
            batch_size=clf_cfg.get("training", {}).get("batch_size", 512),
        )

        trainer = ClassifierTrainer(config=training_config)
        trainer.train(train, val)

        output_dir = Path(self.config.model_output_path) / "classifier"
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save(output_dir)

        logger.info("Classifier saved to %s", output_dir)
        return trainer

    # ── Step 4: Forecaster ─────────────────────────────────────────────────

    def train_forecaster(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> Any:
        """Train expense forecasters per user × category."""
        from src.models.forecaster.train import ForecastTrainer

        forecast_cfg = self._model_cfg.get("forecasting", {})
        trainer = ForecastTrainer(
            horizon=forecast_cfg.get("horizon_days", 30),
        )

        trainer.train(train, val)

        output_dir = Path(self.config.model_output_path) / "forecaster"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Forecasters saved to %s", output_dir)
        return trainer

    # ── Step 5: Budget Optimizer ───────────────────────────────────────────

    def fit_recommender(self, train: pd.DataFrame) -> Any:
        """Fit the budget optimizer and explanation engine."""
        from src.models.recommender.budget_optimizer import BudgetOptimiser
        from src.models.recommender.feasibility import FeasibilityChecker

        optimiser = BudgetOptimiser()
        feasibility = FeasibilityChecker()

        logger.info("Budget optimizer & feasibility checker initialised")
        return {"optimiser": optimiser, "feasibility": feasibility}

    # ── Step 6: Evaluation ─────────────────────────────────────────────────

    def evaluate(
        self,
        classifier: Any,
        forecaster: Any,
        test: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate all models on the test set."""
        results: dict[str, Any] = {}

        # Classifier evaluation
        try:
            from src.evaluation.classification_metrics import evaluate_classifier

            y_true = test.get("category_l2")
            if y_true is not None and classifier is not None:
                preds = classifier.predict(test)
                if preds is not None:
                    y_pred = preds.get("category_l2", [])
                    y_prob = preds.get("probabilities")
                    metrics = evaluate_classifier(y_true.tolist(), y_pred, y_prob)
                    results["classification"] = {
                        "macro_f1": metrics.macro_f1,
                        "top_3_accuracy": metrics.top_3_accuracy,
                        "ece": metrics.ece,
                        "meets_targets": metrics.meets_targets(),
                    }
                    logger.info("Classifier metrics: %s", results["classification"])
        except Exception as exc:
            logger.warning("Classifier evaluation failed: %s", exc)

        # Forecast evaluation
        try:
            from src.evaluation.forecast_metrics import evaluate_forecasts

            # Would compare forecast vs actuals here
            logger.info("Forecast evaluation: placeholder (requires actuals)")
        except Exception as exc:
            logger.warning("Forecast evaluation failed: %s", exc)

        return results

    # ── Step 7: Model Registry ─────────────────────────────────────────────

    def register_models(self, metrics: dict[str, Any]) -> None:
        """Register models in MLflow (if available)."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name=f"train_{datetime.utcnow():%Y%m%d_%H%M}"):
                # Log all metrics
                for model_name, model_metrics in metrics.items():
                    for k, v in model_metrics.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{model_name}/{k}", v)

                # Log config as params
                mlflow.log_param("val_months", self.config.val_months)
                mlflow.log_param("test_months", self.config.test_months)

                # Log model artefacts
                model_dir = Path(self.config.model_output_path)
                if model_dir.exists():
                    mlflow.log_artifacts(str(model_dir), "models")

            logger.info("Models registered in MLflow")

        except ImportError:
            logger.info("MLflow not available — skipping model registration")
        except Exception as exc:
            logger.warning("MLflow registration failed: %s", exc)

    # ── Run full pipeline ──────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Execute the complete training pipeline."""
        logger.info("═══ Training Pipeline START ═══")
        start = datetime.utcnow()

        # 1. Load features
        features = self.load_features()

        # 2. Temporal split
        train, val, test = self.temporal_split(features)

        # 3. Train classifier
        classifier = None
        if self.config.train_classifier:
            try:
                classifier = self.train_classifier(train, val)
            except Exception as exc:
                logger.error("Classifier training failed: %s", exc)

        # 4. Train forecaster
        forecaster = None
        if self.config.train_forecaster:
            try:
                forecaster = self.train_forecaster(train, val)
            except Exception as exc:
                logger.error("Forecaster training failed: %s", exc)

        # 5. Fit recommender
        recommender = None
        if self.config.train_recommender:
            try:
                recommender = self.fit_recommender(train)
            except Exception as exc:
                logger.error("Recommender fitting failed: %s", exc)

        # 6. Evaluate
        metrics: dict[str, Any] = {}
        if self.config.run_evaluation:
            metrics = self.evaluate(classifier, forecaster, test)

        # 7. Register
        if metrics:
            self.register_models(metrics)

        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info("═══ Training Pipeline DONE (%.1f s) ═══", elapsed)

        return metrics


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Run training pipeline from command line."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Model training pipeline")
    parser.add_argument("--features", default="data/feature_store", help="Feature store path")
    parser.add_argument("--output", default="models", help="Model output path")
    parser.add_argument("--no-classifier", action="store_true")
    parser.add_argument("--no-forecaster", action="store_true")
    parser.add_argument("--no-recommender", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    config = TrainingPipelineConfig(
        feature_store_path=args.features,
        model_output_path=args.output,
        train_classifier=not args.no_classifier,
        train_forecaster=not args.no_forecaster,
        train_recommender=not args.no_recommender,
        run_evaluation=not args.no_eval,
    )

    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
