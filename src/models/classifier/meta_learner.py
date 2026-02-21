"""
LightGBM Meta-Learner for the Transaction Classifier (SPEC §6.2).

Stacks on top of MLP logits + raw features to produce calibrated
probabilities. Acts as a correction layer that captures residual
patterns — especially for long-tail merchants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    LightGBM meta-learner that consumes MLP logits concatenated with
    selected raw features to produce final class probabilities.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    params : dict
        LightGBM parameters (see model_config.yaml).
    n_folds : int
        Number of CV folds for out-of-fold prediction generation.
    """

    def __init__(
        self,
        n_classes: int = 30,
        params: dict | None = None,
        n_folds: int = 5,
    ) -> None:
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.params = params or {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "num_leaves": 127,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "random_state": 42,
        }
        self._model: Any = None
        self._fold_models: list[Any] = []

    def _get_lgb(self) -> Any:
        """Lazy-import LightGBM."""
        try:
            import lightgbm as lgb

            return lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required for the meta-learner. "
                "pip install lightgbm"
            )

    def fit(
        self,
        X_logits: np.ndarray,
        X_raw: np.ndarray,
        y: np.ndarray,
    ) -> MetaLearner:
        """
        Train the meta-learner using out-of-fold predictions.

        Parameters
        ----------
        X_logits : (N, n_classes)
            MLP output logits.
        X_raw : (N, D)
            Selected raw features (amount, MCC, etc.).
        y : (N,)
            Integer class labels.
        """
        lgb = self._get_lgb()
        X = np.hstack([X_logits, X_raw])

        # Generate out-of-fold predictions to avoid overfitting
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=42
        )

        self._fold_models = []
        oof_preds = np.zeros((len(y), self.n_classes))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**self.params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
            )
            oof_preds[val_idx] = model.predict_proba(X_val)
            self._fold_models.append(model)

            logger.info(
                "Fold %d/%d complete — val logloss: %.4f",
                fold + 1,
                self.n_folds,
                float(
                    -np.mean(
                        np.log(
                            oof_preds[val_idx][np.arange(len(y_val)), y_val]
                            + 1e-10
                        )
                    )
                ),
            )

        # Final model on full data
        self._model = lgb.LGBMClassifier(**self.params)
        self._model.fit(X, y)

        logger.info("MetaLearner training complete — %d folds.", self.n_folds)
        return self

    def predict_proba(
        self,
        X_logits: np.ndarray,
        X_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Uses ensemble averaging across fold models for more robust estimates.
        """
        X = np.hstack([X_logits, X_raw])

        if self._fold_models:
            preds = np.stack(
                [m.predict_proba(X) for m in self._fold_models], axis=0
            )
            return preds.mean(axis=0)

        if self._model is not None:
            return self._model.predict_proba(X)

        raise RuntimeError("MetaLearner not fitted. Call fit() first.")

    def predict(
        self,
        X_logits: np.ndarray,
        X_raw: np.ndarray,
    ) -> np.ndarray:
        """Return predicted class indices."""
        return self.predict_proba(X_logits, X_raw).argmax(axis=-1)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from the full-data model."""
        if self._model is None:
            raise RuntimeError("MetaLearner not fitted.")

        importance = self._model.feature_importances_
        return pd.DataFrame(
            {"feature_idx": range(len(importance)), "importance": importance}
        ).sort_values("importance", ascending=False)

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "fold_models": self._fold_models},
            path,
        )
        logger.info("MetaLearner saved to %s", path)

    def load(self, path: str | Path) -> MetaLearner:
        """Load a trained model from disk."""
        import joblib

        data = joblib.load(path)
        self._model = data["model"]
        self._fold_models = data.get("fold_models", [])
        logger.info("MetaLearner loaded from %s", path)
        return self
