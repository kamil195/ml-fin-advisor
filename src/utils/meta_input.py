"""
Canonical meta-learner input construction (train/serve contract).

The stacked classifier's meta-learner consumes:

    [ base-model class probabilities | original feature vector ]

This order is part of the model contract: ``run_pipeline.train_classifier``
fits the XGBoost meta-learner on this layout, and the serving route must
reproduce it exactly. Both call sites MUST go through ``build_meta_input``
so the ordering can never drift again (dimension totals alone cannot catch
a transposed stack, because n_probs + n_features is symmetric).
"""

from __future__ import annotations

import numpy as np


def build_meta_input(probabilities: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Stack base-model class probabilities with the original features.

    Parameters
    ----------
    probabilities : (N, n_classes) array-like
        Base model ``predict_proba`` output (row-wise class probabilities).
    features : (N, D) array-like
        The same feature matrix that was fed to the base model.

    Returns
    -------
    (N, n_classes + D) ndarray
        ``[probabilities | features]`` — the canonical training order.
    """
    probabilities = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
    features = np.atleast_2d(np.asarray(features, dtype=np.float64))

    if probabilities.shape[0] != features.shape[0]:
        raise ValueError(
            "Meta-input row mismatch: probabilities "
            f"{probabilities.shape} vs features {features.shape}"
        )

    return np.hstack([probabilities, features])
