"""
Contract tests for the meta-learner train/serve input ordering.

Regression guard for the bug where training stacked the meta input as
``[probabilities | features]`` but serving stacked ``[features | probabilities]``.
Both have n_classes + n_features columns, so the mismatch was silent — the
XGBoost meta model happily consumed transposed blocks and produced wrong
probabilities whenever the base model's confidence was < 0.90.

The canonical order is ``[probabilities | features]`` and both call sites
(``run_pipeline.train_classifier`` and ``serving.routes.classify``) must go
through ``src.utils.meta_input.build_meta_input``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.meta_input import build_meta_input  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVE_DIR = PROJECT_ROOT / "models" / "serving"


# ── Helper contract ─────────────────────────────────────────────────────────────


def test_build_meta_input_puts_probabilities_first():
    probs = np.arange(1, 7, dtype=float).reshape(2, 3) / 6.0  # two rows, 3 classes
    feats = np.arange(10, 16, dtype=float).reshape(2, 3)  # two rows, 3 features

    meta = build_meta_input(probs, feats)

    assert meta.shape == (2, 6)
    assert np.array_equal(meta[:, :3], probs), "probabilities must occupy the FIRST block"
    assert np.array_equal(meta[:, 3:], feats), "features must occupy the LAST block"


def test_build_meta_input_accepts_single_row_1d_probabilities():
    """Serving path passes ``proba.reshape(1, -1)``; a 1-D vector must also work."""
    probs = np.array([0.1, 0.7, 0.2])
    feats = np.array([[1.0, 2.0, 3.0, 4.0]])

    meta = build_meta_input(probs, feats)

    assert meta.shape == (1, 7)
    assert np.array_equal(meta[0, :3], probs)


def test_build_meta_input_rejects_row_mismatch():
    with pytest.raises(ValueError, match="row mismatch"):
        build_meta_input(np.zeros((2, 3)), np.zeros((1, 4)))


# ── Served-artifact contract ────────────────────────────────────────────────────


def _load_served():
    meta_path = SERVE_DIR / "classifier_meta.joblib"
    le_path = SERVE_DIR / "label_encoder.joblib"
    cols_path = SERVE_DIR / "feature_columns.json"
    if not (meta_path.exists() and le_path.exists() and cols_path.exists()):
        pytest.skip("serving artifacts not present in models/serving/")
    import joblib

    with open(cols_path) as f:
        feature_cols = json.load(f)
    return joblib.load(meta_path), joblib.load(le_path), feature_cols


def test_served_meta_model_expects_canonical_layout():
    """The served meta model's input width must equal probs-block + feature-block."""
    meta_model, label_encoder, feature_cols = _load_served()

    n_classes = len(label_encoder.classes_)
    n_features = len(feature_cols)
    expected = n_classes + n_features

    assert meta_model.n_features_in_ == expected, (
        f"meta model expects {meta_model.n_features_in_} columns but the "
        f"canonical layout provides {expected} ({n_classes} probs + {n_features} features)"
    )

    # Helper output must be directly consumable by the served artifact.
    probs = np.full((1, n_classes), 1.0 / n_classes)
    feats = np.zeros((1, n_features))
    out = meta_model.predict(build_meta_input(probs, feats))
    assert out.shape == (1,)


# ── Call-site guard ─────────────────────────────────────────────────────────────


def test_train_and_serve_use_the_shared_helper():
    """
    Source-level contract: both meta-input call sites must construct the
    vector via ``build_meta_input`` and must NOT hand-roll an np.hstack with
    a divergent order.
    """
    pipeline_src = (PROJECT_ROOT / "run_pipeline.py").read_text(encoding="utf-8")
    classify_src = (
        PROJECT_ROOT / "src" / "serving" / "routes" / "classify.py"
    ).read_text(encoding="utf-8")

    # Training: all three meta inputs built through the helper (probs first).
    assert "build_meta_input(oof_probs, X_train)" in pipeline_src
    assert "build_meta_input(base_val_probs, X_val)" in pipeline_src
    assert "build_meta_input(base_test_probs, X_test)" in pipeline_src
    assert "np.hstack([oof_probs" not in pipeline_src

    # Serving: probabilities FIRST, features LAST.
    assert "build_meta_input(proba.reshape(1, -1), X)" in classify_src
    assert "np.hstack([X, proba" not in classify_src