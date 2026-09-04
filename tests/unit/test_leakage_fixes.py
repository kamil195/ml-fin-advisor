"""
Focused regression tests for the classifier evaluation-leakage fixes.

Covers:
1. Target-derived ``is_discretionary`` never reaches the model feature matrix.
2. ``amount_zscore_user`` uses only *prior* transactions (no future leakage).
3. ClassifierResult carries the new honest-evaluation metric fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_pipeline as rp  # noqa: E402
from src.features.numerical_features import amount_zscore_user  # noqa: E402


def _hist_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1"] * 5,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "amount": [-10.0, -20.0, -30.0, -40.0, -50.0],
        }
    )


def test_get_feature_columns_excludes_target_derived():
    df = pd.DataFrame(
        {
            "amount": [-5.0],
            "mcc": [5732.0],
            "is_debit": [1.0],
            "is_discretionary": [1.0],  # target-derived impostor
            "user_id": ["u1"],
            "timestamp": pd.Timestamp("2024-01-01"),
            "merchant_name": ["m"],
            "raw_description": ["d"],
            "location_city": ["c"],
            "location_country": ["co"],
            "category_l1": ["A"],
            "category_l2": ["B"],
            "is_pending": [0.0],
            "merchant_mcc": [5732],
            "account_type": ["CHECKING"],
            "channel": ["POS"],
        }
    )
    cols = rp._get_feature_columns(df)
    assert "is_discretionary" not in cols
    assert {"amount", "mcc", "is_debit"}.issubset(set(cols))


def test_engineer_features_has_no_target_derived_columns():
    df = rp.generate_data(n_users=2, months=2)
    feats = rp.engineer_features(df)
    assert "is_discretionary" not in feats.columns
    # Label columns must remain available for splitting / evaluation.
    assert "category_l2" in feats.columns


def test_amount_zscore_user_is_prior_only():
    df = _hist_df()
    z = amount_zscore_user(df)
    # Fewer than two prior transactions → no z-score → 0.0
    assert z.iloc[0] == 0.0
    assert z.iloc[1] == 0.0
    assert z.iloc[2] != 0.0


def test_amount_zscore_user_unaffected_by_future_amounts():
    df = _hist_df()
    z_before = amount_zscore_user(df)

    df_future = df.copy()
    df_future.loc[df_future.index[-1], "amount"] = -5_000.0
    z_after = amount_zscore_user(df_future)

    # Earlier rows identical; only the changed row differs.
    pd.testing.assert_series_equal(z_before.iloc[:-1], z_after.iloc[:-1])
    assert z_after.iloc[-1] != z_before.iloc[-1]


def test_amount_zscore_user_isolated_across_users():
    df = _hist_df()
    other = _hist_df()
    other["user_id"] = "u2"
    other.loc[other.index[-1], "amount"] = -9_999.0  # extreme tail user
    combined = pd.concat([df, other], ignore_index=True)

    z = amount_zscore_user(combined)
    z_solo = amount_zscore_user(df)
    pd.testing.assert_series_equal(z.iloc[:5].reset_index(drop=True), z_solo.reset_index(drop=True))


def test_classifier_result_reports_new_metrics():
    res = rp.ClassifierResult()
    assert res.accuracy == 0.0
    assert res.weighted_f1 == 0.0
    assert res.baseline_majority_accuracy == 0.0