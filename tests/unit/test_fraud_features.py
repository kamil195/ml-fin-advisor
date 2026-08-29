"""
Unit tests for the fraud-detection features added alongside classify.

Covers ``calculate_velocity`` (the shared feature module) and the rule-based
``_compute_fraud_analysis`` helper in the classify route. These run without a
trained model or Redis — pure logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.models import Transaction
from src.features.numerical_features import calculate_velocity
from src.serving.routes.classify import _compute_fraud_analysis
from src.utils.constants import AccountType, CategoryL1, CategoryL2, Channel


def test_velocity_single_txn_is_zero():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "timestamp": [pd.Timestamp("2026-01-01 12:00:00")],
        }
    )
    result = calculate_velocity(df, window_minutes=10)
    assert int(result.iloc[0]) == 0


def test_velocity_excludes_current_row():
    df = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "timestamp": [
                pd.Timestamp("2026-01-01 12:00:00"),
                pd.Timestamp("2026-01-01 12:02:00"),
                pd.Timestamp("2026-01-01 12:04:00"),
                pd.Timestamp("2026-01-01 12:30:00"),
            ],
        }
    )
    result = calculate_velocity(df, window_minutes=10)
    values = result.tolist()
    # first has no prior; second/third see 1 and 2 prior within 10min;
    # fourth is outside the window from previous ones
    assert values[0] == 0
    assert values[1] == 1
    assert values[2] == 2
    assert values[3] == 0


def test_velocity_handles_tz_aware_and_naive():
    naive = pd.DataFrame(
        {"user_id": ["a", "a"], "timestamp": ["2026-01-01 12:00:00", "2026-01-01 12:05:00"]}
    )
    aware = naive.copy()
    aware["timestamp"] = pd.to_datetime(aware["timestamp"]).dt.tz_localize("UTC")
    r1 = calculate_velocity(naive, window_minutes=10)
    r2 = calculate_velocity(aware, window_minutes=10)
    assert r1.tolist() == r2.tolist()


def test_velocity_distinguishes_users():
    df = pd.DataFrame(
        {
            "user_id": ["a", "a", "b"],
            "timestamp": [
                pd.Timestamp("2026-01-01 12:00:00"),
                pd.Timestamp("2026-01-01 12:01:00"),
                pd.Timestamp("2026-01-01 12:01:00"),
            ],
        }
    )
    result = calculate_velocity(df, window_minutes=10)
    assert result.tolist() == [0, 1, 0]


def test_fraud_low_risk_by_default():
    txn = _txn(amount=-25.0, channel="POS", hour=12)
    fa = _compute_fraud_analysis(txn)
    assert fa.fraud_score < 0.5
    assert fa.is_suspicious is False
    assert isinstance(fa.velocity_flags, list)


def test_fraud_high_amount_flags():
    txn = _txn(amount=-25_000.0, channel="POS", hour=12)
    fa = _compute_fraud_analysis(txn)
    assert "high_amount" in fa.velocity_flags
    assert fa.is_suspicious is True


def test_fraud_velocity_burst_from_history():
    txn = _txn(amount=-50.0, channel="POS", hour=10)
    history = [
        _txn(amount=-20.0, channel="POS", hour=10, minutes=0),
        _txn(amount=-15.0, channel="POS", hour=10, minutes=1),
        _txn(amount=-12.0, channel="POS", hour=10, minutes=2),
        _txn(amount=-18.0, channel="POS", hour=10, minutes=3),
        _txn(amount=-30.0, channel="POS", hour=10, minutes=4),
    ]
    fa = _compute_fraud_analysis(txn, history=history, max_expected_window=3)
    assert any(f.startswith("velocity_burst_") for f in fa.velocity_flags)
    assert fa.is_suspicious is True


def _txn(amount, channel="POS", hour=12, minutes=30):
    return Transaction(
        user_id="u1",
        timestamp=pd.Timestamp(f"2026-01-01 {hour:02d}:{minutes:02d}:00"),
        amount=amount,
        merchant_name="test",
        merchant_mcc=5999,
        account_type=AccountType.CHECKING,
        channel=Channel(channel),
        category_l1=CategoryL1.FINANCIAL,
        category_l2=CategoryL2.FEES_CHARGES,
    )