"""
Parity tests: serving feature path must equal training feature definitions.
Proves the train/serve feature skew is resolved by routing serving through
src/services/feature_service.FeatureService, which calls the exact training
functions in src/features.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.models import Transaction  # noqa: E402
from src.features.numerical_features import (  # noqa: E402
    amount_pct_of_income,
    amount_zscore_user,
    extract_numerical_features,
    log_amount,
    rolling_spend,
    txn_count_24h,
)
from src.features.temporal_features import (  # noqa: E402
    extract_temporal_features,
)

TS0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_txn(
    amount: float,
    ts: datetime,
    *,
    user_id: str = "u1",
    merchant: str = "Test Merchant",
    mcc: int = 5411,
    channel: str = "POS",
    account_type: str = "CHECKING",
) -> Transaction:
    return Transaction(
        transaction_id=f"txn-{ts.isoformat()}",
        user_id=user_id,
        timestamp=ts,
        amount=amount,
        currency="USD",
        merchant_name=merchant,
        merchant_mcc=mcc,
        account_type=account_type,
        channel=channel,
        raw_description="",
    )


def _to_df(txns: list[Transaction]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(txns):
        rows.append(
            {
                "index": i,
                "user_id": t.user_id,
                "timestamp": pd.Timestamp(t.timestamp).tz_convert("UTC").tz_localize(None),
                "amount": float(t.amount),
                "merchant_mcc": float(t.merchant_mcc),
                "channel": t.channel.value,
                "account_type": t.account_type.value,
            }
        )
    return pd.DataFrame(rows).set_index("index")


def _service_build(txn: Transaction, history: list[Transaction]) -> dict[str, float]:
    from src.services.feature_service import FeatureService

    cols = [
        "log_amount", "amount_zscore_user", "amount_pct_of_income",
        "rolling_spend_7d", "rolling_spend_30d", "txn_count_24h",
        "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos",
        "day_of_month_sin", "day_of_month_cos", "is_weekend", "is_holiday",
        "month_phase_early", "month_phase_late", "month_phase_mid", "days_since_payday",
                "mcc", "is_debit", "ch_POS",
    ]
    return FeatureService(cols).build(txn, history)


# --------------------------------------------------------------------------- #
# 1. log_amount parity
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def test_parity_log_amount():
    import numpy as np  # noqa: F401  (mirrors training import context)

    t = _make_txn(-50.0, TS0)
    feats = _service_build(t, [])
    training_val = float(log_amount(pd.Series([float(t.amount)], index=[0])).iloc[0])
    assert feats["log_amount"] == pytest.approx(training_val, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. amount_zscore_user parity
# --------------------------------------------------------------------------- #


def test_parity_amount_zscore_user_single_transaction():
    t = _make_txn(-50.0, TS0)
    feats = _service_build(t, [])
    assert feats["amount_zscore_user"] == pytest.approx(0.0, abs=1e-9)
    df = _to_df([t])
    training_val = float(amount_zscore_user(df).iloc[0])
    assert training_val == pytest.approx(0.0, abs=1e-9)


def test_parity_amount_zscore_user_with_history():
    prior_ts = [TS0 - timedelta(days=i) for i in range(5, 0, -1)]
    history = [
        _make_txn(-amt, ts)
        for amt, ts in zip([100, 120, 80, 90, 110], prior_ts)
    ]
    current = _make_txn(-150.0, TS0)
    feats = _service_build(current, history)
    df = _to_df(history + [current])
    training_z = float(amount_zscore_user(df).iloc[-1])
    assert feats["amount_zscore_user"] == pytest.approx(training_z, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. amount_pct_of_income parity
# --------------------------------------------------------------------------- #


def test_parity_amount_pct_of_income():
    t = _make_txn(-250.0, TS0)
    income = 5_000.0
    feats = _service_build(t, [])
    expected = 250.0 / max(income, 1.0)
    assert feats["amount_pct_of_income"] == pytest.approx(expected, abs=1e-9)
    df = _to_df([t])
    training_val = float(amount_pct_of_income(df["amount"], income).iloc[0])
    assert feats["amount_pct_of_income"] == pytest.approx(training_val, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. rolling_spend_7d / rolling_spend_30d parity
# --------------------------------------------------------------------------- #


def test_parity_rolling_spend_7d():
    history_ts = [TS0 - timedelta(hours=i) for i in range(12, 0, -1)]
    history = [_make_txn(-10.0, ts) for ts in history_ts]
    current = _make_txn(-40.0, TS0)
    feats = _service_build(current, history)
    df = _to_df(history + [current])
    training_7d = float(rolling_spend(df, window_days=7).iloc[-1])
    assert feats["rolling_spend_7d"] == pytest.approx(training_7d, abs=1e-9)


def test_parity_rolling_spend_30d():
    history_ts = [TS0 - timedelta(days=i) for i in range(30, 0, -1)]
    history = [_make_txn(-100.0, ts) for ts in history_ts]
    current = _make_txn(-50.0, TS0)
    feats = _service_build(current, history)
    df = _to_df(history + [current])
    training_30d = float(rolling_spend(df, window_days=30).iloc[-1])
    assert feats["rolling_spend_30d"] == pytest.approx(training_30d, abs=1e-9)


# --------------------------------------------------------------------------- #
# 5. txn_count_24h parity (includes current row per training semantics)
# --------------------------------------------------------------------------- #


def test_parity_txn_count_24h():
    history_ts = [TS0 - timedelta(hours=i) for i in range(5, 0, -1)]
    history = [_make_txn(-10.0, ts) for ts in history_ts]
    current = _make_txn(-25.0, TS0)
    feats = _service_build(current, history)
    df = _to_df(history + [current])
    training_count = int(txn_count_24h(df).iloc[-1])
    assert feats["txn_count_24h"] == pytest.approx(float(training_count), abs=1e-9)
    assert training_count == 6  # 5 prior + 1 current


# --------------------------------------------------------------------------- #
# 6. Temporal & cyclical parity
# --------------------------------------------------------------------------- #


def test_parity_temporal_features():
    history_ts = [TS0 - timedelta(days=i) for i in range(5, 0, -1)]
    history = [
        _make_txn(-50.0, ts, merchant=f"Store {i}")
        for i, ts in enumerate(history_ts, 1)
    ]
    current = _make_txn(-75.0, TS0)
    feats = _service_build(current, history)
    df = _to_df(history + [current])
    temp = extract_temporal_features(df)
    row = temp.iloc[-1]

    assert feats["hour_of_day_sin"] == pytest.approx(float(row["hour_of_day_sin"]), abs=1e-9)
    assert feats["hour_of_day_cos"] == pytest.approx(float(row["hour_of_day_cos"]), abs=1e-9)
    assert feats["day_of_week_sin"] == pytest.approx(float(row["day_of_week_sin"]), abs=1e-9)
    assert feats["day_of_week_cos"] == pytest.approx(float(row["day_of_week_cos"]), abs=1e-9)
    assert feats["day_of_month_sin"] == pytest.approx(float(row["day_of_month_sin"]), abs=1e-9)
    assert feats["day_of_month_cos"] == pytest.approx(float(row["day_of_month_cos"]), abs=1e-9)
    assert int(feats["is_weekend"]) == int(row["is_weekend"])
    assert int(feats["is_holiday"]) == int(row["is_holiday"])
    assert int(feats["month_phase_early"]) == int(row["month_phase_early"])
    assert int(feats["month_phase_late"]) == int(row["month_phase_late"])
    assert int(feats["month_phase_mid"]) == int(row["month_phase_mid"])
    assert feats["days_since_payday"] == pytest.approx(float(row["days_since_payday"]), abs=1e-9)


# --------------------------------------------------------------------------- #
# 7. Categorical parity
# --------------------------------------------------------------------------- #


def test_parity_categorical_features():
    t = _make_txn(-100.0, TS0, channel="POS", account_type="CHECKING", mcc=5411)
    feats = _service_build(t, [])
    assert feats["mcc"] == pytest.approx(5411.0, abs=1e-9)
    assert feats["is_debit"] == pytest.approx(1.0, abs=1e-9)
    assert feats["ch_POS"] == pytest.approx(1.0, abs=1e-9)


def test_parity_categorical_unseen_channel():
    """Channel not in feature_cols -> all-zero one-hot (trained encoding)."""
    t = _make_txn(-100.0, TS0, channel="ATM", account_type="CHECKING", mcc=5812)
    feats = _service_build(t, [])
    assert feats["is_debit"] == pytest.approx(1.0, abs=1e-9)
    assert feats["ch_POS"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 8. Full numeric-block parity vs extract_numerical_features
# --------------------------------------------------------------------------- #


def test_parity_full_numeric_block():
    history_ts = [TS0 - timedelta(days=i) for i in range(7, 0, -1)]
    history = [
        _make_txn(-amt, ts)
        for amt, ts in zip([100, 120, 80, 90, 110, 95, 105], history_ts)
    ]
    current = _make_txn(-150.0, TS0)
    feats = _service_build(current, history)

    df = _to_df(history + [current])
    num = extract_numerical_features(df, estimated_monthly_income=5_000.0)
    row = num.iloc[-1]

    for col in [
        "log_amount",
        "amount_zscore_user",
        "amount_pct_of_income",
        "rolling_spend_7d",
        "rolling_spend_30d",
        "txn_count_24h",
    ]:
        assert feats[col] == pytest.approx(float(row[col]), abs=1e-9), (
            f"mismatch on {col}: serving={feats[col]} vs training={float(row[col])}"
        )


# --------------------------------------------------------------------------- #
# 9. No future-data leakage from history ordering
# --------------------------------------------------------------------------- #


def test_no_future_data_leakage():
    import random

    rng = random.Random(42)
    history_ts = [TS0 - timedelta(hours=i) for i in range(10, 0, -1)]
    history = [_make_txn(-rng.uniform(50, 500), ts) for ts in history_ts]
    current = _make_txn(-150.0, TS0)

    f_sorted = _service_build(current, history)
    f_shuffled = _service_build(current, list(reversed(history)))

    for col, val in f_sorted.items():
        assert f_shuffled[col] == pytest.approx(val, abs=1e-9), (
            f"history ordering affected {col}"
        )


# --------------------------------------------------------------------------- #
# 10. Determinism
# --------------------------------------------------------------------------- #


def test_determinism():
    history = [_make_txn(-100 * i, TS0 - timedelta(days=i)) for i in range(1, 4)]
    current = _make_txn(-500.0, TS0)
    f1 = _service_build(current, history)
    f2 = _service_build(current, history)
    assert f1 == f2

# --------------------------------------------------------------------------- #
# 11. CSV history accumulation (ingest.py batch semantics)
# --------------------------------------------------------------------------- #


def test_csv_history_accumulation():
    """Ingest path accumulates per-user history so later rows see prior ones."""
    # Three transactions for the same user, 1 day apart.
    t1 = _make_txn(-100.0, TS0 - timedelta(days=2))
    t2 = _make_txn(-200.0, TS0 - timedelta(days=1))
    t3 = _make_txn(-300.0, TS0)

    # First transaction: no history → z-score = 0, rolling = abs(amount).
    fa = _service_build(t1, [])
    assert fa["amount_zscore_user"] == pytest.approx(0.0, abs=1e-9)
    assert fa["rolling_spend_7d"] == pytest.approx(100.0, abs=1e-9)

    # Second transaction: history = [t1] → z-score still 0 (only 1 prior,
    # min_periods=2), rolling includes t1 + t2.
    fb = _service_build(t2, [t1])
    assert fb["amount_zscore_user"] == pytest.approx(0.0, abs=1e-9)
    assert fb["rolling_spend_7d"] == pytest.approx(300.0, abs=1e-9)

    # Third transaction: history = [t1, t2] → z-score uses 2 prior values.
    f3 = _service_build(t3, [t1, t2])
    # Prior mean = (-100 + -200) / 2 = -150, std (ddof=0) = 50
    # z = (-300 - (-150)) / 50 = -3.0
    assert f3["amount_zscore_user"] == pytest.approx(-3.0, abs=1e-9)
    assert f3["rolling_spend_7d"] == pytest.approx(600.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 12. Per-user isolation
# --------------------------------------------------------------------------- #


def test_per_user_isolation():
    """One user's history must not affect another user's features."""
    u1_history = [_make_txn(-1000.0, TS0 - timedelta(days=i), user_id="u1") for i in range(1, 4)]
    u2_txn = _make_txn(-50.0, TS0, user_id="u2")

    # u2 with no history → z-score = 0
    f_u2_alone = _service_build(u2_txn, [])
    # u2 with u1's history mixed in → still computed only over u2's prior
    # (which is empty), so z-score must remain 0 (groupby user_id isolates).
    f_u2_with_others = _service_build(u2_txn, u1_history)

    assert f_u2_alone["amount_zscore_user"] == pytest.approx(0.0, abs=1e-9)
    assert f_u2_with_others["amount_zscore_user"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 13. ATM / unseen-channel behaviour
# --------------------------------------------------------------------------- #


def test_atm_unseen_channel_all_zero_onehot():
    """ATM channel not in feature_cols → all-zero one-hot (trained encoding)."""
    t = _make_txn(-500.0, TS0, channel="ATM", account_type="CHECKING", mcc=6011)
    feats = _service_build(t, [])
    # ch_POS should be 0 for an ATM transaction.
    assert feats["ch_POS"] == pytest.approx(0.0, abs=1e-9)
    # is_debit still reflects the negative amount.
    assert feats["is_debit"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 14. Missing-history (first-transaction) behaviour
# --------------------------------------------------------------------------- #


def test_missing_history_first_transaction():
    """A user's first transaction gets safe defaults, matching training."""
    t = _make_txn(-250.0, TS0)
    feats = _service_build(t, [])
    assert feats["amount_zscore_user"] == pytest.approx(0.0, abs=1e-9)
    assert feats["txn_count_24h"] == pytest.approx(1.0, abs=1e-9)  # current only
    assert feats["rolling_spend_7d"] == pytest.approx(250.0, abs=1e-9)
    assert feats["amount_pct_of_income"] == pytest.approx(250.0 / 5000.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 15. txn_count_24h with history
# --------------------------------------------------------------------------- #


def test_txn_count_24h_with_history():
    """txn_count_24h counts transactions within the trailing 24h window."""
    # Two prior transactions within 24h + current = 3
    h1 = _make_txn(-50.0, TS0 - timedelta(hours=2))
    h2 = _make_txn(-75.0, TS0 - timedelta(hours=5))
    current = _make_txn(-100.0, TS0)
    feats = _service_build(current, [h1, h2])
    assert feats["txn_count_24h"] == pytest.approx(3.0, abs=1e-9)

    # Prior transaction outside 24h window → not counted
    h_far = _make_txn(-50.0, TS0 - timedelta(hours=25))
    feats2 = _service_build(current, [h_far])
    assert feats2["txn_count_24h"] == pytest.approx(1.0, abs=1e-9)

