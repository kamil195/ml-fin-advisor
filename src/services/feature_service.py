"""Shared, history-aware transaction feature generation (train/serve parity).

Single source of truth for the *numerical/temporal* classifier features.

Training (``run_pipeline.engineer_features``) computes these features on the
time-sorted per-user transaction history using:

* ``src.features.numerical_features.extract_numerical_features``
* ``src.features.temporal_features.extract_temporal_features``

plus the merchant/derived one-hot encodings (`mcc`, `is_debit`, `ch_*`,
`acct_*`) and the raw signed ``amount``.

This service replays exactly those same functions on the caller-supplied prior
history plus the current transaction (which is inserted as the *last* row even
when timestamps tie), guaranteeing the produced feature vector is identical to
what training would compute for that transaction. The text features
(TF-IDF → SVD) are handled separately by the serving route with the trained
artifacts; this service only produces the numeric block aligned to
``feature_columns.json``.

History semantics (must match training):

* ``amount_zscore_user`` — prior-only expanding z-score (current excluded);
  0.0 when fewer than 2 prior transactions or zero prior std (training
  ``min_periods=2`` behaviour).
* ``amount_pct_of_income`` — ``abs(amount) / max(income_basis, 1)`` with the
  same constant income basis used by training.
* ``rolling_spend_7d/30d`` — trailing-window sum of debits, *including* the
  current transaction (training ``min_periods=1`` behaviour).
* ``txn_count_24h`` — trailing-window transaction count, *including* the
  current transaction (training ``min_periods=1`` behaviour).
* ``is_holiday`` / ``days_since_payday`` — computed with the exact same
  temporal-feature functions as training.

Categorical one-hots (`ch_*`, `acct_*`) are aligned to the *trained* feature
columns (``feature_columns.json``), so channels/accounts that never appear in
training (e.g. ATM, if absent from the trained columns) produce all-zero
one-hot vectors — exactly the trained encoding — rather than inventing a new
model feature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data.models import Transaction
from src.features.numerical_features import (
    amount_pct_of_income as _amount_pct_of_income,
    amount_zscore_user as _amount_zscore_user,
    rolling_spend as _rolling_spend,
    txn_count_24h as _txn_count_24h,
)
from src.features.temporal_features import extract_temporal_features as _extract_temporal_features

# The exact income basis used during training (run_pipeline.engineer_features).
TRAINING_INCOME_BASIS = 5_000.0


class FeatureService:
    """Compute the numeric feature vector for a transaction given prior history."""

    def __init__(
        self,
        feature_cols: Sequence[str],
        estimated_monthly_income: float = TRAINING_INCOME_BASIS,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.num_cols = [c for c in self.feature_cols if not c.startswith("text_svd_")]
        self.text_cols = [c for c in self.feature_cols if c.startswith("text_svd_")]
        self.estimated_monthly_income = estimated_monthly_income

    # ── Public API ──────────────────────────────────────────────────────────

    def build(
        self,
        txn: Transaction,
        history: Sequence[Transaction] | None = None,
    ) -> dict[str, float]:
        """Return the numeric feature vector for ``txn`` aligned to trained columns.

        ``history`` must contain only strictly-prior transactions (same user).
        Empty history is first-transaction semantics (matches training).
        """
        frame = self._build_frame(txn, history or [])
        feats = self._compute_frame_features(frame)
        cur = int(frame.index[frame["_is_current"]][0])
        return {
            col: (float(feats.loc[cur, col]) if col in feats.columns else 0.0)
            for col in self.num_cols
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _row(self, t: Transaction, is_current: bool) -> dict:
        return {
            "user_id": t.user_id,
            "timestamp": pd.Timestamp(t.timestamp),
            "amount": float(t.amount),
            "merchant_mcc": float(t.merchant_mcc),
            "channel": t.channel.value,
            "account_type": t.account_type.value,
            "is_pending": t.is_pending,
            "_is_current": is_current,
        }

    def _build_frame(
        self,
        txn: Transaction,
        history: Sequence[Transaction],
    ) -> pd.DataFrame:
        rows = [self._row(t, is_current=False) for t in history]
        rows.append(self._row(txn, is_current=True))
        frame = pd.DataFrame(rows)
        # Sort chronologically per user; the current transaction is always last
        # even when timestamps tie (the batch processes rows sequentially and
        # never lets later rows influence earlier ones).
        frame = frame.sort_values(
            ["user_id", "timestamp", "_is_current"]
        ).reset_index(drop=True)
        return frame

    def _compute_frame_features(self, df: pd.DataFrame) -> pd.DataFrame:
        num = pd.DataFrame(index=df.index)
        num["amount"] = df["amount"].astype(float)
        num["log_amount"] = np.log1p(df["amount"].abs())
        num["amount_zscore_user"] = _amount_zscore_user(df)
        num["amount_pct_of_income"] = _amount_pct_of_income(
            df["amount"], self.estimated_monthly_income
        )
        num["rolling_spend_7d"] = _rolling_spend(df, window_days=7)
        num["rolling_spend_30d"] = _rolling_spend(df, window_days=30)
        num["txn_count_24h"] = _txn_count_24h(df)

        temp = _extract_temporal_features(df)

        merchant = pd.DataFrame(index=df.index)
        merchant["mcc"] = df["merchant_mcc"].astype(float)
        merchant["is_debit"] = (df["amount"] < 0).astype(float)
        for col in self.num_cols:
            if col.startswith("ch_"):
                merchant[col] = (df["channel"] == col[3:]).astype(float)
            elif col.startswith("acct_"):
                merchant[col] = (df["account_type"] == col[5:]).astype(float)

        merged = pd.concat([num, temp, merchant], axis=1)
        # Keep only trained numeric columns (in trained order). Anything
        # missing (shouldn't be, but defensive) defaults to 0.0 at build time.
        return merged[[c for c in self.num_cols if c in merged.columns]]