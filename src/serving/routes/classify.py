"""
POST /v1/classify — Transaction classification endpoint (SPEC §11.2.1).

Runs a real LightGBM + XGBoost meta-learner pipeline on the transaction,
returning predicted category, confidence, top-3, and SHAP feature attributions.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.data.models import ClassificationResult, Transaction
from src.utils.constants import (
    CategoryL1,
    CategoryL2,
    CATEGORY_HIERARCHY,
    DISCRETIONARY_CATEGORIES,
    lookup_category_by_mcc,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────


class SHAPAttribution(BaseModel):
    """Single SHAP feature attribution."""
    feature: str
    value: float = Field(description="Feature value used for prediction")
    shap_value: float = Field(description="SHAP attribution (positive pushes toward this class)")


class ClassifyRequest(BaseModel):
    """Request body for transaction classification."""
    transaction: Transaction


class ClassifyResponse(ClassificationResult):
    """Response body — ClassificationResult + SHAP explanations."""
    shap_features: list[SHAPAttribution] = Field(
        default_factory=list,
        description="Top SHAP feature attributions for the predicted class",
    )
    anchor_rule: str = Field(
        default="",
        description="Human-readable IF-THEN anchor rule",
    )


# ── Feature engineering helpers (mirrors run_pipeline.py logic) ────────────────


def _build_features(txn: Transaction, feature_cols: list[str]) -> dict[str, float]:
    """Build the numerical feature vector for a single transaction."""
    ts = txn.timestamp
    features: dict[str, float] = {}

    # Numerical
    features["amount"] = txn.amount
    features["log_amount"] = math.log1p(abs(txn.amount))
    features["amount_zscore_user"] = 0.0  # single txn — no user history
    features["amount_pct_of_income"] = 0.0
    features["rolling_spend_7d"] = abs(txn.amount)
    features["rolling_spend_30d"] = abs(txn.amount)

    # Temporal
    features["txn_count_24h"] = 1
    hour = ts.hour if ts else 12
    features["hour_of_day_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_of_day_cos"] = math.cos(2 * math.pi * hour / 24)
    dow = ts.weekday() if ts else 0
    features["day_of_week_sin"] = math.sin(2 * math.pi * dow / 7)
    features["day_of_week_cos"] = math.cos(2 * math.pi * dow / 7)
    dom = ts.day if ts else 15
    features["day_of_month_sin"] = math.sin(2 * math.pi * dom / 31)
    features["day_of_month_cos"] = math.cos(2 * math.pi * dom / 31)
    features["is_weekend"] = 1.0 if dow >= 5 else 0.0
    features["is_holiday"] = 0.0
    features["month_phase_early"] = 1.0 if dom <= 10 else 0.0
    features["month_phase_mid"] = 1.0 if 11 <= dom <= 20 else 0.0
    features["month_phase_late"] = 1.0 if dom > 20 else 0.0
    features["days_since_payday"] = min(dom, 30 - dom)

    # Merchant / channel
    features["mcc"] = txn.merchant_mcc
    mcc_cat = lookup_category_by_mcc(txn.merchant_mcc)
    features["is_discretionary"] = 1.0 if mcc_cat in DISCRETIONARY_CATEGORIES else 0.0
    features["is_debit"] = 1.0 if txn.amount < 0 else 0.0

    # One-hot channels
    for ch in ("ONLINE", "POS", "RECURRING", "TRANSFER"):
        features[f"ch_{ch}"] = 1.0 if txn.channel.value == ch else 0.0

    # One-hot accounts
    for at in ("CHECKING", "CREDIT", "INVESTMENT", "SAVINGS"):
        features[f"acct_{at}"] = 1.0 if txn.account_type.value == at else 0.0

    # Align to trained feature columns (text_svd handled separately)
    result = {}
    for col in feature_cols:
        result[col] = features.get(col, 0.0)

    return result


# ── Endpoint ───────────────────────────────────────────────────────────────────


@router.post("/classify", response_model=ClassifyResponse)
async def classify_transaction(request: ClassifyRequest, req: Request):
    """
    Classify a single transaction into a spending category.

    Returns the predicted L1/L2 category, confidence, top-3 predictions,
    SHAP feature attributions, and an anchor rule.
    """
    txn = request.transaction
    state = req.app.state

    # Guard: model must be loaded
    if state.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model not loaded. Run the training pipeline first.",
        )

    try:
        # ── 1. Build numerical feature vector ─────────────────────
        # feature_cols contains [text_svd_0..127, numerical_col_0..29]
        all_cols = state.feature_cols
        text_cols = [c for c in all_cols if c.startswith("text_svd_")]
        num_cols = [c for c in all_cols if not c.startswith("text_svd_")]

        feat_dict = _build_features(txn, num_cols)
        X_num = np.array([[feat_dict.get(c, 0.0) for c in num_cols]], dtype=np.float64)

        # ── 2. TF-IDF text features ──────────────────────────────
        if state.tfidf is not None and state.svd is not None:
            text = (txn.merchant_name or "") + " " + (txn.raw_description or "")
            tfidf_vec = state.tfidf.transform([text])
            X_text = state.svd.transform(tfidf_vec)
        else:
            X_text = np.zeros((1, len(text_cols)), dtype=np.float64)

        # ── 3. Scale numerical features only ──────────────────────
        if state.scaler is not None:
            X_num = state.scaler.transform(X_num)

        # ── 4. Concatenate [text, numerical] to match training order ──
        X = np.hstack([X_text, X_num])
        feature_names = text_cols + num_cols

        # ── 5. Predict with base model ────────────────────────────
        proba = state.classifier.predict_proba(X)[0]
        classes = state.label_encoder.classes_

        # ── 6. Meta-learner stacking (only if base is uncertain) ──────
        if state.meta_model is not None and proba.max() < 0.90:
            meta_input = np.hstack([X, proba.reshape(1, -1)])
            proba = state.meta_model.predict_proba(meta_input)[0]

        # ── 7. Top-3 predictions ──────────────────────────────────
        top_indices = np.argsort(proba)[::-1][:3]
        top_3 = []
        for idx in top_indices:
            cat_name = classes[idx]
            try:
                cat_enum = CategoryL2(cat_name)
            except ValueError:
                cat_enum = CategoryL2.FEES_CHARGES
            top_3.append({"category": cat_enum, "confidence": round(float(proba[idx]), 4)})

        predicted_idx = top_indices[0]
        predicted_cat = classes[predicted_idx]
        confidence = float(proba[predicted_idx])

        try:
            l2 = CategoryL2(predicted_cat)
        except ValueError:
            l2 = CategoryL2.FEES_CHARGES
        l1 = CATEGORY_HIERARCHY.get(l2, CategoryL1.FINANCIAL)

        # ── 8. SHAP feature attributions ──────────────────────────
        shap_features = _compute_shap(
            state.classifier, X, feature_names, predicted_idx
        )

        # ── 9. Anchor rule ────────────────────────────────────────
        anchor_rule = _build_anchor_rule(feat_dict, l2, txn)

        # ── 10. Impulse assessment ────────────────────────────────
        is_impulse = (
            l2 in DISCRETIONARY_CATEGORIES
            and abs(txn.amount) > 50
            and (txn.timestamp.hour >= 22 or txn.timestamp.hour <= 5)
        )
        impulse_score = 0.7 if is_impulse else 0.1

        return ClassifyResponse(
            category_l1=l1,
            category_l2=l2,
            confidence=round(confidence, 4),
            top_3=top_3,
            is_impulse=is_impulse,
            impulse_score=round(impulse_score, 2),
            shap_features=shap_features,
            anchor_rule=anchor_rule,
        )

    except Exception as exc:
        logger.exception("Classification failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _compute_shap(
    model,
    X: np.ndarray,
    feature_names: list[str],
    class_idx: int,
    top_k: int = 5,
) -> list[SHAPAttribution]:
    """
    Compute SHAP values using TreeExplainer when available,
    falling back to gain-based feature importance.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # shap_values shape: (n_classes, n_samples, n_features) or (n_samples, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]
        else:
            sv = shap_values[0]
    except Exception:
        # Fallback: use gain-based feature importance
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            sv = fi.copy()
        else:
            return []

    # Top-k features by absolute SHAP value
    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[::-1][:top_k]

    result = []
    for i in top_idx:
        if i < len(feature_names):
            result.append(SHAPAttribution(
                feature=feature_names[i],
                value=round(float(X[0, i]), 4),
                shap_value=round(float(sv[i]), 4),
            ))
    return result


def _build_anchor_rule(
    features: dict[str, float],
    predicted_cat: CategoryL2,
    txn: Transaction,
) -> str:
    """Build a human-readable IF-THEN anchor rule."""
    conditions = []

    # MCC range
    conditions.append(f"MCC = {txn.merchant_mcc}")

    # Amount range
    amt = abs(txn.amount)
    if amt < 20:
        conditions.append("amount < $20")
    elif amt < 100:
        conditions.append("$20 ≤ amount < $100")
    elif amt < 500:
        conditions.append("$100 ≤ amount < $500")
    else:
        conditions.append("amount ≥ $500")

    # Channel
    conditions.append(f"channel = {txn.channel.value}")

    # Time of day
    hour = txn.timestamp.hour
    if hour < 6:
        conditions.append("time = late-night")
    elif hour < 12:
        conditions.append("time = morning")
    elif hour < 18:
        conditions.append("time = afternoon")
    else:
        conditions.append("time = evening")

    rule = f"IF {' AND '.join(conditions)} THEN category = {predicted_cat.value}"
    return rule
