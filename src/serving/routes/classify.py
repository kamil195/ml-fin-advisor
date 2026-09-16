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
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.data.models import ClassificationResult, FraudAnalysis, Transaction
from src.serving.auth import AuthPrincipal, require_auth
from src.features.numerical_features import calculate_velocity
from src.utils.constants import (
    CategoryL1,
    CategoryL2,
    CATEGORY_HIERARCHY,
    Channel,
    DISCRETIONARY_CATEGORIES,
    lookup_category_by_mcc,
)
from src.utils.meta_input import build_meta_input

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
    attribution_method: str = Field(
        default="shap",
        description=(
            "How shap_features was computed. 'shap' = real per-transaction "
            "TreeExplainer attributions. 'global_importance' = fallback to the "
            "model's global feature importance, which is IDENTICAL for every "
            "request and is not an explanation of this prediction."
        ),
    )
    anchor_rule: str = Field(
        default="",
        description="Human-readable IF-THEN anchor rule",
    )
    fraud_analysis: FraudAnalysis = Field(
        default_factory=lambda: FraudAnalysis(velocity_flags=["no_history"]),
        description=(
            "Rule-based fraud-risk assessment: velocity counts, flags, score, "
            "and an is_suspicious verdict. Built from the transaction's own "
            "velocity window (no cross-user data)."
        ),
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
async def classify_transaction(
    request: ClassifyRequest,
    req: Request,
    principal: AuthPrincipal = Depends(require_auth),
):
    """HTTP entry point — history stays server-side, so it is always None here.

    Ownership (AUTH STEP 3): the transaction's ``user_id`` must equal the
    authenticated subject. A caller cannot classify a transaction attributed
    to another user (403).
    """
    if request.transaction.user_id != principal.sub:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: transaction user_id does not match the authenticated user.",
        )
    return await _classify(request, req, history=None)


async def classify_transaction_with_history(
    request: ClassifyRequest,
    req: Request,
    history: list[Transaction] | None = None,
) -> ClassifyResponse:
    """Server-side entry point for batch ingestion (NOT exposed via HTTP).

    ``history`` must contain the caller's strictly-prior transactions for the
    same user. Kept out of the decorated route's signature because a second
    Pydantic-model parameter would make FastAPI embed the request body and
    break the public ``{"transaction": {...}}`` contract.
    """
    return await _classify(request, req, history=history)


async def _classify(
    request: ClassifyRequest,
    req: Request,
    *,
    history: list[Transaction] | None,
) -> ClassifyResponse:
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

        from src.services.feature_service import FeatureService

        # C1 train/serve parity: the served feature vector MUST be produced by
        # the same history-aware feature functions the training pipeline uses.
        # ``history`` carries the strictly-prior transactions for this user
        # (single-txn callers pass None → first-transaction semantics).
        prior = history if history is not None else []
        service = FeatureService(all_cols)
        feat_dict = service.build(txn, prior)
        X_num = np.array([[feat_dict[c] for c in num_cols]], dtype=np.float64)

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
            meta_input = build_meta_input(proba.reshape(1, -1), X)
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
        shap_features, attribution_method = _compute_shap(
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

        # ── 11. Fraud-risk assessment (velocity + rule-based signals) ─────
        fraud_analysis = _compute_fraud_analysis(
            txn,
            velocity_window_minutes=10,
            max_expected_window=3,
        )

        return ClassifyResponse(
            category_l1=l1,
            category_l2=l2,
            confidence=round(confidence, 4),
            top_3=top_3,
            is_impulse=is_impulse,
            impulse_score=round(impulse_score, 2),
            shap_features=shap_features,
            attribution_method=attribution_method,
            anchor_rule=anchor_rule,
            fraud_analysis=fraud_analysis,
        )

    except Exception:
        # STEP 5: diagnostics stay server-side; clients get a generic message.
        logger.exception("Classification failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error.") from None


def _compute_shap(
    model,
    X: np.ndarray,
    feature_names: list[str],
    class_idx: int,
    top_k: int = 5,
) -> tuple[list[SHAPAttribution], str]:
    """
    Compute SHAP values using TreeExplainer when available,
    falling back to gain-based feature importance.

    Returns ``(attributions, method)`` where ``method`` is ``"shap"`` for real
    per-transaction attributions or ``"global_importance"`` for the fallback.
    The distinction matters: the fallback is a single global ranking that is
    *identical for every request*, so callers must not read it as an
    explanation of this particular prediction.
    """
    sv = None
    method = "shap"

    try:
        import shap
    except ImportError:
        # shap is not a declared dependency in requirements-serve.txt, so this
        # is the path production actually takes. Keep it separate from genuine
        # SHAP failures below rather than swallowing both in one except.
        method = "global_importance"
    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                # shap < ~0.45: list of (n_samples, n_features), one per class.
                sv = np.asarray(shap_values[class_idx])[0]
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    # shap >= ~0.45 multiclass: (n_samples, n_features, n_classes).
                    # Must select the predicted class's column, otherwise `sv`
                    # stays 2-D and the top-k loop below raises
                    # "truth value of an array ... is ambiguous".
                    sv = arr[0, :, class_idx]
                elif arr.ndim == 2:
                    # Binary / regression: (n_samples, n_features).
                    sv = arr[0]
                else:
                    sv = arr.ravel()
        except Exception:
            logger.warning("SHAP computation failed; using global importance", exc_info=True)
            sv = None
            method = "global_importance"

    if sv is None:
        if hasattr(model, "feature_importances_"):
            sv = np.asarray(model.feature_importances_, dtype=np.float64)
            method = "global_importance"
        else:
            return [], "global_importance"

    sv = np.asarray(sv, dtype=np.float64).ravel()
    if sv.shape[0] != len(feature_names):
        logger.warning(
            "Attribution length %d != feature count %d; skipping attributions",
            sv.shape[0],
            len(feature_names),
        )
        return [], method

    # Top-k features by absolute attribution
    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[::-1][:top_k]

    result = [
        SHAPAttribution(
            feature=feature_names[int(i)],
            value=round(float(X[0, int(i)]), 4),
            shap_value=round(float(sv[int(i)]), 4),
        )
        for i in top_idx
    ]
    return result, method


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


def _compute_fraud_analysis(
    txn: Transaction,
    history: list[Transaction] | None = None,
    velocity_window_minutes: int = 10,
    max_expected_window: int = 3,
) -> FraudAnalysis:
    """
    Rule-based fraud-risk assessment for a single transaction.

    Uses ``calculate_velocity`` (rolling count of prior transactions inside a
    short window) when ``history`` is supplied, plus transparent heuristics on
    amount, channel and time-of-day. The classify request currently carries a
    single transaction, so velocity is 0 unless combined with a batch view —
    the flags below therefore reflect amount/channel/timing risk factors that
    are meaningful without cross-transaction context.

    Cache-compatible: every field is JSON-serialisable and the result never
    depends on shared state between requests.
    """
    flags: list[str] = []
    velocity_count = 0

    if history:
        history_df = pd.DataFrame(
            [
                {"user_id": t.user_id, "timestamp": t.timestamp}
                for t in history
            ]
        )
        if not history_df.empty:
            prior = calculate_velocity(
                history_df, window_minutes=velocity_window_minutes
            )
            velocity_count = int(prior.max())

    # Rule-based signals — transparent, no PII
    amount = abs(txn.amount)
    hour = txn.timestamp.hour

    if amount >= 10_000:
        flags.append("high_amount")
    if txn.channel == Channel.ATM and amount >= 2_000:
        flags.append("large_atm_withdrawal")
    if amount >= 5_000 and (hour >= 22 or hour <= 4):
        flags.append("late_night_high_value")
    if amount <= 0:
        flags.append("non_spend_amount")
    if velocity_count > max_expected_window:
        flags.append(f"velocity_burst_{velocity_window_minutes}m")

    # Score: start 0, add signal weights, clamp to [0,1].
    # Hard flags (high amount, ATM, velocity burst) are each sufficient to
    # cross the suspicion threshold on their own; softer signals stack.
    score = 0.0
    if "high_amount" in flags:
        score += 0.55
    if "large_atm_withdrawal" in flags:
        score += 0.55
    if "late_night_high_value" in flags:
        score += 0.35
    if "non_spend_amount" in flags:
        score += 0.05
    if velocity_count > max_expected_window:
        # velocity burst is the strongest single indicator
        score += 0.55

    fraud_score = round(min(score, 1.0), 4)
    is_suspicious = fraud_score >= 0.5

    return FraudAnalysis(
        fraud_score=fraud_score,
        velocity_flags=flags,
        is_suspicious=is_suspicious,
    )
