"""
GET /v1/budget/{user_id} — Budget recommendations endpoint (SPEC §11.2.3).

Uses pre-computed budget optimisation results and generates real
SHAP attributions, anchor rules, and counterfactual explanations
via the ``ExplanationEngine``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from src.data.models import BudgetRecommendation, BudgetResult, SHAPFeature
from src.models.recommender.explanations import ExplanationEngine
from src.utils.constants import CategoryL2, DISCRETIONARY_CATEGORIES

logger = logging.getLogger(__name__)

router = APIRouter()

_explanation_engine = ExplanationEngine()


@router.get("/budget/{user_id}", response_model=BudgetResult)
async def get_budget(user_id: str, req: Request):
    """
    Retrieve personalised budget recommendations for a user.

    Each recommendation includes SHAP feature attributions,
    an anchor rule, and a counterfactual explanation.
    """
    state = req.app.state
    cache = state.cache

    # Check cache
    cached = cache.get("budgets", user_id)
    if cached is not None:
        logger.info("Cache HIT for budget %s", user_id)
        return BudgetResult(**cached)

    try:
        budget_data = state.budget_data
        if budget_data is None:
            raise HTTPException(
                status_code=503,
                detail="Budget data not available. Run the training pipeline first.",
            )

        # Find user-specific budget or use aggregate
        user_budgets = budget_data.get("user_budgets", {})

        # Try matching user ID prefix
        user_budget = None
        for uid, bdata in user_budgets.items():
            if uid == user_id or uid.startswith(user_id[:12]):
                user_budget = bdata
                break

        # If no exact match, use first available user as demo
        if user_budget is None and user_budgets:
            first_key = next(iter(user_budgets))
            user_budget = user_budgets[first_key]
            logger.info("User %s not found, using %s as demo", user_id, first_key)

        if user_budget is None:
            raise HTTPException(status_code=404, detail=f"No budget data for user {user_id}")

        income = user_budget.get("income", 0)
        total_budget = user_budget.get("total_budget", 0)
        category_budgets = user_budget.get("category_budgets", {})
        category_actuals = user_budget.get("category_actuals", {})

        savings_target = round(income * 0.10, 2)

        recommendations = []
        for cat_name, budget_val in category_budgets.items():
            actual = category_actuals.get(cat_name, budget_val)

            try:
                cat_enum = CategoryL2(cat_name)
            except ValueError:
                continue

            # Build feature dict for SHAP explanation
            features = {
                "monthly_spend": actual,
                "budget_ceiling": budget_val,
                "spend_to_income_ratio": actual / max(income, 1) * 100,
                "is_discretionary": 1.0 if cat_enum in DISCRETIONARY_CATEGORIES else 0.0,
                "over_budget_pct": max(0, (actual - budget_val) / max(budget_val, 1) * 100),
            }

            user_stats = {
                "monthly_spend": actual,
                "frequency": actual / max(abs(budget_val / 30), 1),
                "avg_transaction": actual / max(actual / 50, 1),
            }

            # Generate full explanation via ExplanationEngine
            explanation = _explanation_engine.explain(
                category=cat_name,
                baseline=actual,
                budget=budget_val,
                features=features,
                user_stats=user_stats,
            )

            # Convert SHAP to response format
            shap_features = [
                SHAPFeature(feature=f["feature"], impact=f["impact"])
                for f in explanation.shap.top_k(5)
            ]

            cut_pct = (actual - budget_val) / max(actual, 1) * 100
            confidence = min(0.95, 0.70 + abs(cut_pct) / 100)

            recommendations.append(BudgetRecommendation(
                category=cat_enum,
                recommended_budget=round(budget_val, 2),
                current_trend=round(actual, 2),
                confidence=round(confidence, 2),
                explanation=explanation.narrative,
                shap_top_features=shap_features,
                anchor_rule=explanation.anchor.rule,
                counterfactual=explanation.counterfactual.text,
            ))

        # Sort by largest potential savings first
        recommendations.sort(
            key=lambda r: r.current_trend - r.recommended_budget,
            reverse=True,
        )

        result = BudgetResult(
            user_id=user_id,
            period=datetime.now(timezone.utc).strftime("%Y-%m"),
            income_estimate=round(income, 2),
            savings_target=savings_target,
            recommendations=recommendations[:10],  # top-10 categories
        )

        # Cache the result
        cache.set("budgets", user_id, value=result.model_dump(mode="json"))

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Budget generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
