"""
Live, key-free endpoints for external consumer apps (e.g. Planwisely).

Unlike /v1/forecast and /v1/budget (which only replay pre-computed
training-time demo data), these endpoints run real inference on
whatever transaction history the caller posts:

  POST /consumer/classify/live  - runs the trained LightGBM+meta-learner
                                   classifier on a single merchant/amount/date
  POST /consumer/forecast/live  - fits Prophet (statistical time-series model)
                                   per category on the posted transaction history
  POST /consumer/budget/live    - runs the scipy.optimize.linprog constraint
                                   budget optimiser on real category baselines

No LLM, no hardcoded "insight" text - every number returned here comes out
of an actual model or solver run against the caller's data.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.models.recommender.budget_optimizer import BudgetOptimiser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consumer", tags=["Live Forecast & Budget"])


# -- Category mapping: 30-class model taxonomy <-> Planwisely's 10 categories --

L2_TO_PLANWISELY: dict[str, str] = {
    "Rent/Mortgage": "Housing",
    "Home Insurance": "Housing",
    "Maintenance & Repairs": "Housing",
    "Utilities": "Utilities",
    "Groceries": "Food & Dining",
    "Restaurants": "Food & Dining",
    "Coffee Shops": "Food & Dining",
    "Food Delivery": "Food & Dining",
    "Alcohol & Bars": "Food & Dining",
    "Fuel": "Transport",
    "Public Transit": "Transport",
    "Ride-Share": "Transport",
    "Parking & Tolls": "Transport",
    "Vehicle Maintenance": "Transport",
    "Clothing & Accessories": "Shopping",
    "Electronics": "Shopping",
    "Books & Media": "Shopping",
    "Gifts & Donations": "Shopping",
    "Subscriptions & Streaming": "Subscriptions",
    "Hobbies & Sports": "Entertainment",
    "Healthcare & Pharmacy": "Health",
    "Fitness & Gym": "Health",
    "Personal Care": "Health",
    "Pet Care": "Health",
    "Savings & Investments": "Other",
    "Loan Payments": "Other",
    "Insurance Premiums": "Other",
    "Fees & Charges": "Other",
    "Taxes": "Other",
    "Income": "Other",
}

PLANWISELY_DISCRETIONARY: dict[str, bool] = {
    "Housing": False,
    "Food & Dining": True,
    "Transport": True,
    "Utilities": False,
    "Entertainment": True,
    "Subscriptions": True,
    "Health": False,
    "Shopping": True,
    "Crypto": True,
    "Other": False,
}

# Merchant-name keyword -> representative MCC, used only to satisfy the
# trained model's required feature input when the caller has no MCC.
_MCC_KEYWORDS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"grocery|mart|supermarket|whole\s?foods|carrefour|imtiaz|metro\b", re.I), 5411),
    (re.compile(r"restaurant|kitchen|grill|diner|biryani|bbq", re.I), 5812),
    (re.compile(r"bar|pub|brewery|wine", re.I), 5813),
    (re.compile(r"foodpanda|uber\s?eats|doordash|delivery|deliveroo", re.I), 5814),
    (re.compile(r"shell|total|petrol|gas\s?station|caltex|pso\b", re.I), 5541),
    (re.compile(r"metro\s?bus|train|transit|railway", re.I), 4111),
    (re.compile(r"uber|careem|lyft|taxi|ride", re.I), 4121),
    (re.compile(r"parking|toll", re.I), 7523),
    (re.compile(r"garage|auto\s?shop|car\s?service|mechanic", re.I), 7538),
    (re.compile(r"zara|h&m|clothing|fashion|apparel|outfitters", re.I), 5311),
    (re.compile(r"apple|samsung|electronics|laptop|phone\s?store", re.I), 5732),
    (re.compile(r"bookstore|kindle|book\b", re.I), 5942),
    (re.compile(r"gym|fitness|yoga", re.I), 7941),
    (re.compile(r"pharmacy|chemist|hospital|clinic|doctor", re.I), 8011),
    (re.compile(r"salon|spa|barber", re.I), 7298),
    (re.compile(r"vet|pet\s?store|petco", re.I), 742),
    (re.compile(r"electricity|water\s?bill|k-electric|sui\s?gas|wapda", re.I), 4900),
    (re.compile(r"netflix|spotify|streaming|subscription|prime\s?video", re.I), 5815),
    (re.compile(r"bank\s?fee|atm|service\s?charge", re.I), 6011),
]
_DEFAULT_MCC = 5999  # general retail / miscellaneous


def _guess_mcc(merchant: str) -> int:
    for pattern, mcc in _MCC_KEYWORDS:
        if pattern.search(merchant or ""):
            return mcc
    return _DEFAULT_MCC


def _guess_channel(merchant: str) -> str:
    if re.search(r"online|\.com|app|delivery|stream|subscription", merchant or "", re.I):
        return "ONLINE"
    return "POS"


# -- Shared request schema for posted transaction history ----------------------


class LiveTransaction(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    merchant: str = ""
    amount: float = Field(..., description="Negative = spend, positive = income")
    category: str | None = Field(default=None, description="Planwisely category, if already known")


# ============================================================================
# POST /consumer/classify/live
# ============================================================================


class ClassifyLiveRequest(BaseModel):
    merchant: str = Field(..., min_length=1)
    amount: float
    date: str = Field(..., description="YYYY-MM-DD")


@router.post("/classify/live")
async def classify_live(body: ClassifyLiveRequest, request: Request):
    """Run the real trained classifier on a single merchant/amount/date."""
    from src.data.models import Transaction
    from src.serving.routes.classify import ClassifyRequest, classify_transaction
    from src.utils.constants import AccountType, Channel

    try:
        ts = datetime.strptime(body.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        ts = datetime.now(timezone.utc)

    txn = Transaction(
        user_id="planwise-live",
        timestamp=ts,
        amount=-abs(body.amount),
        merchant_name=body.merchant,
        merchant_mcc=_guess_mcc(body.merchant),
        account_type=AccountType.CHECKING,
        channel=Channel(_guess_channel(body.merchant)),
        raw_description=body.merchant,
    )

    result = await classify_transaction(ClassifyRequest(transaction=txn), request)
    data = result.model_dump(mode="json")

    predicted_l2 = data["category_l2"]
    planwisely_category = L2_TO_PLANWISELY.get(predicted_l2, "Other")

    return {
        "category": planwisely_category,
        "model_category": predicted_l2,
        "confidence": data["confidence"],
        "top_3": data["top_3"],
        "is_impulse": data["is_impulse"],
        "anchor_rule": data["anchor_rule"],
        "shap_features": data["shap_features"],
        "attribution_method": data["attribution_method"],
    }


# ============================================================================
# POST /consumer/forecast/live
# ============================================================================


class ForecastLiveRequest(BaseModel):
    transactions: list[LiveTransaction]
    horizon_days: int = Field(default=30, ge=7, le=90)


@router.post("/forecast/live")
async def forecast_live(body: ForecastLiveRequest):
    """Fit Prophet per category on the caller's real transaction history."""
    from src.models.forecaster.prophet_model import ProphetModel

    rows = []
    for t in body.transactions:
        if t.amount >= 0:
            continue  # only spend, not income
        cat = t.category or "Other"
        try:
            d = datetime.strptime(t.date, "%Y-%m-%d")
        except ValueError:
            continue
        rows.append({"user_id": "planwise-live", "timestamp": d, "amount": t.amount, "category_l2": cat})

    if not rows:
        raise HTTPException(status_code=422, detail="No spend transactions to forecast from.")

    df = pd.DataFrame(rows)
    categories = sorted(df["category_l2"].unique())
    horizon_weeks = max(1, round(body.horizon_days / 7))

    results = []
    weekly_projected = [0.0] * horizon_weeks  # p50, summed across categories, by future week index
    for cat in categories:
        model = ProphetModel()
        model.fit(df, user_id="planwise-live", category=cat)
        fc = model.predict(user_id="planwise-live", category=cat, horizon_weeks=horizon_weeks)

        if not fc.p50:
            continue

        for i, v in enumerate(fc.p50):
            if i < horizon_weeks:
                weekly_projected[i] += v

        p10 = round(sum(fc.p10), 2)
        p50 = round(sum(fc.p50), 2)
        p90 = round(sum(fc.p90), 2)

        recent_actual = df[df["category_l2"] == cat]["amount"].abs().sum()
        span_days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)
        scaled_actual = recent_actual * (body.horizon_days / span_days)
        trend = "stable"
        if scaled_actual > 0:
            ratio = p50 / scaled_actual
            trend = "increasing" if ratio > 1.10 else "decreasing" if ratio < 0.90 else "stable"

        results.append({"category": cat, "p10": p10, "p50": p50, "p90": p90, "trend": trend})

    total_p10 = round(sum(r["p10"] for r in results), 2)
    total_p50 = round(sum(r["p50"] for r in results), 2)
    total_p90 = round(sum(r["p90"] for r in results), 2)

    # ── Weekly chart series: real historical actuals (cumulative) + real
    #    Prophet-projected future weeks (cumulative), so the frontend chart
    #    plots genuine numbers instead of a mock curve. ──
    df["week"] = pd.to_datetime(df["timestamp"]).dt.to_period("W").dt.start_time
    weekly_actual = df.groupby("week")["amount"].apply(lambda s: round(float(s.abs().sum()), 2)).sort_index()

    weekly_series = []
    cum = 0.0
    for week_start, amt in weekly_actual.items():
        cum += amt
        weekly_series.append({
            "label": week_start.strftime("%b %-d"),
            "actual": round(cum, 2),
            "projected": round(cum, 2),
        })

    last_actual_date = weekly_actual.index.max() if len(weekly_actual) else pd.Timestamp.now()
    for i, v in enumerate(weekly_projected):
        cum += v
        week_label = (last_actual_date + pd.Timedelta(weeks=i + 1)).strftime("%b %-d")
        weekly_series.append({
            "label": week_label,
            "actual": None,
            "projected": round(cum, 2),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": body.horizon_days,
        "weekly_series": weekly_series,
        "categories": results,
        "total_spend": {"p10": total_p10, "p50": total_p50, "p90": total_p90},
    }


# ============================================================================
# POST /consumer/budget/live
# ============================================================================


class BudgetLiveRequest(BaseModel):
    transactions: list[LiveTransaction]
    income: float = Field(..., ge=0)
    savings_target: float = Field(default=0, ge=0)


@router.post("/budget/live")
async def budget_live(body: BudgetLiveRequest):
    """Run the real scipy.optimize.linprog constraint optimiser on actual category spend."""
    baselines: dict[str, float] = defaultdict(float)
    for t in body.transactions:
        if t.amount >= 0:
            continue
        cat = t.category or "Other"
        baselines[cat] += abs(t.amount)

    if not baselines:
        raise HTTPException(status_code=422, detail="No spend transactions to budget from.")

    is_discretionary = {cat: PLANWISELY_DISCRETIONARY.get(cat, True) for cat in baselines}

    optimiser = BudgetOptimiser()
    result = optimiser.optimise(
        income=body.income,
        savings_target=body.savings_target,
        category_baselines=dict(baselines),
        is_discretionary=is_discretionary,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "income": body.income,
        "savings_target": body.savings_target,
        "solver_status": result.solver_status,
        "total_budget": result.total_budget,
        "savings_achieved": result.savings_achieved,
        "allocations": [
            {
                "category": a.category,
                "recommended_budget": a.budget,
                "current_spend": a.baseline,
                "cut_amount": a.cut_amount,
                "cut_pct": a.cut_pct,
                "is_discretionary": a.is_discretionary,
            }
            for a in sorted(result.allocations, key=lambda a: a.cut_amount, reverse=True)
        ],
    }
