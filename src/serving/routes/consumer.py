"""
Consumer-facing API endpoints for the freemium "5 free then pay" model.

Routes:
  POST /consumer/register      — create or retrieve user by email
  GET  /consumer/status/{uid}  — usage count, remaining, subscription info
  POST /consumer/analyse       — proxies to /v1/classify with free-tier gating
  POST /consumer/upgrade       — returns Lemon Squeezy checkout URL
  GET  /consumer/history/{uid} — past analysis results
"""

from __future__ import annotations

import logging
import os
import time


from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from src.serving.consumer import (
    CONSUMER_ANNUAL_ID,
    CONSUMER_MONTHLY_ID,
    FREE_TIER_LIMIT,
    LEMON_SQUEEZY_CHECKOUT_URL,
    AnalysisResult,
    SubStatus,
    consumer_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consumer", tags=["Consumer"])

# ── Request / Response schemas ──────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User e-mail address")
    display_name: str = Field("", description="Optional display name")


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    subscription_status: str
    free_tier_used: int
    free_tier_limit: int
    remaining_free: int


class StatusResponse(BaseModel):
    user_id: str
    email: str
    subscription_status: str
    free_tier_used: int
    free_tier_limit: int
    remaining_free: int
    can_analyse: bool
    total_analyses: int


class AnalyseRequest(BaseModel):
    user_id: str
    transaction: dict


class UpgradeRequest(BaseModel):
    user_id: str


class UpgradeResponse(BaseModel):
    monthly_url: str
    annual_url: str
    monthly_price: str
    annual_price: str
    annual_savings: str


class HistoryItem(BaseModel):
    id: str
    analysis_type: str
    request_summary: dict
    response_data: dict
    created_at: float


# ── Routes ──────────────────────────────────────────────────────


@router.post("/register", response_model=RegisterResponse)
async def register_user(body: RegisterRequest):
    """Register a new consumer user (or return existing by email)."""
    user = consumer_store.register(email=body.email, display_name=body.display_name)
    return RegisterResponse(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        subscription_status=user.subscription_status.value,
        free_tier_used=user.free_tier_used,
        free_tier_limit=FREE_TIER_LIMIT,
        remaining_free=user.remaining_free,
    )


@router.get("/status/{user_id}", response_model=StatusResponse)
async def get_status(user_id: str):
    """Return a user's usage stats and subscription status."""
    user = consumer_store.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return StatusResponse(
        user_id=user.user_id,
        email=user.email,
        subscription_status=user.subscription_status.value,
        free_tier_used=user.free_tier_used,
        free_tier_limit=FREE_TIER_LIMIT,
        remaining_free=user.remaining_free,
        can_analyse=user.can_analyse,
        total_analyses=len(user.analysis_history),
    )


@router.post("/analyse")
async def analyse_transaction(body: AnalyseRequest, request: Request):
    """
    Classify a transaction with free-tier gating.

    - Free users: up to 5 analyses total.
    - Paid users: unlimited.
    - Returns 402 Payment Required when free tier exhausted.
    """
    user = consumer_store.get(body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found. Register first.")

    if not user.can_analyse:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "free_tier_exhausted",
                "message": (
                    f"You've used all {FREE_TIER_LIMIT} free analyses. "
                    "Upgrade to a paid plan for unlimited access."
                ),
                "free_tier_used": user.free_tier_used,
                "free_tier_limit": FREE_TIER_LIMIT,
                "monthly_url": _build_checkout_url(user.user_id, user.email, CONSUMER_MONTHLY_ID),
                "annual_url": _build_checkout_url(user.user_id, user.email, CONSUMER_ANNUAL_ID),
            },
        )

    # ── Call the classify handler directly ─────────────────
    from src.data.models import Transaction
    from src.serving.routes.classify import ClassifyRequest, classify_transaction

    try:
        txn = Transaction(**body.transaction)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid transaction: {exc}")

    classify_req = ClassifyRequest(transaction=txn)
    result_response = await classify_transaction(classify_req, request)

    # ClassifyResponse is a Pydantic model; convert to dict
    result_data = result_response.model_dump(mode="json")

    # Record usage
    analysis = AnalysisResult(
        analysis_type="classify",
        request_summary={
            "merchant": body.transaction.get("merchant_name", ""),
            "amount": body.transaction.get("amount", 0),
        },
        response_data=result_data,
    )
    user.record_analysis(analysis)
    consumer_store.update(user)

    return {
        **result_data,
        "_consumer": {
            "free_tier_used": user.free_tier_used,
            "free_tier_limit": FREE_TIER_LIMIT,
            "remaining_free": user.remaining_free,
            "subscription_status": user.subscription_status.value,
        },
    }


@router.post("/upgrade", response_model=UpgradeResponse)
async def upgrade(body: UpgradeRequest):
    """Return Lemon Squeezy checkout URLs for monthly and annual plans."""
    user = consumer_store.get(body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return UpgradeResponse(
        monthly_url=_build_checkout_url(user.user_id, user.email, CONSUMER_MONTHLY_ID),
        annual_url=_build_checkout_url(user.user_id, user.email, CONSUMER_ANNUAL_ID),
        monthly_price="$4.99/month",
        annual_price="$49/year",
        annual_savings="Save 18%",
    )


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 20):
    """Return a user's past analysis results."""
    user = consumer_store.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    items = sorted(user.analysis_history, key=lambda a: a.created_at, reverse=True)[:limit]
    return {
        "user_id": user_id,
        "total": len(user.analysis_history),
        "items": [
            {
                "id": a.id,
                "analysis_type": a.analysis_type,
                "request_summary": a.request_summary,
                "response_data": a.response_data,
                "created_at": a.created_at,
            }
            for a in items
        ],
    }


# ── Helpers ─────────────────────────────────────────────────────


def _build_checkout_url(user_id: str, email: str, variant_id: str) -> str:
    """Build a Lemon Squeezy checkout URL with pre-fill parameters."""
    base = LEMON_SQUEEZY_CHECKOUT_URL
    # Lemon Squeezy checkout supports ?checkout[email]=…&checkout[custom][user_id]=…
    return (
        f"{base}/{variant_id}"
        f"?checkout[email]={email}"
        f"&checkout[custom][user_id]={user_id}"
    )
