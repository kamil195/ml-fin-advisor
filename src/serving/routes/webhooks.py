"""
Lemon Squeezy webhook handler.

Receives subscription lifecycle events, verifies HMAC signatures,
and updates the in-memory subscription / API-key stores.

Endpoint: POST /webhooks/lemonsqueezy
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time

from fastapi import APIRouter, HTTPException, Request

from src.serving.middleware import generate_api_key
from src.serving.subscriptions import (
    PlanTier,
    Subscription,
    plan_for_variant,
    subscription_store,
    usage_tracker,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Webhooks"])

# Events we care about
_HANDLED_EVENTS = {
    "subscription_created",
    "subscription_updated",
    "subscription_cancelled",
    "subscription_payment_success",
    "subscription_payment_failed",
    "subscription_expired",
}


# ── Signature verification ──────────────────────────────────────


def _verify_signature(payload: bytes, signature: str | None) -> bool:
    """
    Verify the Lemon Squeezy HMAC-SHA256 webhook signature.

    Lemon Squeezy sends the hex-encoded HMAC in the
    ``X-Signature`` header.
    """
    secret = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("LEMONSQUEEZY_WEBHOOK_SECRET not set — skipping verification")
        return True  # dev/test convenience; disable in prod

    if not signature:
        return False

    expected = hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# ── Helpers ─────────────────────────────────────────────────────


def _extract_subscription_fields(data: dict) -> dict:
    """Pull the fields we need from the webhook payload."""
    attrs = data.get("data", {}).get("attributes", {})
    relationships = data.get("data", {}).get("relationships", {})

    # variant_id can be in attributes or first_subscription_item
    variant_id = attrs.get("variant_id") or attrs.get("first_subscription_item", {}).get("variant_id")
    if variant_id is None:
        # Try relationships
        variant_data = relationships.get("variant", {}).get("data", {})
        variant_id = variant_data.get("id")

    return {
        "subscription_id": int(data.get("data", {}).get("id", 0)),
        "customer_email": attrs.get("user_email", ""),
        "variant_id": int(variant_id) if variant_id else None,
        "status": attrs.get("status", ""),
    }


def _get_or_provision_key(email: str) -> str:
    """Return existing API key for email, or generate a new one."""
    existing = subscription_store.get_by_email(email)
    if existing:
        return existing.api_key
    return generate_api_key(prefix="fina")


# ── Webhook route ───────────────────────────────────────────────


@router.post("/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request):
    """
    Handle Lemon Squeezy subscription webhook events.

    Events handled:
      - subscription_created → provision API key, set plan
      - subscription_updated → update plan / status
      - subscription_cancelled → deactivate key
      - subscription_payment_success → ensure active status
      - subscription_payment_failed → mark as past_due
      - subscription_expired → deactivate key
    """
    body = await request.body()
    signature = request.headers.get("X-Signature")

    if not _verify_signature(body, signature):
        logger.warning("Webhook signature verification failed")
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = await request.json()
    meta = payload.get("meta", {})
    event_name = meta.get("event_name", "")

    if event_name not in _HANDLED_EVENTS:
        logger.debug("Ignoring unhandled webhook event: %s", event_name)
        return {"status": "ignored", "event": event_name}

    fields = _extract_subscription_fields(payload)
    sub_id = fields["subscription_id"]
    email = fields["customer_email"]
    variant_id = fields["variant_id"]
    ls_status = fields["status"]

    logger.info(
        "Webhook [%s] sub_id=%d email=%s variant=%s ls_status=%s",
        event_name, sub_id, email, variant_id, ls_status,
    )

    # ── subscription_created ────────────────────────────────
    if event_name == "subscription_created":
        if variant_id is None:
            logger.error("subscription_created without variant_id")
            return {"status": "error", "detail": "missing variant_id"}

        plan = plan_for_variant(variant_id)
        api_key = _get_or_provision_key(email)

        sub = Subscription(
            api_key=api_key,
            customer_email=email,
            plan=plan,
            subscription_id=sub_id,
            status="active",
        )
        subscription_store.upsert(sub)

        # Dynamically add the key to the valid-keys set
        _add_to_api_keys_env(api_key)

        logger.info(
            "Provisioned key=%s…%s plan=%s for %s",
            api_key[:8], api_key[-4:], plan.name, email,
        )
        return {
            "status": "ok",
            "event": event_name,
            "api_key": api_key,
            "plan": plan.name,
        }

    # ── subscription_updated ────────────────────────────────
    if event_name == "subscription_updated":
        existing = subscription_store.get_by_sub_id(sub_id)
        if not existing:
            logger.warning("subscription_updated for unknown sub_id=%d", sub_id)
            return {"status": "ignored", "detail": "unknown subscription"}

        # Plan change
        if variant_id is not None:
            try:
                new_plan = plan_for_variant(variant_id)
                existing.plan = new_plan
                logger.info("Plan changed to %s for sub_id=%d", new_plan.name, sub_id)
            except ValueError:
                logger.warning("Unknown variant_id %s in update", variant_id)

        # Status change
        if ls_status in ("active", "paused", "cancelled", "expired", "past_due"):
            existing.status = ls_status
            if ls_status not in ("active",):
                _remove_from_api_keys_env(existing.api_key)
            else:
                _add_to_api_keys_env(existing.api_key)

        existing.updated_at = time.time()
        subscription_store.upsert(existing)
        return {"status": "ok", "event": event_name}

    # ── subscription_cancelled / subscription_expired ───────
    if event_name in ("subscription_cancelled", "subscription_expired"):
        existing = subscription_store.get_by_sub_id(sub_id)
        if existing:
            existing.status = "cancelled" if "cancelled" in event_name else "expired"
            existing.updated_at = time.time()
            subscription_store.upsert(existing)
            _remove_from_api_keys_env(existing.api_key)
            logger.info("Deactivated key for sub_id=%d (%s)", sub_id, event_name)
        return {"status": "ok", "event": event_name}

    # ── subscription_payment_success ────────────────────────
    if event_name == "subscription_payment_success":
        existing = subscription_store.get_by_sub_id(sub_id)
        if existing:
            existing.status = "active"
            existing.updated_at = time.time()
            subscription_store.upsert(existing)
            _add_to_api_keys_env(existing.api_key)
            logger.info("Payment success → reactivated sub_id=%d", sub_id)
        return {"status": "ok", "event": event_name}

    # ── subscription_payment_failed ─────────────────────────
    if event_name == "subscription_payment_failed":
        existing = subscription_store.get_by_sub_id(sub_id)
        if existing:
            existing.status = "past_due"
            existing.updated_at = time.time()
            subscription_store.upsert(existing)
            logger.warning("Payment failed → past_due for sub_id=%d", sub_id)
        return {"status": "ok", "event": event_name}

    return {"status": "ignored", "event": event_name}


# ── Runtime API_KEYS env-var management ─────────────────────────


def _add_to_api_keys_env(api_key: str) -> None:
    """Append a key to the live API_KEYS env var."""
    raw = os.environ.get("API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    keys.add(api_key)
    os.environ["API_KEYS"] = ",".join(keys)


def _remove_from_api_keys_env(api_key: str) -> None:
    """Remove a key from the live API_KEYS env var."""
    raw = os.environ.get("API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    keys.discard(api_key)
    os.environ["API_KEYS"] = ",".join(keys)
