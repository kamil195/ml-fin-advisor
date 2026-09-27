"""
GET /consumer/forecast — user-specific spend forecast (STEP 13).

Answers a product-relevant question with the caller's **own** data:

    "What spending is likely for me over the next N days?"

Pipeline (all owner-scoped, deterministic):

1. ``principal.sub`` (verified JWT subject) → ``store.fetch_spend_history``
   (Step 12 repository; every query scoped by ``owner_sub``).
2. Rows → zero-filled daily spend series (``personal_forecast.build_daily_series``).
3. Series → explicit quality tier. Insufficient history returns
   ``personalization_status="insufficient_history"`` with **no** number and a
   ``requirements`` block — a global artifact is never substituted.
4. Tier A/B → ``personal_forecast.predict`` (method selected by the offline
   benchmark; both the primary and the fallback are user-specific).

Truthfulness (STEP 13K): the response always carries ``personalization_status``,
``method``, ``engine_version``, ``history_days_used`` and
``transaction_count_used``. ``personalized`` is used only when Tier A history of
the caller's own was actually used.

Cache (STEP 13I): ``forecasts:<sub>:personal:<horizon>:<engine_version>`` — user
scoped, horizon scoped, engine-version aware, TTL-bounded. Step 12's
``purge_user`` (used by ingest, delete and profile mutations) already clears the
``forecasts:<sub>:*`` namespace, so a mutation cannot leave a stale forecast.
Insufficient-history responses are deliberately not cached (they change as soon
as the user ingests data).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.data.models import PersonalForecastResult
from src.models.forecaster import personal_forecast as pf
from src.serving.auth import AuthPrincipal, require_auth
from src.serving.cache import CACHE_TTLS
from src.serving.persistence import (
    DETAIL_503,
    DETAIL_NOT_CONFIGURED,
    PersistenceUnavailable,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consumer", tags=["Forecasting"])

#: History window fetched from Postgres. Bounded so a long-standing account
#: cannot pull an unbounded number of rows into one request.
LOOKBACK_DAYS = 120

CACHE_NAMESPACE = "forecasts"

_MSG_INSUFFICIENT = (
    "Not enough of your own transaction history for a personalized forecast yet. "
    "Ingest more transactions to enable it. No global forecast and no other user's "
    "data is substituted."
)
_MSG_LIMITED = (
    "Limited history (less than four weeks of your own data): the forecast is "
    "personalized but low-confidence."
)
_MSG_FALLBACK = (
    "The preferred method could not be applied to this history; another "
    "user-specific method was used instead (never a global forecast)."
)


def _cache_parts(sub: str, horizon_days: int) -> tuple[str, ...]:
    """Cache key parts: identity, scope, horizon and engine version only.

    Never contains bearer tokens, emails, or transaction text.
    """
    return (sub, "personal", str(horizon_days), pf.FORECAST_ENGINE_VERSION)


@router.get("/forecast", response_model=PersonalForecastResult)
def get_personal_forecast(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
    horizon_days: int = Query(
        default=pf.DEFAULT_HORIZON_DAYS,
        ge=pf.MIN_HORIZON_DAYS,
        le=pf.MAX_HORIZON_DAYS,
        description="Forecast horizon in days (7-90).",
    ),
) -> PersonalForecastResult:
    """Forecast the caller's own future spend from their persisted history.

    Identity is ``principal.sub`` only: there is no user id in the path, query
    or body, so no request field can select whose history is forecast.
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=DETAIL_NOT_CONFIGURED)

    cache = getattr(request.app.state, "cache", None)
    parts = _cache_parts(principal.sub, horizon_days)
    if cache is not None:
        cached = cache.get(CACHE_NAMESPACE, *parts)
        if cached is not None:
            logger.info("Cache HIT for personalized forecast (user-scoped)")
            return PersonalForecastResult(**cached)

    since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    try:
        rows = store.fetch_spend_history(principal.sub, since)
    except PersistenceUnavailable:
        # Fail closed with a generic message: no SQL, DSN or table names.
        raise HTTPException(status_code=503, detail=DETAIL_503) from None

    try:
        series = pf.build_daily_series(
            [(row["occurred_at"], row["amount"]) for row in rows]
        )
        now = datetime.now(timezone.utc)
        tier = pf.quality_tier(series)

        if tier == pf.STATUS_INSUFFICIENT:
            # No number is produced and nothing is cached: the honest answer is
            # "not enough of your own data yet", never a borrowed forecast.
            return PersonalForecastResult(
                user_id=principal.sub,
                generated_at=now,
                engine_version=pf.FORECAST_ENGINE_VERSION,
                method=pf.PRIMARY_METHOD,
                personalization_status=pf.STATUS_INSUFFICIENT,
                fallback_status=pf.FALLBACK_NONE,
                horizon_days=horizon_days,
                interval_width=0.80,
                expected_spend=None,
                total={},
                daily=[],
                history_days_used=series.history_days,
                transaction_count_used=series.transaction_count,
                distinct_spend_days_used=series.distinct_spend_days,
                residual_sigma=0.0,
                message=_MSG_INSUFFICIENT,
                requirements=pf.insufficient_history_detail(series),
            )

        payload: dict[str, Any] = pf.predict(series, horizon_days)
        if tier == pf.STATUS_LIMITED:
            payload["message"] = _MSG_LIMITED
        elif payload.get("fallback_status") != pf.FALLBACK_NONE:
            payload["message"] = _MSG_FALLBACK

        result = PersonalForecastResult(
            user_id=principal.sub, generated_at=now, **payload
        )
    except HTTPException:
        raise
    except Exception:
        # STEP 5 convention: diagnostics server-side only, generic client error.
        logger.exception("Personalized forecast generation failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error.") from None

    if cache is not None:
        cache.set(
            CACHE_NAMESPACE,
            *parts,
            value=result.model_dump(mode="json"),
            ttl=CACHE_TTLS["forecasts"],
        )
    return result