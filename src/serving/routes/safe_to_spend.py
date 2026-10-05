"""
Safe-to-Spend endpoints (STEP 14).

    GET  /consumer/safe-to-spend           — "what can I safely spend before payday?"
    POST /consumer/safe-to-spend/scenario  — same, with an explicit scenario

Answers the core product question **"Know what you can safely spend before
payday."** entirely from the caller's own persisted data:

1. ``principal.sub`` (verified JWT subject) → ``get_profile`` (cash position,
   payday, safety buffer) and ``fetch_financial_history`` (transactions). No
   request field can select whose data is used — identity is never in the URL.
2. Payday: the profile's ``next_payday`` when it is in the future, else a cadence
   measured from the caller's own income deposits, else the ``missing_payday``
   status. A payday is never assumed.
3. Expected spending: the STEP 13 **user-specific** forecast over
   ``today+1 … payday`` (never the legacy global/static artifact, never an
   arbitrary 30-day window).
4. The arithmetic itself lives in ``src/services/safe_to_spend.py`` — this module
   only wires owner-scoped inputs to it and maps the result to the response
   model. The backend is authoritative: nothing is recalculated in the frontend.

Honesty rules (enforced by tests):

* A missing input produces an explicit status (``missing_payday`` /
  ``missing_balance`` / ``missing_profile_data``), never a fabricated number.
* ``insufficient_history`` produces no expected spending and therefore no amount.
* ``limited_history`` amounts are labelled as low-confidence.
* A negative amount is returned as-is (never floored to zero) with
  ``is_negative`` set, because it states a real shortfall.

Cache (STEP 14J): ``safe_to_spend:<sub>:<payday>:<scenario_hash>:<engine>`` —
user-scoped, payday-scoped, engine-version aware and TTL-bounded (1 h). Cached
entries are only reused while their ``as_of`` is the current date, so a cached
response can never carry a stale "today". Statuses without an amount are never
cached. Step 12's ``purge_user`` (ingest, delete, profile mutation) clears the
whole ``safe_to_spend:<sub>:*`` prefix, so a changed input cannot leave a stale
amount behind.

Scenarios never persist anything: the request body is used for the arithmetic of
that single response only, and no store method is called for it.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.data.models import (
    SafeToSpendComponent,
    SafeToSpendObligation,
    SafeToSpendResult,
    SafeToSpendScenario,
    SafeToSpendScenarioRequest,
)
from src.models.forecaster import personal_forecast as pf
from src.services import safe_to_spend as sts
from src.serving.auth import AuthPrincipal, require_auth
from src.serving.cache import CACHE_TTLS
from src.serving.persistence import (
    DETAIL_503,
    DETAIL_NOT_CONFIGURED,
    PersistenceUnavailable,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consumer", tags=["Safe to Spend"])

#: Bounded history window (same as STEP 13): enough for weekday profiles and
#: monthly obligation cadences, without pulling an unbounded number of rows.
LOOKBACK_DAYS = 120

CACHE_NAMESPACE = "safe_to_spend"

#: Cache-key component for the no-scenario GET.
SCENARIO_HASH_NONE = "none"

_MSG_LIMITED = (
    "Limited history (less than four weeks of your own non-obligation spending): "
    "this amount is personalized but low-confidence."
)
_MSG_NO_AMOUNT = (
    "Safe to Spend is not available yet. No amount is shown because a required "
    "input or enough of your own history is missing — see 'requirements'."
)
_MSG_FALLBACK = (
    "Another user-specific forecasting method was used for your expected spending "
    "(never a global forecast)."
)


def _call_store(store: Any, method: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run an owner-scoped store operation; translate failures to generic 503s."""
    try:
        return getattr(store, method)(*args, **kwargs)
    except PersistenceUnavailable:
        raise HTTPException(status_code=503, detail=DETAIL_503) from None


def _num(value: Any) -> float | None:
    """JSON-safe money value (Postgres NUMERIC arrives as Decimal)."""
    return None if value is None else round(float(value), 2)


def _as_date(value: Any) -> date | None:
    """Parse a profile DATE column (date/datetime/ISO string) or ``None``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _scenario_hash(amount: float) -> str:
    """Stable, non-reversible cache-key component for a scenario amount.

    Contains no identity and no financial history — only the requested amount.
    """
    if amount <= 0:
        return SCENARIO_HASH_NONE
    return hashlib.sha256(f"{amount:.2f}".encode()).hexdigest()[:12]


def _cache_parts(sub: str, payday: date | None, scenario_hash: str) -> tuple[str, ...]:
    """User-scoped, payday-scoped, scenario-scoped, engine-versioned key parts.

    The owner's ``sub`` is the first component, so one user's cache entry can
    never be served to (or purged on behalf of) another user.
    """
    return (
        sub,
        payday.isoformat() if payday else "none",
        scenario_hash,
        sts.SAFE_TO_SPEND_ENGINE_VERSION,
    )





def _breakdown(comp: sts.SafeToSpendComputation) -> list[SafeToSpendComponent]:
    """The exact component-by-component arithmetic, in formula order."""
    return [
        SafeToSpendComponent(
            label="Current available funds",
            amount=comp.current_available_funds,
            sign="+",
            source="user_profiles.liquid_buffer (your own saved cash position)",
        ),
        SafeToSpendComponent(
            label="Protected obligations before payday",
            amount=comp.protected_obligations,
            sign="−",
            source=(
                "your own observed monthly cadence for the six protected "
                "categories (Uncategorized is never protected)"
            ),
        ),
        SafeToSpendComponent(
            label="Expected spending before payday",
            amount=comp.expected_spending_before_payday,
            sign="−",
            source=(
                f"your own user-specific forecast ({comp.forecast_method}); "
                "non-protected spending only"
            ),
        ),
        SafeToSpendComponent(
            label="Safety buffer",
            amount=comp.safety_buffer,
            sign="−",
            source=comp.safety_buffer_source,
        ),
        SafeToSpendComponent(
            label="Scenario adjustment",
            amount=comp.scenario_adjustment,
            sign="−",
            source="this request only (never persisted)",
        ),
    ]


def _compute(
    request: Request,
    sub: str,
    *,
    scenario_amount: float,
    scenario_label: str | None,
) -> SafeToSpendResult:
    """Owner-scoped inputs → deterministic engine → public response.

    The same path serves both endpoints; the scenario only changes the
    arithmetic of the returned response and never touches persisted data.
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=DETAIL_NOT_CONFIGURED)
    cache = getattr(request.app.state, "cache", None)

    now = datetime.now(timezone.utc)
    today = now.date()
    since = now - timedelta(days=LOOKBACK_DAYS)

    profile_row = _call_store(store, "get_profile", sub)
    rows = _call_store(store, "fetch_financial_history", sub, since)

    try:
        configured_payday = (
            _as_date(profile_row.get("next_payday")) if profile_row else None
        )
        settled_rows = sts.settled(sts.normalise_history(rows))
        payday_res = sts.resolve_payday(configured_payday, settled_rows, today)

        s_hash = _scenario_hash(scenario_amount)
        parts = _cache_parts(sub, payday_res.payday, s_hash)
        if cache is not None and payday_res.resolved:
            cached = cache.get(CACHE_NAMESPACE, *parts)
            if cached is not None and cached.get("as_of") == today.isoformat():
                # A cached entry is only reused while "today" is still today, so
                # a response can never carry a stale horizon or as_of date.
                logger.info("Cache HIT for Safe-to-Spend (user-scoped)")
                return SafeToSpendResult(**cached)

        comp = sts.compute_safe_to_spend(
            rows=rows,
            today=today,
            available_funds=(
                _num(profile_row.get("liquid_buffer")) if profile_row else None
            ),
            safety_buffer=(
                _num(profile_row.get("safety_buffer")) if profile_row else None
            ),
            configured_payday=configured_payday,
            scenario_amount=scenario_amount,
            scenario_label=scenario_label,
            profile_present=profile_row is not None,
        )
        result = _to_model(
            comp, scenario_amount=scenario_amount, scenario_label=scenario_label
        )
    except HTTPException:
        raise
    except Exception:
        # Diagnostics stay server-side; the client gets a generic error and the
        # store is untouched (this endpoint is read-only).
        logger.exception("Safe-to-Spend computation failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error.") from None

    if cache is not None and comp.has_amount:
        cache.set(
            CACHE_NAMESPACE,
            *parts,
            value=result.model_dump(mode="json"),
            ttl=CACHE_TTLS["safe_to_spend"],
        )
    return result


@router.get("/safe-to-spend", response_model=SafeToSpendResult)
def get_safe_to_spend(
    request: Request,
    principal: AuthPrincipal = Depends(require_auth),
) -> SafeToSpendResult:
    """How much the caller can safely spend between today and their next payday.

    Identity is ``principal.sub`` only — there is no user id in the path, query
    or body, so no request field can select whose finances are computed.
    """
    return _compute(
        request, principal.sub, scenario_amount=0.0, scenario_label=None
    )


@router.post("/safe-to-spend/scenario", response_model=SafeToSpendResult)
def post_safe_to_spend_scenario(
    request: Request,
    body: SafeToSpendScenarioRequest,
    principal: AuthPrincipal = Depends(require_auth),
) -> SafeToSpendResult:
    """Safe-to-Spend after an explicit, hypothetical spend before payday.

    The scenario is applied to this response only (``scenario`` carries
    before/after/delta); nothing is persisted and no store method is called with
    the scenario, so persisted data cannot be modified by a scenario request.
    """
    return _compute(
        request,
        principal.sub,
        scenario_amount=body.scenario_amount,
        scenario_label=body.label,
    )



def _to_model(
    comp: sts.SafeToSpendComputation,
    *,
    scenario_amount: float,
    scenario_label: str | None,
) -> SafeToSpendResult:
    """Map the engine's outcome onto the public response contract."""
    scenario: SafeToSpendScenario | None = None
    if scenario_amount > 0 or scenario_label:
        after = comp.safe_to_spend
        # The formula is linear in the scenario adjustment, so the "before"
        # value is exactly after + amount; nothing is recomputed or stored.
        before = None if after is None else round(after + scenario_amount, 2)
        scenario = SafeToSpendScenario(
            label=scenario_label,
            amount=comp.scenario_adjustment,
            safe_to_spend_before=before,
            safe_to_spend_after=after,
            delta=None if before is None else round(before - after, 2),
        )

    message: str | None = None
    if comp.status == sts.STATUS_LIMITED_HISTORY:
        message = _MSG_LIMITED
    elif comp.status not in sts.STATUSES_WITH_AMOUNT:
        message = _MSG_NO_AMOUNT
    elif comp.forecast_fallback_status != pf.FALLBACK_NONE:
        message = _MSG_FALLBACK

    return SafeToSpendResult(
        status=comp.status,
        as_of=comp.as_of.isoformat(),
        currency=comp.currency,
        next_payday=comp.next_payday.isoformat() if comp.next_payday else None,
        days_to_payday=comp.days_to_payday,
        current_available_funds=comp.current_available_funds,
        protected_obligations=comp.protected_obligations,
        protected_obligations_detail=[
            SafeToSpendObligation(**item.as_dict())
            for item in comp.protected_obligation_items
        ],
        expected_spending_before_payday=comp.expected_spending_before_payday,
        expected_spending_p10=comp.expected_spending_p10,
        expected_spending_p90=comp.expected_spending_p90,
        safety_buffer=comp.safety_buffer,
        safety_buffer_source=comp.safety_buffer_source,
        scenario_adjustment=comp.scenario_adjustment,
        scenario=scenario,
        safe_to_spend=comp.safe_to_spend,
        is_negative=comp.is_negative,
        forecast_status=comp.forecast_status,
        forecast_method=comp.forecast_method,
        forecast_fallback_status=comp.forecast_fallback_status,
        payday_source=comp.payday_source,
        history_days_used=comp.history_days,
        transaction_count_used=comp.transaction_count,
        distinct_spend_days_used=comp.distinct_spend_days,
        excluded_pending_transactions=comp.excluded_pending_transactions,
        breakdown=_breakdown(comp),
        assumptions=list(comp.assumptions),
        requirements=comp.requirements,
        explanation=comp.explanation,
        engine_version=comp.engine_version,
        message=message,
    )

