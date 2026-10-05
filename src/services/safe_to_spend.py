"""
Safe-to-Spend (STEP 14) — deterministic, explainable, user-specific.

Core product promise: **"Know what you can safely spend before payday."**

    safe_to_spend = current_available_funds
                  − protected_obligations_before_payday
                  − expected_spending_before_payday
                  − safety_buffer
                  − scenario_adjustment

This module is the single source of truth for that arithmetic. It is a *pure*
function layer: no database access, no cache access, no network, no wall-clock
reads, no randomness and no logging of financial values. The serving route
(``src/serving/routes/safe_to_spend.py``) is the only caller and supplies
owner-scoped, persisted inputs. Identical inputs always produce identical
output, so the number is reproducible and explainable.

Dependencies:
  * STEP 13 (``personal_forecast``) — the caller's OWN expected spending.
    The legacy global/static forecast is never used here.
  * ``HARD_PROTECTED_CATEGORIES`` — the existing six protected obligations
    (Rent/Mortgage, Utilities, Home Insurance, Insurance Premiums, Loan
    Payments, Taxes). ``Uncategorized`` is never protected.

Financial policy decisions (each one is *explicit*, not implied by code):

1. **Available funds** come from the user's own persisted
   ``user_profiles.liquid_buffer`` — documented in ``FinancialProfile`` as
   "Cash/liquid assets available today". Nothing is derived from transaction
   history (a derived figure would be a fabricated balance). Missing → the
   ``missing_balance`` status, never a guessed number.
2. **Payday** is either the user's persisted ``next_payday`` (a future date) or
   measured from the cadence of the user's own income deposits. If neither
   exists the ``missing_payday`` status is returned. A payday is never assumed.
3. **Protected obligations** are inferred from the user's own observed monthly
   cadence per protected category; only a due date strictly after *today* and on
   or before payday is counted. An obligation already paid in this cycle is
   therefore never counted twice, and no bill schedule is invented.
4. **Expected spending** is the STEP 13 user-specific daily forecast summed over
   ``today+1 … payday``. It covers **non-protected** spending only: the six
   protected categories are subtracted separately, so keeping them in the
   forecast would count a bill twice and let one large monthly bill distort the
   spending profile. If STEP 13 cannot produce a forecast for the user (governing
   tier ``insufficient_history`` — the worse of the full-history and
   non-protected-history tiers) no number is produced at all.
5. **Safety buffer** is only what the user configured (``safety_buffer``). When
   unset it is ``0.00`` and reported as ``not_configured`` — the amount is then
   an *upper bound*, which the response states in ``assumptions``.
6. **Negative results are not floored.** A negative value means the user's own
   commitments exceed their available funds; hiding that by clamping to zero
   would be a false statement about their position. ``is_negative`` is returned
   so a client can label it as a shortfall.
7. **Pending transactions** (``is_pending``) are excluded from history,
   obligations and the forecast; the count of exclusions is reported so nothing
   is hidden.

Money representation: plain ``float`` rounded to 2 decimals, in the user's own
observed currency (``currency``); this matches the project's existing money
convention. Floats are used consistently, so the identity in the docstring above
holds exactly at 2-decimal precision.

This is deterministic decision-support output, not financial advice: no
accuracy, safety or "guaranteed" claim is made anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any, Sequence

from src.models.forecaster import personal_forecast as pf
from src.utils.constants import HARD_PROTECTED_CATEGORIES

logger = logging.getLogger(__name__)

#: Engine identifier — returned with every response and used in cache keys.
SAFE_TO_SPEND_ENGINE_VERSION = "safe-to-spend-v1"

#: Horizon bounds for a resolved payday (days from today). A payday further away
#: than this is treated as not determinable rather than forecasting a quarter of
#: uncertain daily spend; nearer than 1 day is not a "future payday".
MIN_DAYS_TO_PAYDAY = 1
MAX_DAYS_TO_PAYDAY = 45

# ── Status vocabulary (documented; returned verbatim) ────────────────────────
STATUS_READY = "ready"
STATUS_LIMITED_HISTORY = "limited_history"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
STATUS_MISSING_PAYDAY = "missing_payday"
STATUS_MISSING_BALANCE = "missing_balance"
STATUS_MISSING_PROFILE_DATA = "missing_profile_data"

#: Every status a response can carry, in the documented precedence order the
#: engine evaluates them (first match wins).
STATUSES = (
    STATUS_MISSING_PROFILE_DATA,
    STATUS_MISSING_BALANCE,
    STATUS_MISSING_PAYDAY,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_LIMITED_HISTORY,
    STATUS_READY,
)

#: Statuses where a numeric ``safe_to_spend`` is produced.
STATUSES_WITH_AMOUNT = (STATUS_READY, STATUS_LIMITED_HISTORY)

# ── Provenance labels ───────────────────────────────────────────────────────
PAYDAY_SOURCE_PROFILE = "profile"
PAYDAY_SOURCE_MEASURED = "measured_from_income_history"

BUFFER_SOURCE_PROFILE = "profile"
BUFFER_SOURCE_NOT_CONFIGURED = "not_configured"


# ── Obligation-inference constants (deterministic, no per-user tuning) ───────
#: Categories treated as protected obligations. Single source of truth shared
#: with the budget optimiser (src/utils/constants.py). "Uncategorized" is not
#: in this set and can therefore never be treated as protected.
PROTECTED_CATEGORIES: tuple[str, ...] = tuple(sorted(HARD_PROTECTED_CATEGORIES))

#: Minimum number of paid occurrences needed before a cadence is meaningful.
OBLIGATION_MIN_OBSERVATIONS = 2

#: Accepted monthly cadence band (days between occurrences). Wider gaps (e.g.
#: quarterly) are reported as ``irregular_cadence`` and excluded: the engine does
#: not guess when a bill is next due.
OBLIGATION_MIN_GAP_DAYS = 25
OBLIGATION_MAX_GAP_DAYS = 35

#: Accepted income-cadence bands (days between income deposits).
PAYDAY_MONTHLY_BAND = (25, 35)
PAYDAY_SEMI_MONTHLY_BAND = (11, 20)
PAYDAY_MIN_OBSERVATIONS = 2

#: Money rounding (project convention: 2-decimal floats).
MONEY_DP = 2

#: Reasons attached to each protected category in the response (explainability).
REASON_INCLUDED = "included"
REASON_INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
REASON_IRREGULAR_CADENCE = "irregular_cadence"
REASON_ALREADY_PAID_THIS_CYCLE = "already_paid_this_cycle"
REASON_DUE_AFTER_PAYDAY = "due_after_payday"

#: Reason attached to payday resolution.
PAYDAY_REASON_CONFIGURED = "configured_next_payday"
PAYDAY_REASON_STALE_CONFIG = "configured_payday_already_passed"
PAYDAY_REASON_MEASURED = "measured_income_cadence"
PAYDAY_REASON_NO_INCOME_HISTORY = "no_income_history"
PAYDAY_REASON_IRREGULAR_INCOME = "irregular_income"
PAYDAY_REASON_OUT_OF_RANGE = "payday_out_of_supported_range"


def money(value: float | None) -> float | None:
    """Round a money value to 2 decimals (``None`` passes through)."""
    return None if value is None else round(float(value), MONEY_DP)


# ── Persisted-history normalisation (dict rows or positional tuples) ─────────

@dataclass(frozen=True)
class HistoryRow:
    """One persisted transaction row, as the engine needs to see it."""

    occurred_at: datetime
    amount: float
    currency: str | None
    category_l2: str | None
    is_pending: bool

    @property
    def day(self) -> date:
        if isinstance(self.occurred_at, datetime):
            return self.occurred_at.date()
        return self.occurred_at


def normalise_history(rows: Sequence[Any]) -> list[HistoryRow]:
    """Normalise repository rows (dict or tuple) into :class:`HistoryRow`.

    Positional rows must follow the ``fetch_financial_history`` column order:
    ``occurred_at, amount, currency, category_l2, channel, is_pending``.
    Unknown/odd rows are skipped rather than guessed — an unreadable row is not
    silently turned into a zero-value observation.
    """
    out: list[HistoryRow] = []
    for row in rows:
        try:
            if isinstance(row, dict):
                occurred_at = row["occurred_at"]
                amount = row["amount"]
                currency = row.get("currency")
                category_l2 = row.get("category_l2")
                is_pending = bool(row.get("is_pending") or False)
            else:
                occurred_at = row[0]
                amount = row[1]
                currency = row[2] if len(row) > 2 else None
                category_l2 = row[3] if len(row) > 3 else None
                is_pending = bool(row[5]) if len(row) > 5 else False
        except (KeyError, IndexError, TypeError):
            continue
        if occurred_at is None or amount is None:
            continue
        if isinstance(occurred_at, str):
            try:
                occurred_at = datetime.fromisoformat(occurred_at)
            except ValueError:
                continue
        out.append(
            HistoryRow(
                occurred_at=occurred_at,
                amount=float(amount),
                currency=currency,
                category_l2=category_l2,
                is_pending=is_pending,
            )
        )
    return out


def observed_currency(rows: Sequence[HistoryRow]) -> str | None:
    """Most frequent currency in the user's own rows (ties → alphabetical).

    Returns ``None`` when the user has no rows or none carry a currency: a
    currency is never assumed. Same rule as
    ``src/services/financial_profile.build_financial_profile``.
    """
    counts: dict[str, int] = {}
    for row in rows:
        if row.currency:
            key = str(row.currency).upper()
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def settled(rows: Sequence[HistoryRow]) -> list[HistoryRow]:
    """Rows usable as evidence: pending transactions are excluded (rule 7)."""
    return [row for row in rows if not row.is_pending]


#: Membership test for the protected taxonomy (see ``PROTECTED_CATEGORIES``).
PROTECTED_CATEGORY_SET: frozenset[str] = frozenset(PROTECTED_CATEGORIES)


def non_protected_spend_rows(
    rows: Sequence[HistoryRow],
) -> list[HistoryRow]:
    """The caller's own rows excluding protected-category spending.

    Protected obligations are subtracted **separately** in the formula, so
    leaving rent/utilities/etc. inside the spending forecast would count them
    twice and let a single large monthly bill distort the weekday profile. This
    exclusion is the anti-double-counting rule (reported in ``assumptions``).
    ``Uncategorized`` is not a protected category, so uncategorized spending
    stays in the forecast as ordinary spend.
    """
    return [row for row in rows if (row.category_l2 or "") not in PROTECTED_CATEGORY_SET]


#: Rank used to compare two quality tiers; higher is less trustworthy.
_TIER_RANK = {
    pf.STATUS_PERSONALIZED: 0,
    pf.STATUS_LIMITED: 1,
    pf.STATUS_INSUFFICIENT: 2,
}


def worse_tier(*tiers: str) -> str:
    """The least trustworthy of the given STEP 13 tiers (never optimistic)."""
    return max(tiers, key=lambda tier: _TIER_RANK.get(tier, 2))



# ── Payday resolution: today → next payday ───────────────────────────────────

@dataclass(frozen=True)
class PaydayResolution:
    """Outcome of resolving the next payday (never guessed)."""

    payday: date | None
    source: str | None
    reason: str
    detail: str
    cadence_days: int | None = None
    income_observations: int = 0
    last_income_date: date | None = None
    configured_payday_stale: bool = False

    @property
    def resolved(self) -> bool:
        return self.payday is not None


#: Largest allowed deviation of an individual gap from the median gap when a
#: cadence is claimed. Payroll dates drift by a day or two (weekends, holidays);
#: anything less consistent is treated as irregular income, not as a payday.
CADENCE_TOLERANCE_DAYS = 4


def _cadence_from_gaps(gaps: Sequence[int]) -> int | None:
    """Median cadence in days when every gap fits one accepted band.

    Returns ``None`` when the gaps are inconsistent or outside both the monthly
    (25–35 d) and semi-monthly (11–20 d) bands — the engine then reports
    ``irregular_income`` instead of inventing a payday.
    """
    if not gaps:
        return None
    med = int(round(median(gaps)))
    band = None
    for candidate in (PAYDAY_MONTHLY_BAND, PAYDAY_SEMI_MONTHLY_BAND):
        low, high = candidate
        if all(low <= gap <= high for gap in gaps):
            band = candidate
            break
    if band is None:
        return None
    if any(abs(gap - med) > CADENCE_TOLERANCE_DAYS for gap in gaps):
        return None
    return med


def _income_days(rows: Sequence[HistoryRow]) -> list[date]:
    """Distinct days on which the user received a classified income credit."""
    income_label = "Income"
    days = {
        row.day
        for row in rows
        if row.amount > 0 and (row.category_l2 or "") == income_label
    }
    return sorted(days)


def detect_payday_from_income(
    rows: Sequence[HistoryRow], today: date
) -> PaydayResolution:
    """Measure the next payday from the user's OWN income-deposit cadence.

    Requires at least three classified income deposits (two observed gaps), all
    consistent within one accepted cadence band. This is the salary-cycle
    *evidence* rule: a single or double deposit is not enough to claim a payday.
    """
    days = _income_days(rows)
    if len(days) < PAYDAY_MIN_OBSERVATIONS + 1:
        return PaydayResolution(
            payday=None,
            source=None,
            reason=PAYDAY_REASON_NO_INCOME_HISTORY,
            detail=(
                "Fewer than three classified income deposits were found in your "
                "own history, so a payday cadence cannot be measured."
            ),
            income_observations=len(days),
            last_income_date=days[-1] if days else None,
        )

    gaps = [(days[i] - days[i - 1]).days for i in range(1, len(days))]
    cadence = _cadence_from_gaps(gaps)
    if cadence is None:
        return PaydayResolution(
            payday=None,
            source=None,
            reason=PAYDAY_REASON_IRREGULAR_INCOME,
            detail=(
                "Your income deposits do not follow a consistent monthly or "
                "semi-monthly cadence, so no payday is assumed."
            ),
            income_observations=len(days),
            last_income_date=days[-1],
        )

    candidate = days[-1] + timedelta(days=cadence)
    age = (candidate - today).days
    if not (MIN_DAYS_TO_PAYDAY <= age <= MAX_DAYS_TO_PAYDAY):
        return PaydayResolution(
            payday=None,
            source=None,
            reason=PAYDAY_REASON_OUT_OF_RANGE,
            detail=(
                "The payday implied by your income cadence is not within the "
                f"{MIN_DAYS_TO_PAYDAY}–{MAX_DAYS_TO_PAYDAY} day window from "
                "today, so it is not used."
            ),
            cadence_days=cadence,
            income_observations=len(days),
            last_income_date=days[-1],
        )

    return PaydayResolution(
        payday=candidate,
        source=PAYDAY_SOURCE_MEASURED,
        reason=PAYDAY_REASON_MEASURED,
        detail=(
            f"Measured from {len(days)} of your own income deposits "
            f"(median cadence {cadence} days)."
        ),
        cadence_days=cadence,
        income_observations=len(days),
        last_income_date=days[-1],
    )



def resolve_payday(
    configured_payday: date | None,
    rows: Sequence[HistoryRow],
    today: date,
) -> PaydayResolution:
    """Resolve the next payday: explicit profile value first, else measured.

    A configured date that is not strictly in the future is treated as stale and
    ignored (reported via ``configured_payday_stale``); it is never silently
    reused. The measured cadence is used only when it resolves cleanly, and the
    response always states which source was used.
    """
    if configured_payday is not None:
        age = (configured_payday - today).days
        if MIN_DAYS_TO_PAYDAY <= age <= MAX_DAYS_TO_PAYDAY:
            return PaydayResolution(
                payday=configured_payday,
                source=PAYDAY_SOURCE_PROFILE,
                reason=PAYDAY_REASON_CONFIGURED,
                detail="Taken from the payday saved in your own profile.",
            )
        stale = age < MIN_DAYS_TO_PAYDAY
        measured = detect_payday_from_income(rows, today)
        if measured.resolved:
            return replace(measured, configured_payday_stale=True)
        return PaydayResolution(
            payday=None,
            source=None,
            reason=(
                PAYDAY_REASON_STALE_CONFIG if stale else PAYDAY_REASON_OUT_OF_RANGE
            ),
            detail=(
                "The payday saved in your profile is no longer in the future; "
                "update it to get a Safe-to-Spend amount."
                if stale
                else (
                    "The payday saved in your profile is further away than the "
                    f"supported {MAX_DAYS_TO_PAYDAY}-day window."
                )
            ),
            configured_payday_stale=stale,
        )

    return detect_payday_from_income(rows, today)



# ── Protected obligations before payday ──────────────────────────────────────

@dataclass(frozen=True)
class ObligationItem:
    """One protected category and the conclusion drawn from the user's history."""

    category: str
    included: bool
    reason: str
    amount: float | None = None          # counted amount (None when not included)
    due_date: date | None = None
    observations: int = 0
    median_gap_days: int | None = None
    last_paid_on: date | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "included": self.included,
            "reason": self.reason,
            "amount": money(self.amount),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "observations": self.observations,
            "median_gap_days": self.median_gap_days,
            "last_paid_on": self.last_paid_on.isoformat() if self.last_paid_on else None,
        }


@dataclass(frozen=True)
class ObligationPlan:
    """Every protected category's evaluation plus the counted total."""

    items: list[ObligationItem] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(
            sum(item.amount or 0.0 for item in self.items if item.included), MONEY_DP
        )

    @property
    def included(self) -> list[ObligationItem]:
        return [item for item in self.items if item.included]


def _debit_amounts_by_day(
    rows: Sequence[HistoryRow], category: str
) -> tuple[list[date], dict[date, float]]:
    """Occurrence days and per-day totals for one category (debits only)."""
    per_day: dict[date, float] = {}
    for row in rows:
        if row.amount >= 0 or (row.category_l2 or "") != category:
            continue
        per_day[row.day] = per_day.get(row.day, 0.0) + abs(row.amount)
    days = sorted(per_day)
    return days, per_day


def upcoming_protected_obligations(
    rows: Sequence[HistoryRow],
    today: date,
    payday: date,
) -> ObligationPlan:
    """Estimate protected obligations falling after today and on/before payday.

    Per protected category the engine requires at least two observed payments
    (``insufficient_observations``) spaced on a monthly cadence in the accepted
    band (``irregular_cadence``). The next due date is the last observed payment
    plus the median gap; the counted amount is the **median** observed payment
    (robust to one-off spikes).

    Nothing is counted when the implied next due date is already in the past
    (``already_paid_this_cycle`` — that cycle's bill is treated as settled and is
    never re-counted) or lands after payday (``due_after_payday``). All six
    protected categories are always reported with their reason, so every
    exclusion is visible rather than silent.
    """
    plan = ObligationPlan()
    for category in PROTECTED_CATEGORIES:
        days, per_day = _debit_amounts_by_day(rows, category)
        if len(days) < OBLIGATION_MIN_OBSERVATIONS:
            plan.items.append(
                ObligationItem(
                    category=category,
                    included=False,
                    reason=REASON_INSUFFICIENT_OBSERVATIONS,
                    observations=len(days),
                    last_paid_on=days[-1] if days else None,
                )
            )
            continue

        gaps = [(days[i] - days[i - 1]).days for i in range(1, len(days))]
        med_gap = int(round(median(gaps)))
        if not (OBLIGATION_MIN_GAP_DAYS <= med_gap <= OBLIGATION_MAX_GAP_DAYS):
            plan.items.append(
                ObligationItem(
                    category=category,
                    included=False,
                    reason=REASON_IRREGULAR_CADENCE,
                    observations=len(days),
                    median_gap_days=med_gap,
                    last_paid_on=days[-1],
                )
            )
            continue

        due = days[-1] + timedelta(days=med_gap)
        amount = round(median([per_day[day] for day in days]), MONEY_DP)
        if due <= today:
            reason = REASON_ALREADY_PAID_THIS_CYCLE
            include = False
        elif due > payday:
            reason = REASON_DUE_AFTER_PAYDAY
            include = False
        else:
            reason = REASON_INCLUDED
            include = True
        plan.items.append(
            ObligationItem(
                category=category,
                included=include,
                reason=reason,
                amount=amount if include else None,
                due_date=due,
                observations=len(days),
                median_gap_days=med_gap,
                last_paid_on=days[-1],
            )
        )
    return plan


# ── Expected spending before payday (STEP 13 user-specific forecast) ─────────

@dataclass(frozen=True)
class ExpectedSpending:
    """The caller's own forecast summed over ``today+1 … payday``."""

    total: float | None
    p10: float | None
    p90: float | None
    days: int
    method: str
    fallback_status: str
    status: str
    residual_sigma: float
    first_day: date
    last_day: date
    history_days: int = 0
    transaction_count: int = 0
    distinct_spend_days: int = 0
    requirements: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.total is not None


def expected_spending_for_horizon(
    series: pf.DailySeries,
    first_day: date,
    payday: date,
    *,
    tier_series: pf.DailySeries | None = None,
) -> ExpectedSpending:
    """Sum the user's OWN STEP 13 forecast over the payday horizon.

    ``pf.predict`` is called with an explicit ``start`` anchor so the horizon is
    exactly ``today+1 … payday`` — never an arbitrary 30-day window and never the
    series end. The legacy global/static artifact is not reachable from here.

    ``series`` is the non-protected spending series (see
    :func:`non_protected_spend_rows`). ``tier_series`` is the same user's *full*
    settled history; the reported status is the **worse** of the two tiers, so a
    user whose non-obligation history alone would look thin is never labelled
    more confidently than their data supports (and vice versa).

    When the governing tier is ``insufficient_history`` no number is produced:
    the caller receives ``total=None`` plus the requirements block explaining
    what is missing.
    """
    days = max(0, (payday - first_day).days + 1)
    governing = tier_series if tier_series is not None else series
    base = ExpectedSpending(
        total=None,
        p10=None,
        p90=None,
        days=days,
        method=pf.PRIMARY_METHOD,
        fallback_status=pf.FALLBACK_NONE,
        status=pf.STATUS_INSUFFICIENT,
        residual_sigma=0.0,
        first_day=first_day,
        last_day=payday,
        history_days=series.history_days,
        transaction_count=series.transaction_count,
        distinct_spend_days=series.distinct_spend_days,
    )

    tier = worse_tier(pf.quality_tier(series), pf.quality_tier(governing))
    detail_series = governing if pf.quality_tier(governing) == tier else series
    if tier == pf.STATUS_INSUFFICIENT or series.history_days == 0 or days == 0:
        return replace(
            base,
            status=tier,
            requirements=pf.insufficient_history_detail(detail_series),
        )

    payload = pf.predict(series, days, start=first_day)
    total = payload.get("total") or {}
    requirements = (
        None
        if tier == pf.STATUS_PERSONALIZED
        else {
            "note": (
                "Personalized but based on limited history "
                "(less than four weeks of your own transactions)."
            ),
            **pf.insufficient_history_detail(detail_series)["for_full_confidence"],
        }
    )
    return ExpectedSpending(
        total=money(payload.get("expected_spend")),
        p10=money(total.get("p10")),
        p90=money(total.get("p90")),
        days=days,
        method=str(payload.get("method")),
        fallback_status=str(payload.get("fallback_status")),
        status=tier,
        residual_sigma=float(payload.get("residual_sigma") or 0.0),
        first_day=first_day,
        last_day=payday,
        history_days=series.history_days,
        transaction_count=series.transaction_count,
        distinct_spend_days=series.distinct_spend_days,
        requirements=requirements,
    )


# ── Explanation & assumptions (templated, deterministic — no LLM) ────────────

def format_money(value: float | None, currency: str | None) -> str:
    """Format a money value for the human-readable explanation."""
    if value is None:
        return "not available"
    text = f"{value:,.2f}"
    return f"{currency} {text}" if currency else f"{text} (currency not observed)"


def build_explanation(
    *,
    status: str,
    currency: str | None,
    available_funds: float | None,
    protected_obligations: float | None,
    expected_spending: float | None,
    safety_buffer: float | None,
    scenario_adjustment: float | None,
    safe_to_spend: float | None,
) -> str:
    """Deterministic, templated breakdown of the reported amount.

    Produced from the numbers themselves — never by a language model, and never
    describing a computation that did not happen. When a required input is
    missing the explanation says so instead of showing fabricated components.
    """
    if status not in STATUSES_WITH_AMOUNT or safe_to_spend is None:
        return (
            f"Safe to Spend is not available ({status}). "
            "No amount is reported because a required input or enough of your own "
            "history is missing; see 'requirements' for what is needed."
        )
    lines = [f"Safe to Spend = {format_money(safe_to_spend, currency)}", "Breakdown:"]
    lines.append(
        f"  Current available funds: {format_money(available_funds, currency)}"
    )
    lines.append(
        f"  − Protected obligations before payday: "
        f"{format_money(protected_obligations or 0.0, currency)}"
    )
    lines.append(
        f"  − Expected spending before payday: "
        f"{format_money(expected_spending or 0.0, currency)}"
    )
    lines.append(f"  − Safety buffer: {format_money(safety_buffer or 0.0, currency)}")
    lines.append(
        f"  − Scenario adjustment: {format_money(scenario_adjustment or 0.0, currency)}"
    )
    lines.append(f"  = {format_money(safe_to_spend, currency)}")
    if safe_to_spend < 0:
        lines.append(
            "This is negative: your protected commitments and expected spending "
            "exceed your available funds before payday. The amount is reported "
            "as-is (never clamped to zero)."
        )
    return "\n".join(lines)


def build_assumptions(
    *,
    currency: str | None,
    expected: ExpectedSpending,
    payday: PaydayResolution,
    buffer_source: str,
    excluded_pending: int,
    scenario_amount: float,
    safe_to_spend: float | None,
) -> list[str]:
    """Explicit, ordered list of every assumption used to produce the result."""
    notes: list[str] = []
    notes.append(
        "Money values are floats rounded to 2 decimals"
        + (f" in {currency}." if currency else "; no currency has been observed yet.")
    )
    if expected.available:
        notes.append(
            f"Expected spending is your own user-specific forecast "
            f"(method: {expected.method}, {expected.days} day(s) from "
            f"{expected.first_day.isoformat()} to {expected.last_day.isoformat()}); "
            "the legacy global/static forecast is never used."
        )
        notes.append(
            "Expected spending covers non-protected spending only: the six "
            "protected categories are excluded from it and counted once, in "
            "protected obligations, so a bill is never subtracted twice and a "
            "single large bill cannot distort the spending profile."
        )
    else:
        notes.append(
            "No expected-spending amount is produced: your own history is not "
            "sufficient for a user-specific forecast, and nothing is fabricated."
        )
    notes.append(
        "Protected obligations are inferred from your own observed monthly "
        "cadence for the six protected categories; only a due date after today "
        "and on or before payday is counted (a bill already paid this cycle is "
        "never counted twice)."
    )
    notes.append(
        "Pending transactions are excluded from history, obligations and the "
        f"forecast ({excluded_pending} excluded)."
    )
    if payday.source == PAYDAY_SOURCE_MEASURED:
        notes.append(
            f"Payday was measured from your own income deposits "
            f"(median cadence {payday.cadence_days} days, "
            f"{payday.income_observations} observations)."
        )
    if payday.configured_payday_stale:
        notes.append(
            "The payday saved in your profile is no longer in the future and was "
            "not used."
        )
    if buffer_source == BUFFER_SOURCE_NOT_CONFIGURED:
        notes.append(
            "No safety buffer is configured for you, so 0.00 is assumed and the "
            "reported amount is an upper bound."
        )
    if scenario_amount > 0:
        notes.append(
            "The scenario adjustment is subtracted from this result only: no "
            "persisted data was read-modify-written or changed."
        )
    if safe_to_spend is not None:
        notes.append(
            "A negative result is reported as-is (never floored at zero) because "
            "it states a real shortfall."
        )
    return notes


# ── Requirements for the missing-input statuses ──────────────────────────────

MISSING_INPUTS: dict[str, list[str]] = {
    STATUS_MISSING_PROFILE_DATA: ["a saved financial profile (income and your cash position)"],
    STATUS_MISSING_BALANCE: [
        "liquid_buffer — your cash / liquid assets available today (a saved profile field)"
    ],
    STATUS_MISSING_PAYDAY: [
        "next_payday (a future date) saved in your profile, or at least three "
        "classified income deposits so a payday cadence can be measured"
    ],
    STATUS_INSUFFICIENT_HISTORY: [
        "more of your own classified transaction history (four weeks or more for "
        "a full-confidence forecast)"
    ],
}

HOW_TO_RESOLVE: dict[str, list[str]] = {
    STATUS_MISSING_PROFILE_DATA: ["PUT /consumer/profile"],
    STATUS_MISSING_BALANCE: ["PUT /consumer/profile with liquid_buffer"],
    STATUS_MISSING_PAYDAY: [
        "PUT /consumer/profile with next_payday (YYYY-MM-DD)",
        "POST /consumer/transactions/ingest-csv with salary history",
    ],
}


def build_requirements(
    status: str,
    *,
    payday: PaydayResolution,
    expected: ExpectedSpending,
) -> dict[str, Any] | None:
    """What the caller must provide to receive a Safe-to-Spend amount.

    ``None`` is returned only for statuses that already carry a number.
    """
    if status in STATUSES_WITH_AMOUNT:
        return None
    if status == STATUS_INSUFFICIENT_HISTORY:
        return {
            "required": MISSING_INPUTS[STATUS_INSUFFICIENT_HISTORY],
            "detail": expected.requirements,
            "how_to_resolve": ["POST /consumer/transactions/ingest-csv"],
        }
    if status.startswith("missing_"):
        block: dict[str, Any] = {
            "required": MISSING_INPUTS.get(status, []),
            "how_to_resolve": HOW_TO_RESOLVE.get(status, []),
        }
        if status == STATUS_MISSING_PAYDAY:
            block["payday_detail"] = {
                "reason": payday.reason,
                "detail": payday.detail,
                "income_observations": payday.income_observations,
                "last_income_date": (
                    payday.last_income_date.isoformat()
                    if payday.last_income_date
                    else None
                ),
            }
        return block
    return None


# ── Result contract ──────────────────────────────────────────────────────────

@dataclass
class SafeToSpendComputation:
    """Complete, explainable outcome of the STEP 14 arithmetic."""

    status: str
    as_of: date
    currency: str | None
    next_payday: date | None
    days_to_payday: int | None
    current_available_funds: float | None
    protected_obligations: float | None
    protected_obligation_items: list[ObligationItem] = field(default_factory=list)
    expected_spending_before_payday: float | None = None
    expected_spending_p10: float | None = None
    expected_spending_p90: float | None = None
    safety_buffer: float | None = None
    safety_buffer_source: str = BUFFER_SOURCE_NOT_CONFIGURED
    scenario_adjustment: float = 0.0
    scenario_label: str | None = None
    safe_to_spend: float | None = None
    forecast_status: str = STATUS_INSUFFICIENT_HISTORY
    forecast_method: str = ""
    forecast_fallback_status: str = pf.FALLBACK_NONE
    payday_source: str | None = None
    history_days: int = 0
    transaction_count: int = 0
    distinct_spend_days: int = 0
    excluded_pending_transactions: int = 0
    assumptions: list[str] = field(default_factory=list)
    requirements: dict[str, Any] | None = None
    explanation: str = ""
    engine_version: str = SAFE_TO_SPEND_ENGINE_VERSION

    @property
    def is_negative(self) -> bool:
        """True when the reported amount is a shortfall (never clamped to 0)."""
        return self.safe_to_spend is not None and self.safe_to_spend < 0

    @property
    def has_amount(self) -> bool:
        return self.status in STATUSES_WITH_AMOUNT and self.safe_to_spend is not None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready view (used by tests and the offline evaluation harness)."""
        return {
            "status": self.status,
            "as_of": self.as_of.isoformat(),
            "currency": self.currency,
            "next_payday": self.next_payday.isoformat() if self.next_payday else None,
            "days_to_payday": self.days_to_payday,
            "current_available_funds": self.current_available_funds,
            "protected_obligations": self.protected_obligations,
            "expected_spending_before_payday": self.expected_spending_before_payday,
            "safety_buffer": self.safety_buffer,
            "scenario_adjustment": self.scenario_adjustment,
            "safe_to_spend": self.safe_to_spend,
            "is_negative": self.is_negative,
            "forecast_status": self.forecast_status,
            "forecast_method": self.forecast_method,
            "engine_version": self.engine_version,
        }


def compute_safe_to_spend(
    *,
    rows: Sequence[Any],
    today: date,
    available_funds: float | None,
    safety_buffer: float | None,
    configured_payday: date | None,
    scenario_amount: float = 0.0,
    scenario_label: str | None = None,
    profile_present: bool = True,
) -> SafeToSpendComputation:
    """Compute Safe-to-Spend from persisted, owner-scoped inputs only.

    Status precedence (first match wins, documented in ``STATUSES``):
    ``missing_profile_data`` → ``missing_balance`` → ``missing_payday`` →
    ``insufficient_history`` → ``limited_history`` → ``ready``. The first three
    produce no amount at all; ``insufficient_history`` produces no expected
    spending and therefore no amount either. Nothing is fabricated and no global
    or static forecast is ever substituted for the caller's own data.
    """
    history = normalise_history(rows)
    settled_rows = settled(history)
    excluded_pending = len(history) - len(settled_rows)
    currency = observed_currency(settled_rows)

    funds = money(available_funds)
    buffer_value = 0.0 if safety_buffer is None else money(safety_buffer)
    buffer_source = (
        BUFFER_SOURCE_PROFILE
        if safety_buffer is not None
        else BUFFER_SOURCE_NOT_CONFIGURED
    )
    scenario = money(max(float(scenario_amount or 0.0), 0.0)) or 0.0

    payday_res = resolve_payday(configured_payday, settled_rows, today)
    expected = ExpectedSpending(
        total=None,
        p10=None,
        p90=None,
        days=0,
        method=pf.PRIMARY_METHOD,
        fallback_status=pf.FALLBACK_NONE,
        status=pf.STATUS_INSUFFICIENT,
        residual_sigma=0.0,
        first_day=today + timedelta(days=1),
        last_day=today,
    )
    obligation_plan = ObligationPlan()
    status = STATUS_READY

    if not profile_present:
        status = STATUS_MISSING_PROFILE_DATA
    elif funds is None:
        status = STATUS_MISSING_BALANCE
    elif not payday_res.resolved:
        status = STATUS_MISSING_PAYDAY
    else:
        series = pf.build_daily_series(
            [
                (row.occurred_at, row.amount)
                for row in non_protected_spend_rows(settled_rows)
            ]
        )
        full_series = pf.build_daily_series(
            [(row.occurred_at, row.amount) for row in settled_rows]
        )
        expected = expected_spending_for_horizon(
            series,
            today + timedelta(days=1),
            payday_res.payday,
            tier_series=full_series,
        )
        if not expected.available:
            status = STATUS_INSUFFICIENT_HISTORY
        else:
            obligation_plan = upcoming_protected_obligations(
                settled_rows, today, payday_res.payday
            )
            status = (
                STATUS_LIMITED_HISTORY
                if expected.status == pf.STATUS_LIMITED
                else STATUS_READY
            )

    days_to_payday = (
        (payday_res.payday - today).days if payday_res.payday is not None else None
    )
    obligations = obligation_plan.total if status in STATUSES_WITH_AMOUNT else None
    safe_to_spend = None
    if status in STATUSES_WITH_AMOUNT:
        safe_to_spend = round(
            (funds or 0.0)
            - (obligations or 0.0)
            - (expected.total or 0.0)
            - (buffer_value or 0.0)
            - scenario,
            MONEY_DP,
        )

    result = SafeToSpendComputation(
        status=status,
        as_of=today,
        currency=currency,
        next_payday=payday_res.payday,
        days_to_payday=days_to_payday,
        current_available_funds=funds,
        protected_obligations=obligations,
        protected_obligation_items=obligation_plan.items,
        expected_spending_before_payday=expected.total,
        expected_spending_p10=expected.p10,
        expected_spending_p90=expected.p90,
        safety_buffer=buffer_value,
        safety_buffer_source=buffer_source,
        scenario_adjustment=scenario,
        scenario_label=scenario_label,
        safe_to_spend=safe_to_spend,
        forecast_status=expected.status,
        forecast_method=expected.method,
        forecast_fallback_status=expected.fallback_status,
        payday_source=payday_res.source,
        history_days=expected.history_days,
        transaction_count=expected.transaction_count,
        distinct_spend_days=expected.distinct_spend_days,
        excluded_pending_transactions=excluded_pending,
    )
    result.assumptions = build_assumptions(
        currency=currency,
        expected=expected,
        payday=payday_res,
        buffer_source=buffer_source,
        excluded_pending=excluded_pending,
        scenario_amount=scenario,
        safe_to_spend=safe_to_spend,
    )
    result.requirements = build_requirements(
        status, payday=payday_res, expected=expected
    )
    result.explanation = build_explanation(
        status=status,
        currency=currency,
        available_funds=funds,
        protected_obligations=obligations,
        expected_spending=expected.total,
        safety_buffer=buffer_value,
        scenario_adjustment=scenario,
        safe_to_spend=safe_to_spend,
    )
    return result
