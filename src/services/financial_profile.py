"""
FinancialProfile aggregation service.

Turns already-classified transactions into the canonical
``FinancialProfile`` (``src.data.models``).

Determinism & honesty rules (hard requirements):

* No randomness, no wall-clock reads — the same inputs always produce the
  same profile (``generated_at`` is only set if the caller passes it).
* Income is ONLY credits labelled ``CategoryL2.INCOME``. Refunds,
  reversals, internal account-to-account transfers (unlabelled
  ``Channel.TRANSFER``) and savings & investment movements never inflate
  income or expenses — they are reported informationally via
  ``transfers_total``, ``savings_transfers_total`` and ``refunds_total``.
  The service NEVER fabricates income — in particular it never derives
  income from spending (no ``total_spend * 1.3`` heuristics).
* Category-labelled transactions moving over the TRANSFER rail (salary
  deposits, rent payments) keep their semantic meaning: the channel alone
  never reclassifies a transaction.
* Missing information is represented as ``None`` (e.g. ``savings_rate``
  when no income was observed), never as a guessed number.

Bucket semantics (matches the ``FinancialProfile`` docstring):

* ``fixed_expenses``        — debits whose channel is RECURRING or whose L2
  category is a core obligation (see ``FIXED_CATEGORIES``). The recurring
  signal wins over the discretionary taxonomy.
* ``discretionary_expenses`` — debits in ``DISCRETIONARY_CATEGORIES`` that
  are not fixed. A subset of ``variable_expenses``.
* ``variable_expenses``     — ALL non-fixed debits (includes discretionary).
* ``recurring_expenses``    — debits with ``Channel.RECURRING``.

Monthly normalisation: raw totals are divided by the number of distinct
calendar months covered by the transaction window (minimum 1). Callers
should pass complete calendar months for best fidelity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from src.data.models import FinancialProfile, Transaction
from src.utils.constants import CategoryL2, Channel, DISCRETIONARY_CATEGORIES

DAYS_PER_MONTH = 30.4375  # mean Gregorian month length (informational)

# Core-obligation L2 categories treated as fixed even without a RECURRING
# channel signal. Deliberately small; the channel signal carries the rest.
FIXED_CATEGORIES: frozenset[CategoryL2] = frozenset(
    {
        CategoryL2.RENT_MORTGAGE,
        CategoryL2.UTILITIES,
        CategoryL2.HOME_INSURANCE,
        CategoryL2.INSURANCE_PREMIUMS,
        CategoryL2.LOAN_PAYMENTS,
        CategoryL2.TAXES,
    }
)

UNCATEGORIZED_KEY = "Uncategorized"


def _as_utc(ts: datetime) -> datetime:
    """Normalise naive timestamps to UTC (project convention) for comparisons."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def build_financial_profile(
    transactions: Sequence[Transaction],
    *,
    user_id: str | None = None,
    period: str | None = None,
    observation_days: int | None = None,
    liquid_buffer: float | None = None,
    total_debt: float | None = None,
    monthly_debt_payments: float | None = None,
    generated_at: datetime | None = None,
) -> FinancialProfile:
    """
    Aggregate transactions into a canonical :class:`FinancialProfile`.

    Parameters
    ----------
    transactions:
        Classified transactions (``category_l2`` set where known).
        The caller controls the window; all supplied transactions are used.
    user_id:
        Profile owner. Defaults to the first transaction's ``user_id``.
        Required when ``transactions`` is empty.
    period:
        Optional window label (e.g. ``"2026-03"``) written to the profile.
    observation_days:
        Optional trailing-window filter (relative to the newest transaction).
    liquid_buffer, total_debt, monthly_debt_payments:
        Balance-sheet inputs the transaction stream cannot provide. Passed
        through as-is or left ``None`` — never invented.
    generated_at:
        Optional stamp; omitted for deterministic output.

    Returns
    -------
    FinancialProfile
        ``monthly_income == 0.0`` means *no INCOME-labelled credits were
        observed* (not "the user earns nothing"); ``savings_rate`` is
        ``None`` in that case.
    """
    txns = list(transactions)
    if not txns and user_id is None:
        raise ValueError(
            "user_id is required when aggregating an empty transaction list."
        )

    # ── Optional trailing-window filter ──────────────────────────────────
    if txns and observation_days is not None:
        newest = max(_as_utc(t.timestamp) for t in txns)
        cutoff = newest - timedelta(days=observation_days)
        txns = [t for t in txns if _as_utc(t.timestamp) >= cutoff]

    # ── Window length in distinct calendar months (min 1) ────────────────
    if txns:
        stamps = [_as_utc(t.timestamp) for t in txns]
        month_keys = {(s.year, s.month) for s in stamps}
        n_months = float(len(month_keys))
    else:
        n_months = 1.0

    # ── Money-movement classification. Precedence (documented in the module
    #    docstring):
    #      1. category SAVINGS_INVESTMENTS  → savings movement (either side)
    #      2. credit with category INCOME   → genuine income (any channel)
    #      3. channel TRANSFER + unlabelled → internal transfer (either side)
    #      4. remaining credits             → refunds/other credits (NOT income)
    #      5. remaining debits              → ordinary expenses
    income_total = 0.0
    transfers_total = 0.0
    savings_transfers_total = 0.0
    refunds_total = 0.0

    # ── Expense buckets (ordinary debits only) ───────────────────────────
    fixed_total = 0.0
    variable_total = 0.0
    discretionary_total = 0.0
    recurring_total = 0.0
    category_totals: dict[str, float] = {}

    for t in txns:
        cat = t.category_l2
        if t.amount > 0:
            # ── Credits ──
            if cat == CategoryL2.INCOME:
                income_total += t.amount
            elif cat == CategoryL2.SAVINGS_INVESTMENTS:
                savings_transfers_total += t.amount
            elif t.channel == Channel.TRANSFER and cat is None:
                transfers_total += t.amount
            else:
                # Refunds / reversals / unidentified credits — never income.
                refunds_total += t.amount
            continue
        if t.amount == 0:
            continue
        spend = abs(t.amount)
        # ── Debits ──
        if cat == CategoryL2.SAVINGS_INVESTMENTS:
            savings_transfers_total += spend
            continue
        if t.channel == Channel.TRANSFER and cat is None:
            transfers_total += spend
            continue
        # Precedence: the recurring/fixed signal wins over the discretionary
        # taxonomy, so a RECURRING subscription lands in `fixed`, never in
        # `discretionary`.
        is_fixed = t.channel == Channel.RECURRING or cat in FIXED_CATEGORIES
        is_discretionary = cat is not None and cat in DISCRETIONARY_CATEGORIES

        if is_fixed:
            fixed_total += spend
        else:
            variable_total += spend
            if is_discretionary:
                discretionary_total += spend
        if t.channel == Channel.RECURRING:
            recurring_total += spend

        key = cat.value if cat is not None else UNCATEGORIZED_KEY
        category_totals[key] = category_totals.get(key, 0.0) + spend

    # ── Monthly normalisation ────────────────────────────────────────────
    monthly_income = income_total / n_months
    monthly_fixed = fixed_total / n_months
    monthly_variable = variable_total / n_months
    monthly_discretionary = discretionary_total / n_months
    monthly_recurring = recurring_total / n_months
    monthly_transfers = transfers_total / n_months
    monthly_savings_transfers = savings_transfers_total / n_months
    monthly_refunds = refunds_total / n_months
    monthly_total_expenses = monthly_fixed + monthly_variable

    monthly_savings = monthly_income - monthly_total_expenses
    savings_rate = monthly_savings / monthly_income if monthly_income > 0 else None
    months_of_buffer = (
        liquid_buffer / monthly_total_expenses
        if liquid_buffer is not None and monthly_total_expenses > 0
        else None
    )

    # ── Currency: most frequent observed (ties → alphabetical) ──────────
    if txns:
        counts = Counter(t.currency.upper() for t in txns)
        currency = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    else:
        currency = "USD"

    category_spending = {k: v / n_months for k, v in sorted(category_totals.items())}

    return FinancialProfile(
        user_id=user_id if user_id is not None else txns[0].user_id,
        currency=currency,
        period=period,
        monthly_income=monthly_income,
        fixed_expenses=monthly_fixed,
        variable_expenses=monthly_variable,
        discretionary_expenses=monthly_discretionary,
        recurring_expenses=monthly_recurring,
        total_expenses=monthly_total_expenses,
        category_spending=category_spending,
        transfers_total=monthly_transfers,
        savings_transfers_total=monthly_savings_transfers,
        refunds_total=monthly_refunds,
        monthly_savings=monthly_savings,
        savings_rate=savings_rate,
        total_debt=total_debt,
        monthly_debt_payments=monthly_debt_payments,
        monthly_net_cash_flow=monthly_savings,
        liquid_buffer=liquid_buffer,
        months_of_buffer=months_of_buffer,
        generated_at=generated_at,
    )


def expense_ratios(profile: FinancialProfile) -> dict[str, float | None]:
    """
    Derived expense ratios (share of total monthly outflow).

    Purely derived from the profile — returns ``None`` components when the
    profile has no observed expenses (avoids divide-by-zero and keeps
    missing information explicit).
    """
    total = profile.total_expenses
    if total is None or total <= 0:
        return {"fixed": None, "variable": None, "discretionary": None}
    return {
        "fixed": profile.fixed_expenses / total,
        "variable": profile.variable_expenses / total,
        "discretionary": profile.discretionary_expenses / total,
    }