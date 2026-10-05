"""
Pydantic data models for the ML Fin-Advisor project.

Implements the raw transaction schema from SPEC §5.1 and related models.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.constants import (
    AccountType,
    CategoryL1,
    CategoryL2,
    Channel,
)


# ── Raw Transaction Schema (SPEC §5.1) ────────────────────────────────────────


class Transaction(BaseModel):
    """
    A single financial transaction as ingested from bank feeds, CSV uploads,
    or manual entry.

    Mirrors the schema defined in SPEC §5.1. Amounts are signed:
    negative = debit (money out), positive = credit (money in).
    """

    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transaction identifier (UUID).",
    )
    user_id: str = Field(
        ...,
        description="Unique user identifier (UUID).",
    )
    timestamp: datetime = Field(
        ...,
        description="Transaction timestamp in UTC.",
    )
    amount: float = Field(
        ...,
        description="Signed amount. Negative = debit, positive = credit.",
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code.",
    )
    merchant_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Merchant or payee name.",
    )
    merchant_mcc: int = Field(
        ...,
        ge=0,
        le=9999,
        description="4-digit Merchant Category Code (ISO 18245).",
    )
    account_type: AccountType = Field(
        ...,
        description="Type of account the transaction belongs to.",
    )
    channel: Channel = Field(
        ...,
        description="Transaction channel (POS, ONLINE, ATM, TRANSFER, RECURRING).",
    )
    location_city: str | None = Field(
        default=None,
        description="City where the transaction occurred (nullable).",
    )
    location_country: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    raw_description: str = Field(
        default="",
        max_length=2000,
        description="Raw transaction description from bank feed.",
    )
    is_pending: bool = Field(
        default=False,
        description="Whether the transaction is still pending settlement.",
    )

    # ── Optional label (populated after classification or user correction) ──

    category_l1: CategoryL1 | None = Field(
        default=None,
        description="Level-1 category label (assigned by classifier or user).",
    )
    category_l2: CategoryL2 | None = Field(
        default=None,
        description="Level-2 category label (assigned by classifier or user).",
    )

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("location_country")
    @classmethod
    def country_uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def validate_category_consistency(self) -> Transaction:
        """If both L1 and L2 are set, ensure L2 belongs to L1."""
        from src.utils.constants import CATEGORY_HIERARCHY

        if self.category_l1 is not None and self.category_l2 is not None:
            expected_l1 = CATEGORY_HIERARCHY.get(self.category_l2)
            if expected_l1 != self.category_l1:
                raise ValueError(
                    f"Category mismatch: L2 '{self.category_l2.value}' belongs to "
                    f"'{expected_l1.value if expected_l1 else 'unknown'}', "
                    f"not '{self.category_l1.value}'."
                )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "user_id": "u-00000001-0000-0000-0000-000000000001",
                    "timestamp": "2025-11-15T14:32:00Z",
                    "amount": -42.50,
                    "currency": "USD",
                    "merchant_name": "Whole Foods Market",
                    "merchant_mcc": 5411,
                    "account_type": "CHECKING",
                    "channel": "POS",
                    "location_city": "San Francisco",
                    "location_country": "US",
                    "raw_description": "WHOLE FOODS MKT #10234 SAN FRANCISCO CA",
                    "is_pending": False,
                    "category_l1": "FOOD & DINING",
                    "category_l2": "Groceries",
                }
            ]
        }
    }


# ── Batch Wrapper ──────────────────────────────────────────────────────────────


class TransactionBatch(BaseModel):
    """A batch of transactions for bulk ingestion."""

    transactions: list[Transaction] = Field(
        ...,
        min_length=1,
        description="List of transactions in the batch.",
    )
    source: str = Field(
        default="unknown",
        description="Source identifier (e.g., 'plaid', 'csv_upload', 'manual').",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the batch was ingested.",
    )

    @property
    def size(self) -> int:
        return len(self.transactions)

    @property
    def user_ids(self) -> set[str]:
        return {t.user_id for t in self.transactions}


# ── Classification Result ──────────────────────────────────────────────────────


class CategoryPrediction(BaseModel):
    """Single category prediction with confidence."""

    category: CategoryL2
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class ClassificationResult(BaseModel):
    """
    Output of the transaction classifier (SPEC §11.2.1).
    """

    category_l1: CategoryL1
    category_l2: CategoryL2
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    top_3: list[CategoryPrediction] = Field(max_length=3)
    is_impulse: bool = False
    impulse_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


# ── Fraud Analysis ────────────────────────────────────────────────────────────


class FraudAnalysis(BaseModel):
    """
    Fraud-risk assessment for a single transaction (SPEC §11.2.4).

    Built from velocity features (rolling transaction counts over short
    windows) plus transparent rule-based heuristics on amount, channel and
    time of day. No opaque model — every flag maps to a specific signal.
    """

    fraud_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    velocity_flags: list[str] = Field(
        default_factory=list,
        description="Human-readable velocity/anomaly signals detected.",
    )
    is_suspicious: bool = Field(
        default=False,
        description="True when fraud_score crosses the suspicion threshold.",
    )


# ── Forecast Result ────────────────────────────────────────────────────────────


class CategoryForecast(BaseModel):
    """Probabilistic forecast for a single category (SPEC §11.2.2)."""

    category: CategoryL2
    p10: float = Field(description="10th percentile forecast")
    p50: float = Field(description="Median (50th percentile) forecast")
    p90: float = Field(description="90th percentile forecast")
    trend: str = Field(description="Trend direction: stable, increasing, decreasing")
    regime: str = Field(description="Current spending regime: normal, elevated, reduced, irregular")


class ForecastResult(BaseModel):
    """Full forecast response for a user.

    STEP 13K — truthfulness: this endpoint replays the **global/static**
    training artifact, so every label below states that explicitly. A client can
    never mistake this response for a personalized forecast, and the
    personalized contract lives in :class:`PersonalForecastResult`.
    """

    user_id: str
    generated_at: datetime
    horizon_days: int
    forecasts: list[CategoryForecast]
    total_spend: dict[str, float] = Field(
        description="Aggregate spend forecast: p10, p50, p90"
    )
    personalization_status: str = Field(
        default="not_personalized",
        description="Always 'not_personalized' here: output comes from a global artifact.",
    )
    fallback_status: str = Field(
        default="global_reference",
        description="Marks this output explicitly as a global (non-personal) reference.",
    )
    method: str = Field(
        default="global_static_artifact",
        description="This endpoint replays a static training artifact; no per-user model runs.",
    )
    engine_version: str = Field(
        default="global-static-v1",
        description="Identifier of the static artifact pipeline.",
    )
    history_days_used: int = Field(
        default=0, description="Always 0 here: no user transaction history is used."
    )
    transaction_count_used: int = Field(
        default=0, description="Always 0 here: no user transaction history is used."
    )


class PersonalForecastDaily(BaseModel):
    """One day of a user-specific forecast (STEP 13A)."""

    date: str = Field(description="ISO date (YYYY-MM-DD).")
    p50: float = Field(description="Expected spend for the day.")
    p10: float = Field(description="Lower bound of the 80% interval.")
    p90: float = Field(description="Upper bound of the 80% interval.")


class PersonalForecastResult(BaseModel):
    """User-specific spend forecast built from the caller's own history.

    ``personalization_status`` is authoritative and honest: ``personalized``
    (Tier A), ``limited_history`` (Tier B) or ``insufficient_history`` (Tier C,
    where no numeric forecast is produced and ``requirements`` explains what is
    needed). Nothing global is ever substituted for a user's own data.
    """

    user_id: str
    generated_at: datetime
    engine_version: str = Field(description="Forecast engine identifier (cache/reproducibility).")
    method: str = Field(description="Method that produced this forecast.")
    personalization_status: str = Field(
        description="personalized | limited_history | insufficient_history"
    )
    fallback_status: str = Field(
        description=(
            "none | user_recent_baseline | primary_substituted — every value is "
            "user-specific; a global fallback is never used."
        )
    )
    forecast_start: str | None = Field(default=None, description="First forecast day (ISO).")
    forecast_end: str | None = Field(default=None, description="Last forecast day (ISO).")
    horizon_days: int
    interval_width: float = Field(description="Width of the reported interval (0.80).")
    expected_spend: float | None = Field(
        default=None, description="Total expected spend over the horizon (None when insufficient)."
    )
    total: dict[str, float] = Field(description="Horizon totals: p10, p50, p90.")
    daily: list[PersonalForecastDaily] = Field(default_factory=list)
    history_days_used: int = Field(description="Days of the caller's own history used.")
    transaction_count_used: int = Field(description="Caller's own spend transactions used.")
    distinct_spend_days_used: int
    residual_sigma: float = Field(description="Measured one-step residual dispersion.")
    message: str | None = Field(default=None, description="Non-misleading note when limited.")
    requirements: dict[str, object] | None = Field(
        default=None, description="What is needed when history is insufficient."
    )


# ── Safe to Spend (STEP 14) ───────────────────────────────────────────────────
# Deterministic decision-support output: the arithmetic is authoritative on the
# backend (never recomputed in frontend JavaScript) and every component is
# returned so a client can show the exact breakdown. No accuracy, safety or
# "guaranteed" claim is made about the number.


class SafeToSpendComponent(BaseModel):
    """One line of the deterministic Safe-to-Spend breakdown."""

    label: str = Field(description="Human-readable component name.")
    amount: float | None = Field(
        default=None, description="Component amount in the response currency."
    )
    sign: str = Field(description="'+' for the funds term, '−' for subtractions.")
    source: str = Field(description="Where the component comes from.")


class SafeToSpendObligation(BaseModel):
    """One protected category and what the caller's own history implies."""

    category: str = Field(description="Protected category (CategoryL2 value).")
    included: bool = Field(description="True when counted in this result.")
    reason: str = Field(
        description=(
            "included | insufficient_observations | irregular_cadence | "
            "already_paid_this_cycle | due_after_payday"
        )
    )
    amount: float | None = Field(default=None, description="Counted amount.")
    due_date: str | None = Field(default=None, description="Predicted due date (ISO).")
    observations: int = Field(description="Payments observed in the caller's history.")
    median_gap_days: int | None = Field(default=None, description="Measured cadence.")
    last_paid_on: str | None = Field(default=None, description="Last observed payment.")


class SafeToSpendScenario(BaseModel):
    """Before/after view of an explicit, non-persisted scenario adjustment."""

    label: str | None = Field(default=None, description="Scenario label, echoed back.")
    amount: float = Field(description="Amount subtracted by the scenario.")
    safe_to_spend_before: float | None
    safe_to_spend_after: float | None
    delta: float | None = Field(
        default=None, description="before − after (equal to the amount)."
    )
    persisted: bool = Field(
        default=False, description="Always false: scenarios never modify stored data."
    )


class SafeToSpendScenarioRequest(BaseModel):
    """Optional 'what if I spend X before payday?' request body."""

    scenario_amount: float = Field(
        ...,
        ge=0,
        le=1_000_000_000,
        description="Planned spend before payday (same currency as the response).",
    )
    label: str | None = Field(
        default=None, max_length=120, description="Optional label, e.g. 'weekend trip'."
    )


class SafeToSpendResult(BaseModel):
    """Deterministic, explainable Safe-to-Spend response (STEP 14).

    ``status`` is authoritative and honest. Amounts are present only for
    ``ready`` / ``limited_history``; every other status returns no number plus a
    ``requirements`` block saying what is missing. ``explanation`` is a templated
    breakdown of the arithmetic — it is never produced by a language model.
    """

    status: str = Field(
        description=(
            "ready | limited_history | insufficient_history | missing_payday | "
            "missing_balance | missing_profile_data"
        )
    )
    as_of: str = Field(description="Computation date (ISO), the 'today' anchor.")
    currency: str | None = Field(
        default=None,
        description="Currency of every money field (from your own data; None if unobserved).",
    )
    next_payday: str | None = Field(default=None, description="Horizon end (ISO date).")
    days_to_payday: int | None = Field(default=None, description="Days from today.")
    current_available_funds: float | None = Field(
        default=None, description="Your saved cash/liquid position (liquid_buffer)."
    )
    protected_obligations: float | None = Field(
        default=None, description="Protected obligations due before payday."
    )
    protected_obligations_detail: list[SafeToSpendObligation] = Field(
        default_factory=list, description="Every protected category and its reason."
    )
    expected_spending_before_payday: float | None = Field(
        default=None,
        description="Your own forecast of non-protected spending before payday (p50).",
    )
    expected_spending_p10: float | None = Field(default=None)
    expected_spending_p90: float | None = Field(default=None)
    safety_buffer: float | None = Field(
        default=None, description="Reserve held back (0.00 when not configured)."
    )
    safety_buffer_source: str = Field(
        description="profile | not_configured (an unconfigured buffer is never invented)."
    )
    scenario_adjustment: float = Field(
        default=0.0, description="Amount subtracted by the requested scenario."
    )
    scenario: SafeToSpendScenario | None = Field(
        default=None, description="Before/after view when a scenario was supplied."
    )
    safe_to_spend: float | None = Field(
        default=None,
        description=(
            "The amount, or None when an input/history is missing. Can be "
            "negative — it is never floored, and is_negative flags a shortfall."
        ),
    )
    is_negative: bool = Field(
        default=False, description="True when safe_to_spend is a shortfall (< 0)."
    )
    forecast_status: str = Field(
        description="STEP 13 status of the user-specific forecast used (personalized/limited/insufficient)."
    )
    forecast_method: str = Field(
        description="Forecasting method used; the global/static forecast is never used here."
    )
    forecast_fallback_status: str = Field(
        description="STEP 13 fallback label (none | user_recent_baseline | primary_substituted)."
    )
    payday_source: str | None = Field(
        default=None, description="profile | measured_from_income_history | None."
    )
    history_days_used: int = Field(description="Days of your own non-obligation history used.")
    transaction_count_used: int = Field(description="Your own spend transactions used.")
    distinct_spend_days_used: int
    excluded_pending_transactions: int = Field(
        description="Pending rows excluded from history, obligations and the forecast."
    )
    breakdown: list[SafeToSpendComponent] = Field(
        default_factory=list, description="Exact component-by-component arithmetic."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Every assumption used, stated explicitly."
    )
    requirements: dict[str, object] | None = Field(
        default=None, description="What is needed to produce an amount (when missing)."
    )
    explanation: str = Field(description="Templated, deterministic breakdown text.")
    engine_version: str = Field(description="Engine identifier (cache/reproducibility).")
    message: str | None = Field(default=None, description="Optional clarifying note.")


# ── Budget Recommendation ─────────────────────────────────────────────────────


class SHAPFeature(BaseModel):
    """A single SHAP feature-impact pair."""

    feature: str
    impact: float


class BudgetRecommendation(BaseModel):
    """Single category budget recommendation with explanation (SPEC §11.2.3)."""

    category: CategoryL2
    recommended_budget: float
    current_trend: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    explanation: str
    shap_top_features: list[SHAPFeature] = Field(default_factory=list)
    anchor_rule: str = ""
    counterfactual: str = ""


class BudgetResult(BaseModel):
    """Full budget recommendation response for a user."""

    user_id: str
    period: str = Field(description="Budget period, e.g. '2026-03'")
    income_estimate: float
    savings_target: float
    recommendations: list[BudgetRecommendation]


# ── Validation Report ──────────────────────────────────────────────────────────


class ValidationIssue(BaseModel):
    """A single data validation issue."""

    row_index: int | None = None
    field: str
    issue: str
    severity: str = Field(description="'error' or 'warning'")


class ValidationReport(BaseModel):
    """Result of validating a transaction batch."""

    total_rows: int
    valid_rows: int
    invalid_rows: int
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.invalid_rows == 0


# ── Canonical Planning Contracts (Financial Intelligence layer) ───────────────
#
# Minimal, extensible schemas for the upcoming product layer:
#
#   Transactions → Classification → FinancialProfile
#     → Behaviour / Forecast / Budget → Scenario → Decision
#     → AI Copilot → Outcome Tracking
#
# Contracts only — no route wiring or engine yet. Money fields are *monthly*
# amounts in the profile currency unless stated otherwise.


class FinancialProfile(BaseModel):
    """
    Canonical, deterministic snapshot of a user's monthly financial state.

    Aggregation input (transactions) and consumers (behaviour, forecast,
    budget, scenario, decision) all speak this one representation.
    Expense components are non-overlapping: ``discretionary_expenses`` is a
    subset of ``variable_expenses``.
    """

    user_id: str = Field(description="Owner of the profile.")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    period: str | None = Field(
        default=None,
        description="Label of the aggregation window, e.g. '2026-03' or 'last_90d'.",
    )

    # ── Income & expenses ────────────────────────────────────────────────
    monthly_income: Annotated[float, Field(ge=0)]
    fixed_expenses: Annotated[float, Field(ge=0)] = 0.0
    variable_expenses: Annotated[float, Field(ge=0)] = 0.0
    discretionary_expenses: Annotated[float, Field(ge=0)] = 0.0
    recurring_expenses: Annotated[float, Field(ge=0)] = 0.0
    total_expenses: float | None = Field(
        default=None,
        description=(
            "Total monthly outflow. Optional — consumers may derive it as "
            "fixed + variable when omitted."
        ),
    )
    category_spending: dict[str, float] = Field(
        default_factory=dict,
        description="Average monthly spend by category name (e.g. CategoryL2 values).",
    )

    # ── Savings, debt & obligations ──────────────────────────────────────
    monthly_savings: float | None = Field(
        default=None,
        description="Monthly amount saved (may be negative when outflow exceeds income).",
    )
    savings_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="monthly_savings / monthly_income as a fraction [0, 1].",
    )
    total_debt: Annotated[float, Field(ge=0)] | None = None
    monthly_debt_payments: Annotated[float, Field(ge=0)] | None = None

    # ── Cash-flow & resilience ───────────────────────────────────────────
    monthly_net_cash_flow: float | None = Field(
        default=None,
        description="Inflow − outflow per month; negative indicates a deficit.",
    )
    liquid_buffer: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="Cash/liquid assets available today.",
    )
    months_of_buffer: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="liquid_buffer ÷ total monthly expenses (resilience).",
    )

    # ── Money movement (informational, monthly-normalised, gross) ────────
    transfers_total: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description=(
            "Gross internal account-to-account transfers observed in the "
            "window (unlabelled Channel.TRANSFER). Excluded from income "
            "and expenses."
        ),
    )
    savings_transfers_total: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description=(
            "Gross movements to/from savings & investments (either "
            "direction). Excluded from income and expenses."
        ),
    )
    refunds_total: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description=(
            "Gross credits that are not classified income (refunds, "
            "reversals, unidentified credits). Excluded from income; "
            "reported separately rather than netted against expenses."
        ),
    )

    generated_at: datetime | None = None

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper()


class BehavioralProfile(BaseModel):
    """
    Interpretable behavioural summary for a user.

    Deliberately excludes raw model internals (posteriors, changepoint
    distributions): every field is a human-readable signal the decision and
    AI layers can consume directly.
    """

    user_id: str
    generated_at: datetime | None = None
    observation_days: Annotated[int, Field(ge=1)] | None = Field(
        default=None,
        description="Length of the history window the profile summarises.",
    )

    spending_regime: Literal["normal", "elevated", "reduced", "irregular"] | None = Field(
        default=None,
        description="Current spending regime (same vocabulary as CategoryForecast.regime).",
    )
    discretionary_ratio: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Share of spending that is discretionary [0, 1].",
    )
    impulse_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    spending_volatility: Annotated[float, Field(ge=0.0)] | None = Field(
        default=None,
        description="Spending variability (e.g. coefficient of variation; unitless).",
    )
    savings_behavior_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="0 = spends everything, 1 = consistently saves.",
    )
    habit_strength: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict,
        description="Per-category habit strength [0, 1] (regularity × consistency × duration).",
    )

    top_patterns: list[str] = Field(
        default_factory=list,
        description="Human-readable behaviour summaries (for the AI copilot).",
    )
    warnings: list[str] = Field(default_factory=list)


class ScenarioParams(BaseModel):
    """
    User-selectable assumptions for a what-if scenario.

    Percentage changes are relative to the baseline FinancialProfile; a
    default-constructed instance is the identity (no-change) scenario.
    """

    scenario_id: str | None = None
    name: str | None = None
    horizon_months: Annotated[int, Field(ge=1, le=36)] = 1

    income_change_pct: float = Field(
        default=0.0,
        ge=-100.0,
        description="Relative income change; −100 eliminates income, +25 is a raise.",
    )
    expense_change_pct: float = Field(
        default=0.0,
        ge=-100.0,
        description="Relative change applied to total expenses.",
    )
    savings_target: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="Absolute monthly savings target in currency units (overrides rate).",
    )
    category_changes: dict[str, float] = Field(
        default_factory=dict,
        description="Category name → % change in that category's monthly spend.",
    )

    @field_validator("category_changes")
    @classmethod
    def validate_category_changes(cls, v: dict[str, float]) -> dict[str, float]:
        for cat, pct in v.items():
            if pct < -100.0:
                raise ValueError(
                    f"category_changes[{cat!r}] = {pct}: cannot reduce below −100%."
                )
        return v


class MetricChange(BaseModel):
    """A single before/after delta produced by a scenario."""

    metric: str = Field(description="FinancialProfile field name, e.g. 'monthly_income'.")
    before: float | None = None
    after: float | None = None
    delta: float | None = None


class ScenarioResult(BaseModel):
    """Outcome of applying ScenarioParams to a baseline FinancialProfile."""

    scenario_id: str | None = None
    params: ScenarioParams | None = None
    status: Literal["feasible", "infeasible", "partial"] = Field(
        description=(
            "feasible = applies cleanly; infeasible = violates constraints; "
            "partial = applied with adjustments (see warnings)."
        )
    )
    resulting_profile: FinancialProfile = Field(
        description="Projected financial state after the scenario.",
    )
    key_changes: list[MetricChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DecisionResult(BaseModel):
    """
    A single, explainable recommendation produced by the decision layer.

    Deterministic fields (recommendation, reasoning, metrics) are computed by
    rules/optimisers; ``confidence`` is optional so purely rule-based
    decisions remain representable.
    """

    decision_id: str | None = None
    scenario_id: str | None = None
    decision_type: str = Field(
        description="Coarse kind of decision, e.g. 'budget_adjustment', 'savings_goal'.",
    )
    recommendation: str = Field(
        description="Primary human-readable action to take.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of why this decision was reached.",
    )
    supporting_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics that back the recommendation (name → value).",
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    alternatives: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime | None = None
