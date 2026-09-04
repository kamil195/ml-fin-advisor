"""
Thin API adapter for the DecisionEngine.

    POST /consumer/advise

Bridges the serving layer to the deterministic orchestration service
(``src.services.decision_engine.advise``) without changing any existing
contract:

* accepts the richer *classified* transaction payload (the existing
  ``Transaction`` schema fields plus the classifier's ``confidence``);
* applies the confidence gate before anything reaches the financial layer;
* converts gated payloads into existing ``Transaction`` objects;
* delegates all financial arithmetic to the existing services and returns
  the existing :class:`DecisionResult` contract.

Confidence gate (deterministic policy; threshold 0.85):

* ``confidence >= 0.85`` → the classifier's category is used as-is.
* ``confidence <  0.85`` → the category is cleared unless the transaction
  carries an *authoritative* signal that is input data rather than a
  classifier guess:

  - ``Channel.TRANSFER``               (unlabelled = internal transfer,
                                        labelled = semantic rail, e.g. rent);
  - ``CategoryL2.SAVINGS_INVESTMENTS`` (savings/investment movement);
  - ``CategoryL2.INCOME`` on a credit  (observed income).

Cleared transactions flow into the existing FinancialProfile semantics:
debits → ``Uncategorized`` (visible and counted but never a cut target);
unlabelled credits → refunds (never invented income).

No randomness, no wall-clock reads, no LLM calls: identical requests
produce identical responses.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from src.data.models import DecisionResult, ScenarioParams, Transaction
from src.services.decision_engine import advise as advise_engine
from src.utils.constants import (
    CATEGORY_HIERARCHY,
    AccountType,
    CategoryL1,
    CategoryL2,
    Channel,
)

router = APIRouter(prefix="/consumer", tags=["Decision Engine"])

# Confidence gate policy (see module docstring).
CONFIDENCE_GATE_THRESHOLD = 0.85


class ClassifiedTransactionIn(BaseModel):
    """One classified transaction: the existing ``Transaction`` contract + ``confidence``."""

    user_id: str
    timestamp: datetime
    amount: float = Field(
        ..., description="Signed amount. Negative = debit, positive = credit."
    )
    currency: str = "USD"
    merchant_name: str = Field(..., min_length=1)
    merchant_mcc: int = Field(..., ge=0, le=9999)
    account_type: AccountType
    channel: Channel
    location_city: str | None = None
    location_country: str = "US"
    raw_description: str = ""
    is_pending: bool = False
    category_l1: CategoryL1 | None = None
    category_l2: CategoryL2 | None = None
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence (max predicted probability) from /v1/classify.",
    )

    @model_validator(mode="after")
    def validate_category_hierarchy(self) -> ClassifiedTransactionIn:
        """Mirror Transaction's hierarchy rule so mismatches fail as 422, not 500."""
        if self.category_l1 is not None and self.category_l2 is not None:
            expected_l1 = CATEGORY_HIERARCHY.get(self.category_l2)
            if expected_l1 != self.category_l1:
                raise ValueError(
                    f"Category mismatch: L2 '{self.category_l2.value}' belongs to "
                    f"'{expected_l1.value if expected_l1 else 'unknown'}', "
                    f"not '{self.category_l1.value}'."
                )
        return self


class AdviseRequest(BaseModel):
    """Request body for ``POST /consumer/advise``."""

    transactions: list[ClassifiedTransactionIn] = Field(
        ...,
        min_length=1,
        description="Classified transaction history for the aggregation window.",
    )
    params: ScenarioParams = Field(
        default_factory=ScenarioParams,
        description="Scenario assumptions (default = identity/no-change scenario).",
    )
    user_id: str | None = None
    period: str | None = None
    observation_days: int | None = Field(default=None, ge=1)
    liquid_buffer: float | None = Field(default=None, ge=0)
    total_debt: float | None = Field(default=None, ge=0)
    monthly_debt_payments: float | None = Field(default=None, ge=0)
    habit_strengths: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional per-category habit strengths [0, 1] forwarded to the "
            "behavioral feasibility screen."
        ),
    )
    compliance_history: dict[str, float] | None = None


def _is_authoritative(p: ClassifiedTransactionIn) -> bool:
    """True when the low-confidence payload carries an input-side (not predicted) signal."""
    return (
        p.channel == Channel.TRANSFER
        or p.category_l2 == CategoryL2.SAVINGS_INVESTMENTS
        or (p.category_l2 == CategoryL2.INCOME and p.amount > 0)
    )


def _to_gated_transaction(p: ClassifiedTransactionIn) -> Transaction:
    """Apply the confidence gate and convert to the existing ``Transaction`` contract."""
    fields = p.model_dump(exclude={"confidence"})
    if p.confidence < CONFIDENCE_GATE_THRESHOLD and not _is_authoritative(p):
        fields["category_l1"] = None
        fields["category_l2"] = None
    return Transaction(**fields)


@router.post("/advise", response_model=DecisionResult)
def advise_endpoint(req: AdviseRequest) -> DecisionResult:
    """Deterministic financial decision for the posted classified history."""
    transactions = [_to_gated_transaction(p) for p in req.transactions]
    return advise_engine(
        transactions,
        req.params,
        user_id=req.user_id,
        period=req.period,
        observation_days=req.observation_days,
        liquid_buffer=req.liquid_buffer,
        total_debt=req.total_debt,
        monthly_debt_payments=req.monthly_debt_payments,
        habit_strengths=req.habit_strengths,
        compliance_history=req.compliance_history,
    )